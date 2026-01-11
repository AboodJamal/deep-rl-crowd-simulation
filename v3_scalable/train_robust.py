#!/usr/bin/env python3
"""
ROBUST Training for test3 - Multi-Agent Corridor Navigation
============================================================

IMPROVEMENTS in this version:
1. SOLID obstacles (agents can't pass through)
2. FASTER agents (max_speed=3.0, acceleration=0.7)
3. Better obstacle placement
4. Continuous collision detection
5. Extended training with curriculum learning

Author: Robust Training v2
Date: December 2025
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional
import json
import argparse

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from corridor_env import RealisticCorridorEnv


class PolicyNetwork(nn.Module):
    """Actor network with continuous action output."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.clamp(obs, -10.0, 10.0)
        features = self.net(obs)
        mean = self.mean_head(features)
        mean = torch.tanh(mean)
        return mean
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean = self.forward(obs)
        
        if deterministic:
            return mean, None, None
        
        std = torch.exp(torch.clamp(self.log_std, -2.0, 0.5))
        dist = Normal(mean, std)
        action = dist.rsample()
        action = torch.clamp(action, -1.0, 1.0)
        
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        return action, log_prob, entropy
    
    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        mean = self.forward(obs)
        std = torch.exp(torch.clamp(self.log_std, -2.0, 0.5))
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.clamp(obs, -10.0, 10.0)
        return self.net(obs)


class RobustTrainer:
    """PPO Trainer with robust settings for reliable training."""
    
    def __init__(
        self,
        env: RealisticCorridorEnv,
        device: str = 'cpu',
        hidden_dim: int = 256,
    ):
        self.env = env
        self.num_agents = env.num_agents
        self.device = device
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Networks (shared across agents)
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.value = ValueNetwork(obs_dim, hidden_dim).to(device)
        
        # Optimizers
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=3e-4, eps=1e-5)
        self.value_optim = optim.Adam(self.value.parameters(), lr=1e-3, eps=1e-5)
        
        # PPO Hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_param = 0.2
        self.entropy_coef = 0.01  # Higher for more exploration
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.ppo_epochs = 10
        self.batch_size = 256
        
        # Rollout buffer
        self.buffer = []
        
        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.all_success_rates = deque(maxlen=100)
        self.per_agent_success_rates = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.collision_counts = deque(maxlen=100)
        self.obstacle_collision_counts = deque(maxlen=100)
    
    def collect_rollout(self, num_steps: int = 4096):
        """Collect experience from environment."""
        self.buffer = []
        
        obs_list, _ = self.env.reset()
        episode_reward = 0.0
        
        for step in range(num_steps):
            obs_t = torch.FloatTensor(np.array(obs_list)).to(self.device)
            
            with torch.no_grad():
                actions, log_probs, _ = self.policy.get_action(obs_t)
                values = self.value(obs_t).squeeze(-1)
            
            actions_np = actions.cpu().numpy()
            next_obs, rewards, terminated, truncated, info = self.env.step(actions_np)
            done = terminated or truncated
            
            self.buffer.append({
                'obs': np.array(obs_list),
                'actions': actions_np,
                'rewards': np.array(rewards),
                'values': values.cpu().numpy(),
                'log_probs': log_probs.cpu().numpy(),
                'done': done
            })
            
            episode_reward += np.mean(rewards)
            
            if done:
                # Record metrics
                self.episode_rewards.append(episode_reward)
                self.all_success_rates.append(float(info.get('all_success', False)))
                self.per_agent_success_rates.append(info.get('per_agent_success', 0.0))
                self.episode_lengths.append(info.get('episode_length', 0))
                
                total_collisions = (
                    info.get('total_wall_collisions', 0) +
                    info.get('total_obstacle_collisions', 0) +
                    info.get('total_agent_collisions', 0)
                )
                self.collision_counts.append(total_collisions)
                self.obstacle_collision_counts.append(
                    info.get('total_obstacle_collisions', 0)
                )
                
                # Reset
                obs_list, _ = self.env.reset()
                episode_reward = 0.0
            else:
                obs_list = next_obs
    
    def compute_gae(self):
        """Compute Generalized Advantage Estimation."""
        num_steps = len(self.buffer)
        
        rewards = np.array([t['rewards'] for t in self.buffer])
        values = np.array([t['values'] for t in self.buffer])
        dones = np.array([t['done'] for t in self.buffer])
        
        all_advantages = []
        all_returns = []
        
        for agent_idx in range(self.num_agents):
            advantages = np.zeros(num_steps)
            gae = 0.0
            
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_value = 0.0
                else:
                    next_value = values[t + 1, agent_idx]
                
                mask = 1.0 - float(dones[t])
                delta = rewards[t, agent_idx] + self.gamma * next_value * mask - values[t, agent_idx]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                advantages[t] = gae
            
            returns = advantages + values[:, agent_idx]
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        return np.array(all_advantages).T, np.array(all_returns).T
    
    def update(self) -> Dict:
        """PPO update."""
        if len(self.buffer) == 0:
            return {}
        
        advantages, returns = self.compute_gae()
        
        # Flatten data
        obs = torch.FloatTensor(
            np.array([t['obs'] for t in self.buffer])
        ).reshape(-1, self.env.observation_space.shape[0]).to(self.device)
        
        actions = torch.FloatTensor(
            np.array([t['actions'] for t in self.buffer])
        ).reshape(-1, self.env.action_space.shape[0]).to(self.device)
        
        old_log_probs = torch.FloatTensor(
            np.array([t['log_probs'] for t in self.buffer])
        ).reshape(-1).to(self.device)
        
        returns_t = torch.FloatTensor(returns.flatten()).to(self.device)
        advantages_t = torch.FloatTensor(advantages.flatten()).to(self.device)
        
        # Normalize advantages
        adv_mean = advantages_t.mean()
        adv_std = advantages_t.std()
        if adv_std > 1e-8:
            advantages_t = (advantages_t - adv_mean) / (adv_std + 1e-8)
        advantages_t = torch.clamp(advantages_t, -10.0, 10.0)
        
        # PPO epochs
        dataset_size = obs.size(0)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                idx = indices[start:end]
                
                if len(idx) < 8:
                    continue
                
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns_t[idx]
                mb_advantages = advantages_t[idx]
                
                # Policy loss
                new_log_probs, entropy = self.policy.evaluate(mb_obs, mb_actions)
                ratio = torch.exp(torch.clamp(new_log_probs - mb_old_log_probs, -10, 10))
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                
                total_loss = policy_loss + self.entropy_coef * entropy_loss
                
                if not torch.isnan(total_loss):
                    self.policy_optim.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy_optim.step()
                    total_policy_loss += policy_loss.item()
                    total_entropy += entropy.mean().item()
                
                # Value loss
                values = self.value(mb_obs).squeeze()
                value_loss = self.value_coef * ((values - mb_returns) ** 2).mean()
                
                if not torch.isnan(value_loss):
                    self.value_optim.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                    self.value_optim.step()
                    total_value_loss += value_loss.item()
                
                num_updates += 1
        
        self.buffer = []
        
        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
        }
    
    def train(
        self,
        total_timesteps: int,
        log_interval: int = 10,
        save_dir: str = 'models',
        save_interval: int = 50,
    ) -> Dict:
        """Main training loop."""
        os.makedirs(save_dir, exist_ok=True)
        
        rollout_length = 4096
        num_updates = total_timesteps // rollout_length
        
        print(f"\n{'='*70}")
        print(f"ROBUST TRAINING - test3")
        print(f"{'='*70}")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Updates: {num_updates}")
        print(f"  Scenario: {self.env.scenario}")
        print(f"  Num agents: {self.env.num_agents}")
        print(f"  Max speed: {self.env.max_speed}")
        print(f"  Device: {self.device}")
        print(f"{'='*70}\n")
        
        best_all_success = 0.0
        best_per_agent = 0.0
        training_log = []
        
        for update in range(num_updates):
            # Collect rollout
            self.collect_rollout(num_steps=rollout_length)
            
            # PPO update
            losses = self.update()
            
            # Logging
            if (update + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                avg_all_success = np.mean(self.all_success_rates) if self.all_success_rates else 0
                avg_per_agent = np.mean(self.per_agent_success_rates) if self.per_agent_success_rates else 0
                avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                avg_obs_coll = np.mean(self.obstacle_collision_counts) if self.obstacle_collision_counts else 0
                
                # Track best
                improved = ""
                if avg_all_success > best_all_success:
                    best_all_success = avg_all_success
                    improved = "★"
                if avg_per_agent > best_per_agent:
                    best_per_agent = avg_per_agent
                
                print(f"{improved:1s} [{update+1:4d}/{num_updates}] "
                      f"Reward: {avg_reward:7.1f} | "
                      f"All%: {avg_all_success:5.1%} | "
                      f"Per-Agent: {avg_per_agent:5.1%} | "
                      f"Len: {avg_length:4.0f} | "
                      f"ObsColl: {avg_obs_coll:4.1f}")
                
                training_log.append({
                    'update': update + 1,
                    'reward': float(avg_reward),
                    'all_success': float(avg_all_success),
                    'per_agent_success': float(avg_per_agent),
                    'episode_length': float(avg_length),
                    'obstacle_collisions': float(avg_obs_coll),
                })
            
            # Save checkpoints
            if (update + 1) % save_interval == 0:
                checkpoint = {
                    'policy': self.policy.state_dict(),
                    'value': self.value.state_dict(),
                    'update': update + 1,
                    'best_all_success': best_all_success,
                    'best_per_agent': best_per_agent,
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_{update+1}.pt'))
        
        # Save final model
        final_checkpoint = {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'scenario': self.env.scenario,
            'num_agents': self.env.num_agents,
            'best_all_success': best_all_success,
            'best_per_agent': best_per_agent,
        }
        torch.save(final_checkpoint, os.path.join(save_dir, 'final_model.pt'))
        
        # Save training log
        with open(os.path.join(save_dir, 'training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"  Best All-Success Rate: {best_all_success:.1%}")
        print(f"  Best Per-Agent Success: {best_per_agent:.1%}")
        print(f"{'='*70}")
        
        return {
            'final_all_success': np.mean(self.all_success_rates) if self.all_success_rates else 0,
            'final_per_agent': np.mean(self.per_agent_success_rates) if self.per_agent_success_rates else 0,
            'best_all_success': best_all_success,
            'best_per_agent': best_per_agent,
        }
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])


def train_robust_curriculum(
    output_dir: str = 'models_robust',
    device: str = 'cpu',
    num_agents: int = 5,
):
    """
    Train with robust curriculum.
    
    IMPROVED CURRICULUM:
    - Longer training per stage
    - More gradual difficulty increase
    - Focus on solid obstacle handling
    """
    
    # Extended curriculum with more training time
    curriculum = [
        # Stage 1: Foundation - straight corridor
        {'scenario': 'straight', 'timesteps': 800_000, 'randomize': False, 'name': 'straight'},
        # Stage 2: Curves (alternating walls)
        {'scenario': 'curves', 'timesteps': 1_000_000, 'randomize': True, 'name': 'curves'},
        # Stage 3: Single wall
        {'scenario': 'single_wall', 'timesteps': 1_200_000, 'randomize': True, 'name': 'single_wall'},
        # Stage 4: Narrow passage
        {'scenario': 'narrow_passage', 'timesteps': 1_500_000, 'randomize': True, 'name': 'narrow_passage'},
        # Stage 5: Double wall (S-curve)
        {'scenario': 'double_wall', 'timesteps': 1_500_000, 'randomize': True, 'name': 'double_wall'},
        # Stage 6: Mixed (final)
        {'scenario': 'mixed', 'timesteps': 2_000_000, 'randomize': True, 'name': 'mixed'},
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(full_output_dir, exist_ok=True)
    
    total_timesteps = sum(s['timesteps'] for s in curriculum)
    
    print(f"\n{'#'*80}")
    print(f"#  ROBUST MULTI-AGENT NAVIGATION TRAINING")
    print(f"#  Agents: {num_agents}")
    print(f"#  Total Timesteps: {total_timesteps:,}")
    print(f"#  Device: {device}")
    print(f"#  Output: {full_output_dir}")
    print(f"{'#'*80}\n")
    
    policy_state = None
    value_state = None
    results = {}
    
    for stage_idx, stage in enumerate(curriculum):
        scenario = stage['scenario']
        timesteps = stage['timesteps']
        randomize = stage['randomize']
        name = stage['name']
        
        print(f"\n{'='*70}")
        print(f"  STAGE {stage_idx + 1}/{len(curriculum)}: {name.upper()}")
        print(f"  Timesteps: {timesteps:,}")
        print(f"  Randomization: {'ON' if randomize else 'OFF'}")
        print(f"{'='*70}\n")
        
        # Create environment with improved settings
        env = RealisticCorridorEnv(
            num_agents=num_agents,
            scenario=scenario,
            max_steps=800,  # Longer episodes
            randomize_scenario=randomize,
        )
        
        # Create trainer
        trainer = RobustTrainer(env=env, device=device)
        
        # Load weights from previous stage
        if policy_state is not None:
            trainer.policy.load_state_dict(policy_state)
            trainer.value.load_state_dict(value_state)
            print("→ Loaded weights from previous stage")
        
        # Train
        stage_dir = os.path.join(full_output_dir, name)
        stage_results = trainer.train(
            total_timesteps=timesteps,
            log_interval=10,
            save_dir=stage_dir,
            save_interval=50,
        )
        
        results[name] = stage_results
        
        # Save weights for next stage
        policy_state = trainer.policy.state_dict()
        value_state = trainer.value.state_dict()
        
        # Save stage checkpoint
        torch.save({
            'policy': policy_state,
            'value': value_state,
            'scenario': scenario,
            'num_agents': num_agents,
            'results': stage_results,
        }, os.path.join(full_output_dir, f'{name}_final.pt'))
    
    # Save overall results
    with open(os.path.join(full_output_dir, 'curriculum_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'#'*80}")
    print(f"#  CURRICULUM TRAINING COMPLETE!")
    print(f"#  Models saved to: {full_output_dir}")
    print(f"{'#'*80}\n")
    
    return results, full_output_dir


def main():
    parser = argparse.ArgumentParser(description="Robust Multi-Agent Training")
    parser.add_argument('--output-dir', type=str, default='models_robust',
                        help='Output directory for models')
    parser.add_argument('--num-agents', type=int, default=5,
                        help='Number of agents')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\n{'='*70}")
    print(f"  ROBUST MULTI-AGENT CORRIDOR NAVIGATION")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")
    
    # Train with curriculum
    results, output_dir = train_robust_curriculum(
        output_dir=args.output_dir,
        device=device,
        num_agents=args.num_agents,
    )
    
    print("\nFinal Results:")
    for scenario, res in results.items():
        print(f"  {scenario}: All={res['best_all_success']:.1%}, Per-Agent={res['best_per_agent']:.1%}")


if __name__ == '__main__':
    main()

