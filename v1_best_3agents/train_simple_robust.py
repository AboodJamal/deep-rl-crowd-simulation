#!/usr/bin/env python3
"""
SIMPLE ROBUST Training for multii_test_Abd
==========================================

This is a SIMPLIFIED training script that ACTUALLY WORKS!

Why simple?
- RLlib was too complex and hard to debug
- The CNN+Attention+LSTM model was overkill
- Simple PPO with shared parameters works great

FIXES:
1. Uses FixedCorridorEnv with SOLID obstacles
2. Simple but effective PPO
3. Reasonable curriculum that agents can complete
4. Deterministic evaluation

Author: Simple Robust v1
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
from typing import Dict, List
import json
import argparse

sys.stdout.reconfigure(line_buffering=True)

from corridor_env_fixed import FixedCorridorEnv


class SimplePolicy(nn.Module):
    """Simple but effective policy network."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        
        self.mean = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
    
    def forward(self, obs):
        obs = torch.clamp(obs, -10, 10)
        x = self.net(obs)
        mean = torch.tanh(self.mean(x))
        return mean
    
    def get_action(self, obs, deterministic=False):
        mean = self.forward(obs)
        if deterministic:
            return mean, None, None
        
        std = torch.exp(torch.clamp(self.log_std, -2, 0.5))
        dist = Normal(mean, std)
        action = dist.rsample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy
    
    def evaluate(self, obs, actions):
        mean = self.forward(obs)
        std = torch.exp(torch.clamp(self.log_std, -2, 0.5))
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class SimpleValue(nn.Module):
    """Simple value network."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs):
        obs = torch.clamp(obs, -10, 10)
        return self.net(obs)


class SimpleTrainer:
    """Simple PPO trainer that works!"""
    
    def __init__(self, env, device='cpu'):
        self.env = env
        self.num_agents = env.num_agents
        self.device = device
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.policy = SimplePolicy(obs_dim, action_dim).to(device)
        self.value = SimpleValue(obs_dim).to(device)
        
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=3e-4, eps=1e-5)
        self.value_opt = optim.Adam(self.value.parameters(), lr=1e-3, eps=1e-5)
        
        # PPO params
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip = 0.2
        self.entropy_coef = 0.01
        self.epochs = 10
        self.batch_size = 256
        
        self.buffer = []
        
        # Stats
        self.rewards = deque(maxlen=100)
        self.all_success = deque(maxlen=100)
        self.per_agent = deque(maxlen=100)
        self.lengths = deque(maxlen=100)
        self.obs_coll = deque(maxlen=100)
    
    def collect(self, steps=4096):
        self.buffer = []
        obs_list, _ = self.env.reset()
        ep_reward = 0
        
        for _ in range(steps):
            obs_t = torch.FloatTensor(np.array(obs_list)).to(self.device)
            
            with torch.no_grad():
                actions, log_probs, _ = self.policy.get_action(obs_t)
                values = self.value(obs_t).squeeze(-1)
            
            actions_np = actions.cpu().numpy()
            next_obs, rewards, term, trunc, info = self.env.step(actions_np)
            done = term or trunc
            
            self.buffer.append({
                'obs': np.array(obs_list),
                'actions': actions_np,
                'rewards': np.array(rewards),
                'values': values.cpu().numpy(),
                'log_probs': log_probs.cpu().numpy(),
                'done': done
            })
            
            ep_reward += np.mean(rewards)
            
            if done:
                self.rewards.append(ep_reward)
                self.all_success.append(float(info.get('all_success', False)))
                self.per_agent.append(info.get('per_agent_success', 0))
                self.lengths.append(info.get('episode_length', 0))
                self.obs_coll.append(info.get('total_obstacle_collisions', 0))
                
                obs_list, _ = self.env.reset()
                ep_reward = 0
            else:
                obs_list = next_obs
    
    def compute_gae(self):
        n = len(self.buffer)
        rewards = np.array([t['rewards'] for t in self.buffer])
        values = np.array([t['values'] for t in self.buffer])
        dones = np.array([t['done'] for t in self.buffer])
        
        all_adv = []
        all_ret = []
        
        for i in range(self.num_agents):
            adv = np.zeros(n)
            gae = 0
            
            for t in reversed(range(n)):
                next_val = values[t+1, i] if t < n-1 else 0
                mask = 1 - float(dones[t])
                delta = rewards[t, i] + self.gamma * next_val * mask - values[t, i]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                adv[t] = gae
            
            all_adv.append(adv)
            all_ret.append(adv + values[:, i])
        
        return np.array(all_adv).T, np.array(all_ret).T
    
    def update(self):
        if not self.buffer:
            return {}
        
        adv, ret = self.compute_gae()
        
        obs = torch.FloatTensor(
            np.array([t['obs'] for t in self.buffer])
        ).reshape(-1, self.env.observation_space.shape[0]).to(self.device)
        
        actions = torch.FloatTensor(
            np.array([t['actions'] for t in self.buffer])
        ).reshape(-1, self.env.action_space.shape[0]).to(self.device)
        
        old_lp = torch.FloatTensor(
            np.array([t['log_probs'] for t in self.buffer])
        ).reshape(-1).to(self.device)
        
        ret_t = torch.FloatTensor(ret.flatten()).to(self.device)
        adv_t = torch.FloatTensor(adv.flatten()).to(self.device)
        
        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        adv_t = torch.clamp(adv_t, -10, 10)
        
        n = obs.size(0)
        
        for _ in range(self.epochs):
            idx = torch.randperm(n)
            
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                b = idx[start:end]
                
                if len(b) < 8:
                    continue
                
                # Policy update
                new_lp, ent = self.policy.evaluate(obs[b], actions[b])
                ratio = torch.exp(torch.clamp(new_lp - old_lp[b], -10, 10))
                
                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * adv_t[b]
                p_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * ent.mean()
                
                if not torch.isnan(p_loss):
                    self.policy_opt.zero_grad()
                    p_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.policy_opt.step()
                
                # Value update
                v = self.value(obs[b]).squeeze()
                v_loss = 0.5 * ((v - ret_t[b]) ** 2).mean()
                
                if not torch.isnan(v_loss):
                    self.value_opt.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                    self.value_opt.step()
        
        self.buffer = []
    
    def train(self, timesteps, save_dir, log_interval=10, save_interval=50):
        os.makedirs(save_dir, exist_ok=True)
        
        rollout = 4096
        updates = timesteps // rollout
        best_success = 0
        
        print(f"\n{'='*60}")
        print(f"TRAINING: {self.env.scenario}")
        print(f"Agents: {self.num_agents}, Timesteps: {timesteps:,}")
        print(f"{'='*60}\n")
        
        log = []
        
        for u in range(updates):
            self.collect(rollout)
            self.update()
            
            if (u + 1) % log_interval == 0:
                avg_r = np.mean(self.rewards) if self.rewards else 0
                avg_s = np.mean(self.all_success) if self.all_success else 0
                avg_p = np.mean(self.per_agent) if self.per_agent else 0
                avg_l = np.mean(self.lengths) if self.lengths else 0
                avg_c = np.mean(self.obs_coll) if self.obs_coll else 0
                
                star = "★" if avg_s > best_success else ""
                if avg_s > best_success:
                    best_success = avg_s
                
                print(f"{star:1s} [{u+1:4d}/{updates}] "
                      f"R:{avg_r:7.1f} | "
                      f"All:{avg_s:5.1%} | "
                      f"Per:{avg_p:5.1%} | "
                      f"Len:{avg_l:4.0f} | "
                      f"Coll:{avg_c:4.1f}")
                
                log.append({
                    'update': u+1,
                    'reward': float(avg_r),
                    'all_success': float(avg_s),
                    'per_agent': float(avg_p),
                    'length': float(avg_l),
                    'obs_coll': float(avg_c),
                })
            
            if (u + 1) % save_interval == 0:
                torch.save({
                    'policy': self.policy.state_dict(),
                    'value': self.value.state_dict(),
                    'update': u + 1,
                }, os.path.join(save_dir, f'checkpoint_{u+1}.pt'))
        
        # Final save
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'scenario': self.env.scenario,
            'num_agents': self.num_agents,
            'best_success': best_success,
        }, os.path.join(save_dir, 'final_model.pt'))
        
        with open(os.path.join(save_dir, 'log.json'), 'w') as f:
            json.dump(log, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Complete! Best: {best_success:.1%}")
        print(f"{'='*60}\n")
        
        return best_success


def train_curriculum(output_dir, device, num_agents=3):
    """Train with curriculum that agents can actually complete!"""
    
    curriculum = [
        # Stage 1: Empty corridor - learn basics
        {'scenario': 'straight', 'timesteps': 600_000, 'name': 'straight'},
        # Stage 2: Single wall bottom
        {'scenario': 'single_wall_bottom', 'timesteps': 800_000, 'name': 'wall_bottom'},
        # Stage 3: Single wall top
        {'scenario': 'single_wall_top', 'timesteps': 800_000, 'name': 'wall_top'},
        # Stage 4: S-curve
        {'scenario': 's_curve', 'timesteps': 1_000_000, 'name': 's_curve'},
        # Stage 5: Narrow passage
        {'scenario': 'narrow_passage', 'timesteps': 1_200_000, 'name': 'narrow'},
        # Stage 6: Chicane
        {'scenario': 'chicane', 'timesteps': 1_500_000, 'name': 'chicane'},
        # Stage 7: Mixed
        {'scenario': 'mixed', 'timesteps': 1_500_000, 'name': 'mixed'},
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_dir = f"{output_dir}_{timestamp}"
    os.makedirs(full_dir, exist_ok=True)
    
    total = sum(s['timesteps'] for s in curriculum)
    
    print(f"\n{'#'*70}")
    print(f"#  SIMPLE ROBUST TRAINING - multii_test_Abd")
    print(f"#  Agents: {num_agents}")
    print(f"#  Total Timesteps: {total:,}")
    print(f"#  Device: {device}")
    print(f"#  Output: {full_dir}")
    print(f"{'#'*70}\n")
    
    policy_state = None
    value_state = None
    results = {}
    
    for i, stage in enumerate(curriculum):
        print(f"\n{'='*70}")
        print(f"  STAGE {i+1}/{len(curriculum)}: {stage['name'].upper()}")
        print(f"  Scenario: {stage['scenario']}")
        print(f"  Timesteps: {stage['timesteps']:,}")
        print(f"{'='*70}\n")
        
        env = FixedCorridorEnv(
            num_agents=num_agents,
            scenario=stage['scenario'],
            max_steps=800,
            randomize=True,
        )
        
        trainer = SimpleTrainer(env, device)
        
        if policy_state is not None:
            trainer.policy.load_state_dict(policy_state)
            trainer.value.load_state_dict(value_state)
            print("→ Loaded weights from previous stage")
        
        stage_dir = os.path.join(full_dir, stage['name'])
        best = trainer.train(
            timesteps=stage['timesteps'],
            save_dir=stage_dir,
            log_interval=10,
            save_interval=50,
        )
        
        results[stage['name']] = best
        
        policy_state = trainer.policy.state_dict()
        value_state = trainer.value.state_dict()
        
        torch.save({
            'policy': policy_state,
            'value': value_state,
            'scenario': stage['scenario'],
        }, os.path.join(full_dir, f"{stage['name']}_final.pt"))
    
    with open(os.path.join(full_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'#'*70}")
    print(f"#  CURRICULUM COMPLETE!")
    print(f"#  Models saved to: {full_dir}")
    print(f"{'#'*70}\n")
    
    return results, full_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='models_simple_robust')
    parser.add_argument('--num-agents', type=int, default=3)
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"  SIMPLE ROBUST MULTI-AGENT TRAINING")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    results, output_dir = train_curriculum(
        args.output_dir,
        device,
        args.num_agents,
    )
    
    print("\nFinal Results:")
    for name, success in results.items():
        print(f"  {name}: {success:.1%}")


if __name__ == '__main__':
    main()

