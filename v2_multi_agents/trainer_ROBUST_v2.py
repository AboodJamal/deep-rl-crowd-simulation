"""
ROBUST Multi-Agent Navigation Trainer v2.0
==========================================
Complete rewrite with fixes for:
1. Direct velocity control (NO rotation-based movement)
2. Strong goal-seeking behavior
3. Anti-hesitation rewards
4. Smooth navigation
5. Robust training

Author: GitHub Copilot
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from collections import deque

sys.stdout.reconfigure(line_buffering=True)

print("="*70)
print("  ROBUST Multi-Agent Navigation Trainer v2.0")
print("  Direct Velocity Control - No Rotation Issues")
print("="*70)


class RobustCorridorEnv(gym.Env):
    """
    Robust corridor environment with DIRECT velocity control.
    
    Key Design Decisions:
    1. Actions are [vx, vy] directly - NO angular velocity!
    2. Goal-centric observations (not agent-angle-centric)
    3. Strong goal rewards + anti-hesitation penalties
    4. Smooth movement with momentum
    """
    
    def __init__(self, num_agents=3, corridor_type='straight', max_steps=800):
        super().__init__()
        
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.current_step = 0
        self.corridor_type = corridor_type
        
        # Corridor dimensions
        self.width = 50.0
        self.height = 12.0
        
        # Agent properties
        self.agent_radius = 0.3
        self.max_speed = 1.5
        self.acceleration = 0.3  # Smooth acceleration
        
        # Raycasting (world-aligned, NOT agent-aligned!)
        self.num_rays = 16
        self.ray_length = 8.0
        
        # Goal - large and easy to reach
        self.goal_radius = 3.0
        self.goal_position = np.array([self.width - 5.0, self.height / 2])
        
        # Obstacles
        self.obstacles = self._create_obstacles(corridor_type)
        
        # Observation space:
        # - Position (2): x, y normalized
        # - Velocity (2): vx, vy normalized  
        # - Goal direction (2): unit vector to goal
        # - Goal distance (1): normalized distance
        # - Rays (16): obstacle distances
        # - Other agents (num_agents-1)*4: relative pos + vel
        other_obs = (num_agents - 1) * 4
        obs_dim = 2 + 2 + 2 + 1 + self.num_rays + other_obs  # = 23 + other_obs
        
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: direct velocity control [vx, vy]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # State
        self.agents = []
        self._prev_distances = []
        self._prev_velocities = []
        
    def _create_obstacles(self, corridor_type):
        """Create obstacles."""
        obstacles = []
        
        if corridor_type == 'straight':
            pass
        elif corridor_type == 'single_wall':
            obstacles.append({
                'x': self.width * 0.5 - 1.0,
                'y': 0,
                'width': 2.0,
                'height': self.height * 0.55
            })
        elif corridor_type == 'double_wall':
            obstacles.append({
                'x': self.width * 0.35 - 1.0,
                'y': 0,
                'width': 2.0,
                'height': self.height * 0.45
            })
            obstacles.append({
                'x': self.width * 0.65 - 1.0,
                'y': self.height * 0.55,
                'width': 2.0,
                'height': self.height * 0.45
            })
        elif corridor_type == 'narrow_passage':
            # Two walls creating a narrow passage
            obstacles.append({
                'x': self.width * 0.5 - 1.0,
                'y': 0,
                'width': 2.0,
                'height': self.height * 0.35
            })
            obstacles.append({
                'x': self.width * 0.5 - 1.0,
                'y': self.height * 0.65,
                'width': 2.0,
                'height': self.height * 0.35
            })
        
        return obstacles
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.agents = []
        self._prev_distances = []
        self._prev_velocities = []
        
        # Spread agents in start area
        y_spacing = self.height / (self.num_agents + 1)
        
        for i in range(self.num_agents):
            # Start on left side, spread vertically
            x = np.random.uniform(2.0, 6.0)
            y = y_spacing * (i + 1) + np.random.uniform(-0.5, 0.5)
            y = np.clip(y, self.agent_radius + 0.5, self.height - self.agent_radius - 0.5)
            
            agent = {
                'x': x,
                'y': y,
                'vx': 0.0,
                'vy': 0.0,
                'reached_goal': False,
                'collided': False,
                'steps_taken': 0
            }
            self.agents.append(agent)
            
            dist = np.linalg.norm(self.goal_position - np.array([x, y]))
            self._prev_distances.append(dist)
            self._prev_velocities.append(np.array([0.0, 0.0]))
        
        return self._get_observations(), {}
    
    def _cast_ray(self, agent_idx, angle):
        """Cast ray in WORLD coordinates (not relative to agent heading)."""
        agent = self.agents[agent_idx]
        x, y = agent['x'], agent['y']
        
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Check incrementally
        for dist in np.linspace(0.2, self.ray_length, 30):
            cx = x + dx * dist
            cy = y + dy * dist
            
            # Walls
            if cx < 0 or cx > self.width or cy < 0 or cy > self.height:
                return dist / self.ray_length
            
            # Obstacles
            for obs in self.obstacles:
                if (obs['x'] <= cx <= obs['x'] + obs['width'] and
                    obs['y'] <= cy <= obs['y'] + obs['height']):
                    return dist / self.ray_length
            
            # Other agents
            for j, other in enumerate(self.agents):
                if j != agent_idx:
                    d = np.sqrt((cx - other['x'])**2 + (cy - other['y'])**2)
                    if d < self.agent_radius * 2:
                        return dist / self.ray_length
        
        return 1.0
    
    def _get_observations(self):
        """Get observations for all agents."""
        observations = []
        
        for i, agent in enumerate(self.agents):
            obs = []
            
            pos = np.array([agent['x'], agent['y']])
            vel = np.array([agent['vx'], agent['vy']])
            
            # Position (normalized to [-1, 1])
            obs.append((agent['x'] / self.width) * 2 - 1)
            obs.append((agent['y'] / self.height) * 2 - 1)
            
            # Velocity (normalized)
            obs.append(agent['vx'] / self.max_speed)
            obs.append(agent['vy'] / self.max_speed)
            
            # Goal direction (unit vector)
            goal_vec = self.goal_position - pos
            goal_dist = np.linalg.norm(goal_vec)
            if goal_dist > 0.01:
                goal_dir = goal_vec / goal_dist
            else:
                goal_dir = np.array([1.0, 0.0])
            obs.append(goal_dir[0])
            obs.append(goal_dir[1])
            
            # Goal distance (normalized, capped)
            obs.append(min(goal_dist / self.width, 1.0))
            
            # Raycasting - WORLD-ALIGNED (fixed angles, not relative to heading)
            for j in range(self.num_rays):
                angle = j * (2 * np.pi / self.num_rays)  # 0 to 2*pi
                obs.append(self._cast_ray(i, angle))
            
            # Other agents (relative position and velocity)
            for j, other in enumerate(self.agents):
                if i != j:
                    rel_x = (other['x'] - agent['x']) / 10.0  # Scale for neural network
                    rel_y = (other['y'] - agent['y']) / 10.0
                    rel_vx = (other['vx'] - agent['vx']) / self.max_speed
                    rel_vy = (other['vy'] - agent['vy']) / self.max_speed
                    obs.extend([
                        np.clip(rel_x, -2, 2),
                        np.clip(rel_y, -2, 2),
                        np.clip(rel_vx, -2, 2),
                        np.clip(rel_vy, -2, 2)
                    ])
            
            observations.append(np.array(obs, dtype=np.float32))
        
        return observations
    
    def _check_collision(self, agent_idx):
        """Check collision with walls, obstacles, and other agents."""
        agent = self.agents[agent_idx]
        x, y = agent['x'], agent['y']
        r = self.agent_radius
        
        # Walls
        if x < r or x > self.width - r or y < r or y > self.height - r:
            return True
        
        # Obstacles
        for obs in self.obstacles:
            # Expanded obstacle bounds
            if (obs['x'] - r < x < obs['x'] + obs['width'] + r and
                obs['y'] - r < y < obs['y'] + obs['height'] + r):
                return True
        
        # Other agents
        for j, other in enumerate(self.agents):
            if j != agent_idx and not other['reached_goal']:
                dist = np.sqrt((x - other['x'])**2 + (y - other['y'])**2)
                if dist < self.agent_radius * 2:
                    return True
        
        return False
    
    def step(self, actions):
        """Execute one step with direct velocity control."""
        self.current_step += 1
        dt = 0.1  # Time step
        
        for i, agent in enumerate(self.agents):
            if agent['reached_goal'] or agent['collided']:
                continue
            
            agent['steps_taken'] += 1
            
            # Actions are target velocity direction [vx, vy]
            target_vx = actions[i][0] * self.max_speed
            target_vy = actions[i][1] * self.max_speed
            
            # Smooth acceleration towards target velocity
            agent['vx'] += (target_vx - agent['vx']) * self.acceleration
            agent['vy'] += (target_vy - agent['vy']) * self.acceleration
            
            # Clamp velocity magnitude
            speed = np.sqrt(agent['vx']**2 + agent['vy']**2)
            if speed > self.max_speed:
                agent['vx'] = (agent['vx'] / speed) * self.max_speed
                agent['vy'] = (agent['vy'] / speed) * self.max_speed
            
            # Update position
            new_x = agent['x'] + agent['vx'] * dt
            new_y = agent['y'] + agent['vy'] * dt
            
            # Boundary clamping
            new_x = np.clip(new_x, self.agent_radius, self.width - self.agent_radius)
            new_y = np.clip(new_y, self.agent_radius, self.height - self.agent_radius)
            
            agent['x'] = new_x
            agent['y'] = new_y
        
        # Check collisions and goal
        for i, agent in enumerate(self.agents):
            if agent['reached_goal'] or agent['collided']:
                continue
            
            # Collision check
            if self._check_collision(i):
                agent['collided'] = True
                continue
            
            # Goal check
            dist = np.linalg.norm(self.goal_position - np.array([agent['x'], agent['y']]))
            if dist < self.goal_radius:
                agent['reached_goal'] = True
        
        # Calculate rewards
        rewards = [self._calculate_reward(i) for i in range(self.num_agents)]
        
        # Episode termination
        all_done = all(a['reached_goal'] or a['collided'] for a in self.agents)
        truncated = self.current_step >= self.max_steps
        done = all_done or truncated
        
        success_count = sum(1 for a in self.agents if a['reached_goal'])
        
        info = {
            'success_rate': success_count / self.num_agents,
            'success_count': success_count,
            'collision_count': sum(1 for a in self.agents if a['collided']),
            'episode_length': self.current_step
        }
        
        return self._get_observations(), rewards, done, False, info
    
    def _calculate_reward(self, agent_idx):
        """
        Robust reward function designed for smooth goal-reaching.
        
        Key principles:
        1. MASSIVE goal reward
        2. Strong progress reward
        3. Velocity alignment reward (move TOWARDS goal)
        4. Anti-hesitation: penalize stopping, penalize slow movement
        5. Penalize erratic movement
        """
        agent = self.agents[agent_idx]
        
        # Already done
        if agent['reached_goal']:
            return 500.0  # Big bonus for reaching goal
        
        if agent['collided']:
            return -50.0  # Penalty for collision
        
        reward = 0.0
        
        pos = np.array([agent['x'], agent['y']])
        vel = np.array([agent['vx'], agent['vy']])
        
        # === 1. PROGRESS REWARD (most important) ===
        curr_dist = np.linalg.norm(self.goal_position - pos)
        prev_dist = self._prev_distances[agent_idx]
        progress = prev_dist - curr_dist
        
        # Strong reward for moving closer
        reward += progress * 50.0
        
        self._prev_distances[agent_idx] = curr_dist
        
        # === 2. VELOCITY ALIGNMENT REWARD ===
        # Reward moving in the direction of the goal
        goal_dir = self.goal_position - pos
        goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-8)
        
        speed = np.linalg.norm(vel)
        if speed > 0.01:
            vel_dir = vel / speed
            alignment = np.dot(goal_dir, vel_dir)  # -1 to 1
            
            # Reward aligned velocity, penalize opposite
            reward += alignment * speed * 10.0
        
        # === 3. SPEED REWARD (anti-hesitation) ===
        # Encourage maintaining good speed
        if speed > 0.3 * self.max_speed:
            reward += 1.0  # Bonus for moving
        elif speed < 0.1 * self.max_speed:
            reward -= 0.5  # Penalty for being too slow
        
        # === 4. DISTANCE-BASED POTENTIAL ===
        # Closer = higher reward (encourages finishing)
        reward += 2.0 / (1.0 + curr_dist)
        
        # === 5. SMOOTHNESS REWARD ===
        # Penalize erratic velocity changes
        prev_vel = self._prev_velocities[agent_idx]
        vel_change = np.linalg.norm(vel - prev_vel)
        if vel_change > 0.5:
            reward -= vel_change * 0.5  # Penalize jerky motion
        self._prev_velocities[agent_idx] = vel.copy()
        
        # === 6. TIME PENALTY ===
        reward -= 0.05  # Small time penalty
        
        # === 7. GOAL PROXIMITY BONUS ===
        if curr_dist < self.goal_radius * 2:
            reward += 5.0  # Getting close!
        if curr_dist < self.goal_radius * 1.5:
            reward += 10.0  # Very close!
        
        return reward


class PolicyNetwork(nn.Module):
    """
    Actor network with tanh output for bounded actions.
    Uses LayerNorm for stability.
    """
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
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
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable but shared
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # Small initial output
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
    
    def forward(self, obs):
        obs = torch.clamp(obs, -5.0, 5.0)
        features = self.net(obs)
        mean = self.mean_head(features)
        mean = torch.tanh(mean)  # Bounded [-1, 1]
        return mean
    
    def get_action(self, obs, deterministic=False):
        mean = self.forward(obs)
        
        if deterministic:
            return mean, None, None
        
        std = torch.exp(torch.clamp(self.log_std, -2.0, 0.5))  # Reasonable std range
        dist = Normal(mean, std)
        action = dist.rsample()
        action = torch.clamp(action, -1.0, 1.0)
        
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        return action, log_prob, entropy
    
    def evaluate(self, obs, actions):
        mean = self.forward(obs)
        std = torch.exp(torch.clamp(self.log_std, -2.0, 0.5))
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(self, obs_dim, hidden_dim=256):
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
    
    def forward(self, obs):
        obs = torch.clamp(obs, -5.0, 5.0)
        return self.net(obs)


class RobustTrainer:
    """
    Robust PPO trainer with proper hyperparameters.
    """
    
    def __init__(self, env, device='cpu'):
        self.env = env
        self.num_agents = env.num_agents
        self.device = device
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dim=256).to(device)
        self.value = ValueNetwork(obs_dim, hidden_dim=256).to(device)
        
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=3e-4, eps=1e-5)
        self.value_optim = optim.Adam(self.value.parameters(), lr=1e-3, eps=1e-5)
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_param = 0.2
        self.entropy_coef = 0.005  # Lower entropy for more deterministic behavior
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.ppo_epochs = 10
        self.batch_size = 256
        
        # Rollout buffer
        self.buffer = []
        
        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def collect_rollout(self, num_steps=4096):
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
            next_obs, rewards, done, _, info = self.env.step(actions_np)
            
            # Store transition
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
                self.episode_rewards.append(episode_reward)
                self.success_rates.append(info.get('success_rate', 0.0))
                self.episode_lengths.append(info.get('episode_length', 0))
                
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
    
    def update(self):
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
        ).reshape(-1, 2).to(self.device)
        
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
            'value_loss': total_value_loss / max(num_updates, 1)
        }
    
    def train(self, total_timesteps, log_interval=10, save_dir='models_robust'):
        """Main training loop."""
        os.makedirs(save_dir, exist_ok=True)
        
        rollout_length = 4096
        num_updates = total_timesteps // rollout_length
        
        print(f"\n{'='*50}")
        print(f"Training Configuration:")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Updates: {num_updates}")
        print(f"  Rollout length: {rollout_length}")
        print(f"  Corridor: {self.env.corridor_type}")
        print(f"  Size: {self.env.width}x{self.env.height}")
        print(f"  Goal radius: {self.env.goal_radius}")
        print(f"  Device: {self.device}")
        print(f"{'='*50}\n")
        
        best_success = 0.0
        
        for update in range(num_updates):
            self.collect_rollout(num_steps=rollout_length)
            losses = self.update()
            
            if (update + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                avg_success = np.mean(self.success_rates) if self.success_rates else 0
                avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                
                status = "✓" if avg_success > best_success else " "
                if avg_success > best_success:
                    best_success = avg_success
                
                print(f"{status} Update {update+1:4d}/{num_updates} | "
                      f"Reward: {avg_reward:8.1f} | "
                      f"Success: {avg_success:5.1%} | "
                      f"Length: {avg_length:5.0f}")
            
            # Save checkpoints
            if (update + 1) % 25 == 0:
                checkpoint = {
                    'policy': self.policy.state_dict(),
                    'value': self.value.state_dict(),
                    'update': update + 1
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_{update+1}.pt'))
        
        # Save final model
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict()
        }, os.path.join(save_dir, 'final_model.pt'))
        
        final_success = np.mean(self.success_rates) if self.success_rates else 0
        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Final Success Rate: {final_success:.1%}")
        print(f"Best Success Rate: {best_success:.1%}")
        print(f"{'='*50}")
        
        return best_success


def main():
    """Main training with curriculum learning."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f'models_robust_{timestamp}'
    os.makedirs(base_dir, exist_ok=True)
    
    # Curriculum: start easy, get harder
    curriculum = [
        ('straight', 600000),        # Stage 1: Open corridor
        ('single_wall', 600000),     # Stage 2: One wall
        ('double_wall', 600000),     # Stage 3: Two walls (zigzag)
        ('narrow_passage', 400000),  # Stage 4: Narrow passage
    ]
    
    policy_state = None
    value_state = None
    
    for stage_idx, (corridor_type, timesteps) in enumerate(curriculum):
        print(f"\n{'#'*70}")
        print(f"#  STAGE {stage_idx + 1}: {corridor_type.upper()}")
        print(f"#  Timesteps: {timesteps:,}")
        print(f"{'#'*70}")
        
        env = RobustCorridorEnv(
            num_agents=3,
            corridor_type=corridor_type,
            max_steps=800
        )
        
        trainer = RobustTrainer(env=env, device=device)
        
        # Load weights from previous stage
        if policy_state is not None:
            trainer.policy.load_state_dict(policy_state)
            trainer.value.load_state_dict(value_state)
            print("→ Loaded weights from previous stage")
        
        save_dir = os.path.join(base_dir, corridor_type)
        best_success = trainer.train(
            total_timesteps=timesteps,
            log_interval=10,
            save_dir=save_dir
        )
        
        # Save weights for next stage
        policy_state = trainer.policy.state_dict()
        value_state = trainer.value.state_dict()
        
        # Save stage final
        torch.save({
            'policy': policy_state,
            'value': value_state,
            'corridor_type': corridor_type,
            'best_success': best_success
        }, os.path.join(base_dir, f'{corridor_type}_final.pt'))
    
    print(f"\n{'='*70}")
    print(f"  ALL STAGES COMPLETE!")
    print(f"  Models saved to: {base_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
