"""
PROPER DYNAMIC OBSTACLE TRAINING
- Raycasts like static environment
- Collisions DON'T terminate - agent learns to avoid
- Better reward structure
- Stronger training
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import warnings
warnings.filterwarnings('ignore')


class DynamicCorridorEnv(gym.Env):
    """
    Corridor with dynamic obstacles and raycasts.
    Collisions give penalty but don't terminate!
    """
    
    def __init__(self, max_steps=400):
        super().__init__()
        self.max_steps = max_steps
        
        # Corridor dimensions
        self.width = 12.0
        self.height = 6.0
        
        # Robot parameters
        self.robot_radius = 0.25
        self.max_linear_vel = 1.5
        self.max_angular_vel = 2.0
        self.dt = 0.1
        
        # Goal
        self.goal_threshold = 0.4
        
        # Dynamic obstacles (only 2-3)
        self.num_dynamic_obs = 3
        
        # Raycasts (like static env)
        self.n_rays = 12
        self.ray_length = 3.0
        
        # Observation: [dx, dy, dist, heading_error, vel_x, vel_y, raycasts(12)]
        # = 6 + 12 = 18
        obs_dim = 6 + self.n_rays
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Agent starts LEFT
        self.pos = np.array([1.0, self.height / 2.0])
        self.theta = 0.0  # Facing right
        self.vel = np.array([0.0, 0.0])
        
        # Goal on RIGHT
        self.goal = np.array([self.width - 1.0, self.height / 2.0])
        
        # Only 3 dynamic obstacles in middle
        self.dynamic_obstacles = []
        x_positions = [4.0, 6.5, 9.0]  # Spread across corridor
        
        for x_pos in x_positions:
            y_pos = np.random.uniform(1.5, self.height - 1.5)
            vel_y = np.random.choice([-1, 1]) * np.random.uniform(0.5, 0.9)
            
            self.dynamic_obstacles.append({
                'pos': np.array([x_pos, y_pos]),
                'radius': 0.35,
                'vel_y': vel_y,
                'y_min': 0.8,
                'y_max': self.height - 0.8
            })
        
        self.steps = 0
        self.prev_dist = np.linalg.norm(self.goal - self.pos)
        self.trajectory = [self.pos.copy()]
        self.collision_count = 0
        
        return self._get_obs(), {}
    
    def _cast_rays(self):
        """Cast rays to detect obstacles and walls"""
        rays = np.ones(self.n_rays) * self.ray_length
        
        for i in range(self.n_rays):
            angle = self.theta + (2 * np.pi * i / self.n_rays)
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            
            # Check walls
            for step in np.linspace(0, self.ray_length, 30):
                check_pos = self.pos + ray_dir * step
                if (check_pos[0] < 0 or check_pos[0] > self.width or
                    check_pos[1] < 0 or check_pos[1] > self.height):
                    rays[i] = step
                    break
            
            # Check dynamic obstacles
            for obs in self.dynamic_obstacles:
                to_obs = obs['pos'] - self.pos
                proj_len = np.dot(to_obs, ray_dir)
                
                if proj_len > 0 and proj_len < rays[i]:
                    perp_dist = np.linalg.norm(to_obs - proj_len * ray_dir)
                    if perp_dist < obs['radius']:
                        hit_dist = proj_len - np.sqrt(obs['radius']**2 - perp_dist**2)
                        if hit_dist > 0:
                            rays[i] = min(rays[i], hit_dist)
        
        # Normalize rays
        return rays / self.ray_length
    
    def _get_obs(self):
        # Goal info
        dx = self.goal[0] - self.pos[0]
        dy = self.goal[1] - self.pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        goal_angle = np.arctan2(dy, dx)
        heading_error = goal_angle - self.theta
        while heading_error > np.pi:
            heading_error -= 2*np.pi
        while heading_error < -np.pi:
            heading_error += 2*np.pi
        
        base_obs = [
            dx / 15.0,
            dy / 10.0,
            dist / 15.0,
            heading_error / np.pi,
            self.vel[0] / self.max_linear_vel,
            self.vel[1] / self.max_angular_vel
        ]
        
        # Raycasts
        rays = self._cast_rays()
        
        return np.array(base_obs + rays.tolist(), dtype=np.float32)
    
    def _update_dynamic_obstacles(self):
        """Move obstacles up/down"""
        for obs in self.dynamic_obstacles:
            obs['pos'][1] += obs['vel_y'] * self.dt
            
            # Bounce at boundaries
            if obs['pos'][1] <= obs['y_min']:
                obs['pos'][1] = obs['y_min']
                obs['vel_y'] = abs(obs['vel_y'])
            elif obs['pos'][1] >= obs['y_max']:
                obs['pos'][1] = obs['y_max']
                obs['vel_y'] = -abs(obs['vel_y'])
    
    def _check_collision(self):
        """Check collision with obstacles or walls"""
        # Wall collision
        if (self.pos[0] < self.robot_radius or 
            self.pos[0] > self.width - self.robot_radius or
            self.pos[1] < self.robot_radius or 
            self.pos[1] > self.height - self.robot_radius):
            return True
        
        # Obstacle collision
        for obs in self.dynamic_obstacles:
            dist = np.linalg.norm(self.pos - obs['pos'])
            if dist < self.robot_radius + obs['radius']:
                return True
        
        return False
    
    def step(self, action):
        self.steps += 1
        
        # Update obstacles
        self._update_dynamic_obstacles()
        
        # Apply action
        linear_vel = (action[0] + 1.0) / 2.0 * self.max_linear_vel
        angular_vel = action[1] * self.max_angular_vel
        
        # Update robot
        self.theta += angular_vel * self.dt
        self.pos[0] += linear_vel * np.cos(self.theta) * self.dt
        self.pos[1] += linear_vel * np.sin(self.theta) * self.dt
        self.vel = np.array([linear_vel, angular_vel])
        
        self.trajectory.append(self.pos.copy())
        
        # Calculate distance to goal
        dist = np.linalg.norm(self.goal - self.pos)
        
        # === REWARD STRUCTURE ===
        reward = 0.0
        
        # 1. Progress reward (STRONG)
        progress = self.prev_dist - dist
        reward += progress * 30.0
        
        # 2. Forward motion bonus
        if linear_vel > 0.8:
            reward += 0.5
        
        # 3. Ray-based obstacle avoidance
        rays = self._cast_rays()
        min_ray = np.min(rays)
        if min_ray < 0.2:  # Very close to obstacle
            reward -= 3.0
        elif min_ray < 0.4:  # Getting close
            reward -= 1.0
        
        # 4. Collision penalty (BUT DON'T TERMINATE!)
        collision = self._check_collision()
        if collision:
            reward -= 20.0  # Strong penalty
            self.collision_count += 1
        
        # 5. Small time penalty
        reward -= 0.02
        
        self.prev_dist = dist
        
        # === TERMINATION ===
        done = False
        truncated = False
        success = False
        
        # ONLY terminate on success or timeout
        # Collisions DON'T terminate!
        if dist < self.goal_threshold:
            reward += 300.0  # HUGE success bonus
            done = True
            success = True
        elif self.steps >= self.max_steps:
            truncated = True
        
        return self._get_obs(), reward, done, truncated, {
            'is_success': success,
            'collision_count': self.collision_count,
            'distance': dist
        }


class BetterCallback(BaseCallback):
    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_success_rate = 0.0
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                successes = [ep_info.get('is_success', 0) for ep_info in self.model.ep_info_buffer]
                success_rate = np.mean(successes) if successes else 0.0
                
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    # Save best model
                    self.model.save("drl_vga_experiments/models_dynamic/best_model")
                
                if self.verbose > 0:
                    print(f"\n[Progress] Success: {success_rate*100:.1f}%, Best: {self.best_success_rate*100:.1f}%")
        
        return True


def train():
    print("\n" + "="*70)
    print("STRONG TRAINING - DYNAMIC OBSTACLES WITH RAYCASTS")
    print("="*70)
    print("Environment:")
    print("  - Corridor: 12m × 6m")
    print("  - 3 dynamic obstacles moving vertically")
    print("  - 12 raycasts for obstacle detection")
    print("  - Collisions = PENALTY (not termination!)")
    print("  - Agent learns to avoid obstacles")
    print("="*70 + "\n")
    
    # Create environment
    env = DynamicCorridorEnv()
    
    # Create model with STRONGER hyperparameters
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=4096,        # Larger rollout
        batch_size=128,      # Larger batch
        n_epochs=15,         # More epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),  # Bigger network
        verbose=1,
        tensorboard_log="./logs"
    )
    
    # Train LONGER
    callback = BetterCallback(check_freq=8192)
    
    print("Starting STRONG training for 1,000,000 timesteps...")
    print("This will take about 30-40 minutes.\n")
    
    model.learn(
        total_timesteps=1000000,
        callback=callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("drl_vga_experiments/models_dynamic/final_model")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print(f"✓ Best success rate: {callback.best_success_rate*100:.1f}%")
    print("✓ Models saved:")
    print("  - drl_vga_experiments/models_dynamic/best_model")
    print("  - drl_vga_experiments/models_dynamic/final_model")
    print("="*70)


if __name__ == "__main__":
    import os
    os.makedirs("drl_vga_experiments/models_dynamic", exist_ok=True)
    train()
