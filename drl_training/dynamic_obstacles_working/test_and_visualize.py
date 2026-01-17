"""
Test with proper collision handling - agent CANNOT pass through obstacles!
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path


class DynamicCorridorEnv(gym.Env):
    """Corridor with dynamic obstacles - agent CANNOT pass through!"""
    
    def __init__(self, max_steps=400):
        super().__init__()
        self.max_steps = max_steps
        
        self.width = 12.0
        self.height = 6.0
        
        self.robot_radius = 0.25
        self.max_linear_vel = 1.5
        self.max_angular_vel = 2.0
        self.dt = 0.1
        
        self.goal_threshold = 0.4
        self.num_dynamic_obs = 3
        
        self.n_rays = 12
        self.ray_length = 3.0
        
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
        
        self.pos = np.array([1.0, self.height / 2.0])
        self.theta = 0.0
        self.vel = np.array([0.0, 0.0])
        
        self.goal = np.array([self.width - 1.0, self.height / 2.0])
        
        self.dynamic_obstacles = []
        x_positions = [4.0, 6.5, 9.0]
        
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
        self.dynamic_trajectories = [[o['pos'].copy()] for o in self.dynamic_obstacles]
        
        return self._get_obs(), {}
    
    def _cast_rays(self):
        rays = np.ones(self.n_rays) * self.ray_length
        
        for i in range(self.n_rays):
            angle = self.theta + (2 * np.pi * i / self.n_rays)
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            
            for step in np.linspace(0, self.ray_length, 30):
                check_pos = self.pos + ray_dir * step
                if (check_pos[0] < 0 or check_pos[0] > self.width or
                    check_pos[1] < 0 or check_pos[1] > self.height):
                    rays[i] = step
                    break
            
            for obs in self.dynamic_obstacles:
                to_obs = obs['pos'] - self.pos
                proj_len = np.dot(to_obs, ray_dir)
                
                if proj_len > 0 and proj_len < rays[i]:
                    perp_dist = np.linalg.norm(to_obs - proj_len * ray_dir)
                    if perp_dist < obs['radius']:
                        hit_dist = proj_len - np.sqrt(obs['radius']**2 - perp_dist**2)
                        if hit_dist > 0:
                            rays[i] = min(rays[i], hit_dist)
        
        return rays / self.ray_length
    
    def _get_obs(self):
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
        
        rays = self._cast_rays()
        
        return np.array(base_obs + rays.tolist(), dtype=np.float32)
    
    def _update_dynamic_obstacles(self):
        for i, obs in enumerate(self.dynamic_obstacles):
            obs['pos'][1] += obs['vel_y'] * self.dt
            
            if obs['pos'][1] <= obs['y_min']:
                obs['pos'][1] = obs['y_min']
                obs['vel_y'] = abs(obs['vel_y'])
            elif obs['pos'][1] >= obs['y_max']:
                obs['pos'][1] = obs['y_max']
                obs['vel_y'] = -abs(obs['vel_y'])
            
            self.dynamic_trajectories[i].append(obs['pos'].copy())
    
    def _check_collision(self, new_pos):
        """Check if new position would collide"""
        # Wall collision
        if (new_pos[0] < self.robot_radius or 
            new_pos[0] > self.width - self.robot_radius or
            new_pos[1] < self.robot_radius or 
            new_pos[1] > self.height - self.robot_radius):
            return True
        
        # Obstacle collision
        for obs in self.dynamic_obstacles:
            dist = np.linalg.norm(new_pos - obs['pos'])
            if dist < self.robot_radius + obs['radius']:
                return True
        
        return False
    
    def step(self, action):
        self.steps += 1
        self._update_dynamic_obstacles()
        
        linear_vel = (action[0] + 1.0) / 2.0 * self.max_linear_vel
        angular_vel = action[1] * self.max_angular_vel
        
        # Update heading
        self.theta += angular_vel * self.dt
        
        # Calculate NEW position
        new_pos = self.pos.copy()
        new_pos[0] += linear_vel * np.cos(self.theta) * self.dt
        new_pos[1] += linear_vel * np.sin(self.theta) * self.dt
        
        # Check collision BEFORE moving
        collision = self._check_collision(new_pos)
        
        # Only move if NO collision!
        if not collision:
            self.pos = new_pos
        else:
            # STOP! Don't move through obstacle
            self.collision_count += 1
        
        self.vel = np.array([linear_vel, angular_vel])
        self.trajectory.append(self.pos.copy())
        
        dist = np.linalg.norm(self.goal - self.pos)
        
        # Rewards
        reward = 0.0
        progress = self.prev_dist - dist
        reward += progress * 30.0
        
        if linear_vel > 0.8 and not collision:
            reward += 0.5
        
        rays = self._cast_rays()
        min_ray = np.min(rays)
        if min_ray < 0.2:
            reward -= 3.0
        elif min_ray < 0.4:
            reward -= 1.0
        
        if collision:
            reward -= 20.0
        
        reward -= 0.02
        
        self.prev_dist = dist
        
        done = False
        truncated = False
        success = False
        
        if dist < self.goal_threshold:
            reward += 300.0
            done = True
            success = True
        elif self.steps >= self.max_steps:
            truncated = True
        
        return self._get_obs(), reward, done, truncated, {
            'is_success': success,
            'collision_count': self.collision_count,
            'distance': dist
        }


print("Loading model...")
model = PPO.load("drl_vga_experiments/models_dynamic/final_model")

print("\nTesting with PROPER collision handling...")
env = DynamicCorridorEnv()

results = []
for ep in range(20):
    obs, _ = env.reset()
    done = False
    steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
    
    results.append({
        'success': info['is_success'],
        'collisions': info['collision_count'],
        'distance': info['distance'],
        'steps': steps
    })
    
    status = "✓ SUCCESS" if info['is_success'] else "✗ TIMEOUT"
    print(f"Ep {ep+1}/20: {status} | Steps: {steps} | Collisions: {info['collision_count']} | Dist: {info['distance']:.2f}m")

success_count = sum(1 for r in results if r['success'])
print("\n" + "="*70)
print(f"Success: {success_count}/20 ({success_count*5}%)")
print(f"Avg collisions: {np.mean([r['collisions'] for r in results]):.1f}")
print("="*70)

# Create MULTIPLE videos - different episodes
num_videos = 5
print(f"\nCreating {num_videos} videos (ONE agent per video)...")

video_data_list = []

for vid_idx in range(num_videos):
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    video_data = {
        'trajectory': env.trajectory.copy(),
        'dynamic_trajectories': [t.copy() for t in env.dynamic_trajectories],
        'dynamic_obstacles': [{'pos': o['pos'].copy(), 'radius': o['radius'], 'vel_y': o['vel_y']} 
                              for o in env.dynamic_obstacles],
        'goal': env.goal.copy(),
        'success': info['is_success'],
        'collisions': info['collision_count'],
        'steps': len(env.trajectory)
    }
    video_data_list.append(video_data)
    
    print(f"Video {vid_idx+1}/{num_videos}: {'✓ SUCCESS' if info['is_success'] else '✗ FAILED'} ({len(env.trajectory)} steps, {info['collision_count']} collisions)")

# Generate all videos
for vid_idx, video_data in enumerate(video_data_list):
    trajectory = video_data['trajectory']
    dynamic_trajectories = video_data['dynamic_trajectories']
    dynamic_obstacles = video_data['dynamic_obstacles']
    goal = video_data['goal']
    success = video_data['success']
    collisions = video_data['collisions']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(-0.5, 12.5)
        ax.set_ylim(-0.5, 6.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linewidth=1)
        
        title = f'DRL Agent Navigation - Dynamic Obstacles (Episode {vid_idx+1})\n'
        title += f"{'✓ SUCCESS' if success else '✗ FAILED'} | Frame {frame+1}/{len(trajectory)} | Collisions: {collisions}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X position (m)', fontsize=14)
        ax.set_ylabel('Y position (m)', fontsize=14)
        
        # Walls
        ax.plot([0, 12], [0, 0], 'k-', linewidth=5, alpha=0.7)
        ax.plot([0, 12], [6, 6], 'k-', linewidth=5, alpha=0.7)
        ax.plot([0, 0], [0, 6], 'k-', linewidth=5, alpha=0.7)
        ax.plot([12, 12], [0, 6], 'k-', linewidth=5, alpha=0.7)
        
        # Dynamic obstacles
        for i, dyn_traj in enumerate(dynamic_trajectories):
            if frame < len(dyn_traj):
                pos = dyn_traj[frame]
                radius = dynamic_obstacles[i]['radius']
                vel_y = dynamic_obstacles[i]['vel_y']
                
                # Obstacle
                circle = plt.Circle(pos, radius, color='#ff4444', alpha=0.8, linewidth=3, edgecolor='#cc0000')
                ax.add_patch(circle)
                
                # Label
                ax.text(pos[0], pos[1], 'OBS', ha='center', va='center', 
                       fontsize=9, fontweight='bold', color='white')
                
                # Velocity arrow
                arrow_scale = 0.25
                ax.arrow(pos[0], pos[1], 0, vel_y*arrow_scale,
                        head_width=0.25, head_length=0.18, fc='#cc0000', ec='#cc0000', 
                        alpha=0.9, linewidth=2)
        
        # Goal
        goal_circle = plt.Circle(goal, 0.4, color='#44ff44', alpha=0.4, linewidth=3, edgecolor='#00cc00')
        ax.add_patch(goal_circle)
        ax.plot(goal[0], goal[1], 'g*', markersize=35, markeredgecolor='#00cc00', markeredgewidth=3)
        ax.text(goal[0], goal[1]-0.8, 'GOAL', ha='center', fontsize=11, fontweight='bold', color='green')
        
        # Trajectory
        if frame > 0:
            traj_array = np.array(trajectory[:frame+1])
            ax.plot(traj_array[:, 0], traj_array[:, 1], 
                   color='#2196F3', alpha=0.6, linewidth=3, linestyle='--')
        
        # Agent
        pos = trajectory[frame]
        agent_circle = plt.Circle(pos, 0.25, color='#2196F3', alpha=0.95, linewidth=3, edgecolor='black')
        ax.add_patch(agent_circle)
        
        # Direction arrow - points in actual movement direction
        arrow_len = 0.4
        if frame > 0:
            # Calculate direction from previous position
            prev_pos = trajectory[frame-1]
            dx = pos[0] - prev_pos[0]
            dy = pos[1] - prev_pos[1]
            # Normalize and scale
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0.01:  # Only draw arrow if agent moved
                dx = dx / dist * arrow_len
                dy = dy / dist * arrow_len
                ax.arrow(pos[0], pos[1], dx, dy,
                        head_width=0.2, head_length=0.15, fc='#2196F3', ec='black', 
                        alpha=0.9, linewidth=2)
        else:
            # First frame - point toward goal
            dx = goal[0] - pos[0]
            dy = goal[1] - pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            dx = dx / dist * arrow_len
            dy = dy / dist * arrow_len
            ax.arrow(pos[0], pos[1], dx, dy,
                    head_width=0.2, head_length=0.15, fc='#2196F3', ec='black', 
                    alpha=0.9, linewidth=2)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2196F3', alpha=0.95, edgecolor='black', linewidth=2, label='Agent'),
            Patch(facecolor='#44ff44', alpha=0.4, edgecolor='#00cc00', linewidth=2, label='Goal'),
            Patch(facecolor='#ff4444', alpha=0.8, edgecolor='#cc0000', linewidth=2, label='Dynamic Obstacle')
        ]
        ax.legend(handles=legend_elements, loc='upper center', fontsize=12, ncol=3, 
                 framealpha=0.95, edgecolor='black', fancybox=True)
    
    anim = FuncAnimation(fig, animate, frames=len(trajectory), interval=100, repeat=True)
    
    output_path = f"drl_vga_experiments/test_results/episode_{vid_idx+1:02d}_{'success' if success else 'failed'}.mp4"
    Path("drl_vga_experiments/test_results").mkdir(exist_ok=True)
    
    writer = FFMpegWriter(fps=10, bitrate=3000)
    anim.save(output_path, writer=writer)
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")

print(f"\n✓ Generated {num_videos} videos!")
print("✓ ONE agent per video")
print("✓ Agent CANNOT pass through obstacles")

