"""
Generate PROPER Animated Videos for Multi-Agent Navigation
==========================================================
Uses matplotlib animation to create real animated files.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import os

print("="*70)
print("  REAL Animated Video Generation")
print("="*70)

# ==========================================
# ENVIRONMENT
# ==========================================

class RobustCorridorEnv:
    def __init__(self, num_agents=3, corridor_type='straight', max_steps=800):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.current_step = 0
        self.corridor_type = corridor_type
        
        self.width = 50.0
        self.height = 12.0
        self.agent_radius = 0.3
        self.max_speed = 1.5
        self.acceleration = 0.3
        self.num_rays = 16
        self.ray_length = 8.0
        self.goal_radius = 3.0
        self.goal_position = np.array([self.width - 5.0, self.height / 2])
        
        self.obstacles = self._create_obstacles(corridor_type)
        self.agents = []
        self._prev_distances = []
        self._prev_velocities = []
        
        other_obs = (num_agents - 1) * 4
        self.obs_dim = 2 + 2 + 2 + 1 + self.num_rays + other_obs
        self.action_dim = 2
        
    def _create_obstacles(self, corridor_type):
        obstacles = []
        if corridor_type == 'single_wall':
            obstacles.append({'x': self.width * 0.5 - 1.0, 'y': 0, 'width': 2.0, 'height': self.height * 0.55})
        elif corridor_type == 'double_wall':
            obstacles.append({'x': self.width * 0.35 - 1.0, 'y': 0, 'width': 2.0, 'height': self.height * 0.45})
            obstacles.append({'x': self.width * 0.65 - 1.0, 'y': self.height * 0.55, 'width': 2.0, 'height': self.height * 0.45})
        elif corridor_type == 'narrow_passage':
            obstacles.append({'x': self.width * 0.5 - 1.0, 'y': 0, 'width': 2.0, 'height': self.height * 0.35})
            obstacles.append({'x': self.width * 0.5 - 1.0, 'y': self.height * 0.65, 'width': 2.0, 'height': self.height * 0.35})
        return obstacles
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.agents = []
        self._prev_distances = []
        self._prev_velocities = []
        
        y_spacing = self.height / (self.num_agents + 1)
        for i in range(self.num_agents):
            x = np.random.uniform(2.0, 6.0)
            y = y_spacing * (i + 1) + np.random.uniform(-0.5, 0.5)
            y = np.clip(y, self.agent_radius + 0.5, self.height - self.agent_radius - 0.5)
            self.agents.append({'x': x, 'y': y, 'vx': 0.0, 'vy': 0.0, 'reached_goal': False, 'collided': False})
            self._prev_distances.append(np.linalg.norm(self.goal_position - np.array([x, y])))
            self._prev_velocities.append(np.array([0.0, 0.0]))
        return self._get_observations()
    
    def _cast_ray(self, agent_idx, angle):
        agent = self.agents[agent_idx]
        x, y = agent['x'], agent['y']
        dx, dy = np.cos(angle), np.sin(angle)
        for dist in np.linspace(0.2, self.ray_length, 30):
            cx, cy = x + dx * dist, y + dy * dist
            if cx < 0 or cx > self.width or cy < 0 or cy > self.height:
                return dist / self.ray_length
            for obs in self.obstacles:
                if obs['x'] <= cx <= obs['x'] + obs['width'] and obs['y'] <= cy <= obs['y'] + obs['height']:
                    return dist / self.ray_length
            for j, other in enumerate(self.agents):
                if j != agent_idx and np.sqrt((cx - other['x'])**2 + (cy - other['y'])**2) < self.agent_radius * 2:
                    return dist / self.ray_length
        return 1.0
    
    def _get_observations(self):
        observations = []
        for i, agent in enumerate(self.agents):
            obs = []
            pos = np.array([agent['x'], agent['y']])
            obs.extend([(agent['x'] / self.width) * 2 - 1, (agent['y'] / self.height) * 2 - 1])
            obs.extend([agent['vx'] / self.max_speed, agent['vy'] / self.max_speed])
            goal_vec = self.goal_position - pos
            goal_dist = np.linalg.norm(goal_vec)
            goal_dir = goal_vec / goal_dist if goal_dist > 0.01 else np.array([1.0, 0.0])
            obs.extend([goal_dir[0], goal_dir[1], min(goal_dist / self.width, 1.0)])
            for j in range(self.num_rays):
                obs.append(self._cast_ray(i, j * (2 * np.pi / self.num_rays)))
            for j, other in enumerate(self.agents):
                if i != j:
                    obs.extend([np.clip((other['x'] - agent['x']) / 10.0, -2, 2),
                               np.clip((other['y'] - agent['y']) / 10.0, -2, 2),
                               np.clip((other['vx'] - agent['vx']) / self.max_speed, -2, 2),
                               np.clip((other['vy'] - agent['vy']) / self.max_speed, -2, 2)])
            observations.append(np.array(obs, dtype=np.float32))
        return observations
    
    def _check_collision(self, agent_idx):
        agent = self.agents[agent_idx]
        x, y, r = agent['x'], agent['y'], self.agent_radius
        if x < r or x > self.width - r or y < r or y > self.height - r:
            return True
        for obs in self.obstacles:
            if obs['x'] - r < x < obs['x'] + obs['width'] + r and obs['y'] - r < y < obs['y'] + obs['height'] + r:
                return True
        for j, other in enumerate(self.agents):
            if j != agent_idx and not other['reached_goal']:
                if np.sqrt((x - other['x'])**2 + (y - other['y'])**2) < self.agent_radius * 2:
                    return True
        return False
    
    def step(self, actions):
        self.current_step += 1
        dt = 0.1
        for i, agent in enumerate(self.agents):
            if agent['reached_goal'] or agent['collided']:
                continue
            target_vx, target_vy = actions[i][0] * self.max_speed, actions[i][1] * self.max_speed
            agent['vx'] += (target_vx - agent['vx']) * self.acceleration
            agent['vy'] += (target_vy - agent['vy']) * self.acceleration
            speed = np.sqrt(agent['vx']**2 + agent['vy']**2)
            if speed > self.max_speed:
                agent['vx'], agent['vy'] = agent['vx'] / speed * self.max_speed, agent['vy'] / speed * self.max_speed
            agent['x'] = np.clip(agent['x'] + agent['vx'] * dt, self.agent_radius, self.width - self.agent_radius)
            agent['y'] = np.clip(agent['y'] + agent['vy'] * dt, self.agent_radius, self.height - self.agent_radius)
        
        for i, agent in enumerate(self.agents):
            if agent['reached_goal'] or agent['collided']:
                continue
            if self._check_collision(i):
                agent['collided'] = True
            elif np.linalg.norm(self.goal_position - np.array([agent['x'], agent['y']])) < self.goal_radius:
                agent['reached_goal'] = True
        
        done = all(a['reached_goal'] or a['collided'] for a in self.agents) or self.current_step >= self.max_steps
        info = {'success_count': sum(1 for a in self.agents if a['reached_goal'])}
        return self._get_observations(), done, info


# ==========================================
# POLICY NETWORK
# ==========================================

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs):
        return torch.tanh(self.mean_head(self.net(torch.clamp(obs, -5.0, 5.0))))


# ==========================================
# VIDEO GENERATOR - PROPER ANIMATION
# ==========================================

def generate_video(env, policy, device, output_path, seed=42):
    """Generate REAL animated video."""
    
    print(f"\n  Recording episode...")
    
    # Run episode and collect ALL states
    obs = env.reset(seed=seed)
    trajectory = []
    done = False
    
    while not done:
        # Save current state
        state = [(a['x'], a['y'], a['vx'], a['vy'], a['reached_goal'], a['collided']) for a in env.agents]
        trajectory.append(state)
        
        # Get actions
        obs_tensor = torch.FloatTensor(np.array(obs)).to(device)
        with torch.no_grad():
            actions = [policy(obs_tensor[i]).cpu().numpy() for i in range(env.num_agents)]
        
        obs, done, info = env.step(actions)
    
    # Final state
    trajectory.append([(a['x'], a['y'], a['vx'], a['vy'], a['reached_goal'], a['collided']) for a in env.agents])
    
    success = info['success_count']
    print(f"    Episode: {len(trajectory)} frames, {success}/{env.num_agents} reached goal")
    
    # Subsample for reasonable file size (every 2nd frame)
    trajectory = trajectory[::2]
    print(f"    Rendering {len(trajectory)} frames...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=80)
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # Store trajectory trails
    trails = [[] for _ in range(env.num_agents)]
    
    def animate(frame_idx):
        ax.clear()
        
        state = trajectory[frame_idx]
        
        # Update trails
        for i, (x, y, vx, vy, reached, collided) in enumerate(state):
            trails[i].append((x, y))
            if len(trails[i]) > 60:
                trails[i] = trails[i][-60:]
        
        # Setup axes
        ax.set_xlim(-1, env.width + 1)
        ax.set_ylim(-1, env.height + 1)
        ax.set_aspect('equal')
        ax.set_facecolor('#fafafa')
        
        # Corridor
        ax.fill([0, env.width, env.width, 0], [0, 0, env.height, env.height],
                color='white', edgecolor='#333', linewidth=2)
        
        # Start zone
        ax.fill([0, 7, 7, 0], [0, 0, env.height, env.height], color='#3498db', alpha=0.1)
        ax.text(3.5, env.height + 0.5, 'START', ha='center', fontsize=10, fontweight='bold', color='#2980b9')
        
        # Goal
        goal_circle = plt.Circle(env.goal_position, env.goal_radius, color='#27ae60', alpha=0.3)
        ax.add_patch(goal_circle)
        ax.plot(env.goal_position[0], env.goal_position[1], '*', color='#27ae60', markersize=20)
        ax.text(env.goal_position[0], env.height + 0.5, 'GOAL', ha='center', fontsize=10, fontweight='bold', color='#27ae60')
        
        # Obstacles
        for obs in env.obstacles:
            rect = patches.Rectangle((obs['x'], obs['y']), obs['width'], obs['height'],
                                     color='#795548', ec='#5d4037', linewidth=2)
            ax.add_patch(rect)
        
        # Draw trails
        for i, trail in enumerate(trails):
            if len(trail) > 1:
                xs, ys = zip(*trail)
                for j in range(len(xs) - 1):
                    alpha = 0.1 + 0.6 * (j / len(xs))
                    ax.plot(xs[j:j+2], ys[j:j+2], color=colors[i], alpha=alpha, linewidth=2.5)
        
        # Draw agents
        for i, (x, y, vx, vy, reached, collided) in enumerate(state):
            if reached:
                color, marker = '#27ae60', 's'
            elif collided:
                color, marker = '#c0392b', 'X'
            else:
                color, marker = colors[i], 'o'
            
            ax.scatter(x, y, c=color, s=300, marker=marker, edgecolors='black', linewidths=2, zorder=10)
            ax.text(x, y, str(i+1), ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=11)
            
            # Velocity arrow
            speed = np.sqrt(vx**2 + vy**2)
            if speed > 0.05 and not reached and not collided:
                ax.arrow(x, y, vx*2.5, vy*2.5, head_width=0.3, head_length=0.15,
                        fc=color, ec='black', linewidth=0.5, zorder=9)
        
        # Title and info
        reached_count = sum(1 for s in state if s[4])
        collided_count = sum(1 for s in state if s[5])
        ax.set_title(f'{env.corridor_type.upper()} | Frame {frame_idx+1}/{len(trajectory)} | '
                    f'Reached: {reached_count}/3 | Collided: {collided_count}/3',
                    fontsize=11, fontweight='bold')
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        return []
    
    # Create animation with matplotlib
    print(f"    Creating animation...")
    anim = FuncAnimation(fig, animate, frames=len(trajectory), interval=100, blit=True)
    
    # Save with PillowWriter (creates REAL animated GIF)
    print(f"    Saving to {output_path}...")
    writer = PillowWriter(fps=15)
    anim.save(output_path, writer=writer)
    
    plt.close(fig)
    
    # Reset trails for next video
    for t in trails:
        t.clear()
    
    print(f"    Done! {os.path.getsize(output_path)/1024:.1f} KB")
    return success


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model_dir = "models_robust_20251201_071452"
    output_dir = "videos"
    os.makedirs(output_dir, exist_ok=True)
    
    corridors = ['straight', 'single_wall', 'double_wall', 'narrow_passage']
    
    for corridor in corridors:
        model_path = f"{model_dir}/{corridor}_final.pt"
        
        if not os.path.exists(model_path):
            print(f"\nModel not found: {model_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"  {corridor.upper()}")
        print(f"{'='*60}")
        
        env = RobustCorridorEnv(num_agents=3, corridor_type=corridor, max_steps=500)
        policy = PolicyNetwork(env.obs_dim, env.action_dim).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint['policy'])
        policy.eval()
        print(f"  Model loaded: {model_path}")
        
        # Generate 2 videos
        for ep, seed in enumerate([42, 999]):
            video_path = f"{output_dir}/{corridor}_{ep+1}.gif"
            generate_video(env, policy, device, video_path, seed=seed)
    
    print(f"\n{'='*60}")
    print(f"  ALL VIDEOS SAVED TO: {output_dir}/")
    print(f"{'='*60}")
    
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f)) / 1024
        print(f"  - {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
