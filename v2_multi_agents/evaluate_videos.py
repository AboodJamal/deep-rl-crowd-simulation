"""
Evaluation and Video Generation for multii_test_Abd2 ROBUST v2
==============================================================
Generate GIF videos showing how agents navigate corridors
Self-contained - all classes defined here
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from datetime import datetime

# ==========================================
# ENVIRONMENT (matches trainer_ROBUST_v2.py)
# ==========================================

class RobustCorridorEnv:
    """Robust corridor environment with DIRECT velocity control."""
    
    def __init__(self, num_agents=3, corridor_type='straight', max_steps=800):
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
        self.acceleration = 0.3
        
        # Raycasting
        self.num_rays = 16
        self.ray_length = 8.0
        
        # Goal
        self.goal_radius = 3.0
        self.goal_threshold = self.goal_radius  # For compatibility
        self.goal_position = np.array([self.width - 5.0, self.height / 2])
        
        # Obstacles
        self.obstacles = self._create_obstacles(corridor_type)
        
        # State
        self.agents = []
        self._prev_distances = []
        self._prev_velocities = []
        
        # Observation/Action specs
        other_obs = (num_agents - 1) * 4
        self.obs_dim = 2 + 2 + 2 + 1 + self.num_rays + other_obs
        self.action_dim = 2
        
        # Namespace for gym-like interface
        class Space:
            def __init__(self, shape):
                self.shape = shape
        self.observation_space = Space((self.obs_dim,))
        self.action_space = Space((self.action_dim,))
        
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
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.agents = []
        self._prev_distances = []
        self._prev_velocities = []
        
        # Spread agents in start area
        y_spacing = self.height / (self.num_agents + 1)
        
        for i in range(self.num_agents):
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
        """Cast ray in WORLD coordinates."""
        agent = self.agents[agent_idx]
        x, y = agent['x'], agent['y']
        
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        for dist in np.linspace(0.2, self.ray_length, 30):
            cx = x + dx * dist
            cy = y + dy * dist
            
            if cx < 0 or cx > self.width or cy < 0 or cy > self.height:
                return dist / self.ray_length
            
            for obs in self.obstacles:
                if (obs['x'] <= cx <= obs['x'] + obs['width'] and
                    obs['y'] <= cy <= obs['y'] + obs['height']):
                    return dist / self.ray_length
            
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
            
            # Position (normalized)
            obs.append((agent['x'] / self.width) * 2 - 1)
            obs.append((agent['y'] / self.height) * 2 - 1)
            
            # Velocity (normalized)
            obs.append(agent['vx'] / self.max_speed)
            obs.append(agent['vy'] / self.max_speed)
            
            # Goal direction
            goal_vec = self.goal_position - pos
            goal_dist = np.linalg.norm(goal_vec)
            if goal_dist > 0.01:
                goal_dir = goal_vec / goal_dist
            else:
                goal_dir = np.array([1.0, 0.0])
            obs.append(goal_dir[0])
            obs.append(goal_dir[1])
            
            # Goal distance
            obs.append(min(goal_dist / self.width, 1.0))
            
            # Raycasting
            for j in range(self.num_rays):
                angle = j * (2 * np.pi / self.num_rays)
                obs.append(self._cast_ray(i, angle))
            
            # Other agents
            for j, other in enumerate(self.agents):
                if i != j:
                    rel_x = (other['x'] - agent['x']) / 10.0
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
        """Check collision."""
        agent = self.agents[agent_idx]
        x, y = agent['x'], agent['y']
        r = self.agent_radius
        
        if x < r or x > self.width - r or y < r or y > self.height - r:
            return True
        
        for obs in self.obstacles:
            if (obs['x'] - r < x < obs['x'] + obs['width'] + r and
                obs['y'] - r < y < obs['y'] + obs['height'] + r):
                return True
        
        for j, other in enumerate(self.agents):
            if j != agent_idx and not other['reached_goal']:
                dist = np.sqrt((x - other['x'])**2 + (y - other['y'])**2)
                if dist < self.agent_radius * 2:
                    return True
        
        return False
    
    def step(self, actions):
        """Execute one step."""
        self.current_step += 1
        dt = 0.1
        
        for i, agent in enumerate(self.agents):
            if agent['reached_goal'] or agent['collided']:
                continue
            
            agent['steps_taken'] += 1
            
            target_vx = actions[i][0] * self.max_speed
            target_vy = actions[i][1] * self.max_speed
            
            agent['vx'] += (target_vx - agent['vx']) * self.acceleration
            agent['vy'] += (target_vy - agent['vy']) * self.acceleration
            
            speed = np.sqrt(agent['vx']**2 + agent['vy']**2)
            if speed > self.max_speed:
                agent['vx'] = (agent['vx'] / speed) * self.max_speed
                agent['vy'] = (agent['vy'] / speed) * self.max_speed
            
            new_x = agent['x'] + agent['vx'] * dt
            new_y = agent['y'] + agent['vy'] * dt
            
            new_x = np.clip(new_x, self.agent_radius, self.width - self.agent_radius)
            new_y = np.clip(new_y, self.agent_radius, self.height - self.agent_radius)
            
            agent['x'] = new_x
            agent['y'] = new_y
        
        for i, agent in enumerate(self.agents):
            if agent['reached_goal'] or agent['collided']:
                continue
            
            if self._check_collision(i):
                agent['collided'] = True
                continue
            
            dist = np.linalg.norm(self.goal_position - np.array([agent['x'], agent['y']]))
            if dist < self.goal_radius:
                agent['reached_goal'] = True
        
        dones = [a['reached_goal'] or a['collided'] for a in self.agents]
        truncs = [self.current_step >= self.max_steps] * self.num_agents
        
        success_count = sum(1 for a in self.agents if a['reached_goal'])
        
        info = {
            'success_rate': success_count / self.num_agents,
            'success_count': success_count,
            'collision_count': sum(1 for a in self.agents if a['collided']),
            'episode_length': self.current_step
        }
        
        rewards = [0.0] * self.num_agents  # Not used for eval
        
        return self._get_observations(), rewards, dones, truncs, info


# ==========================================
# POLICY NETWORK (matches trainer_ROBUST_v2.py)
# ==========================================

class PolicyNetwork(nn.Module):
    """Policy network - must match trainer_ROBUST_v2.py exactly."""
    
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
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs):
        obs = torch.clamp(obs, -5.0, 5.0)
        features = self.net(obs)
        mean = self.mean_head(features)
        mean = torch.tanh(mean)
        return mean
    
    def get_action(self, obs, deterministic=True):
        mean = self.forward(obs)
        if deterministic:
            return mean
        std = torch.exp(torch.clamp(self.log_std, -2.0, 0.5))
        dist = Normal(mean, std)
        return torch.clamp(dist.rsample(), -1.0, 1.0)


# ==========================================
# VIDEO EVALUATOR
# ==========================================

class VideoEvaluator:
    def __init__(self, model_path, corridor_type, device='cpu'):
        self.device = device
        self.corridor_type = corridor_type
        
        # Create environment
        self.env = RobustCorridorEnv(num_agents=3, corridor_type=corridor_type, max_steps=800)
        
        # Load model
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.policy = PolicyNetwork(obs_dim, action_dim).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Load the policy weights - checkpoint has 'policy' key
        if 'policy' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy'])
        else:
            self.policy.load_state_dict(checkpoint)
        
        self.policy.eval()
        print(f"Loaded model: {model_path}")
    
    def run_episode(self):
        """Run episode and collect trajectory data"""
        obs, _ = self.env.reset()
        
        trajectory = {
            'positions': [],
            'velocities': [],
            'reached': [],
            'collisions': []
        }
        
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < self.env.max_steps:
            # Record state
            positions = [(a['x'], a['y']) for a in self.env.agents]
            velocities = [(a['vx'], a['vy']) for a in self.env.agents]
            reached = [a['reached_goal'] for a in self.env.agents]
            
            trajectory['positions'].append(positions)
            trajectory['velocities'].append(velocities)
            trajectory['reached'].append(reached)
            
            # Get actions - PolicyNetwork.forward returns just mean
            obs_tensor = torch.FloatTensor(np.array(obs)).to(self.device)
            
            with torch.no_grad():
                actions = []
                for i in range(self.env.num_agents):
                    mean = self.policy(obs_tensor[i])  # forward returns mean only
                    action = mean.cpu().numpy()
                    action = np.clip(action, -1, 1)
                    actions.append(action)
            
            obs, rewards, dones, truncs, info = self.env.step(actions)
            total_reward += sum(rewards)
            done = all(dones) or all(truncs)
            step += 1
        
        # Final state
        positions = [(a['x'], a['y']) for a in self.env.agents]
        trajectory['positions'].append(positions)
        trajectory['reached'].append([a['reached_goal'] for a in self.env.agents])
        
        success_rate = sum(1 for a in self.env.agents if a['reached_goal']) / self.env.num_agents
        
        return trajectory, total_reward, success_rate, step
    
    def generate_video(self, output_path, fps=20):
        """Generate GIF video of an episode"""
        print(f"Running episode for {self.corridor_type}...")
        trajectory, reward, success, steps = self.run_episode()
        
        print(f"  Success: {success*100:.0f}%, Steps: {steps}, Reward: {reward:.1f}")
        
        positions = trajectory['positions']
        num_frames = len(positions)
        
        # Colors for agents
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        fig, ax = plt.subplots(figsize=(14, 5), dpi=100)
        
        frames = []
        
        for frame in range(num_frames):
            ax.clear()
            
            # Set limits
            ax.set_xlim(-2, self.env.width + 2)
            ax.set_ylim(-2, self.env.height + 2)
            ax.set_aspect('equal')
            
            # Draw corridor
            ax.fill([0, self.env.width, self.env.width, 0], 
                    [0, 0, self.env.height, self.env.height], 
                    color='#f5f5f5', alpha=0.8)
            ax.plot([0, self.env.width, self.env.width, 0, 0], 
                    [0, 0, self.env.height, self.env.height, 0], 
                    'k-', linewidth=2)
            
            # Draw goal
            goal = self.env.goal_position
            goal_circle = plt.Circle(goal, self.env.goal_threshold, 
                                    color='#2ecc71', alpha=0.4)
            ax.add_patch(goal_circle)
            ax.plot(goal[0], goal[1], 'g*', markersize=20)
            ax.annotate('GOAL', (goal[0], goal[1] + 2), ha='center', 
                       fontsize=10, fontweight='bold', color='green')
            
            # Draw obstacles
            for obs in self.env.obstacles:
                rect = patches.Rectangle(
                    (obs['x'], obs['y']), obs['width'], obs['height'],
                    linewidth=2, edgecolor='#8B4513', facecolor='#CD853F', alpha=0.8
                )
                ax.add_patch(rect)
            
            # Draw trajectories (trail)
            trail_len = min(40, frame)
            for agent_idx in range(self.env.num_agents):
                if frame > 0:
                    start = max(0, frame - trail_len)
                    trail_x = [positions[f][agent_idx][0] for f in range(start, frame + 1)]
                    trail_y = [positions[f][agent_idx][1] for f in range(start, frame + 1)]
                    
                    # Gradient trail
                    for i in range(len(trail_x) - 1):
                        alpha = 0.1 + 0.6 * (i / len(trail_x))
                        ax.plot(trail_x[i:i+2], trail_y[i:i+2], 
                               color=colors[agent_idx], alpha=alpha, linewidth=2.5)
            
            # Draw agents
            current_pos = positions[frame]
            current_reached = trajectory['reached'][min(frame, len(trajectory['reached'])-1)]
            
            for agent_idx, (x, y) in enumerate(current_pos):
                if current_reached[agent_idx]:
                    color = '#2ecc71'  # Green for reached
                    marker = 's'
                    size = 300
                else:
                    color = colors[agent_idx]
                    marker = 'o'
                    size = 250
                
                ax.scatter(x, y, c=color, s=size, marker=marker,
                          edgecolors='black', linewidths=2, zorder=10)
                ax.annotate(f'A{agent_idx+1}', (x, y + 1.5), ha='center',
                           fontsize=9, fontweight='bold')
            
            # Info
            reached_count = sum(current_reached)
            ax.set_title(f'ROBUST v2 - {self.corridor_type.upper()} | Step: {frame}/{num_frames-1} | Reached: {reached_count}/3',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)
            
            # Convert to image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(image[:, :, :3])  # Remove alpha channel
        
        plt.close(fig)
        
        # Save as GIF using PIL (more reliable than imageio)
        print(f"  Saving {len(frames)} frames to {output_path}...")
        
        try:
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000/fps),  # ms per frame
                loop=0
            )
        except ImportError:
            # Fallback to imageio
            import imageio
            imageio.mimsave(output_path, frames, fps=fps, loop=0)
        
        print(f"  ✓ Saved: {output_path}")
        return success, steps
    
    def evaluate(self, num_episodes=20):
        """Evaluate model over multiple episodes"""
        successes = []
        lengths = []
        
        for ep in range(num_episodes):
            _, _, success, length = self.run_episode()
            successes.append(success)
            lengths.append(length)
        
        return {
            'success_rate': np.mean(successes),
            'std_success': np.std(successes),
            'avg_length': np.mean(lengths)
        }


def main():
    print("=" * 70)
    print("  MULTII_TEST_ABD2 ROBUST v2 - Evaluation & Video Generation")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Model directory
    model_dir = "/p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/multii_test_Abd2/models_robust_20251201_071452"
    output_dir = "/p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/multii_test_Abd2/evaluation_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Corridor types and models
    corridors = ['straight', 'single_wall', 'double_wall', 'narrow_passage']
    
    all_results = {}
    
    for corridor in corridors:
        model_path = f"{model_dir}/{corridor}_final.pt"
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        
        print(f"\n{'#' * 60}")
        print(f"  {corridor.upper()}")
        print(f"{'#' * 60}")
        
        evaluator = VideoEvaluator(model_path, corridor, device)
        
        # Generate 3 videos per corridor
        for ep in range(3):
            video_path = f"{output_dir}/{corridor}_episode_{ep+1}.gif"
            evaluator.generate_video(video_path, fps=15)
        
        # Full evaluation
        print(f"\n  Running full evaluation (20 episodes)...")
        results = evaluator.evaluate(20)
        all_results[corridor] = results
        
        print(f"  Success Rate: {results['success_rate']*100:.1f}% ± {results['std_success']*100:.1f}%")
        print(f"  Avg Length: {results['avg_length']:.1f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION SUMMARY")
    print("=" * 70)
    
    for corridor, stats in all_results.items():
        print(f"\n{corridor}:")
        print(f"  Success Rate: {stats['success_rate']*100:.1f}% ± {stats['std_success']*100:.1f}%")
        print(f"  Avg Episode Length: {stats['avg_length']:.1f}")
    
    # Save results
    results_path = f"{output_dir}/evaluation_results.txt"
    with open(results_path, 'w') as f:
        f.write("MULTII_TEST_ABD2 ROBUST v2 EVALUATION RESULTS\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write("=" * 50 + "\n\n")
        
        for corridor, stats in all_results.items():
            f.write(f"{corridor}:\n")
            f.write(f"  Success Rate: {stats['success_rate']*100:.1f}% ± {stats['std_success']*100:.1f}%\n")
            f.write(f"  Avg Episode Length: {stats['avg_length']:.1f}\n\n")
    
    print(f"\nResults saved to: {results_path}")
    print(f"Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
