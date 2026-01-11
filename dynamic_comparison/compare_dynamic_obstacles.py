"""
Dynamic Obstacle Comparison: DRL vs VGA-UPL
===========================================

Compare DRL agent (trained on dynamic obstacles) against VGA-UPL baseline
on the same dynamic obstacle scenarios.

Expected outcome:
- DRL: Should succeed (trained specifically for this)
- VGA-UPL: May fail (not designed for moving obstacles)
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path

# Add paths
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import json
from datetime import datetime

# Import VGA-UPL planner
from vga_upl_baseline.models.vga_upl_planner_v4 import VGAUPLPlannerV4, Obstacle


class DynamicCorridorEnv(gym.Env):
    """
    Corridor with dynamic obstacles.
    This is the environment the DRL agent was trained on.
    """
    
    def __init__(self, max_steps=400, seed=None):
        super().__init__()
        self.max_steps = max_steps
        self.manual_seed = seed
        
        # Corridor dimensions
        self.width = 12.0
        self.height = 6.0
        
        # Robot parameters
        self.robot_radius = 0.25
        self.max_linear_vel = 1.34  # Fair speed for comparison
        self.max_angular_vel = 2.0
        self.dt = 0.1
        
        # Goal
        self.goal_threshold = 0.4
        
        # Dynamic obstacles
        self.num_dynamic_obs = 3
        
        # Raycasts
        self.n_rays = 12
        self.ray_length = 3.0
        
        obs_dim = 6 + self.n_rays
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        elif self.manual_seed is not None:
            np.random.seed(self.manual_seed)
        
        # Varied start and goal positions for different trials
        # Start positions: left side of corridor
        start_y_options = [1.5, 2.5, 3.0, 3.5, 4.5]  # Different vertical positions
        start_y = start_y_options[seed % len(start_y_options)] if seed is not None else self.height / 2.0
        
        self.pos = np.array([1.0, start_y])
        self.theta = 0.0
        self.vel = np.array([0.0, 0.0])
        
        # Goal positions: right side of corridor
        goal_y_options = [1.5, 2.5, 3.0, 3.5, 4.5]  # Different vertical positions
        goal_y = goal_y_options[(seed + 2) % len(goal_y_options)] if seed is not None else self.height / 2.0
        
        self.goal = np.array([self.width - 1.0, goal_y])
        
        # 3 dynamic obstacles
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
        """Move obstacles up/down"""
        for i, obs in enumerate(self.dynamic_obstacles):
            obs['pos'][1] += obs['vel_y'] * self.dt
            
            # Bounce at boundaries
            if obs['pos'][1] <= obs['y_min']:
                obs['pos'][1] = obs['y_min']
                obs['vel_y'] = abs(obs['vel_y'])
            elif obs['pos'][1] >= obs['y_max']:
                obs['pos'][1] = obs['y_max']
                obs['vel_y'] = -abs(obs['vel_y'])
            
            # Record trajectory
            self.dynamic_trajectories[i].append(obs['pos'].copy())
    
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
        
        # Update obstacles first
        self._update_dynamic_obstacles()
        
        # Apply action
        linear_vel = (action[0] + 1.0) / 2.0 * self.max_linear_vel
        angular_vel = action[1] * self.max_angular_vel
        
        # Save previous position for collision handling
        prev_pos = self.pos.copy()
        prev_theta = self.theta
        
        # Update robot
        self.theta += angular_vel * self.dt
        self.pos[0] += linear_vel * np.cos(self.theta) * self.dt
        self.pos[1] += linear_vel * np.sin(self.theta) * self.dt
        self.vel = np.array([linear_vel, angular_vel])
        
        # Collision check - if collision, revert position and FAIL
        collision = self._check_collision()
        if collision:
            # Revert to previous position (agent cannot pass through!)
            self.pos = prev_pos
            self.theta = prev_theta
            self.collision_count += 1
            
            self.trajectory.append(self.pos.copy())
            
            # COLLISION = FAILURE
            return self._get_obs(), 0.0, True, False, {
                'success': False,
                'collision': True,
                'collision_count': self.collision_count,
                'failure_reason': 'collision'
            }
        
        self.trajectory.append(self.pos.copy())
        
        # Calculate distance to goal
        dist = np.linalg.norm(self.goal - self.pos)
        
        self.prev_dist = dist
        
        # Termination
        done = False
        truncated = False
        success = False
        
        if dist < self.goal_threshold:
            done = True
            success = True
        elif self.steps >= self.max_steps:
            truncated = True
        
        return self._get_obs(), 0.0, done, truncated, {
            'success': success,
            'collision': False,
            'collision_count': self.collision_count
        }


def run_drl_trial(model, env, trial_num):
    """Run DRL agent on dynamic obstacle environment"""
    print(f"  Running DRL trial {trial_num}...")
    
    obs, _ = env.reset(seed=trial_num * 100)
    
    trajectory = [env.pos.copy()]
    dynamic_trajectories = [[o['pos'].copy()] for o in env.dynamic_obstacles]
    
    done = False
    truncated = False
    steps = 0
    collisions = 0
    collision_step = None
    
    while not (done or truncated) and steps < env.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(action)
        
        trajectory.append(env.pos.copy())
        for i, obs_traj in enumerate(dynamic_trajectories):
            obs_traj.append(env.dynamic_obstacles[i]['pos'].copy())
        
        if info.get('collision', False):
            collisions += 1
            if collision_step is None:
                collision_step = steps
            # Collision causes failure, so done=True from environment
        
        steps += 1
    
    success = info.get('success', False)
    failure_reason = info.get('failure_reason', 'timeout' if truncated else None)
    
    return {
        'method': 'DRL',
        'trial': trial_num,
        'success': success,
        'steps': steps,
        'collisions': collisions,
        'collision_step': collision_step,
        'failure_reason': failure_reason,
        'trajectory': np.array(trajectory),
        'dynamic_trajectories': [np.array(dt) for dt in dynamic_trajectories],
        'start': trajectory[0],
        'goal': env.goal.copy(),
        'obstacles_start': [o['pos'].copy() for o in env.dynamic_obstacles],
        'obstacles_radius': [o['radius'] for o in env.dynamic_obstacles],
        'obstacles_vel': [o['vel_y'] for o in env.dynamic_obstacles],
        'corridor_size': (env.width, env.height)
    }


def run_vgaupl_trial(env, trial_num):
    """Run VGA-UPL planner on dynamic obstacle environment"""
    print(f"  Running VGA-UPL trial {trial_num}...")
    
    obs, _ = env.reset(seed=trial_num * 100)
    
    # Initialize VGA-UPL planner
    planner = VGAUPLPlannerV4(
        agent_radius=env.robot_radius,
        desired_speed=1.34,  # FAIR: Same max speed as DRL (1.34 m/s)
        dt=env.dt
    )
    
    # Set initial state
    planner.pos = env.pos.copy()
    planner.vel = np.array([0.0, 0.0])
    planner.goal = env.goal.copy()
    
    # VGA-UPL expects static obstacles, but we'll give it the current positions
    # It won't be able to predict their movement!
    planner.obstacles = [
        Obstacle(o['pos'].copy(), o['radius']) for o in env.dynamic_obstacles
    ]
    
    trajectory = [env.pos.copy()]
    dynamic_trajectories = [[o['pos'].copy()] for o in env.dynamic_obstacles]
    
    steps = 0
    collisions = 0
    collision_step = None
    success = False
    failure_reason = None
    
    while steps < env.max_steps:
        # VGA-UPL computes next action
        # It needs to update its obstacle positions (but can't predict movement)
        planner.obstacles = [
            Obstacle(o['pos'].copy(), o['radius']) for o in env.dynamic_obstacles
        ]
        
        try:
            # Get VGA-UPL action
            planner.step()
            
            # Convert VGA-UPL's velocity to action format
            # VGA-UPL uses velocity directly, DRL uses action space [-1, 1]
            # We need to apply VGA-UPL's velocity to the environment
            
            # Get direction and speed from VGA-UPL
            if np.linalg.norm(planner.vel) > 0:
                direction = np.arctan2(planner.vel[1], planner.vel[0])
                speed = np.linalg.norm(planner.vel)
            else:
                direction = 0
                speed = 0
            
            # Convert to action space
            # Action[0] controls linear velocity (mapped from [-1,1] to [0, max_vel])
            # Action[1] controls angular velocity
            
            # Calculate angular difference
            angle_diff = direction - env.theta
            while angle_diff > np.pi:
                angle_diff -= 2*np.pi
            while angle_diff < -np.pi:
                angle_diff += 2*np.pi
            
            # Create action
            linear_action = (speed / env.max_linear_vel) * 2.0 - 1.0
            angular_action = np.clip(angle_diff / env.max_angular_vel, -1.0, 1.0)
            
            action = np.array([linear_action, angular_action])
            
        except Exception as e:
            # If VGA-UPL fails, stop
            print(f"    VGA-UPL planning failed: {e}")
            action = np.array([0.0, 0.0])
        
        # Step environment
        obs, _, done, truncated, info = env.step(action)
        
        # Update planner position
        planner.pos = env.pos.copy()
        
        trajectory.append(env.pos.copy())
        for i, obs_traj in enumerate(dynamic_trajectories):
            obs_traj.append(env.dynamic_obstacles[i]['pos'].copy())
        
        if info.get('collision', False):
            collisions += 1
            if collision_step is None:
                collision_step = steps
            failure_reason = 'collision'
            # Collision causes failure, episode ends
            break
        
        if done:
            success = info.get('success', False)
            break
        
        if truncated:
            failure_reason = 'timeout'
            break
        
        steps += 1
    
    return {
        'method': 'VGA-UPL',
        'trial': trial_num,
        'success': success,
        'steps': steps,
        'collisions': collisions,
        'collision_step': collision_step,
        'failure_reason': failure_reason,
        'trajectory': np.array(trajectory),
        'dynamic_trajectories': [np.array(dt) for dt in dynamic_trajectories],
        'start': trajectory[0],
        'goal': env.goal.copy(),
        'obstacles_start': [o['pos'].copy() for o in env.dynamic_obstacles],
        'obstacles_radius': [o['radius'] for o in env.dynamic_obstacles],
        'obstacles_vel': [o['vel_y'] for o in env.dynamic_obstacles],
        'corridor_size': (env.width, env.height)
    }


def create_comparison_video(drl_result, vgaupl_result, output_path):
    """Create side-by-side comparison video"""
    print(f"  Creating video: {output_path.name}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    max_steps = max(len(drl_result['trajectory']), len(vgaupl_result['trajectory']))
    
    width, height = drl_result['corridor_size']
    
    def init():
        for ax in [ax1, ax2]:
            ax.clear()
            ax.set_xlim(-0.5, width + 0.5)
            ax.set_ylim(-0.5, height + 0.5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Draw corridor walls
            ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'k-', linewidth=2)
        
        ax1.set_title(f'DRL Agent - {"SUCCESS" if drl_result["success"] else "FAILED"}', 
                     fontsize=14, fontweight='bold',
                     color='green' if drl_result['success'] else 'red')
        ax2.set_title(f'VGA-UPL Baseline - {"SUCCESS" if vgaupl_result["success"] else "FAILED"}',
                     fontsize=14, fontweight='bold',
                     color='green' if vgaupl_result['success'] else 'red')
        
        return []
    
    def animate(frame):
        for ax in [ax1, ax2]:
            ax.clear()
            ax.set_xlim(-0.5, width + 0.5)
            ax.set_ylim(-0.5, height + 0.5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Walls
            ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'k-', linewidth=2)
        
        # DRL
        drl_idx = min(frame, len(drl_result['trajectory']) - 1)
        drl_traj = drl_result['trajectory'][:drl_idx+1]
        
        # Plot DRL trajectory
        if len(drl_traj) > 1:
            ax1.plot(drl_traj[:, 0], drl_traj[:, 1], 'b-', alpha=0.6, linewidth=2, label='Path')
        
        # DRL agent
        agent_color = 'blue' if drl_result['success'] or frame < len(drl_result['trajectory']) - 1 else 'red'
        ax1.plot(drl_traj[-1, 0], drl_traj[-1, 1], 'o', color=agent_color, markersize=12, label='Agent')
        ax1.add_patch(plt.Circle(drl_traj[-1], 0.25, color=agent_color, alpha=0.3))
        
        # Mark collision point if exists
        if drl_result.get('collision_step') is not None and frame >= drl_result['collision_step']:
            collision_pos = drl_result['trajectory'][drl_result['collision_step']]
            ax1.plot(collision_pos[0], collision_pos[1], 'rx', markersize=20, markeredgewidth=3, label='Collision!')
        
        # DRL dynamic obstacles
        for i, (obs_traj, radius) in enumerate(zip(drl_result['dynamic_trajectories'], 
                                                    drl_result['obstacles_radius'])):
            obs_idx = min(frame, len(obs_traj) - 1)
            obs_pos = obs_traj[obs_idx]
            ax1.add_patch(plt.Circle(obs_pos, radius, color='red', alpha=0.6))
            # Show obstacle trail
            if len(obs_traj[:obs_idx+1]) > 1:
                ax1.plot(obs_traj[:obs_idx+1, 0], obs_traj[:obs_idx+1, 1], 
                        'r--', alpha=0.3, linewidth=1)
        
        # Goal
        ax1.plot(drl_result['goal'][0], drl_result['goal'][1], 'g*', markersize=20, label='Goal')
        
        status_text = "SUCCESS" if drl_result["success"] else f"FAILED ({drl_result.get('failure_reason', 'unknown')})"
        ax1.set_title(f'DRL - Step {drl_idx}/{len(drl_result["trajectory"])-1} - {status_text}',
                     fontsize=12, fontweight='bold',
                     color='green' if drl_result['success'] else 'red')
        ax1.legend(loc='upper left', fontsize=8)
        
        # VGA-UPL
        vga_idx = min(frame, len(vgaupl_result['trajectory']) - 1)
        vga_traj = vgaupl_result['trajectory'][:vga_idx+1]
        
        # Plot VGA-UPL trajectory
        if len(vga_traj) > 1:
            ax2.plot(vga_traj[:, 0], vga_traj[:, 1], 'purple', alpha=0.6, linewidth=2, label='Path')
        
        # VGA-UPL agent
        agent_color = 'purple' if vgaupl_result['success'] or frame < len(vgaupl_result['trajectory']) - 1 else 'red'
        ax2.plot(vga_traj[-1, 0], vga_traj[-1, 1], 'o', color=agent_color, markersize=12, label='Agent')
        ax2.add_patch(plt.Circle(vga_traj[-1], 0.25, color=agent_color, alpha=0.3))
        
        # Mark collision point if exists
        if vgaupl_result.get('collision_step') is not None and frame >= vgaupl_result['collision_step']:
            collision_pos = vgaupl_result['trajectory'][vgaupl_result['collision_step']]
            ax2.plot(collision_pos[0], collision_pos[1], 'rx', markersize=20, markeredgewidth=3, label='Collision!')
        
        # VGA-UPL dynamic obstacles (same as DRL)
        for i, (obs_traj, radius) in enumerate(zip(vgaupl_result['dynamic_trajectories'],
                                                    vgaupl_result['obstacles_radius'])):
            obs_idx = min(frame, len(obs_traj) - 1)
            obs_pos = obs_traj[obs_idx]
            ax2.add_patch(plt.Circle(obs_pos, radius, color='red', alpha=0.6))
            if len(obs_traj[:obs_idx+1]) > 1:
                ax2.plot(obs_traj[:obs_idx+1, 0], obs_traj[:obs_idx+1, 1],
                        'r--', alpha=0.3, linewidth=1)
        
        # Goal
        ax2.plot(vgaupl_result['goal'][0], vgaupl_result['goal'][1], 'g*', markersize=20, label='Goal')
        
        status_text = "SUCCESS" if vgaupl_result['success'] else f"FAILED ({vgaupl_result.get('failure_reason', 'unknown')})"
        ax2.set_title(f'VGA-UPL - Step {vga_idx}/{len(vgaupl_result["trajectory"])-1} - {status_text}',
                     fontsize=12, fontweight='bold',
                     color='green' if vgaupl_result['success'] else 'red')
        ax2.legend(loc='upper left', fontsize=8)
        
        return []
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=max_steps, interval=50, blit=True)
    
    writer = FFMpegWriter(fps=20, bitrate=2000)
    anim.save(output_path, writer=writer)
    plt.close()


def main():
    print("=" * 70)
    print("DYNAMIC OBSTACLE COMPARISON: DRL vs VGA-UPL")
    print("=" * 70)
    
    # Setup paths
    root_path = Path(__file__).parent.parent
    model_path = root_path / "drl_vga_experiments" / "dynamic_obstacles_working" / "trained_model" / "final_model.zip"
    
    output_dir = Path(__file__).parent
    video_dir = output_dir / "videos"
    results_dir = output_dir / "results"
    
    video_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Load DRL model
    print("\nLoading DRL model...")
    drl_model = PPO.load(model_path)
    print("✓ DRL model loaded")
    
    # Create environment
    env = DynamicCorridorEnv(max_steps=400)
    
    # Run trials
    n_trials = 5
    results = []
    
    print(f"\nRunning {n_trials} trials...")
    for trial in range(1, n_trials + 1):
        print(f"\nTrial {trial}/{n_trials}")
        
        # Run DRL
        drl_result = run_drl_trial(drl_model, env, trial)
        results.append(drl_result)
        
        # Run VGA-UPL (same seed)
        vgaupl_result = run_vgaupl_trial(env, trial)
        results.append(vgaupl_result)
        
        # Create comparison video
        video_path = video_dir / f"trial_{trial:02d}_comparison.mp4"
        create_comparison_video(drl_result, vgaupl_result, video_path)
    
    # Compute statistics
    drl_results = [r for r in results if r['method'] == 'DRL']
    vga_results = [r for r in results if r['method'] == 'VGA-UPL']
    
    drl_success_rate = sum(r['success'] for r in drl_results) / len(drl_results) * 100
    vga_success_rate = sum(r['success'] for r in vga_results) / len(vga_results) * 100
    
    drl_avg_collisions = np.mean([r['collisions'] for r in drl_results])
    vga_avg_collisions = np.mean([r['collisions'] for r in vga_results])
    
    drl_avg_steps = np.mean([r['steps'] for r in drl_results if r['success']])
    vga_avg_steps = np.mean([r['steps'] for r in vga_results if r['success']]) if vga_success_rate > 0 else 0
    
    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_trials': n_trials,
        'collision_causes_failure': True,
        'drl': {
            'success_rate': drl_success_rate,
            'avg_collisions': float(drl_avg_collisions),
            'avg_steps_success': float(drl_avg_steps) if drl_success_rate > 0 else None,
            'trials': [{
                'trial': r['trial'],
                'success': r['success'],
                'steps': r['steps'],
                'collisions': r['collisions'],
                'collision_step': r.get('collision_step'),
                'failure_reason': r.get('failure_reason')
            } for r in drl_results]
        },
        'vgaupl': {
            'success_rate': vga_success_rate,
            'avg_collisions': float(vga_avg_collisions),
            'avg_steps_success': float(vga_avg_steps) if vga_success_rate > 0 else None,
            'trials': [{
                'trial': r['trial'],
                'success': r['success'],
                'steps': r['steps'],
                'collisions': r['collisions'],
                'collision_step': r.get('collision_step'),
                'failure_reason': r.get('failure_reason')
            } for r in vga_results]
        }
    }
    
    results_file = results_dir / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nDRL Agent:")
    print(f"  Success Rate: {drl_success_rate:.1f}%")
    print(f"  Avg Collisions: {drl_avg_collisions:.1f}")
    if drl_success_rate > 0:
        print(f"  Avg Steps (success): {drl_avg_steps:.1f}")
    
    print(f"\nVGA-UPL Baseline:")
    print(f"  Success Rate: {vga_success_rate:.1f}%")
    print(f"  Avg Collisions: {vga_avg_collisions:.1f}")
    if vga_success_rate > 0:
        print(f"  Avg Steps (success): {vga_avg_steps:.1f}")
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Videos saved to: {video_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
