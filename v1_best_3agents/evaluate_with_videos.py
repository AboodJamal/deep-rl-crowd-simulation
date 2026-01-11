#!/usr/bin/env python3
"""
Enhanced Evaluation with Better Videos
======================================
- Faster agents (speedup visualization)
- Direction arrows showing movement
- Velocity vectors
- Goal markers
- Trail paths
- Nice visual effects
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import PatchCollection
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

from corridor_env_fixed import FixedCorridorEnv
from train_simple_robust import SimplePolicy, SimpleValue


def generate_enhanced_video(policy, scenario, num_agents, device, output_path, num_episodes=1, 
                           speedup=1, show_trails=True, show_velocity=True, seed=None):
    """
    Generate enhanced video of agent behavior with:
    - Direction arrows
    - Velocity vectors
    - Trails
    - Speed indicator
    - Better visuals
    """
    # Set seed for reproducibility but different attempts
    if seed is not None:
        np.random.seed(seed)
    
    env = FixedCorridorEnv(
        num_agents=num_agents,
        scenario=scenario,
        max_steps=800,
        randomize=True  # Enable randomization for variety
    )
    
    all_frames = []
    all_trails = [[] for _ in range(num_agents)]  # Store position history
    
    for ep in range(num_episodes):
        obs_list, _ = env.reset()
        done = False
        step = 0
        
        # Reset trails for new episode
        trails = [[] for _ in range(num_agents)]
        
        while not done and step < 800:
            # Record frame with extra info
            frame = {
                'agents': [],
                'goals': env.goal_positions,
                'obstacles': [(o.x, o.y, o.width, o.height) 
                            for o in env.obstacles] if hasattr(env, 'obstacles') and env.obstacles else [],
                'walls': getattr(env, 'walls', []),
                'corridor_length': getattr(env, 'corridor_length', 50),
                'corridor_width': getattr(env, 'corridor_height', 12),
                'step': step,
                'trails': [list(t) for t in trails],  # Copy trails
            }
            
            for i, a in enumerate(env.agents):
                frame['agents'].append({
                    'pos': np.array([a.x, a.y]),
                    'heading': a.heading,
                    'vx': a.vx,
                    'vy': a.vy,
                    'reached_goal': a.reached_goal,
                    'speed': np.sqrt(a.vx**2 + a.vy**2),
                })
                # Add to trail (keep last 30 positions)
                trails[i].append((a.x, a.y))
                if len(trails[i]) > 30:
                    trails[i].pop(0)
            
            all_frames.append(frame)
            
            # Take action
            obs_t = torch.FloatTensor(np.array(obs_list)).to(device)
            with torch.no_grad():
                actions, _, _ = policy.get_action(obs_t, deterministic=True)
            actions_np = actions.cpu().numpy()
            
            next_obs, rewards, terminated, truncated, info = env.step(actions_np)
            done = terminated or truncated
            step += 1
            
            if done:
                break
            obs_list = next_obs
        
        # Add final frames
        all_frames.extend([all_frames[-1]] * 5)
    
    if len(all_frames) == 0:
        print(f"    No frames recorded for {scenario}")
        return
    
    # Apply speedup - skip frames
    if speedup > 1:
        all_frames = all_frames[::speedup]
    
    # Create video with clean white styling
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    
    # Agent colors - nice vibrant colors on white background
    agent_colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
    
    corridor_length = all_frames[0].get('corridor_length', 50)
    corridor_width = all_frames[0].get('corridor_width', 12)
    
    def init():
        ax.set_xlim(-3, corridor_length + 3)
        ax.set_ylim(-3, corridor_width + 3)
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3, color='#cccccc', linestyle='-')
        return []
    
    def animate(frame_idx):
        ax.clear()
        frame = all_frames[frame_idx]
        
        ax.set_xlim(-3, corridor_length + 3)
        ax.set_ylim(-3, corridor_width + 3)
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3, color='#cccccc', linestyle='-')
        
        # Title with step info
        step_num = frame.get('step', frame_idx * speedup)
        ax.set_title(f'🚶 {scenario.upper()} | {num_agents} Agents | Step {step_num}', 
                    fontsize=16, fontweight='bold', color='#2c3e50', pad=10)
        
        # Draw corridor boundaries - clean dark lines
        ax.plot([0, corridor_length], [0, 0], color='#2c3e50', linewidth=3, solid_capstyle='round')
        ax.plot([0, corridor_length], [corridor_width, corridor_width], color='#2c3e50', linewidth=3, solid_capstyle='round')
        ax.plot([0, 0], [0, corridor_width], color='#2c3e50', linewidth=3, solid_capstyle='round')
        ax.plot([corridor_length, corridor_length], [0, corridor_width], color='#2c3e50', linewidth=3, solid_capstyle='round')
        
        # Draw walls - solid gray
        for wall in frame.get('walls', []):
            if len(wall) >= 4:
                rect = patches.FancyBboxPatch(
                    (wall[0], wall[1]), wall[2] - wall[0], wall[3] - wall[1],
                    boxstyle="round,pad=0.02",
                    facecolor='#7f8c8d', edgecolor='#2c3e50', linewidth=2, alpha=0.9
                )
                ax.add_patch(rect)
        
        # Draw obstacles with nice styling
        for obs in frame.get('obstacles', []):
            if len(obs) >= 4:
                x_center, y_center, width, height = obs[0], obs[1], obs[2], obs[3]
                x_bottom_left = x_center - width / 2
                y_bottom_left = y_center - height / 2
                
                # Shadow
                shadow = patches.Rectangle(
                    (x_bottom_left + 0.15, y_bottom_left - 0.15), width, height,
                    facecolor='#bdc3c7', alpha=0.5, zorder=0
                )
                ax.add_patch(shadow)
                
                # Main obstacle - brown/tan color
                rect = patches.FancyBboxPatch(
                    (x_bottom_left, y_bottom_left), width, height,
                    boxstyle="round,pad=0.05",
                    facecolor='#d35400', edgecolor='#a04000', linewidth=2, alpha=1.0, zorder=1
                )
                ax.add_patch(rect)
        
        # Draw goals and agents
        goals = frame.get('goals', [])
        trails = frame.get('trails', [[] for _ in range(num_agents)])
        
        for i, agent_data in enumerate(frame['agents']):
            color = agent_colors[i % len(agent_colors)]
            pos = agent_data['pos']
            heading = agent_data['heading']
            vx = agent_data['vx']
            vy = agent_data['vy']
            reached = agent_data['reached_goal']
            speed = agent_data['speed']
            
            # Draw trail (fading path)
            if show_trails and i < len(trails) and len(trails[i]) > 1:
                trail = trails[i]
                for j in range(len(trail) - 1):
                    alpha = (j + 1) / len(trail) * 0.6
                    ax.plot([trail[j][0], trail[j+1][0]], 
                           [trail[j][1], trail[j+1][1]], 
                           color=color, alpha=alpha, linewidth=2, zorder=2)
            
            # Goal marker
            if i < len(goals) and len(goals[i]) >= 2:
                goal_x, goal_y = goals[i][0], goals[i][1]
                
                # Pulsing goal circle
                pulse = 0.5 + 0.3 * np.sin(frame_idx * 0.3)
                goal_circle = plt.Circle((goal_x, goal_y), 1.5 + pulse, 
                                        fill=False, edgecolor=color, 
                                        linewidth=3, linestyle='--', alpha=0.6, zorder=3)
                ax.add_patch(goal_circle)
                
                # Goal star
                ax.plot(goal_x, goal_y, '*', color=color, markersize=25, 
                       markeredgecolor='#2c3e50', markeredgewidth=1.5, zorder=4)
                
                # Goal label
                ax.text(goal_x, goal_y + 2.2, f'GOAL {i+1}', ha='center', 
                       fontsize=9, color=color, fontweight='bold')
            
            # Agent body
            if reached:
                # Celebration effect for reached goal
                for r in [1.0, 0.7, 0.4]:
                    circle = plt.Circle(pos, r, facecolor=color, alpha=0.3 * r, zorder=8)
                    ax.add_patch(circle)
                ax.plot(pos[0], pos[1], '*', color=color, markersize=30, 
                       markeredgecolor='#2c3e50', markeredgewidth=2, zorder=10)
            else:
                # Agent circle with glow
                glow = plt.Circle(pos, 0.6, facecolor=color, alpha=0.25, zorder=8)
                ax.add_patch(glow)
                agent_circle = plt.Circle(pos, 0.4, facecolor=color, 
                                         edgecolor='#2c3e50', linewidth=2, zorder=9)
                ax.add_patch(agent_circle)
                
                # Agent number
                ax.text(pos[0], pos[1], str(i+1), ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white', zorder=11)
            
            # Direction arrow (heading) - dark arrow
            arrow_len = 1.2
            dx = arrow_len * np.cos(heading)
            dy = arrow_len * np.sin(heading)
            ax.annotate('', xy=(pos[0] + dx, pos[1] + dy), xytext=(pos[0], pos[1]),
                       arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5), zorder=12)
            
            # Velocity vector (if moving) - orange arrow
            if show_velocity and speed > 0.1:
                vel_scale = 0.8
                ax.annotate('', xy=(pos[0] + vx * vel_scale, pos[1] + vy * vel_scale), 
                           xytext=(pos[0], pos[1]),
                           arrowprops=dict(arrowstyle='-|>', color='#e67e22', lw=2, alpha=0.9),
                           zorder=11)
            
            # Speed indicator
            speed_text = f'{speed:.1f} m/s'
            ax.text(pos[0], pos[1] - 1.0, speed_text, ha='center', 
                   fontsize=8, color=color, alpha=0.8)
        
        # Legend
        legend_y = corridor_width + 1.5
        ax.text(2, legend_y, '→ Direction', color='#2c3e50', fontsize=10, fontweight='bold')
        ax.text(12, legend_y, '→ Velocity', color='#e67e22', fontsize=10, fontweight='bold')
        ax.text(22, legend_y, '★ Goal', color='#27ae60', fontsize=10, fontweight='bold')
        ax.text(32, legend_y, '● Agent', color='#3498db', fontsize=10, fontweight='bold')
        
        # Remove axis labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        return []
    
    print(f"    Creating animation with {len(all_frames)} frames...")
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(all_frames), 
                        interval=50, blit=True)
    
    # Save video
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Try MP4 first, then GIF
    try:
        writer = FFMpegWriter(fps=20, metadata=dict(artist='multii_test_Abd'), bitrate=3000)
        anim.save(output_path, writer=writer, dpi=100)
        print(f"    ✅ Video saved: {output_path}")
    except Exception as e:
        print(f"    ⚠️ MP4 failed ({e}), trying GIF...")
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer=PillowWriter(fps=15), dpi=80)
            print(f"    ✅ GIF saved: {gif_path}")
        except Exception as e2:
            print(f"    ❌ Failed to save video: {e2}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Enhanced Evaluation with Videos')
    parser.add_argument('--model', type=str, 
                       default='models/best_model/mixed/final_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--scenarios', type=str, nargs='+',
                       default=['straight', 'single_wall_bottom', 's_curve', 'narrow_passage', 'mixed'],
                       help='Scenarios to evaluate')
    parser.add_argument('--num-agents', type=int, default=3)
    parser.add_argument('--num-episodes', type=int, default=20,
                       help='Number of episodes for evaluation')
    parser.add_argument('--video-episodes', type=int, default=1,
                       help='Number of episodes to record in video')
    parser.add_argument('--num-tries', type=int, default=3,
                       help='Number of video attempts per scenario')
    parser.add_argument('--speedup', type=int, default=1,
                       help='Video speedup factor (skip frames, 1=normal speed)')
    parser.add_argument('--output-dir', type=str, default='videos_enhanced')
    parser.add_argument('--no-trails', action='store_true', help='Disable trail visualization')
    parser.add_argument('--no-velocity', action='store_true', help='Disable velocity arrows')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"🎬 ENHANCED VIDEO EVALUATION")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Agents: {args.num_agents}")
    print(f"Speedup: {args.speedup}x")
    print(f"Scenarios: {args.scenarios}")
    print(f"{'='*70}\n")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model, map_location=device)
    
    env = FixedCorridorEnv(num_agents=args.num_agents, scenario='straight', max_steps=500)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = SimplePolicy(obs_dim, action_dim, hidden_dim=256).to(device)
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()
    print(f"✅ Model loaded\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate and generate videos for each scenario
    results = {}
    
    for scenario in args.scenarios:
        print(f"\n{'─'*70}")
        print(f"📍 Scenario: {scenario}")
        print(f"{'─'*70}")
        
        # Create scenario folder
        scenario_dir = os.path.join(args.output_dir, scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Quick evaluation
        env = FixedCorridorEnv(
            num_agents=args.num_agents,
            scenario=scenario,
            max_steps=800,
            randomize=False
        )
        
        all_success = 0
        total_agent_success = 0
        total_agents = 0
        
        for ep in range(args.num_episodes):
            obs_list, _ = env.reset()
            done = False
            step = 0
            
            while not done and step < 800:
                obs_t = torch.FloatTensor(np.array(obs_list)).to(device)
                with torch.no_grad():
                    actions, _, _ = policy.get_action(obs_t, deterministic=True)
                actions_np = actions.cpu().numpy()
                next_obs, rewards, terminated, truncated, info = env.step(actions_np)
                done = terminated or truncated
                step += 1
                if done:
                    break
                obs_list = next_obs
            
            reached = sum(1 for a in env.agents if a.reached_goal)
            all_success += int(reached == args.num_agents)
            total_agent_success += reached
            total_agents += args.num_agents
        
        success_rate = all_success / args.num_episodes
        per_agent_rate = total_agent_success / total_agents
        results[scenario] = {
            'all_success_rate': success_rate,
            'per_agent_success': per_agent_rate,
        }
        
        print(f"  📊 All-Success: {success_rate:.1%} | Per-Agent: {per_agent_rate:.1%}")
        
        # Generate multiple video tries
        print(f"  🎬 Generating {args.num_tries} video attempts...")
        for try_num in range(1, args.num_tries + 1):
            video_path = os.path.join(scenario_dir, f'attempt_{try_num}.mp4')
            print(f"    📹 Attempt {try_num}/{args.num_tries}...")
            
            generate_enhanced_video(
                policy=policy,
                scenario=scenario,
                num_agents=args.num_agents,
                device=device,
                output_path=video_path,
                num_episodes=args.video_episodes,
                speedup=args.speedup,
                show_trails=not args.no_trails,
                show_velocity=not args.no_velocity,
                seed=try_num * 1000 + hash(scenario) % 1000,  # Different seed per attempt
            )
    
    # Save results
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE!")
    print(f"{'='*70}")
    print(f"Videos saved to: {args.output_dir}/")
    print(f"Results saved to: {results_file}")
    
    print(f"\n📊 SUMMARY:")
    for scenario, res in results.items():
        status = "✅" if res['all_success_rate'] >= 0.9 else "⚠️" if res['all_success_rate'] >= 0.5 else "❌"
        print(f"  {status} {scenario}: {res['all_success_rate']:.1%}")
    
    print(f"\n🎬 Videos created ({args.num_tries} attempts each):")
    for scenario in args.scenarios:
        print(f"  📁 {args.output_dir}/{scenario}/")
        for try_num in range(1, args.num_tries + 1):
            print(f"      • attempt_{try_num}.gif")


if __name__ == '__main__':
    main()
