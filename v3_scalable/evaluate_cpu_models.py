#!/usr/bin/env python3
"""
Evaluate CPU-trained models from test3
======================================
Evaluates the models saved from the CPU training runs.
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
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

from corridor_env import RealisticCorridorEnv
from train_robust import PolicyNetwork


def generate_video(policy, scenario, num_agents, device, output_path, num_episodes=2):
    """Generate video of agent behavior."""
    env = RealisticCorridorEnv(
        num_agents=num_agents,
        scenario=scenario,
        max_steps=800,
        randomize_scenario=False
    )
    
    all_frames = []
    
    for ep in range(num_episodes):
        obs_list, _ = env.reset()
        done = False
        step = 0
        
        frames = []
        while not done and step < 800:
            # Record frame
            frame = {
                'agents': [(np.array([a.x, a.y]), a.heading, a.reached_goal) 
                          for a in env.agents],
                'goals': env.goal_positions,
                'obstacles': [(o.x, o.y, o.width, o.height) 
                            for o in env.obstacles] if hasattr(env, 'obstacles') and env.obstacles else [],
                'corridor_length': env.corridor_length,
                'corridor_height': env.corridor_height,
            }
            frames.append(frame)
            
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
        
        all_frames.extend(frames)
        # Add separator frames
        all_frames.extend([frames[-1]] * 10)
    
    if len(all_frames) == 0:
        print(f"    No frames recorded for {scenario}")
        return
    
    # Create video
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    def init():
        ax.set_xlim(-2, 52)
        ax.set_ylim(-2, 14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{scenario} - {num_agents} agents')
        return []
    
    def animate(frame_idx):
        ax.clear()
        frame = all_frames[frame_idx]
        
        corridor_length = frame.get('corridor_length', 50)
        corridor_height = frame.get('corridor_height', 12)
        
        ax.set_xlim(-2, corridor_length + 2)
        ax.set_ylim(-2, corridor_height + 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{scenario} - {num_agents} agents (Frame {frame_idx+1}/{len(all_frames)})')
        
        # Draw corridor boundaries
        ax.axhline(y=0, color='black', linewidth=2)
        ax.axhline(y=corridor_height, color='black', linewidth=2)
        ax.axvline(x=0, color='black', linewidth=2)
        ax.axvline(x=corridor_length, color='black', linewidth=2)
        
        # Draw obstacles (x, y are center coordinates)
        for obs in frame.get('obstacles', []):
            if len(obs) >= 4:
                x_center, y_center, width, height = obs[0], obs[1], obs[2], obs[3]
                # Convert center to bottom-left corner for Rectangle
                x_bottom_left = x_center - width / 2
                y_bottom_left = y_center - height / 2
                rect = patches.Rectangle(
                    (x_bottom_left, y_bottom_left), width, height,
                    fill=True, facecolor='#8B4513', edgecolor='black', linewidth=2, alpha=1.0, zorder=1
                )
                ax.add_patch(rect)
        
        # Draw agents and goals
        for i, ((pos, heading, reached), goal) in enumerate(zip(frame['agents'], frame['goals'])):
            color = colors[i]
            
            # Goal
            ax.plot(goal[0], goal[1], 'x', color=color, markersize=15, markeredgewidth=3)
            
            # Agent
            if reached:
                marker = '*'
                size = 300
            else:
                marker = 'o'
                size = 200
            ax.scatter(pos[0], pos[1], c=[color], s=size, marker=marker, edgecolors='black', linewidths=2, zorder=10)
            
            # Heading arrow
            dx = 0.5 * np.cos(heading)
            dy = 0.5 * np.sin(heading)
            ax.arrow(pos[0], pos[1], dx, dy, head_width=0.2, head_length=0.1, fc=color, ec=color)
        
        return []
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(all_frames), interval=50, blit=True)
    
    # Save video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        writer = FFMpegWriter(fps=20, metadata=dict(artist='test3'), bitrate=1800)
        anim.save(output_path, writer=writer)
        print(f"    ✓ Video saved: {output_path}")
    except Exception as e:
        # Fallback to gif
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer=PillowWriter(fps=10))
            print(f"    ✓ GIF saved: {gif_path}")
        except Exception as e2:
            print(f"    ✗ Failed to save video: {e2}")
    
    plt.close()


def evaluate_model(model_path, scenarios, num_agents=5, num_episodes=50, device='cuda', generate_videos=True, video_dir='videos_cpu'):
    """Evaluate a trained model on multiple scenarios."""
    
    print(f"\n{'='*70}")
    print(f"EVALUATING CPU-TRAINED MODEL")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Num agents: {num_agents}")
    print(f"Num episodes: {num_episodes}")
    print(f"{'='*70}\n")
    
    # Create video directory
    if generate_videos:
        os.makedirs(video_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Get model dimensions
    env = RealisticCorridorEnv(num_agents=num_agents, scenario='straight', max_steps=500)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create and load policy
    policy = PolicyNetwork(obs_dim, action_dim, hidden_dim=256).to(device)
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()
    
    print(f"✓ Model loaded successfully\n")
    
    # Training info if available
    if 'best_all_success' in checkpoint:
        print(f"Training best All-Success: {checkpoint['best_all_success']:.1%}")
    if 'best_per_agent' in checkpoint:
        print(f"Training best Per-Agent: {checkpoint['best_per_agent']:.1%}")
    print()
    
    # Evaluate on each scenario
    results = {}
    
    for scenario in scenarios:
        print(f"\n{'─'*70}")
        print(f"Scenario: {scenario}")
        print(f"{'─'*70}")
        
        env = RealisticCorridorEnv(
            num_agents=num_agents,
            scenario=scenario,
            max_steps=800,
            randomize_scenario=False  # Deterministic evaluation
        )
        
        all_success = 0
        any_success = 0
        total_agent_success = 0
        total_agents = 0
        episode_lengths = []
        collision_counts = []
        
        for ep in range(num_episodes):
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
                    obs_list = next_obs
                    break
                obs_list = next_obs
            
            # Count successes
            reached = sum(1 for a in env.agents if a.reached_goal)
            all_success += int(reached == num_agents)
            any_success += int(reached > 0)
            total_agent_success += reached
            total_agents += num_agents
            episode_lengths.append(step)
            
            # Count collisions
            total_collisions = (
                info.get('total_wall_collisions', 0) +
                info.get('total_obstacle_collisions', 0) +
                info.get('total_agent_collisions', 0)
            )
            collision_counts.append(total_collisions)
            
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep+1}/{num_episodes} | "
                      f"All: {all_success/(ep+1):.1%} | "
                      f"Per-Agent: {total_agent_success/total_agents:.1%}")
        
        # Calculate final metrics
        results[scenario] = {
            'all_success_rate': all_success / num_episodes,
            'any_success_rate': any_success / num_episodes,
            'per_agent_success': total_agent_success / total_agents,
            'avg_episode_length': np.mean(episode_lengths),
            'avg_collisions': np.mean(collision_counts),
        }
        
        print(f"\nResults for {scenario}:")
        print(f"  All-Success Rate: {results[scenario]['all_success_rate']:.1%}")
        print(f"  Any-Success Rate: {results[scenario]['any_success_rate']:.1%}")
        print(f"  Per-Agent Success: {results[scenario]['per_agent_success']:.1%}")
        print(f"  Avg Episode Length: {results[scenario]['avg_episode_length']:.1f}")
        print(f"  Avg Collisions: {results[scenario]['avg_collisions']:.1f}")
        
        # Generate video
        if generate_videos:
            print(f"\n  Generating video for {scenario}...")
            video_path = os.path.join(video_dir, f"{scenario}_{num_agents}agents.mp4")
            generate_video(policy, scenario, num_agents, device, video_path, num_episodes=2)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for scenario, res in results.items():
        print(f"{scenario:20s} | All: {res['all_success_rate']:6.1%} | "
              f"Per-Agent: {res['per_agent_success']:6.1%}")
    print(f"{'='*70}\n")
    
    # Save results
    output_file = f"eval_cpu_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'model_path': model_path,
            'num_agents': num_agents,
            'num_episodes': num_episodes,
            'results': results,
        }, f, indent=2)
    print(f"Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CPU-trained models")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-agents', type=int, default=5,
                        help='Number of agents')
    parser.add_argument('--num-episodes', type=int, default=50,
                        help='Number of evaluation episodes')
    parser.add_argument('--scenarios', type=str, nargs='+',
                        default=['straight', 'curves', 'single_wall', 'narrow_passage', 'double_wall', 'mixed'],
                        help='Scenarios to evaluate on')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--generate-videos', action='store_true', default=True,
                        help='Generate videos for each scenario')
    parser.add_argument('--no-videos', dest='generate_videos', action='store_false',
                        help='Skip video generation')
    parser.add_argument('--video-dir', type=str, default='videos_cpu',
                        help='Directory to save videos')
    
    args = parser.parse_args()
    
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found: {args.model_path}")
        return
    
    evaluate_model(
        args.model_path,
        args.scenarios,
        num_agents=args.num_agents,
        num_episodes=args.num_episodes,
        device=device,
        generate_videos=args.generate_videos,
        video_dir=args.video_dir
    )


if __name__ == '__main__':
    main()

