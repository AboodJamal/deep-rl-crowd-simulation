#!/usr/bin/env python3
"""
Evaluate CPU-trained models from multii_test_Abd
================================================
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

from corridor_env_fixed import FixedCorridorEnv
from train_simple_robust import SimplePolicy, SimpleValue


def generate_video(policy, scenario, num_agents, device, output_path, num_episodes=2):
    """Generate video of agent behavior."""
    env = FixedCorridorEnv(
        num_agents=num_agents,
        scenario=scenario,
        max_steps=800,
        randomize=False
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
                'goals': env.goal_positions,  # Use goal_positions from environment
                'obstacles': [(o.x, o.y, o.width, o.height) 
                            for o in env.obstacles] if hasattr(env, 'obstacles') and env.obstacles else [],
                'walls': getattr(env, 'walls', []),
                'corridor_length': getattr(env, 'corridor_length', 50),
                'corridor_width': getattr(env, 'corridor_height', 12),
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
    
    corridor_length = all_frames[0].get('corridor_length', 50)
    corridor_width = all_frames[0].get('corridor_width', 12)
    
    def init():
        ax.set_xlim(-2, corridor_length + 2)
        ax.set_ylim(-2, corridor_width + 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{scenario} - {num_agents} agents')
        return []
    
    def animate(frame_idx):
        ax.clear()
        frame = all_frames[frame_idx]
        
        ax.set_xlim(-2, corridor_length + 2)
        ax.set_ylim(-2, corridor_width + 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{scenario} - {num_agents} agents (Frame {frame_idx+1}/{len(all_frames)})')
        
        # Draw corridor boundaries
        ax.axhline(y=0, color='black', linewidth=2)
        ax.axhline(y=corridor_width, color='black', linewidth=2)
        ax.axvline(x=0, color='black', linewidth=2)
        ax.axvline(x=corridor_length, color='black', linewidth=2)
        
        # Draw walls
        for wall in frame.get('walls', []):
            if len(wall) >= 4:
                rect = patches.Rectangle(
                    (wall[0], wall[1]), wall[2] - wall[0], wall[3] - wall[1],
                    fill=True, facecolor='darkgray', edgecolor='black', alpha=0.8
                )
                ax.add_patch(rect)
        
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
        goals = frame.get('goals', [])
        for i, (pos, heading, reached) in enumerate(frame['agents']):
            color = colors[i]
            
            # Goal - make it very visible
            if i < len(goals) and len(goals[i]) >= 2:
                goal_x, goal_y = goals[i][0], goals[i][1]
                # Draw large goal marker
                ax.plot(goal_x, goal_y, 'X', color=color, markersize=20, markeredgewidth=4, 
                       markeredgecolor='black', zorder=5)
                # Add circle around goal for visibility
                circle = plt.Circle((goal_x, goal_y), 1.5, fill=False, edgecolor=color, 
                                  linewidth=3, linestyle='--', alpha=0.8, zorder=4)
                ax.add_patch(circle)
            
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
        writer = FFMpegWriter(fps=20, metadata=dict(artist='multii_test_Abd'), bitrate=1800)
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


def evaluate_model(model_path, scenarios, num_agents=3, num_episodes=50, device='cuda', generate_videos=True, video_dir='videos_cpu'):
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
    env = FixedCorridorEnv(num_agents=num_agents, scenario='straight', max_steps=500)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create and load policy
    policy = SimplePolicy(obs_dim, action_dim, hidden_dim=256).to(device)
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()
    
    print(f"✓ Model loaded successfully\n")
    
    # Training info if available
    if 'best_success' in checkpoint:
        print(f"Training best Success: {checkpoint['best_success']:.1%}")
    print()
    
    # Evaluate on each scenario
    results = {}
    
    for scenario in scenarios:
        print(f"\n{'─'*70}")
        print(f"Scenario: {scenario}")
        print(f"{'─'*70}")
        
        env = FixedCorridorEnv(
            num_agents=num_agents,
            scenario=scenario,
            max_steps=800,
            randomize=False  # Deterministic evaluation
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
            total_collisions = info.get('total_obstacle_collisions', 0)
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
    parser.add_argument('--num-agents', type=int, default=3,
                        help='Number of agents')
    parser.add_argument('--num-episodes', type=int, default=50,
                        help='Number of evaluation episodes')
    parser.add_argument('--scenarios', type=str, nargs='+',
                        default=['straight', 'single_wall_bottom', 'single_wall_top', 
                                's_curve', 'narrow_passage', 'chicane', 'mixed'],
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

