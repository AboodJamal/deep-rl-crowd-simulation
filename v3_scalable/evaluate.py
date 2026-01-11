"""
Evaluation Script for Realistic Multi-Agent Navigation
=======================================================

Features:
1. Evaluate with different number of agents (3, 5, 7, 10)
2. Evaluate across all scenarios
3. Generate videos with raycasting visualization
4. Comprehensive metrics (AllSuccess, AnySuccess, PerAgent)

Author: GitHub Copilot
Date: December 2025
"""

import os
import sys
import numpy as np
import torch
import json
from datetime import datetime
from typing import Dict, List, Optional

from corridor_env import RealisticCorridorEnv
from trainer import PolicyNetwork, ValueNetwork
from visualizer import generate_video, evaluate_and_visualize


def load_model(
    model_path: str,
    obs_dim: int,
    action_dim: int,
    device: str = 'cpu',
) -> PolicyNetwork:
    """Load trained policy model."""
    policy = PolicyNetwork(obs_dim, action_dim, hidden_dim=256).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'policy' in checkpoint:
        policy.load_state_dict(checkpoint['policy'])
    else:
        policy.load_state_dict(checkpoint)
    
    policy.eval()
    return policy


def evaluate_single_config(
    policy: PolicyNetwork,
    scenario: str,
    num_agents: int,
    num_episodes: int = 20,
    device: str = 'cpu',
    generate_videos: bool = True,
    output_dir: str = 'eval_results',
) -> Dict:
    """Evaluate on a single configuration."""
    
    env = RealisticCorridorEnv(
        num_agents=num_agents,
        scenario=scenario,
        max_steps=1000,
        randomize_scenario=True,
    )
    
    results = {
        'scenario': scenario,
        'num_agents': num_agents,
        'all_success': [],
        'any_success': [],
        'per_agent_success': [],
        'episode_lengths': [],
        'wall_collisions': [],
        'obstacle_collisions': [],
        'agent_collisions': [],
        'agents_reached': [],
    }
    
    video_dir = os.path.join(output_dir, f'{scenario}_{num_agents}agents')
    if generate_videos:
        os.makedirs(video_dir, exist_ok=True)
    
    for ep in range(num_episodes):
        # Generate video for first 3 episodes
        if generate_videos and ep < 3:
            output_path = os.path.join(video_dir, f'episode_{ep+1}.gif')
            info = generate_video(
                env, policy, device,
                output_path=output_path,
                show_rays=True,
            )
        else:
            # Run without video
            obs_list, _ = env.reset()
            done = False
            
            while not done:
                obs_t = torch.FloatTensor(np.array(obs_list)).to(device)
                with torch.no_grad():
                    actions, _, _ = policy.get_action(obs_t, deterministic=True)
                obs_list, _, terminated, truncated, info = env.step(actions.cpu().numpy())
                done = terminated or truncated
        
        # Record results
        results['all_success'].append(float(info.get('all_success', False)))
        results['any_success'].append(float(info.get('any_success', False)))
        results['per_agent_success'].append(info.get('per_agent_success', 0.0))
        results['episode_lengths'].append(info.get('episode_length', 0))
        results['wall_collisions'].append(info.get('total_wall_collisions', 0))
        results['obstacle_collisions'].append(info.get('total_obstacle_collisions', 0))
        results['agent_collisions'].append(info.get('total_agent_collisions', 0))
        results['agents_reached'].append(info.get('agents_reached', 0))
    
    # Compute statistics
    results['stats'] = {
        'all_success_rate': np.mean(results['all_success']),
        'all_success_std': np.std(results['all_success']),
        'any_success_rate': np.mean(results['any_success']),
        'per_agent_success_mean': np.mean(results['per_agent_success']),
        'per_agent_success_std': np.std(results['per_agent_success']),
        'avg_episode_length': np.mean(results['episode_lengths']),
        'avg_wall_collisions': np.mean(results['wall_collisions']),
        'avg_obstacle_collisions': np.mean(results['obstacle_collisions']),
        'avg_agent_collisions': np.mean(results['agent_collisions']),
        'avg_agents_reached': np.mean(results['agents_reached']),
    }
    
    return results


def run_comprehensive_evaluation(
    model_path: str,
    output_dir: str = 'evaluation_results',
    device: str = 'cpu',
    num_episodes_per_config: int = 20,
):
    """
    Run comprehensive evaluation across all scenarios and agent counts.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f'eval_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print(f"#  COMPREHENSIVE EVALUATION")
    print(f"#  Model: {model_path}")
    print(f"#  Output: {output_dir}")
    print(f"{'#'*70}\n")
    
    # Scenarios to evaluate
    scenarios = ['straight', 'single_wall', 'double_wall', 'narrow_passage', 'mixed']
    
    # Agent counts to evaluate
    agent_counts = [3, 5, 7, 10]
    
    # Create dummy env to get observation dimension
    dummy_env = RealisticCorridorEnv(num_agents=3, scenario='straight')
    obs_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.shape[0]
    
    # Load model
    policy = load_model(model_path, obs_dim, action_dim, device)
    print(f"Model loaded: obs_dim={obs_dim}, action_dim={action_dim}")
    
    all_results = {}
    
    for num_agents in agent_counts:
        print(f"\n{'='*60}")
        print(f"  Evaluating with {num_agents} AGENTS")
        print(f"{'='*60}")
        
        all_results[f'{num_agents}_agents'] = {}
        
        for scenario in scenarios:
            print(f"\n  → Scenario: {scenario}")
            
            results = evaluate_single_config(
                policy=policy,
                scenario=scenario,
                num_agents=num_agents,
                num_episodes=num_episodes_per_config,
                device=device,
                generate_videos=True,
                output_dir=output_dir,
            )
            
            all_results[f'{num_agents}_agents'][scenario] = results['stats']
            
            # Print summary
            stats = results['stats']
            print(f"     All Success: {stats['all_success_rate']:.1%} ± {stats['all_success_std']:.1%}")
            print(f"     Any Success: {stats['any_success_rate']:.1%}")
            print(f"     Per-Agent:   {stats['per_agent_success_mean']:.1%} ± {stats['per_agent_success_std']:.1%}")
            print(f"     Collisions:  W={stats['avg_wall_collisions']:.1f}, O={stats['avg_obstacle_collisions']:.1f}, A={stats['avg_agent_collisions']:.1f}")
    
    # Save all results
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print(f"\n\n{'#'*70}")
    print(f"#  EVALUATION SUMMARY")
    print(f"{'#'*70}\n")
    
    print(f"{'Scenario':<15} ", end='')
    for n in agent_counts:
        print(f"| {n} Agents       ", end='')
    print()
    print("-" * (15 + 18 * len(agent_counts)))
    
    for scenario in scenarios:
        print(f"{scenario:<15} ", end='')
        for n in agent_counts:
            stats = all_results[f'{n}_agents'][scenario]
            print(f"| {stats['all_success_rate']:5.1%} / {stats['per_agent_success_mean']:5.1%} ", end='')
        print()
    
    print(f"\n(Format: AllSuccess / PerAgentSuccess)")
    print(f"\nResults saved to: {output_dir}")
    
    return all_results


def quick_evaluate(
    model_path: str,
    scenario: str = 'single_wall',
    num_agents: int = 3,
    num_episodes: int = 5,
    device: str = 'cpu',
):
    """Quick evaluation with video generation."""
    
    print(f"\n{'='*50}")
    print(f"Quick Evaluation")
    print(f"  Model: {model_path}")
    print(f"  Scenario: {scenario}")
    print(f"  Agents: {num_agents}")
    print(f"{'='*50}\n")
    
    # Create env
    env = RealisticCorridorEnv(
        num_agents=num_agents,
        scenario=scenario,
        max_steps=1000,
        randomize_scenario=True,
    )
    
    # Load model
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = load_model(model_path, obs_dim, action_dim, device)
    
    # Evaluate with videos
    output_dir = f'quick_eval_{scenario}_{num_agents}agents'
    results = evaluate_and_visualize(
        env, policy, device,
        num_episodes=num_episodes,
        output_dir=output_dir,
        show_rays=True,
    )
    
    return results


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--scenario', type=str, default=None, help='Specific scenario to evaluate')
    parser.add_argument('--num-agents', type=int, default=3, help='Number of agents')
    parser.add_argument('--num-episodes', type=int, default=20, help='Episodes per config')
    parser.add_argument('--quick', action='store_true', help='Quick evaluation mode')
    parser.add_argument('--full', action='store_true', help='Full comprehensive evaluation')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if args.quick:
        # Quick evaluation
        scenario = args.scenario or 'single_wall'
        quick_evaluate(
            model_path=args.model,
            scenario=scenario,
            num_agents=args.num_agents,
            num_episodes=5,
            device=str(device),
        )
    elif args.full:
        # Comprehensive evaluation
        run_comprehensive_evaluation(
            model_path=args.model,
            output_dir=args.output,
            device=str(device),
            num_episodes_per_config=args.num_episodes,
        )
    else:
        # Single config evaluation
        scenario = args.scenario or 'single_wall'
        
        env = RealisticCorridorEnv(
            num_agents=args.num_agents,
            scenario=scenario,
        )
        
        policy = load_model(
            args.model,
            env.observation_space.shape[0],
            env.action_space.shape[0],
            str(device),
        )
        
        results = evaluate_single_config(
            policy=policy,
            scenario=scenario,
            num_agents=args.num_agents,
            num_episodes=args.num_episodes,
            device=str(device),
            generate_videos=True,
            output_dir=args.output,
        )
        
        print(f"\nResults for {scenario} with {args.num_agents} agents:")
        for key, value in results['stats'].items():
            print(f"  {key}: {value:.3f}")


if __name__ == '__main__':
    main()
