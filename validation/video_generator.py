"""
Video Generator for Trajectory Visualization
============================================

Creates animated videos showing agent navigation through obstacles.
Supports both single trajectories and multi-trial comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json


class TrajectoryVideoGenerator:
    """
    Generates MP4 videos of agent trajectories navigating through obstacles.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        fps: int = 30
    ):
        """
        Initialize video generator.
        
        Args:
            figsize: Figure size in inches
            dpi: Dots per inch for video quality
            fps: Frames per second
        """
        self.figsize = figsize
        self.dpi = dpi
        self.fps = fps
    
    def create_single_trajectory_video(
        self,
        positions: np.ndarray,
        obstacles: List[Dict[str, Any]],
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        output_path: str,
        title: str = "Agent Navigation",
        show_velocity: bool = True
    ):
        """
        Create video of a single trajectory.
        
        Args:
            positions: Array of shape (N, 2) with agent positions
            obstacles: List of obstacle dictionaries with 'position' and 'radius'
            start_pos: Starting position
            goal_pos: Goal position
            output_path: Path to save video file
            title: Video title
            show_velocity: Whether to show velocity vectors
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Set up plot
        all_positions = np.vstack([positions, [start_pos], [goal_pos]])
        x_min, x_max = all_positions[:, 0].min() - 1, all_positions[:, 0].max() + 1
        y_min, y_max = all_positions[:, 1].min() - 1, all_positions[:, 1].max() + 1
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Draw obstacles
        for obs in obstacles:
            obs_pos = np.array(obs['position'])
            obs_radius = obs.get('radius', 0.3)
            circle = Circle(obs_pos, obs_radius, color='red', alpha=0.7, zorder=2)
            ax.add_patch(circle)
        
        # Draw start and goal
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=15, label='Start', zorder=3)
        ax.plot(goal_pos[0], goal_pos[1], 'b*', markersize=20, label='Goal', zorder=3)
        
        # Initialize agent and trail
        agent_circle = Circle(positions[0], 0.2, color='blue', alpha=0.8, zorder=4)
        ax.add_patch(agent_circle)
        
        trail_line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=2, label='Path')
        
        # Velocity arrow (if enabled)
        if show_velocity and len(positions) > 1:
            velocity_arrow = ax.arrow(0, 0, 0, 0, head_width=0.15, head_length=0.2,
                                     fc='cyan', ec='cyan', alpha=0.7, zorder=5)
        
        # Time text
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Progress text
        progress_text = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                               fontsize=12, verticalalignment='top',
                               horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.legend(loc='upper right', fontsize=10)
        
        def init():
            """Initialize animation."""
            trail_line.set_data([], [])
            return trail_line, agent_circle, time_text, progress_text
        
        def animate(frame):
            """Animation function."""
            # Update agent position
            agent_circle.center = positions[frame]
            
            # Update trail
            trail_line.set_data(positions[:frame+1, 0], positions[:frame+1, 1])
            
            # Update velocity arrow
            if show_velocity and frame < len(positions) - 1:
                vel = positions[frame + 1] - positions[frame]
                vel_norm = np.linalg.norm(vel)
                if vel_norm > 0.01:
                    vel_normalized = vel / vel_norm * 0.5  # Scale for visibility
                    velocity_arrow.set_data(
                        x=positions[frame, 0],
                        y=positions[frame, 1],
                        dx=vel_normalized[0],
                        dy=vel_normalized[1]
                    )
            
            # Update time
            time_text.set_text(f'Step: {frame}/{len(positions)-1}')
            
            # Update progress
            dist_to_goal = np.linalg.norm(positions[frame] - goal_pos)
            progress_text.set_text(f'Distance to goal: {dist_to_goal:.2f}m')
            
            return trail_line, agent_circle, time_text, progress_text
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(positions), interval=1000/self.fps,
            blit=False, repeat=False
        )
        
        # Save video
        print(f"Saving video to: {output_path}")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.fps, bitrate=1800)
        anim.save(output_path, writer=writer)
        plt.close(fig)
        print(f"Video saved successfully!")
    
    def create_comparison_video(
        self,
        trajectories: Dict[str, np.ndarray],
        obstacles: List[Dict[str, Any]],
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        output_path: str,
        title: str = "Model Comparison"
    ):
        """
        Create side-by-side comparison video of multiple models.
        
        Args:
            trajectories: Dictionary mapping model names to position arrays
            obstacles: List of obstacles
            start_pos: Starting position
            goal_pos: Goal position
            output_path: Path to save video
            title: Video title
        """
        num_models = len(trajectories)
        fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 6), dpi=self.dpi)
        
        if num_models == 1:
            axes = [axes]
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Set up each subplot
        agent_circles = []
        trail_lines = []
        time_texts = []
        
        for idx, (model_name, positions) in enumerate(trajectories.items()):
            ax = axes[idx]
            
            # Set limits
            all_pos = np.vstack([positions, [start_pos], [goal_pos]])
            x_min, x_max = all_pos[:, 0].min() - 1, all_pos[:, 0].max() + 1
            y_min, y_max = all_pos[:, 1].min() - 1, all_pos[:, 1].max() + 1
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(model_name, fontsize=12, fontweight='bold')
            
            # Draw obstacles
            for obs in obstacles:
                obs_pos = np.array(obs['position'])
                obs_radius = obs.get('radius', 0.3)
                circle = Circle(obs_pos, obs_radius, color='red', alpha=0.7)
                ax.add_patch(circle)
            
            # Draw start/goal
            ax.plot(start_pos[0], start_pos[1], 'go', markersize=12)
            ax.plot(goal_pos[0], goal_pos[1], 'b*', markersize=15)
            
            # Initialize agent
            agent = Circle(positions[0], 0.2, color='blue', alpha=0.8)
            ax.add_patch(agent)
            agent_circles.append(agent)
            
            # Initialize trail
            line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=2)
            trail_lines.append(line)
            
            # Time text
            text = ax.text(0.5, 0.95, '', transform=ax.transAxes,
                          ha='center', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            time_texts.append(text)
        
        plt.tight_layout()
        
        # Find max length
        max_len = max(len(traj) for traj in trajectories.values())
        
        def animate(frame):
            """Animation function."""
            for idx, (model_name, positions) in enumerate(trajectories.items()):
                if frame < len(positions):
                    # Update agent
                    agent_circles[idx].center = positions[frame]
                    
                    # Update trail
                    trail_lines[idx].set_data(positions[:frame+1, 0], positions[:frame+1, 1])
                    
                    # Update text
                    time_texts[idx].set_text(f'Step: {frame}')
            
            return agent_circles + trail_lines + time_texts
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=max_len,
            interval=1000/self.fps, blit=False, repeat=False
        )
        
        # Save
        print(f"Saving comparison video to: {output_path}")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.fps, bitrate=1800)
        anim.save(output_path, writer=writer)
        plt.close(fig)
        print("Comparison video saved!")
    
    def create_stochastic_overlay_video(
        self,
        all_trajectories: List[np.ndarray],
        obstacles: List[Dict[str, Any]],
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        output_path: str,
        title: str = "Stochastic Path Variability",
        show_percentages: bool = True
    ):
        """
        Create video showing multiple stochastic runs overlaid (like Figure 7).
        
        Args:
            all_trajectories: List of trajectory arrays from multiple runs
            obstacles: List of obstacles
            start_pos: Starting position
            goal_pos: Goal position
            output_path: Path to save video
            title: Video title
            show_percentages: Show percentage taking each path
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Set limits
        all_pos = np.vstack([np.vstack(all_trajectories), [start_pos], [goal_pos]])
        x_min, x_max = all_pos[:, 0].min() - 1, all_pos[:, 0].max() + 1
        y_min, y_max = all_pos[:, 1].min() - 1, all_pos[:, 1].max() + 1
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Draw obstacles
        for obs in obstacles:
            obs_pos = np.array(obs['position'])
            obs_radius = obs.get('radius', 0.3)
            circle = Circle(obs_pos, obs_radius, color='red', alpha=0.7)
            ax.add_patch(circle)
        
        # Draw start/goal
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=15, label='Start', zorder=10)
        ax.plot(goal_pos[0], goal_pos[1], 'b*', markersize=20, label='Goal', zorder=10)
        
        # Initialize lines for each trajectory
        lines = []
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_trajectories)))
        
        for i, traj in enumerate(all_trajectories):
            line, = ax.plot([], [], alpha=0.3, linewidth=1.5, color=colors[i])
            lines.append(line)
        
        # Count text
        count_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend(loc='upper right')
        
        # Find max length
        max_len = max(len(traj) for traj in all_trajectories)
        
        def animate(frame):
            """Animation function."""
            completed = 0
            for i, traj in enumerate(all_trajectories):
                if frame < len(traj):
                    lines[i].set_data(traj[:frame+1, 0], traj[:frame+1, 1])
                else:
                    lines[i].set_data(traj[:, 0], traj[:, 1])
                    completed += 1
            
            count_text.set_text(f'Trials shown: {len(all_trajectories)}\\nFrame: {frame}/{max_len}')
            
            return lines + [count_text]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=max_len,
            interval=1000/self.fps, blit=False, repeat=False
        )
        
        # Save
        print(f"Saving stochastic overlay video to: {output_path}")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.fps, bitrate=1800)
        anim.save(output_path, writer=writer)
        plt.close(fig)
        print("Stochastic overlay video saved!")


def generate_videos_from_experiment_results(
    results_json_path: str,
    output_dir: str,
    num_stochastic_to_show: int = 20
):
    """
    Generate videos from experiment runner results.
    
    Args:
        results_json_path: Path to experiment_results.json
        output_dir: Directory to save videos
        num_stochastic_to_show: Number of stochastic trials to overlay
    """
    # Load results
    print(f"Loading results from: {results_json_path}")
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = TrajectoryVideoGenerator(fps=30)
    
    # Process each scenario
    for scenario_name, scenario_data in results['scenarios'].items():
        print(f"\\nProcessing scenario: {scenario_name}")
        
        # Get scenario info
        definition = scenario_data['definition']
        start_pos = np.array(definition['start_pos'])
        goal_pos = np.array(definition['goal_pos'])
        
        # Reconstruct obstacles (stored inline in results)
        obstacles = []
        # We'll extract from first trial's positions context if needed
        # For now, use empty list (can be enhanced)
        
        # Generate deterministic video
        if 'vga_upl_deterministic' in scenario_data['models']:
            det_trial = scenario_data['models']['vga_upl_deterministic'][0]
            positions = np.array(det_trial['positions'])
            
            video_path = output_path / f"{scenario_name}_deterministic.mp4"
            generator.create_single_trajectory_video(
                positions=positions,
                obstacles=obstacles,
                start_pos=start_pos,
                goal_pos=goal_pos,
                output_path=str(video_path),
                title=f"{scenario_name} - Deterministic VGA+UPL"
            )
        
        # Generate stochastic overlay video
        if 'vga_upl_stochastic' in scenario_data['models']:
            stoch_trials = scenario_data['models']['vga_upl_stochastic']
            trajectories = [
                np.array(trial['positions'])
                for trial in stoch_trials[:num_stochastic_to_show]
            ]
            
            video_path = output_path / f"{scenario_name}_stochastic_overlay.mp4"
            generator.create_stochastic_overlay_video(
                all_trajectories=trajectories,
                obstacles=obstacles,
                start_pos=start_pos,
                goal_pos=goal_pos,
                output_path=str(video_path),
                title=f"{scenario_name} - Stochastic VGA+UPL ({num_stochastic_to_show} trials)"
            )
    
    print("\\nAll videos generated successfully!")


if __name__ == "__main__":
    # Example: Generate videos from experiment results
    results_path = "validation/results/experiment_001/experiment_results.json"
    output_dir = "validation/results/experiment_001/videos"
    
    generate_videos_from_experiment_results(results_path, output_dir, num_stochastic_to_show=20)
