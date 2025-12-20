"""
Generate trajectory images and videos for each VGA scenario.
Creates both static images and animated videos showing trajectories.
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_trajectory_image(positions, obstacles, start, goal, title, output_path, success=True):
    """Create static image showing complete trajectory."""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    all_x = [p[0] for p in positions] + [start[0], goal[0]] + [o["position"][0] for o in obstacles]
    all_y = [p[1] for p in positions] + [start[1], goal[1]] + [o["position"][1] for o in obstacles]
    
    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    for obs in obstacles:
        circle = patches.Circle(obs["position"], obs["radius"], color="red", alpha=0.6, zorder=2)
        ax.add_patch(circle)
        personal_circle = patches.Circle(obs["position"], obs["radius"] + 0.1,
                                        color="red", alpha=0.15, zorder=1, linestyle="--", linewidth=1)
        ax.add_patch(personal_circle)
    
    if len(positions) > 1:
        traj_array = np.array(positions)
        ax.plot(traj_array[:, 0], traj_array[:, 1], "b-", linewidth=2.5, alpha=0.8, label="Trajectory", zorder=3)
        for i in range(0, len(traj_array)-1, max(1, len(traj_array)//20)):
            dx = traj_array[i+1][0] - traj_array[i][0]
            dy = traj_array[i+1][1] - traj_array[i][1]
            if dx != 0 or dy != 0:
                ax.arrow(traj_array[i][0], traj_array[i][1], dx*0.8, dy*0.8,
                        head_width=0.15, head_length=0.15, fc='blue', ec='blue', alpha=0.6, zorder=4)
    
    ax.plot(*start, "go", markersize=15, label="Start", zorder=5, markeredgecolor="darkgreen", markeredgewidth=2)
    ax.plot(*goal, "*", color="gold", markersize=25, label="Goal", zorder=5, markeredgecolor="orange", markeredgewidth=2)
    
    status_color = "green" if success else "red"
    status_text = "SUCCESS" if success else "FAILED"
    ax.text(0.02, 0.98, f"Status: {status_text}", transform=ax.transAxes, fontsize=12, fontweight='bold',
           color=status_color, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_trajectory_video(positions, obstacles, start, goal, title, output_path, success=True):
    """Create animated video of trajectory."""
    if len(positions) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    all_x = [p[0] for p in positions] + [start[0], goal[0]] + [o["position"][0] for o in obstacles]
    all_y = [p[1] for p in positions] + [start[1], goal[1]] + [o["position"][1] for o in obstacles]
    
    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    for obs in obstacles:
        circle = patches.Circle(obs["position"], obs["radius"], color="red", alpha=0.6, zorder=2)
        ax.add_patch(circle)
        personal_circle = patches.Circle(obs["position"], obs["radius"] + 0.1,
                                        color="red", alpha=0.15, zorder=1, linestyle="--", linewidth=1)
        ax.add_patch(personal_circle)
    
    ax.plot(*start, "go", markersize=15, label="Start", zorder=5, markeredgecolor="darkgreen", markeredgewidth=2)
    ax.plot(*goal, "*", color="gold", markersize=25, label="Goal", zorder=5, markeredgecolor="orange", markeredgewidth=2)
    
    path_line, = ax.plot([], [], "b-", linewidth=2.5, alpha=0.8, label="Trajectory", zorder=3)
    agent_dot, = ax.plot([], [], "bo", markersize=12, zorder=6, markeredgecolor="darkblue", markeredgewidth=2)
    
    status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12, fontweight='bold', va="top",
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    step_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=10, va="top",
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    ax.legend(loc="upper right", fontsize=10)
    
    def init():
        path_line.set_data([], [])
        agent_dot.set_data([], [])
        status_color = "green" if success else "red"
        status_str = "SUCCESS" if success else "FAILED"
        status_text.set_text(f"Status: {status_str}")
        status_text.set_color(status_color)
        step_text.set_text("")
        return path_line, agent_dot, status_text, step_text
    
    def animate(frame):
        if frame < len(positions):
            path_line.set_data([p[0] for p in positions[:frame+1]], [p[1] for p in positions[:frame+1]])
            agent_dot.set_data([positions[frame][0]], [positions[frame][1]])
            step_text.set_text(f"Step: {frame+1}/{len(positions)}")
        return path_line, agent_dot, status_text, step_text
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(positions), interval=50, blit=True, repeat=True)
    
    try:
        anim.save(str(output_path), writer='ffmpeg', fps=20, dpi=100)
    except Exception as e:
        print(f"  Video save error (trying gif): {e}")
        try:
            anim.save(str(output_path).replace('.mp4', '.gif'), writer='pillow', fps=20)
        except:
            pass
    
    plt.close()


def generate_all_visualizations():
    """Generate images and videos for all scenarios."""
    results_dir = Path("validation/results/vga_final_experiment")
    images_dir = results_dir / "trajectory_images"
    videos_dir = results_dir / "trajectory_videos"
    
    images_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("GENERATING TRAJECTORY VISUALIZATIONS")
    print("=" * 80)
    
    print("\nLoading results...")
    with open(results_dir / "comprehensive_results.json") as f:
        all_results = json.load(f)
    
    scenarios = ["SOSP", "MOSP_A", "MOSP_B", "MOSP_C", "MOSP_D"]
    trials_per_scenario = 5
    
    for scenario_name in scenarios:
        print(f"\n{scenario_name}:")
        print("-" * 80)
        
        if scenario_name not in all_results:
            continue
        
        scenario_results = all_results[scenario_name]
        if "VGA_UPL_Det" not in scenario_results:
            continue
        
        trials = scenario_results["VGA_UPL_Det"][:trials_per_scenario]
        
        for trial in trials:
            trial_id = trial["trial_id"]
            success = trial["deterministic"]["success"]
            positions = trial["deterministic"]["positions"]
            
            if len(positions) < 2:
                continue
            
            print(f"  Trial {trial_id}...", end=" ", flush=True)
            
            image_path = images_dir / f"{scenario_name.lower()}_trial{trial_id:03d}.png"
            create_trajectory_image(positions, trial["obstacles"], trial["start_pos"], 
                                  trial["goal_pos"], f"{scenario_name} - Trial {trial_id} {'(SUCCESS)' if success else '(FAILED)'}",
                                  image_path, success)
            
            video_path = videos_dir / f"{scenario_name.lower()}_trial{trial_id:03d}.mp4"
            create_trajectory_video(positions, trial["obstacles"], trial["start_pos"],
                                   trial["goal_pos"], f"{scenario_name} - Trial {trial_id}",
                                   video_path, success)
            
            print(f"✓")
    
    print(f"\n{'='*80}")
    print("VISUALIZATION GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nImages: {images_dir}")
    print(f"Videos: {videos_dir}")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_visualizations()
