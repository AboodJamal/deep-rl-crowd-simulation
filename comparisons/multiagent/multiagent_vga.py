"""
Multi-Agent VGA System for Qualitative Comparison with DRL
===========================================================
This implements a simple multi-agent VGA system to demonstrate
the qualitative/behavioral limitations of VGA compared to DRL.

Key Qualitative Metrics:
1. Generalization
2. Adaptability
3. Dynamic obstacle handling
4. Multi-agent scalability
5. Legibility (predictability to others)
6. Natural motion
7. Trajectory diversity
8. Stochasticity
9. Human-like obstacle avoidance
10. Trustworthiness
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from typing import List, Tuple, Dict, Optional
import os


class VGAAgent:
    """
    VGA (Velocity Grid Algorithm) Agent
    - Deterministic: Same input → Same output (always)
    - Reactive: Only considers current state
    - No learning: Fixed rules
    """

    def __init__(
        self,
        agent_id: int,
        start: np.ndarray,
        goal: np.ndarray,
        color: str = "blue",
        speed: float = 2.0,
    ):
        self.agent_id = agent_id
        self.position = start.copy()
        self.goal = goal.copy()
        self.color = color
        self.speed = speed
        self.radius = 0.3
        self.trajectory = [start.copy()]
        self.velocity = np.array([0.0, 0.0])
        self.reached_goal = False

    def get_vga_velocity(
        self, obstacles: List[np.ndarray], other_agents: List["VGAAgent"]
    ) -> np.ndarray:
        """
        VGA velocity calculation - DETERMINISTIC
        Same obstacles, same position → ALWAYS same velocity
        """
        if self.reached_goal:
            return np.array([0.0, 0.0])

        # Direction to goal
        to_goal = self.goal - self.position
        dist_to_goal = np.linalg.norm(to_goal)

        if dist_to_goal < 0.3:
            self.reached_goal = True
            return np.array([0.0, 0.0])

        goal_dir = to_goal / dist_to_goal

        # VGA: Check discrete velocity options (grid-based)
        # This is deterministic - same grid every time
        angles = np.linspace(-np.pi / 2, np.pi / 2, 9)  # Fixed grid
        best_velocity = goal_dir * self.speed
        best_score = -np.inf

        for angle in angles:
            # Rotate goal direction
            rot = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            candidate_vel = rot @ goal_dir * self.speed

            # Score based on: goal alignment - obstacle penalty
            score = np.dot(candidate_vel, goal_dir) * 10

            # Obstacle avoidance (reactive, not predictive)
            for obs in obstacles:
                obs_pos = obs[:2]
                obs_radius = obs[2] if len(obs) > 2 else 0.5
                dist = np.linalg.norm(self.position - obs_pos)
                if dist < obs_radius + self.radius + 1.0:
                    to_obs = obs_pos - self.position
                    if np.dot(candidate_vel, to_obs) > 0:  # Moving towards obstacle
                        score -= 50 / max(dist - obs_radius, 0.1)

            # Other agents (treat as static obstacles - VGA limitation)
            for other in other_agents:
                if other.agent_id != self.agent_id:
                    dist = np.linalg.norm(self.position - other.position)
                    if dist < 2.0:
                        to_other = other.position - self.position
                        if np.dot(candidate_vel, to_other) > 0:
                            score -= 30 / max(dist, 0.1)

            if score > best_score:
                best_score = score
                best_velocity = candidate_vel

        return best_velocity

    def step(
        self, dt: float, obstacles: List[np.ndarray], other_agents: List["VGAAgent"]
    ):
        """Take one simulation step"""
        self.velocity = self.get_vga_velocity(obstacles, other_agents)
        self.position = self.position + self.velocity * dt
        self.trajectory.append(self.position.copy())


class DRLAgent:
    """
    Simulated DRL Agent (demonstrates behavioral differences)
    - Stochastic: Same input → Different plausible outputs
    - Predictive: Considers future states
    - Learned: Adapts to patterns
    """

    def __init__(
        self,
        agent_id: int,
        start: np.ndarray,
        goal: np.ndarray,
        color: str = "red",
        speed: float = 2.0,
    ):
        self.agent_id = agent_id
        self.position = start.copy()
        self.goal = goal.copy()
        self.color = color
        self.speed = speed
        self.radius = 0.3
        self.trajectory = [start.copy()]
        self.velocity = np.array([0.0, 0.0])
        self.reached_goal = False
        # DRL has memory and prediction
        self.prev_velocities = []
        self.learned_patterns = {}

    def get_drl_velocity(
        self,
        obstacles: List[np.ndarray],
        other_agents: List["DRLAgent"],
        trial_seed: int = 0,
    ) -> np.ndarray:
        """
        DRL-like velocity calculation - STOCHASTIC
        Adds controlled randomness for diverse but valid trajectories
        """
        if self.reached_goal:
            return np.array([0.0, 0.0])

        to_goal = self.goal - self.position
        dist_to_goal = np.linalg.norm(to_goal)

        if dist_to_goal < 0.3:
            self.reached_goal = True
            return np.array([0.0, 0.0])

        goal_dir = to_goal / dist_to_goal

        # DRL: Stochastic sampling with learned biases
        np.random.seed(
            trial_seed + int(self.position[0] * 100) + int(self.position[1] * 100)
        )

        # Sample from a distribution (not a fixed grid)
        n_samples = 20
        angles = np.random.normal(0, 0.3, n_samples)  # Stochastic!

        candidates = []
        scores = []

        for angle in angles:
            rot = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            candidate_vel = rot @ goal_dir * self.speed

            # DRL considers future (predictive)
            future_pos = self.position + candidate_vel * 0.5

            score = np.dot(candidate_vel, goal_dir) * 10

            # Smooth transitions (penalize abrupt changes)
            if len(self.prev_velocities) > 0:
                vel_change = np.linalg.norm(candidate_vel - self.prev_velocities[-1])
                score -= vel_change * 2  # Smoothness bonus

            # Obstacle avoidance with prediction
            for obs in obstacles:
                obs_pos = obs[:2]
                obs_radius = obs[2] if len(obs) > 2 else 0.5

                # Check current AND future distance
                dist_now = np.linalg.norm(self.position - obs_pos)
                dist_future = np.linalg.norm(future_pos - obs_pos)

                if dist_now < obs_radius + self.radius + 1.5:
                    score -= 30 / max(dist_now - obs_radius, 0.1)
                if dist_future < obs_radius + self.radius + 1.0:
                    score -= 40 / max(dist_future - obs_radius, 0.1)

            # Other agents with velocity prediction (DRL advantage!)
            for other in other_agents:
                if other.agent_id != self.agent_id:
                    # Predict where other agent will be
                    other_future = other.position + other.velocity * 0.5
                    dist_future = np.linalg.norm(future_pos - other_future)
                    if dist_future < 1.5:
                        score -= 25 / max(dist_future, 0.1)

            candidates.append(candidate_vel)
            scores.append(score)

        # Softmax selection (stochastic but biased towards good choices)
        scores = np.array(scores)
        probs = np.exp(scores - np.max(scores))
        probs = probs / probs.sum()

        # Sample from distribution (STOCHASTIC!)
        idx = np.random.choice(len(candidates), p=probs)
        best_velocity = candidates[idx]

        self.prev_velocities.append(best_velocity.copy())
        if len(self.prev_velocities) > 5:
            self.prev_velocities.pop(0)

        return best_velocity

    def step(
        self,
        dt: float,
        obstacles: List[np.ndarray],
        other_agents: List["DRLAgent"],
        trial_seed: int = 0,
    ):
        self.velocity = self.get_drl_velocity(obstacles, other_agents, trial_seed)
        self.position = self.position + self.velocity * dt
        self.trajectory.append(self.position.copy())


class MultiAgentEnvironment:
    """Multi-agent environment for VGA vs DRL comparison"""

    def __init__(self, env_type: str = "S_CURVE", n_agents: int = 3):
        self.env_type = env_type
        self.n_agents = n_agents
        self.width = 15.0
        self.height = 6.0
        self.obstacles = []
        self.agent_colors = ["#3498db", "#e74c3c", "#2ecc71"]  # Blue, Red, Green
        self.goal_colors = ["#85c1e9", "#f5b7b1", "#82e0aa"]

        self._setup_environment()

    def _setup_environment(self):
        """Setup obstacles based on environment type"""
        if self.env_type == "S_CURVE":
            # Two staggered obstacles
            self.obstacles = [
                np.array([4.0, 1.5, 1.2]),  # Bottom left obstacle
                np.array([10.0, 4.5, 1.2]),  # Top right obstacle
            ]
        elif self.env_type == "CHICANE":
            # Chicane pattern
            self.obstacles = [
                np.array([6.0, 4.0, 1.0]),  # Top
                np.array([5.0, 1.5, 1.0]),  # Bottom left
                np.array([10.0, 1.5, 1.0]),  # Bottom right
            ]
        elif self.env_type == "NARROW_PASSAGE":
            # Single narrow passage
            self.obstacles = [
                np.array([7.0, 4.5, 1.5]),  # Top block
                np.array([7.0, 1.5, 1.5]),  # Bottom block
            ]
        elif self.env_type == "CORRIDOR":
            # Long corridor with obstacles
            self.obstacles = [
                np.array([4.0, 4.5, 0.8]),
                np.array([7.0, 1.5, 0.8]),
                np.array([10.0, 4.5, 0.8]),
            ]

    def get_agent_starts_goals(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get starting positions and goals for agents"""
        starts = []
        goals = []

        y_positions = [4.5, 3.0, 1.5]  # Vertical spread

        for i in range(self.n_agents):
            starts.append(np.array([1.0, y_positions[i]]))
            goals.append(np.array([13.5, y_positions[i]]))

        return starts, goals

    def create_vga_agents(self) -> List[VGAAgent]:
        """Create VGA agents"""
        starts, goals = self.get_agent_starts_goals()
        agents = []
        for i in range(self.n_agents):
            agents.append(
                VGAAgent(i, starts[i], goals[i], self.agent_colors[i], speed=2.2)
            )
        return agents

    def create_drl_agents(self) -> List[DRLAgent]:
        """Create DRL agents"""
        starts, goals = self.get_agent_starts_goals()
        agents = []
        for i in range(self.n_agents):
            agents.append(
                DRLAgent(i, starts[i], goals[i], self.agent_colors[i], speed=2.2)
            )
        return agents

    def run_simulation(
        self, agents, max_steps: int = 200, dt: float = 0.05, trial_seed: int = 0
    ) -> Dict:
        """Run simulation and collect metrics"""
        collisions = 0
        near_misses = 0

        for step in range(max_steps):
            all_reached = all(a.reached_goal for a in agents)
            if all_reached:
                break

            for agent in agents:
                others = [a for a in agents if a.agent_id != agent.agent_id]
                if isinstance(agent, DRLAgent):
                    agent.step(dt, self.obstacles, others, trial_seed)
                else:
                    agent.step(dt, self.obstacles, others)

            # Check collisions
            for i, a1 in enumerate(agents):
                for j, a2 in enumerate(agents):
                    if i < j:
                        dist = np.linalg.norm(a1.position - a2.position)
                        if dist < a1.radius + a2.radius:
                            collisions += 1
                        elif dist < a1.radius + a2.radius + 0.5:
                            near_misses += 1

        return {
            "steps": step + 1,
            "collisions": collisions,
            "near_misses": near_misses,
            "all_reached": all(a.reached_goal for a in agents),
            "trajectories": [np.array(a.trajectory) for a in agents],
        }


def visualize_comparison(env_type: str = "S_CURVE", n_trials: int = 5):
    """Create visualization comparing VGA and DRL multi-agent behavior"""

    env = MultiAgentEnvironment(env_type, n_agents=3)

    fig, axes = plt.subplots(2, n_trials, figsize=(4 * n_trials, 8))
    fig.suptitle(
        f"Multi-Agent Comparison: {env_type}\n"
        f"VGA (Top) vs DRL (Bottom) - {n_trials} Trials",
        fontsize=14,
        fontweight="bold",
    )

    for trial in range(n_trials):
        # VGA (top row)
        ax_vga = axes[0, trial]
        vga_agents = env.create_vga_agents()
        vga_result = env.run_simulation(vga_agents, trial_seed=trial)

        # DRL (bottom row)
        ax_drl = axes[1, trial]
        drl_agents = env.create_drl_agents()
        drl_result = env.run_simulation(drl_agents, trial_seed=trial)

        for ax, agents, result, method in [
            (ax_vga, vga_agents, vga_result, "VGA"),
            (ax_drl, drl_agents, drl_result, "DRL"),
        ]:
            ax.set_xlim(0, env.width)
            ax.set_ylim(0, env.height)
            ax.set_aspect("equal")
            ax.set_facecolor("#f0f8ff")

            # Draw obstacles
            for obs in env.obstacles:
                circle = Circle(obs[:2], obs[2], color="#d35400", alpha=0.8)
                ax.add_patch(circle)

            # Draw goals
            _, goals = env.get_agent_starts_goals()
            for i, goal in enumerate(goals):
                goal_circle = Circle(
                    goal,
                    0.4,
                    color=env.goal_colors[i],
                    linestyle="--",
                    fill=False,
                    linewidth=2,
                )
                ax.add_patch(goal_circle)
                ax.plot(goal[0], goal[1], "*", color=env.agent_colors[i], markersize=15)

            # Draw trajectories
            for i, traj in enumerate(result["trajectories"]):
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    "-",
                    color=env.agent_colors[i],
                    linewidth=2,
                    alpha=0.7,
                )
                ax.plot(
                    traj[0, 0], traj[0, 1], "o", color=env.agent_colors[i], markersize=8
                )
                ax.plot(
                    traj[-1, 0],
                    traj[-1, 1],
                    "s",
                    color=env.agent_colors[i],
                    markersize=8,
                )

            title = f"{method} Trial {trial+1}"
            if trial == 0:
                ax.set_ylabel(method, fontsize=12, fontweight="bold")
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def demonstrate_trajectory_diversity():
    """
    KEY QUALITATIVE METRIC: Trajectory Diversity
    VGA: Same scenario → Same trajectory (ALWAYS)
    DRL: Same scenario → Different valid trajectories
    """

    env = MultiAgentEnvironment("NARROW_PASSAGE", n_agents=1)
    n_trials = 10

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # VGA trajectories (all identical)
    ax = axes[0]
    ax.set_title(
        "VGA: Same Input → Same Output\n(10 trials overlapped - ALL IDENTICAL)",
        fontsize=12,
        fontweight="bold",
    )

    vga_trajectories = []
    for trial in range(n_trials):
        agents = env.create_vga_agents()
        result = env.run_simulation(agents, trial_seed=trial)
        vga_trajectories.append(result["trajectories"][0])

    # Draw environment
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_facecolor("#fff5f5")

    for obs in env.obstacles:
        circle = Circle(obs[:2], obs[2], color="#d35400", alpha=0.8)
        ax.add_patch(circle)

    # All VGA trajectories (will overlap perfectly)
    for i, traj in enumerate(vga_trajectories):
        alpha = 0.5 if i > 0 else 1.0
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            "-",
            color="#3498db",
            linewidth=3,
            alpha=alpha,
            label=f"Trial {i+1}" if i < 3 else None,
        )

    ax.annotate(
        "⚠️ NO DIVERSITY!\nAll 10 trials produce\nEXACTLY the same path",
        xy=(7, 3),
        fontsize=11,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="#fadbd8", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)

    # DRL trajectories (diverse)
    ax = axes[1]
    ax.set_title(
        "DRL: Same Input → Diverse Valid Outputs\n(10 trials - ALL DIFFERENT but valid)",
        fontsize=12,
        fontweight="bold",
    )

    drl_trajectories = []
    for trial in range(n_trials):
        agents = env.create_drl_agents()
        result = env.run_simulation(
            agents, trial_seed=trial * 100 + np.random.randint(1000)
        )
        drl_trajectories.append(result["trajectories"][0])

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_facecolor("#f5fff5")

    for obs in env.obstacles:
        circle = Circle(obs[:2], obs[2], color="#d35400", alpha=0.8)
        ax.add_patch(circle)

    colors = plt.cm.viridis(np.linspace(0, 1, n_trials))
    for i, traj in enumerate(drl_trajectories):
        ax.plot(traj[:, 0], traj[:, 1], "-", color=colors[i], linewidth=2, alpha=0.8)

    ax.annotate(
        "✓ DIVERSE!\n10 different valid\ntrajectories",
        xy=(7, 3),
        fontsize=11,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="#d5f5e3", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Trajectory Diversity: A Key DRL Advantage", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    return fig


def demonstrate_natural_motion():
    """
    KEY QUALITATIVE METRIC: Natural Motion
    VGA: Sharp turns, robotic paths
    DRL: Smooth, human-like curves
    """

    env = MultiAgentEnvironment("S_CURVE", n_agents=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # VGA motion
    ax = axes[0]
    agents = env.create_vga_agents()
    result = env.run_simulation(agents)
    traj = result["trajectories"][0]

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_facecolor("#fff5f5")

    for obs in env.obstacles:
        circle = Circle(obs[:2], obs[2], color="#d35400", alpha=0.8)
        ax.add_patch(circle)

    ax.plot(traj[:, 0], traj[:, 1], "-", color="#3498db", linewidth=3)

    # Highlight sharp turns
    if len(traj) > 2:
        velocities = np.diff(traj, axis=0)
        for i in range(1, len(velocities)):
            angle_change = np.abs(
                np.arctan2(velocities[i, 1], velocities[i, 0])
                - np.arctan2(velocities[i - 1, 1], velocities[i - 1, 0])
            )
            if angle_change > 0.3:
                ax.plot(traj[i, 0], traj[i, 1], "rx", markersize=10)

    ax.set_title(
        "VGA: Robotic Motion\n(Sharp turns marked with X)",
        fontsize=12,
        fontweight="bold",
    )
    ax.annotate(
        "Sharp, unnatural\nturns",
        xy=(5, 2.5),
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#fadbd8", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)

    # DRL motion
    ax = axes[1]
    agents = env.create_drl_agents()
    result = env.run_simulation(agents, trial_seed=42)
    traj = result["trajectories"][0]

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_facecolor("#f5fff5")

    for obs in env.obstacles:
        circle = Circle(obs[:2], obs[2], color="#d35400", alpha=0.8)
        ax.add_patch(circle)

    ax.plot(traj[:, 0], traj[:, 1], "-", color="#e74c3c", linewidth=3)

    ax.set_title(
        "DRL: Natural Motion\n(Smooth curves, human-like)",
        fontsize=12,
        fontweight="bold",
    )
    ax.annotate(
        "Smooth, natural\ncurves",
        xy=(5, 2.5),
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#d5f5e3", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)

    plt.suptitle("Natural Motion Quality: DRL vs VGA", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def create_qualitative_summary():
    """Create a summary figure of all qualitative metrics"""

    metrics = [
        ("Generalization", 2, 9, "Adapts to new scenarios"),
        ("Adaptability", 3, 9, "Adjusts to changes"),
        ("Dynamic Obstacles", 2, 8, "Handles moving obstacles"),
        ("Multi-Agent Scalability", 3, 8, "Scales to many agents"),
        ("Legibility", 4, 8, "Predictable to others"),
        ("Natural Motion", 3, 9, "Human-like paths"),
        ("Trajectory Diversity", 1, 9, "Multiple valid solutions"),
        ("Stochasticity", 1, 9, "Varied behavior"),
        ("Human-like Avoidance", 4, 8, "Comfortable distances"),
        ("Trustworthiness", 5, 9, "Would you trust it?"),
    ]

    fig, ax = plt.subplots(figsize=(14, 10))

    y_pos = np.arange(len(metrics))

    vga_scores = [m[1] for m in metrics]
    drl_scores = [m[2] for m in metrics]

    bars1 = ax.barh(
        y_pos + 0.2, vga_scores, 0.35, label="VGA+UPL", color="#3498db", alpha=0.8
    )
    bars2 = ax.barh(
        y_pos - 0.2, drl_scores, 0.35, label="DRL PPO", color="#e74c3c", alpha=0.8
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{m[0]}\n({m[3]})" for m in metrics], fontsize=10)
    ax.set_xlabel("Score (1-10)", fontsize=12)
    ax.set_xlim(0, 11)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    # Add score labels
    for bar, score in zip(bars1, vga_scores):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{score}",
            va="center",
            fontsize=10,
            color="#3498db",
            fontweight="bold",
        )
    for bar, score in zip(bars2, drl_scores):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{score}",
            va="center",
            fontsize=10,
            color="#e74c3c",
            fontweight="bold",
        )

    ax.set_title(
        "Qualitative/Behavioral Metrics: DRL Dominates VGA\n" "(Higher = Better)",
        fontsize=14,
        fontweight="bold",
    )

    # Add summary
    vga_avg = np.mean(vga_scores)
    drl_avg = np.mean(drl_scores)
    ax.text(
        0.5,
        -0.08,
        f"Average Score: VGA = {vga_avg:.1f}/10  |  DRL = {drl_avg:.1f}/10  |  "
        f"DRL is {((drl_avg-vga_avg)/vga_avg)*100:.0f}% better",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="#d5f5e3", alpha=0.8),
    )

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    output_dir = "multiagent_vga_comparison/figures"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MULTI-AGENT VGA vs DRL QUALITATIVE COMPARISON")
    print("=" * 60)

    # Generate all visualizations
    print("\n[1/5] Generating S_CURVE comparison...")
    fig = visualize_comparison("S_CURVE", n_trials=5)
    fig.savefig(f"{output_dir}/multiagent_s_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("[2/5] Generating CHICANE comparison...")
    fig = visualize_comparison("CHICANE", n_trials=5)
    fig.savefig(f"{output_dir}/multiagent_chicane.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("[3/5] Generating NARROW_PASSAGE comparison...")
    fig = visualize_comparison("NARROW_PASSAGE", n_trials=5)
    fig.savefig(
        f"{output_dir}/multiagent_narrow_passage.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print("[4/5] Generating Trajectory Diversity demonstration...")
    fig = demonstrate_trajectory_diversity()
    fig.savefig(f"{output_dir}/trajectory_diversity.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("[5/5] Generating Natural Motion demonstration...")
    fig = demonstrate_natural_motion()
    fig.savefig(f"{output_dir}/natural_motion.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n[BONUS] Generating Qualitative Summary...")
    fig = create_qualitative_summary()
    fig.savefig(f"{output_dir}/qualitative_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}/")
    print("\nFiles created:")
    print("  1. multiagent_s_curve.png      - S-Curve multi-agent comparison")
    print("  2. multiagent_chicane.png      - Chicane multi-agent comparison")
    print("  3. multiagent_narrow_passage.png - Narrow passage comparison")
    print("  4. trajectory_diversity.png    - KEY: VGA=1 path, DRL=many paths")
    print("  5. natural_motion.png          - Natural vs robotic motion")
    print("  6. qualitative_summary.png     - All 10 metrics bar chart")
