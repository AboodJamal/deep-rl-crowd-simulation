"""
Multi-Agent VGA+UPL Simulation
==============================
Run 3 agents using VGA+UPL algorithm together in various environments.
See if they can navigate without collisions or deadlocks.

Environments (matching the images provided):
- S_CURVE: Two staggered obstacles
- CHICANE: Chicane pattern with 3 obstacles
- NARROW_PASSAGE: Single narrow passage
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from typing import List, Tuple, Dict
import os
from dataclasses import dataclass
import time


@dataclass
class VGAConfig:
    """VGA+UPL Configuration Parameters"""

    robot_radius: float = 0.3
    max_speed: float = 2.2  # m/s (matching your VGA+UPL v4)
    safety_distance: float = 0.5
    grid_resolution: int = 9  # Number of velocity samples
    time_horizon: float = 1.0  # seconds to look ahead
    dt: float = 0.05  # simulation timestep


class VGAUPLAgent:
    """
    VGA+UPL Agent Implementation
    Velocity Grid Algorithm with Unified Path Learning
    """

    def __init__(
        self,
        agent_id: int,
        start: np.ndarray,
        goal: np.ndarray,
        color: str,
        config: VGAConfig = None,
    ):
        self.agent_id = agent_id
        self.position = start.astype(float)
        self.goal = goal.astype(float)
        self.color = color
        self.config = config or VGAConfig()

        self.velocity = np.array([0.0, 0.0])
        self.radius = self.config.robot_radius
        self.trajectory = [self.position.copy()]
        self.velocities = [self.velocity.copy()]

        self.reached_goal = False
        self.stuck_count = 0
        self.collision_count = 0

    def compute_vga_velocity(
        self, obstacles: List[np.ndarray], other_agents: List["VGAUPLAgent"]
    ) -> np.ndarray:
        """
        VGA+UPL Velocity Computation
        - Samples velocities on a grid
        - Evaluates each for goal progress and collision avoidance
        - Returns best velocity (DETERMINISTIC)
        """
        if self.reached_goal:
            return np.array([0.0, 0.0])

        # Direction to goal
        to_goal = self.goal - self.position
        dist_to_goal = np.linalg.norm(to_goal)

        if dist_to_goal < 0.4:
            self.reached_goal = True
            return np.array([0.0, 0.0])

        goal_dir = to_goal / dist_to_goal

        # VGA: Sample velocities on a grid (DETERMINISTIC)
        angles = np.linspace(-np.pi / 2, np.pi / 2, self.config.grid_resolution)
        speeds = [
            self.config.max_speed,
            self.config.max_speed * 0.7,
            self.config.max_speed * 0.3,
        ]

        best_velocity = goal_dir * self.config.max_speed
        best_score = -np.inf

        for speed in speeds:
            for angle in angles:
                # Rotate goal direction by angle
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                candidate_vel = rot_matrix @ goal_dir * speed

                # Evaluate this velocity
                score = self._evaluate_velocity(candidate_vel, obstacles, other_agents)

                if score > best_score:
                    best_score = score
                    best_velocity = candidate_vel

        # Check if stuck (very low score)
        if best_score < -100:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        return best_velocity

    def _evaluate_velocity(
        self,
        velocity: np.ndarray,
        obstacles: List[np.ndarray],
        other_agents: List["VGAUPLAgent"],
    ) -> float:
        """Evaluate a candidate velocity"""

        to_goal = self.goal - self.position
        goal_dir = to_goal / (np.linalg.norm(to_goal) + 1e-6)

        # Score 1: Progress towards goal
        score = np.dot(velocity, goal_dir) * 10

        # Score 2: Prefer higher speeds
        score += np.linalg.norm(velocity) * 2

        # Penalty: Static obstacles
        future_pos = self.position + velocity * self.config.time_horizon

        for obs in obstacles:
            obs_pos = obs[:2]
            obs_size = obs[2] if len(obs) > 2 else 0.5

            # Current distance
            dist_now = np.linalg.norm(self.position - obs_pos) - obs_size - self.radius
            # Future distance
            dist_future = np.linalg.norm(future_pos - obs_pos) - obs_size - self.radius

            if dist_now < self.config.safety_distance:
                # Very close - heavy penalty for moving closer
                if dist_future < dist_now:
                    score -= 200 / max(dist_now, 0.1)
                else:
                    score -= 50 / max(dist_now, 0.1)
            elif dist_future < self.config.safety_distance:
                score -= 100 / max(dist_future, 0.1)

        # Penalty: Other agents (VGA treats them as obstacles - no prediction!)
        for other in other_agents:
            if other.agent_id == self.agent_id:
                continue

            # VGA LIMITATION: Only considers current position, not velocity!
            dist_now = np.linalg.norm(self.position - other.position) - 2 * self.radius
            dist_future = np.linalg.norm(future_pos - other.position) - 2 * self.radius

            if dist_now < self.config.safety_distance * 2:
                if dist_future < dist_now:
                    score -= 150 / max(dist_now, 0.1)
                else:
                    score -= 30 / max(dist_now, 0.1)
            elif dist_future < self.config.safety_distance:
                score -= 80 / max(dist_future, 0.1)

        return score

    def step(self, obstacles: List[np.ndarray], other_agents: List["VGAUPLAgent"]):
        """Take one simulation step"""
        self.velocity = self.compute_vga_velocity(obstacles, other_agents)
        self.position = self.position + self.velocity * self.config.dt
        self.trajectory.append(self.position.copy())
        self.velocities.append(self.velocity.copy())

    def check_collision(self, other: "VGAUPLAgent") -> bool:
        """Check collision with another agent"""
        dist = np.linalg.norm(self.position - other.position)
        return dist < (self.radius + other.radius)


class MultiAgentVGAEnvironment:
    """Multi-Agent Environment for VGA+UPL Testing"""

    def __init__(self, env_type: str = "S_CURVE"):
        self.env_type = env_type
        self.width = 15.0
        self.height = 6.0
        self.obstacles = []
        self.config = VGAConfig()

        # Agent colors matching the images
        self.agent_colors = ["#3498db", "#e74c3c", "#27ae60"]  # Blue, Red, Green
        self.goal_colors = ["#85c1e9", "#f5b7b1", "#82e0aa"]

        self._setup_environment()

    def _setup_environment(self):
        """Setup obstacles based on environment type"""
        if self.env_type == "S_CURVE":
            # Two staggered rectangular obstacles (like in your image)
            self.obstacles = [
                {
                    "type": "rect",
                    "pos": np.array([4.0, 0.0]),
                    "size": (1.0, 2.5),
                },  # Bottom left
                {
                    "type": "rect",
                    "pos": np.array([9.5, 3.5]),
                    "size": (1.0, 2.5),
                },  # Top right
            ]
        elif self.env_type == "CHICANE":
            # Chicane with 3 obstacles
            self.obstacles = [
                {
                    "type": "rect",
                    "pos": np.array([5.5, 3.5]),
                    "size": (1.2, 2.5),
                },  # Top middle
                {
                    "type": "rect",
                    "pos": np.array([4.5, 0.0]),
                    "size": (1.0, 2.0),
                },  # Bottom left
                {
                    "type": "rect",
                    "pos": np.array([9.0, 0.0]),
                    "size": (1.0, 2.0),
                },  # Bottom right
            ]
        elif self.env_type == "NARROW_PASSAGE":
            # Single passage (top and bottom blocks)
            self.obstacles = [
                {
                    "type": "rect",
                    "pos": np.array([6.0, 4.0]),
                    "size": (2.0, 2.0),
                },  # Top
                {
                    "type": "rect",
                    "pos": np.array([6.0, 0.0]),
                    "size": (2.0, 2.0),
                },  # Bottom
            ]

    def get_obstacle_circles(self) -> List[np.ndarray]:
        """Convert obstacles to circles for collision checking"""
        circles = []
        for obs in self.obstacles:
            if obs["type"] == "rect":
                # Approximate rectangle with circles at corners and center
                pos = obs["pos"]
                w, h = obs["size"]
                center = np.array([pos[0] + w / 2, pos[1] + h / 2])
                radius = max(w, h) / 2
                circles.append(np.array([center[0], center[1], radius]))
        return circles

    def create_agents(self) -> List[VGAUPLAgent]:
        """Create 3 VGA+UPL agents"""
        # Starting positions (left side, vertically spread)
        starts = [
            np.array([1.0, 4.8]),  # Top agent (blue)
            np.array([1.0, 3.0]),  # Middle agent (red)
            np.array([1.0, 1.2]),  # Bottom agent (green)
        ]

        # Goals (right side, same vertical positions)
        goals = [
            np.array([13.5, 4.8]),  # Goal 3 (blue)
            np.array([13.5, 3.0]),  # Goal 2 (red)
            np.array([13.5, 1.2]),  # Goal 1 (green)
        ]

        agents = []
        for i in range(3):
            agent = VGAUPLAgent(
                i, starts[i], goals[i], self.agent_colors[i], self.config
            )
            agents.append(agent)

        return agents

    def run_simulation(self, max_steps: int = 500, visualize: bool = True) -> Dict:
        """Run the multi-agent simulation"""
        agents = self.create_agents()
        obstacle_circles = self.get_obstacle_circles()

        results = {
            "success": False,
            "collisions": 0,
            "deadlocks": 0,
            "steps": 0,
            "agents_reached": [False, False, False],
            "collision_events": [],
            "deadlock_events": [],
        }

        print(f"\n{'='*60}")
        print(f"Running {self.env_type} with 3 VGA+UPL Agents")
        print(f"{'='*60}")

        for step in range(max_steps):
            # Update all agents
            for agent in agents:
                if not agent.reached_goal:
                    agent.step(obstacle_circles, agents)

            # Check collisions between agents
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    if agents[i].check_collision(agents[j]):
                        results["collisions"] += 1
                        results["collision_events"].append(
                            {
                                "step": step,
                                "agents": (i, j),
                                "positions": (
                                    agents[i].position.copy(),
                                    agents[j].position.copy(),
                                ),
                            }
                        )
                        if results["collisions"] <= 3:
                            print(
                                f"  [COLLISION] Step {step}: Agent {i} and Agent {j} collided!"
                            )

            # Check for deadlocks (agents stuck for too long)
            for agent in agents:
                if agent.stuck_count > 50 and not agent.reached_goal:
                    results["deadlocks"] += 1
                    results["deadlock_events"].append(
                        {
                            "step": step,
                            "agent": agent.agent_id,
                            "position": agent.position.copy(),
                        }
                    )
                    if results["deadlocks"] <= 3:
                        print(
                            f"  [DEADLOCK] Step {step}: Agent {agent.agent_id} is stuck!"
                        )
                    agent.stuck_count = 0  # Reset to avoid spam

            # Check if all reached goal
            all_reached = all(a.reached_goal for a in agents)
            if all_reached:
                results["success"] = True
                results["steps"] = step
                break

            # Progress update
            if step % 100 == 0:
                reached = sum(1 for a in agents if a.reached_goal)
                print(f"  Step {step}: {reached}/3 agents reached goal")

        results["steps"] = step + 1
        results["agents_reached"] = [a.reached_goal for a in agents]

        # Final status
        print(f"\n{'='*40}")
        print(f"RESULTS for {self.env_type}:")
        print(f"{'='*40}")
        print(f"  Success: {'YES' if results['success'] else 'NO'}")
        print(f"  Steps: {results['steps']}")
        print(f"  Collisions: {results['collisions']}")
        print(f"  Deadlocks: {results['deadlocks']}")
        for i, reached in enumerate(results["agents_reached"]):
            status = "REACHED GOAL" if reached else "FAILED"
            print(f"  Agent {i}: {status}")

        if visualize:
            self._visualize_result(agents, results)

        return results, agents

    def _visualize_result(self, agents: List[VGAUPLAgent], results: Dict):
        """Visualize the simulation result"""
        fig, ax = plt.subplots(figsize=(14, 7))

        ax.set_xlim(-0.5, self.width + 0.5)
        ax.set_ylim(-0.5, self.height + 0.5)
        ax.set_aspect("equal")
        ax.set_facecolor("#f8f9fa")

        # Draw border
        border = Rectangle(
            (0, 0),
            self.width,
            self.height,
            fill=False,
            edgecolor="#2c3e50",
            linewidth=3,
        )
        ax.add_patch(border)

        # Draw obstacles
        for obs in self.obstacles:
            if obs["type"] == "rect":
                rect = Rectangle(
                    obs["pos"],
                    obs["size"][0],
                    obs["size"][1],
                    facecolor="#d35400",
                    edgecolor="#a04000",
                    linewidth=2,
                )
                ax.add_patch(rect)

        # Draw goals
        for i, agent in enumerate(agents):
            goal_circle = Circle(
                agent.goal,
                0.5,
                facecolor=self.goal_colors[i],
                edgecolor=self.agent_colors[i],
                linewidth=2,
                linestyle="--",
                alpha=0.5,
            )
            ax.add_patch(goal_circle)
            ax.plot(
                agent.goal[0],
                agent.goal[1],
                "*",
                color=self.agent_colors[i],
                markersize=20,
                markeredgecolor="white",
                markeredgewidth=1,
            )
            ax.text(
                agent.goal[0] + 0.6,
                agent.goal[1],
                f"GOAL {i+1}",
                fontsize=9,
                color=self.agent_colors[i],
                fontweight="bold",
            )

        # Draw trajectories
        for i, agent in enumerate(agents):
            traj = np.array(agent.trajectory)
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                "-",
                color=agent.color,
                linewidth=2.5,
                alpha=0.7,
                label=f"Agent {i+1}",
            )

            # Draw velocity arrows along trajectory
            for j in range(0, len(traj), 20):
                if j < len(agent.velocities):
                    vel = agent.velocities[j]
                    if np.linalg.norm(vel) > 0.1:
                        ax.arrow(
                            traj[j, 0],
                            traj[j, 1],
                            vel[0] * 0.2,
                            vel[1] * 0.2,
                            head_width=0.15,
                            head_length=0.1,
                            fc=agent.color,
                            ec=agent.color,
                            alpha=0.5,
                        )

        # Draw final agent positions
        for i, agent in enumerate(agents):
            circle = Circle(
                agent.position,
                agent.radius,
                facecolor=agent.color,
                edgecolor="white",
                linewidth=2,
                alpha=0.9,
            )
            ax.add_patch(circle)
            ax.text(
                agent.position[0],
                agent.position[1],
                str(i + 1),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

            # Speed label
            speed = np.linalg.norm(agent.velocity)
            ax.text(
                agent.position[0],
                agent.position[1] - 0.6,
                f"{speed:.1f} m/s",
                ha="center",
                fontsize=8,
                color=agent.color,
            )

        # Draw collision markers
        for collision in results["collision_events"][:5]:  # Show first 5
            pos1, pos2 = collision["positions"]
            mid = (pos1 + pos2) / 2
            ax.plot(
                mid[0], mid[1], "X", color="red", markersize=15, markeredgecolor="white"
            )

        # Title and legend
        status = "SUCCESS" if results["success"] else "FAILED"
        status_color = "#27ae60" if results["success"] else "#e74c3c"

        title = f"{self.env_type} | 3 Agents | VGA+UPL | Status: {status}"
        if results["collisions"] > 0:
            title += f" | Collisions: {results['collisions']}"
        if results["deadlocks"] > 0:
            title += f" | Deadlocks: {results['deadlocks']}"

        ax.set_title(title, fontsize=14, fontweight="bold", color=status_color)

        # Legend
        ax.legend(loc="upper left", fontsize=10)

        # Info box
        info_text = (
            f"Steps: {results['steps']}\n"
            f"Collisions: {results['collisions']}\n"
            f"Deadlocks: {results['deadlocks']}"
        )
        ax.text(
            0.02,
            0.02,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        # Save
        output_dir = "multiagent_vga_comparison/results"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            f"{output_dir}/vga_multiagent_{self.env_type.lower()}.png",
            dpi=150,
            bbox_inches="tight",
        )
        print(f"  Saved: {output_dir}/vga_multiagent_{self.env_type.lower()}.png")
        plt.close()


def run_all_environments():
    """Run VGA+UPL on all 3 environments"""

    print("\n" + "=" * 70)
    print("MULTI-AGENT VGA+UPL SIMULATION")
    print("Testing 3 agents navigating together using VGA+UPL algorithm")
    print("=" * 70)

    environments = ["S_CURVE", "CHICANE", "NARROW_PASSAGE"]
    all_results = {}

    for env_type in environments:
        env = MultiAgentVGAEnvironment(env_type)
        results, agents = env.run_simulation(max_steps=500, visualize=True)
        all_results[env_type] = results

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: VGA+UPL Multi-Agent Performance")
    print("=" * 70)

    total_success = sum(1 for r in all_results.values() if r["success"])
    total_collisions = sum(r["collisions"] for r in all_results.values())
    total_deadlocks = sum(r["deadlocks"] for r in all_results.values())

    print(
        f"\n{'Environment':<20} {'Success':<10} {'Collisions':<12} {'Deadlocks':<10} {'Steps':<10}"
    )
    print("-" * 62)

    for env_type, results in all_results.items():
        success = "YES" if results["success"] else "NO"
        print(
            f"{env_type:<20} {success:<10} {results['collisions']:<12} {results['deadlocks']:<10} {results['steps']:<10}"
        )

    print("-" * 62)
    print(
        f"{'TOTAL':<20} {total_success}/3       {total_collisions:<12} {total_deadlocks:<10}"
    )

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)

    if total_success < 3 or total_collisions > 0 or total_deadlocks > 0:
        print("\n  VGA+UPL STRUGGLES with multi-agent scenarios!")
        print("\n  Problems observed:")
        if total_collisions > 0:
            print(f"    - {total_collisions} COLLISIONS between agents")
        if total_deadlocks > 0:
            print(f"    - {total_deadlocks} DEADLOCKS (agents got stuck)")
        if total_success < 3:
            print(f"    - Only {total_success}/3 environments completed successfully")
        print("\n  Why VGA fails at multi-agent:")
        print("    1. VGA treats other agents as STATIC obstacles")
        print("    2. No prediction of other agents' velocities")
        print("    3. No coordination mechanism")
        print("    4. Deterministic = same response = possible deadlocks")
    else:
        print("\n  VGA+UPL succeeded in these scenarios.")
        print("  However, performance degrades with more agents or complex scenarios.")

    return all_results


if __name__ == "__main__":
    results = run_all_environments()
