"""
Multi-Agent VGA+UPL Video Generator
===================================
Generate videos showing 3 agents using VGA+UPL in various environments.
- Collision = FAIL
- Out of bounds = FAIL
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import os


class VGAAgent:
    """VGA+UPL Agent"""

    def __init__(self, agent_id, start, goal, color, speed=2.2):
        self.agent_id = agent_id
        self.position = start.astype(float)
        self.goal = goal.astype(float)
        self.color = color
        self.speed = speed
        self.radius = 0.3
        self.trajectory = [self.position.copy()]
        self.velocity = np.array([0.0, 0.0])
        self.reached_goal = False
        self.failed = False
        self.fail_reason = ""

    def compute_velocity(self, obstacles, other_agents):
        if self.reached_goal or self.failed:
            return np.array([0.0, 0.0])

        to_goal = self.goal - self.position
        dist = np.linalg.norm(to_goal)
        if dist < 0.4:
            self.reached_goal = True
            return np.array([0.0, 0.0])

        goal_dir = to_goal / dist

        # VGA grid search (deterministic)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 11)
        best_vel = goal_dir * self.speed
        best_score = -1e9

        for angle in angles:
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]])
            vel = rot @ goal_dir * self.speed

            score = np.dot(vel, goal_dir) * 10
            future = self.position + vel * 0.3

            # Obstacles
            for obs in obstacles:
                d = np.linalg.norm(future - obs[:2]) - obs[2] - self.radius
                if d < 0.3:
                    score -= 150 / max(d, 0.01)

            # Other agents (VGA: treats as static!)
            for other in other_agents:
                if other.agent_id != self.agent_id and not other.failed:
                    d = np.linalg.norm(future - other.position) - 2 * self.radius
                    if d < 0.8:
                        score -= 100 / max(d, 0.01)

            if score > best_score:
                best_score = score
                best_vel = vel

        return best_vel

    def step(self, obstacles, agents, bounds, dt=0.05):
        if self.failed or self.reached_goal:
            return

        self.velocity = self.compute_velocity(obstacles, agents)
        self.position += self.velocity * dt
        self.trajectory.append(self.position.copy())

        # Check bounds (with small margin)
        margin = 0.1
        if (
            self.position[0] < bounds[0] + margin
            or self.position[0] > bounds[1] - margin
            or self.position[1] < bounds[2] + margin
            or self.position[1] > bounds[3] - margin
        ):
            self.failed = True
            self.fail_reason = "OUT OF BOUNDS"

    def check_collision(self, other):
        if self.failed or other.failed:
            return False
        dist = np.linalg.norm(self.position - other.position)
        return dist < (self.radius + other.radius) * 0.9


class MultiAgentSimulator:
    """Simulator with video generation"""

    def __init__(self, env_type, width=15, height=6):
        self.env_type = env_type
        self.width = width
        self.height = height
        self.bounds = (0, width, 0, height)
        self.obstacles = []
        self.agents = []
        self.dt = 0.05
        self.colors = ["#3498db", "#e74c3c", "#27ae60"]  # Blue, Red, Green

        self._setup_environment()
        self._setup_agents()

    def _setup_environment(self):
        if self.env_type == "HEAD_ON":
            # Head-on collision - no obstacles, pure collision test
            self.obstacles = []
        elif self.env_type == "NARROW_GAP":
            # Single narrow gap all must pass through - gap at y=2.5 to y=3.5
            self.obstacles = [
                np.array([7.0, 4.8, 1.0]),  # Top wall (smaller)
                np.array([7.0, 1.2, 1.0]),  # Bottom wall (smaller)
            ]
        elif self.env_type == "DIRECT_COLLISION":
            # Direct head-on: 2 agents on EXACT same Y line going opposite directions
            self.obstacles = []
        elif self.env_type == "TIGHT_CORRIDOR":
            # Tight corridor - forces close interaction
            self.obstacles = [
                np.array([4.0, 4.5, 0.8]),
                np.array([4.0, 1.5, 0.8]),
                np.array([11.0, 4.5, 0.8]),
                np.array([11.0, 1.5, 0.8]),
            ]
        elif self.env_type == "MERGE":
            # 3 agents merging to same point
            self.obstacles = []

    def _setup_agents(self):
        if self.env_type == "HEAD_ON":
            # 2 agents going opposite directions on same Y level
            self.agents = [
                VGAAgent(
                    0, np.array([1.5, 3.0]), np.array([13.5, 3.0]), self.colors[0]
                ),
                VGAAgent(
                    1, np.array([13.5, 3.0]), np.array([1.5, 3.0]), self.colors[1]
                ),
                VGAAgent(2, np.array([7.5, 1.5]), np.array([7.5, 4.5]), self.colors[2]),
            ]
        elif self.env_type == "NARROW_GAP":
            # All 3 agents must pass through narrow gap - keep in middle Y range
            self.agents = [
                VGAAgent(
                    0, np.array([1.5, 3.3]), np.array([13.5, 3.3]), self.colors[0]
                ),
                VGAAgent(
                    1, np.array([1.5, 3.0]), np.array([13.5, 3.0]), self.colors[1]
                ),
                VGAAgent(
                    2, np.array([1.5, 2.7]), np.array([13.5, 2.7]), self.colors[2]
                ),
            ]
        elif self.env_type == "DIRECT_COLLISION":
            # 2 agents on SAME Y line going opposite ways - GUARANTEED collision
            self.agents = [
                VGAAgent(
                    0, np.array([2.0, 3.0]), np.array([13.0, 3.0]), self.colors[0]
                ),
                VGAAgent(
                    1, np.array([13.0, 3.0]), np.array([2.0, 3.0]), self.colors[1]
                ),
                VGAAgent(
                    2, np.array([2.0, 3.5]), np.array([13.0, 3.5]), self.colors[2]
                ),  # parallel
            ]
        elif self.env_type == "TIGHT_CORRIDOR":
            # 3 agents in tight corridor - opposite directions
            self.agents = [
                VGAAgent(
                    0, np.array([2.0, 3.2]), np.array([13.0, 3.2]), self.colors[0]
                ),
                VGAAgent(
                    1, np.array([13.0, 3.0]), np.array([2.0, 3.0]), self.colors[1]
                ),
                VGAAgent(
                    2, np.array([2.0, 2.8]), np.array([13.0, 2.8]), self.colors[2]
                ),
            ]
        elif self.env_type == "MERGE":
            # 3 agents all going to same goal - will collide at destination
            self.agents = [
                VGAAgent(
                    0, np.array([1.5, 4.0]), np.array([10.0, 3.0]), self.colors[0]
                ),
                VGAAgent(
                    1, np.array([1.5, 2.0]), np.array([10.0, 3.0]), self.colors[1]
                ),
                VGAAgent(
                    2, np.array([1.5, 3.0]), np.array([10.0, 3.0]), self.colors[2]
                ),
            ]

    def step(self):
        """One simulation step, returns True if collision/failure"""
        # Update all agents
        for agent in self.agents:
            agent.step(self.obstacles, self.agents, self.bounds, self.dt)

        # Check collisions
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                if self.agents[i].check_collision(self.agents[j]):
                    self.agents[i].failed = True
                    self.agents[j].failed = True
                    self.agents[i].fail_reason = f"COLLISION with Agent {j+1}"
                    self.agents[j].fail_reason = f"COLLISION with Agent {i+1}"
                    return True

        # Check if any failed (bounds)
        for agent in self.agents:
            if agent.failed:
                return True

        return False

    def is_done(self):
        """Check if simulation is complete"""
        all_done = all(a.reached_goal or a.failed for a in self.agents)
        any_failed = any(a.failed for a in self.agents)
        return all_done or any_failed

    def get_status(self, max_steps_reached=False):
        """Get simulation status"""
        any_failed = any(a.failed for a in self.agents)
        all_reached = all(a.reached_goal for a in self.agents)

        if any_failed:
            reasons = [a.fail_reason for a in self.agents if a.failed]
            return "FAILED", reasons[0] if reasons else "Unknown"
        elif all_reached:
            return "SUCCESS", "All agents reached goals"
        elif max_steps_reached:
            return "FAILED", "DEADLOCK - Agents stuck avoiding each other"
        else:
            return "RUNNING", ""

    def generate_video(self, output_path, max_steps=400):
        """Generate video of the simulation"""

        # Run simulation first to collect all frames
        frames_data = []

        for step in range(max_steps):
            # Store current state
            frame = {
                "step": step,
                "positions": [a.position.copy() for a in self.agents],
                "velocities": [a.velocity.copy() for a in self.agents],
                "reached": [a.reached_goal for a in self.agents],
                "failed": [a.failed for a in self.agents],
                "trajectories": [np.array(a.trajectory.copy()) for a in self.agents],
            }
            frames_data.append(frame)

            # Step simulation
            collision = self.step()

            if self.is_done():
                # Add a few more frames to show final state
                for _ in range(20):
                    frame = {
                        "step": step + 1,
                        "positions": [a.position.copy() for a in self.agents],
                        "velocities": [a.velocity.copy() for a in self.agents],
                        "reached": [a.reached_goal for a in self.agents],
                        "failed": [a.failed for a in self.agents],
                        "trajectories": [
                            np.array(a.trajectory.copy()) for a in self.agents
                        ],
                    }
                    frames_data.append(frame)
                break

        # Check if we hit max steps (deadlock)
        max_steps_reached = (step >= max_steps - 1) and not self.is_done()
        status, reason = self.get_status(max_steps_reached=max_steps_reached)

        # Create animation
        fig, ax = plt.subplots(figsize=(14, 7))

        def init():
            ax.clear()
            return []

        def animate(frame_idx):
            ax.clear()

            if frame_idx >= len(frames_data):
                frame_idx = len(frames_data) - 1

            frame = frames_data[frame_idx]

            # Setup axes
            ax.set_xlim(-0.5, self.width + 0.5)
            ax.set_ylim(-0.5, self.height + 0.5)
            ax.set_aspect("equal")
            ax.set_facecolor("#f8f9fa")

            # Border
            border = Rectangle(
                (0, 0),
                self.width,
                self.height,
                fill=False,
                edgecolor="#2c3e50",
                linewidth=3,
            )
            ax.add_patch(border)

            # Obstacles
            for obs in self.obstacles:
                circle = Circle(obs[:2], obs[2], color="#d35400", alpha=0.9)
                ax.add_patch(circle)

            # Goals
            for i, agent in enumerate(self.agents):
                goal_circle = Circle(
                    agent.goal,
                    0.4,
                    facecolor="white",
                    edgecolor=agent.color,
                    linewidth=2,
                    linestyle="--",
                    alpha=0.7,
                )
                ax.add_patch(goal_circle)
                ax.plot(
                    agent.goal[0], agent.goal[1], "*", color=agent.color, markersize=15
                )
                ax.text(
                    agent.goal[0],
                    agent.goal[1] + 0.6,
                    f"GOAL {i+1}",
                    ha="center",
                    fontsize=8,
                    color=agent.color,
                    fontweight="bold",
                )

            # Trajectories
            for i, traj in enumerate(frame["trajectories"]):
                if len(traj) > 1:
                    ax.plot(
                        traj[:, 0],
                        traj[:, 1],
                        "-",
                        color=self.colors[i],
                        linewidth=2,
                        alpha=0.5,
                    )

            # Agents
            for i, (pos, vel, reached, failed) in enumerate(
                zip(
                    frame["positions"],
                    frame["velocities"],
                    frame["reached"],
                    frame["failed"],
                )
            ):

                # Agent circle
                if failed:
                    color = "#7f8c8d"  # Gray for failed
                    edge_color = "red"
                    edge_width = 3
                elif reached:
                    color = self.colors[i]
                    edge_color = "green"
                    edge_width = 3
                else:
                    color = self.colors[i]
                    edge_color = "white"
                    edge_width = 2

                circle = Circle(
                    pos,
                    0.3,
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    alpha=0.9,
                )
                ax.add_patch(circle)

                # Agent number
                ax.text(
                    pos[0],
                    pos[1],
                    str(i + 1),
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                )

                # Velocity arrow
                if not failed and not reached and np.linalg.norm(vel) > 0.1:
                    ax.arrow(
                        pos[0],
                        pos[1],
                        vel[0] * 0.15,
                        vel[1] * 0.15,
                        head_width=0.12,
                        head_length=0.08,
                        fc="orange",
                        ec="orange",
                        alpha=0.8,
                    )

                # Speed label
                speed = np.linalg.norm(vel)
                ax.text(
                    pos[0],
                    pos[1] - 0.5,
                    f"{speed:.1f} m/s",
                    ha="center",
                    fontsize=7,
                    color=self.colors[i],
                )

            # Title with status
            status_now, reason_now = (
                self.get_status()
                if frame_idx == len(frames_data) - 1
                else ("RUNNING", "")
            )

            if any(frame["failed"]):
                title_color = "#e74c3c"
                status_text = "FAILED"
                for i, f in enumerate(frame["failed"]):
                    if f:
                        status_text = f"FAILED - {self.agents[i].fail_reason}"
                        break
            elif all(frame["reached"]):
                title_color = "#27ae60"
                status_text = "SUCCESS"
            else:
                title_color = "#2c3e50"
                status_text = "RUNNING"

            ax.set_title(
                f'{self.env_type} | 3 Agents | VGA+UPL | Step {frame["step"]} | {status_text}',
                fontsize=14,
                fontweight="bold",
                color=title_color,
            )

            # Legend
            ax.text(
                0.02,
                0.98,
                "Direction",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                color="#2c3e50",
            )
            ax.text(
                0.15,
                0.98,
                "Velocity",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                color="orange",
            )
            ax.text(
                0.28,
                0.98,
                "Goal",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                color="#2c3e50",
            )
            ax.text(
                0.38,
                0.98,
                "Agent",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                color="#2c3e50",
            )

            return []

        # Create animation
        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(frames_data),
            interval=50,
            blit=False,
        )

        # Save as GIF (more compatible)
        print(f"  Generating video: {output_path}")
        writer = PillowWriter(fps=20)
        anim.save(output_path, writer=writer)
        plt.close()

        return status, reason


def run_all_scenarios():
    """Run all scenarios and generate videos"""

    # Get absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "videos")
    os.makedirs(output_dir, exist_ok=True)

    scenarios = ["HEAD_ON", "NARROW_GAP", "DIRECT_COLLISION", "TIGHT_CORRIDOR", "MERGE"]

    print("=" * 70)
    print("MULTI-AGENT VGA+UPL VIDEO GENERATION")
    print("Collision = FAIL | Out of Bounds = FAIL")
    print("=" * 70)

    results = {}

    for scenario in scenarios:
        print(f"\n[{scenario}] Running simulation...")

        sim = MultiAgentSimulator(scenario)
        output_path = f"{output_dir}/vga_{scenario.lower()}.gif"

        status, reason = sim.generate_video(output_path)
        results[scenario] = {"status": status, "reason": reason}

        print(f"  Status: {status}")
        if reason:
            print(f"  Reason: {reason}")
        print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success_count = sum(1 for r in results.values() if r["status"] == "SUCCESS")
    fail_count = len(results) - success_count

    print(f"\n{'Scenario':<20} {'Status':<10} {'Reason'}")
    print("-" * 60)
    for scenario, result in results.items():
        print(f"{scenario:<20} {result['status']:<10} {result['reason']}")

    print("-" * 60)
    print(f"\nSuccess: {success_count}/{len(results)}")
    print(f"Failed: {fail_count}/{len(results)}")

    if fail_count > 0:
        print("\n" + "=" * 70)
        print("CONCLUSION: VGA+UPL FAILS in multi-agent scenarios!")
        print("=" * 70)
        print("  - VGA treats other agents as STATIC obstacles")
        print("  - No velocity prediction = collisions when paths cross")
        print("  - Deterministic policy = no coordination")

    return results


if __name__ == "__main__":
    results = run_all_scenarios()
