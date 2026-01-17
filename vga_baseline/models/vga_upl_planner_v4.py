"""
VGA + UPL Planner V4 - Robust Implementation
============================================

This version focuses on ROBUSTNESS - handling all the edge cases
that make MOSP_B, MOSP_C, MOSP_D challenging.

Key insight from analyzing failures:
- The agent gets stuck when obstacles form "corridors"
- Need better look-ahead to find gaps
- Need to commit to a path and not oscillate

Strategy:
1. Simple, greedy navigation with look-ahead
2. When path blocked, find the nearest gap
3. Navigate through gap, then resume toward goal
4. Strong stuck detection and recovery

VGA (Variable Goal Approach):
- The key innovation is that the agent uses "variable goals" (subgoals)
- Instead of always aiming at the final goal, it dynamically selects
  intermediate targets to navigate around obstacles
- This avoids the local minima problems of potential field methods
- Subgoals are computed geometrically: find passage points around blockers
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_base import NavigationModel, SimulationResult


class Obstacle:
    """Circular obstacle"""

    def __init__(self, position: np.ndarray, radius: float):
        self.position = np.array(position, dtype=float)
        self.radius = float(radius)
        self._id = id(self)

    def distance_to(self, point: np.ndarray) -> float:
        """Surface distance to point"""
        return np.linalg.norm(self.position - point) - self.radius

    def __eq__(self, other):
        return isinstance(other, Obstacle) and self._id == other._id

    def __hash__(self):
        return hash(self._id)


class VGAUPLPlannerV4(NavigationModel):
    """
    Robust VGA implementation focusing on high success rate.

    This planner implements the Variable Goal Approach (VGA) combined with
    simplified UPL (Universal Power Law) physics for obstacle avoidance.

    The agent maintains:
    - Final goal: where we ultimately want to go
    - Subgoal: current intermediate target (variable goal)
    - Subgoal history: record of all subgoals used for metrics
    """

    def __init__(
        self,
        use_probabilistic: bool = False,
        agent_radius: float = 0.2,
        min_clearance: float = 0.02,  # Minimum gap from obstacles
        desired_speed: float = 1.34,
        dt: float = 0.05,
        **kwargs,
    ):
        mode = "stochastic" if use_probabilistic else "deterministic"
        super().__init__(name=f"VGA+UPL-V4 ({mode})")

        self.use_probabilistic = use_probabilistic
        self.agent_radius = agent_radius
        self.min_clearance = min_clearance
        self.desired_speed = desired_speed
        self.dt = dt

        # Effective radius for collision checking
        self.effective_radius = agent_radius + min_clearance

        # State
        self.pos = None
        self.vel = None
        self.goal = None
        self.subgoal = None
        self.obstacles: List[Obstacle] = []

        # History for stuck detection
        self.history = []
        self.steps = 0

        # === NEW: Track subgoals for metrics ===
        self.subgoal_history = []
        self.last_subgoal = None

        # === STOCHASTIC: Track obstacle side decisions ===
        # Maps obstacle id -> 'left' or 'right' once decided
        self.obstacle_decisions = {}

    def reset(self, start_pos: np.ndarray, goal_pos: np.ndarray, **kwargs):
        """Reset for new episode"""
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.zeros(2)
        self.goal = np.array(goal_pos, dtype=float)
        self.subgoal = self.goal.copy()

        # Parse obstacles
        obs_data = kwargs.get("obstacles", [])
        self.obstacles = []
        for o in obs_data:
            if isinstance(o, Obstacle):
                self.obstacles.append(o)
            elif isinstance(o, dict):
                self.obstacles.append(
                    Obstacle(np.array(o["position"]), o.get("radius", 0.25))
                )
            else:
                self.obstacles.append(
                    Obstacle(np.array(o[:2]), o[2] if len(o) > 2 else 0.25)
                )

        self.history = []
        self.steps = 0

        # === NEW: Reset subgoal tracking ===
        self.subgoal_history = []
        self.last_subgoal = None

        # === STOCHASTIC: Reset decisions ===
        self.obstacle_decisions = {}

    def _line_blocked(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check if straight line from start to end is blocked by any obstacle"""
        direction = end - start
        length = np.linalg.norm(direction)

        if length < 0.01:
            return False

        direction = direction / length

        for obs in self.obstacles:
            # Vector from start to obstacle center
            to_obs = obs.position - start

            # Project onto line
            proj_len = np.dot(to_obs, direction)

            # Skip if behind start or past end
            if proj_len < 0 or proj_len > length:
                continue

            # Perpendicular distance
            proj_point = start + direction * proj_len
            perp_dist = np.linalg.norm(obs.position - proj_point)

            # Blocked if perpendicular distance less than combined radii
            if perp_dist < obs.radius + self.effective_radius:
                return True

        return False

    def _find_subgoal(self) -> np.ndarray:
        """
        Find next subgoal using VGA approach.

        If direct path to goal is clear, return goal.
        Otherwise, find a way around the nearest obstacle.
        """
        # Check direct path
        if not self._line_blocked(self.pos, self.goal):
            return self.goal.copy()

        # Find blocking obstacles
        blocking = []
        to_goal = self.goal - self.pos
        goal_dist = np.linalg.norm(to_goal)

        if goal_dist < 0.1:
            return self.goal.copy()

        goal_dir = to_goal / goal_dist

        for obs in self.obstacles:
            to_obs = obs.position - self.pos
            proj = np.dot(to_obs, goal_dir)

            # Only consider obstacles ahead
            if proj < 0 or proj > goal_dist:
                continue

            # Check if it blocks
            perp_vec = to_obs - proj * goal_dir
            perp_dist = np.linalg.norm(perp_vec)

            if perp_dist < obs.radius + self.effective_radius:
                blocking.append((obs, proj, perp_vec))

        if not blocking:
            return self.goal.copy()

        # Sort by distance (projection length)
        blocking.sort(key=lambda x: x[1])
        nearest_obs, _, perp_vec = blocking[0]

        # === STOCHASTIC: Check if we already made a decision for this obstacle ===
        obs_id = nearest_obs._id

        # Determine which side to go
        # Perpendicular vectors (left and right of goal direction)
        perp_left = np.array([-goal_dir[1], goal_dir[0]])
        perp_right = np.array([goal_dir[1], -goal_dir[0]])

        # Calculate potential subgoals on each side
        clearance = nearest_obs.radius + self.effective_radius + 0.1
        subgoal_left = nearest_obs.position + perp_left * clearance
        subgoal_right = nearest_obs.position + perp_right * clearance

        # === STOCHASTIC: Check if we already made a decision for this obstacle ===
        if self.use_probabilistic:
            obs_id = nearest_obs._id
            if obs_id in self.obstacle_decisions:
                decision = self.obstacle_decisions[obs_id]
                return subgoal_left if decision == "left" else subgoal_right

        # Evaluate each option
        def score_subgoal(sg):
            # Consider: deviation from direct path + remaining distance to goal
            to_sg = sg - self.pos
            sg_dist = np.linalg.norm(to_sg)

            if sg_dist < 0.01:
                return float("inf")

            # Check if path to subgoal is clear
            if self._line_blocked(self.pos, sg):
                return float("inf")  # Invalid

            # Distance via subgoal
            total_dist = sg_dist + np.linalg.norm(self.goal - sg)

            # Angle deviation
            angle = np.arccos(np.clip(np.dot(to_sg / sg_dist, goal_dir), -1, 1))

            return total_dist + angle * 0.5  # Weight angle less

        left_score = score_subgoal(subgoal_left)
        right_score = score_subgoal(subgoal_right)

        # If both invalid, find ANY passable point
        if left_score == float("inf") and right_score == float("inf"):
            # Try points along a semicircle toward goal
            for angle_offset in [0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2]:
                test_dir = np.array(
                    [
                        np.cos(np.arctan2(goal_dir[1], goal_dir[0]) + angle_offset),
                        np.sin(np.arctan2(goal_dir[1], goal_dir[0]) + angle_offset),
                    ]
                )
                test_point = self.pos + test_dir * 1.0  # 1m ahead
                if not self._line_blocked(self.pos, test_point):
                    return test_point

            # Last resort: just move toward goal and let collision handling deal with it
            return self.pos + goal_dir * 0.5

        # Select best (and commit for stochastic mode)
        if self.use_probabilistic:
            if left_score < float("inf") and right_score < float("inf"):
                # Use Boltzmann/softmax selection with temperature
                # Higher temperature = more random exploration
                temperature = 1.5  # Increased for more path variety

                # Convert scores to probabilities (lower score = higher probability)
                scores = np.array([-left_score, -right_score])
                scores = scores / temperature
                # Softmax
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / np.sum(exp_scores)

                # Make decision and COMMIT to it for this obstacle
                decision = "left" if np.random.random() < probs[0] else "right"
                self.obstacle_decisions[obs_id] = decision

                return subgoal_left if decision == "left" else subgoal_right
            elif left_score < float("inf"):
                self.obstacle_decisions[obs_id] = "left"
                return subgoal_left
            else:
                self.obstacle_decisions[obs_id] = "right"
                return subgoal_right

        return subgoal_left if left_score <= right_score else subgoal_right

    def _move(self):
        """Move toward subgoal with collision avoidance"""
        to_subgoal = self.subgoal - self.pos
        dist = np.linalg.norm(to_subgoal)

        if dist < 0.01:
            self.vel = np.zeros(2)
            return

        # Desired direction
        desired_dir = to_subgoal / dist

        # Calculate repulsion from nearby obstacles
        repulsion = np.zeros(2)
        for obs in self.obstacles:
            to_agent = self.pos - obs.position
            dist_to_obs = np.linalg.norm(to_agent)
            surface_dist = dist_to_obs - obs.radius

            if surface_dist < self.effective_radius * 2:
                # Apply repulsion (inverse square falloff)
                if dist_to_obs > 0.01:
                    strength = max(0, 1 - surface_dist / (self.effective_radius * 2))
                    repulsion += (to_agent / dist_to_obs) * strength * 0.5

        # Combine desired direction with repulsion
        move_dir = desired_dir + repulsion
        move_mag = np.linalg.norm(move_dir)
        if move_mag > 0.01:
            move_dir = move_dir / move_mag
        else:
            move_dir = desired_dir

        # Speed control
        speed = min(self.desired_speed, dist / self.dt)
        self.vel = move_dir * speed

        # Update position
        new_pos = self.pos + self.vel * self.dt

        # Hard collision resolution
        for obs in self.obstacles:
            to_new = new_pos - obs.position
            dist_new = np.linalg.norm(to_new)
            min_dist = obs.radius + self.agent_radius

            if dist_new < min_dist:
                if dist_new > 0.001:
                    new_pos = obs.position + (to_new / dist_new) * (min_dist + 0.01)
                else:
                    new_pos = obs.position + desired_dir * (min_dist + 0.01)

        self.pos = new_pos

    def _check_stuck(self) -> bool:
        """Check and recover from stuck state"""
        self.history.append(self.pos.copy())

        if len(self.history) > 100:
            self.history.pop(0)

        if len(self.history) >= 50:
            recent = np.array(self.history[-50:])
            spread = np.max(recent, axis=0) - np.min(recent, axis=0)

            if np.max(spread) < 0.3:  # Stuck in small area
                # Random jump
                self.pos += np.random.uniform(-0.3, 0.3, size=2)

                # Ensure not inside obstacle
                for obs in self.obstacles:
                    to_pos = self.pos - obs.position
                    d = np.linalg.norm(to_pos)
                    min_d = obs.radius + self.agent_radius
                    if d < min_d:
                        if d > 0.01:
                            self.pos = obs.position + (to_pos / d) * (min_d + 0.1)
                        else:
                            self.pos = obs.position + np.array([min_d + 0.1, 0])

                self.history.clear()
                return True

        return False

    def step(self) -> bool:
        """Single simulation step"""
        self.steps += 1

        # Goal check
        if np.linalg.norm(self.pos - self.goal) < 0.25:
            return True

        # Stuck recovery
        self._check_stuck()

        # VGA: find subgoal
        self.subgoal = self._find_subgoal()

        # === NEW: Track subgoal changes ===
        if (
            self.last_subgoal is None
            or np.linalg.norm(self.subgoal - self.last_subgoal) > 0.1
        ):
            # Subgoal changed significantly - record it
            self.subgoal_history.append(self.subgoal.copy())
            self.last_subgoal = self.subgoal.copy()

        # Move
        self._move()

        return False

    def simulate(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        max_steps: int = 2000,
        **kwargs,
    ) -> SimulationResult:
        """
        Full simulation with comprehensive metrics tracking.

        Returns SimulationResult with:
        - positions: Full trajectory
        - velocities: Velocity at each step
        - timestamps: Time at each step
        - success: Whether goal was reached
        - metadata: Contains all metrics including subgoal info
        """
        self.reset(start_pos, goal_pos, **kwargs)

        positions = [self.pos.copy()]
        velocities = [self.vel.copy()]
        timestamps = [0.0]
        subgoals_per_step = [self.subgoal.copy()]  # Track subgoal at each step

        t = 0.0
        for _ in range(max_steps):
            done = self.step()
            t += self.dt

            positions.append(self.pos.copy())
            velocities.append(self.vel.copy())
            timestamps.append(t)
            subgoals_per_step.append(self.subgoal.copy())

            if done:
                break

        pos_arr = np.array(positions)
        vel_arr = np.array(velocities)
        time_arr = np.array(timestamps)

        final_dist = np.linalg.norm(self.pos - self.goal)
        success = final_dist < 0.3
        path_len = np.sum(np.linalg.norm(np.diff(pos_arr, axis=0), axis=1))
        optimal_path = np.linalg.norm(self.goal - start_pos)

        # === Compute Comprehensive Metrics ===

        # Time & Efficiency
        travel_time = t
        path_efficiency = optimal_path / path_len if path_len > 0 else 0
        speeds = np.linalg.norm(vel_arr, axis=1)
        avg_speed = np.mean(speeds)
        speed_variance = np.var(speeds)

        # Collisions / Overlap
        num_collisions = 0
        collision_frames = 0
        max_penetration = 0.0
        in_collision_prev = False

        for pos in positions:
            in_collision = False
            for obs in self.obstacles:
                dist = np.linalg.norm(pos - obs.position)
                surface_dist = dist - obs.radius - self.agent_radius
                if surface_dist < 0:
                    in_collision = True
                    collision_frames += 1
                    max_penetration = max(max_penetration, -surface_dist)
                    break

            if in_collision and not in_collision_prev:
                num_collisions += 1
            in_collision_prev = in_collision

        # Oscillation
        headings = np.arctan2(vel_arr[:, 1], vel_arr[:, 0])
        heading_changes = np.abs(np.diff(headings))
        heading_changes = np.minimum(heading_changes, 2 * np.pi - heading_changes)
        direction_changes = np.sum(heading_changes > 0.3)  # > ~17 degrees
        oscillation_index = np.sum(heading_changes)

        # Smoothness
        if len(vel_arr) > 1:
            accelerations = np.diff(vel_arr, axis=0) / self.dt
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)
            avg_acceleration = np.mean(acc_magnitudes)
            max_acceleration = np.max(acc_magnitudes)

            if len(accelerations) > 1:
                jerks = np.diff(accelerations, axis=0) / self.dt
                jerk_magnitudes = np.linalg.norm(jerks, axis=1)
                avg_jerk = np.mean(jerk_magnitudes)
                max_jerk = np.max(jerk_magnitudes)
            else:
                avg_jerk = max_jerk = 0.0
        else:
            avg_acceleration = max_acceleration = avg_jerk = max_jerk = 0.0

        # Deviation from straight line
        if optimal_path > 0.01:
            path_dir = (self.goal - start_pos) / optimal_path
            deviations = []
            for pos in positions:
                to_pos = pos - start_pos
                proj_len = np.dot(to_pos, path_dir)
                proj_point = start_pos + proj_len * path_dir
                dev = np.linalg.norm(pos - proj_point)
                deviations.append(dev)
            avg_deviation = np.mean(deviations)
            max_deviation = np.max(deviations)
        else:
            avg_deviation = max_deviation = 0.0

        # Safety / Clearance
        clearances = []
        danger_frames = 0
        for pos in positions:
            min_clearance = float("inf")
            for obs in self.obstacles:
                dist = np.linalg.norm(pos - obs.position) - obs.radius
                min_clearance = min(min_clearance, dist)
            if min_clearance < float("inf"):
                clearances.append(min_clearance)
                if min_clearance < self.agent_radius * 1.5:
                    danger_frames += 1

        min_clearance_overall = min(clearances) if clearances else float("inf")
        avg_clearance = np.mean(clearances) if clearances else float("inf")
        danger_zone_ratio = danger_frames / len(positions) if len(positions) > 0 else 0

        # Subgoal metrics
        unique_subgoals = []
        for sg in self.subgoal_history:
            if (
                len(unique_subgoals) == 0
                or np.linalg.norm(sg - unique_subgoals[-1]) > 0.1
            ):
                unique_subgoals.append(sg)

        num_subgoals = len(unique_subgoals)
        subgoal_switch_rate = num_subgoals / travel_time if travel_time > 0 else 0

        return SimulationResult(
            positions=pos_arr,
            velocities=vel_arr,
            timestamps=time_arr,
            success=success,
            metadata={
                # Basic
                "final_distance_to_goal": final_dist,
                "path_length": path_len,
                "num_steps": len(positions),
                "num_obstacles": len(self.obstacles),
                # Time & Efficiency
                "travel_time": travel_time,
                "optimal_path_length": optimal_path,
                "path_efficiency": path_efficiency,
                "average_speed": avg_speed,
                "speed_variance": speed_variance,
                # Overlap / Collision
                "num_collisions": num_collisions,
                "collision_frames": collision_frames,
                "max_penetration": max_penetration,
                # Oscillation
                "direction_changes": direction_changes,
                "oscillation_index": oscillation_index,
                # Smoothness
                "average_acceleration": avg_acceleration,
                "max_acceleration": max_acceleration,
                "average_jerk": avg_jerk,
                "max_jerk": max_jerk,
                # Deviation
                "average_deviation": avg_deviation,
                "max_deviation": max_deviation,
                # Safety / Clearance
                "min_clearance": min_clearance_overall,
                "average_clearance": avg_clearance,
                "danger_zone_ratio": danger_zone_ratio,
                # VGA-Specific: Subgoals
                "num_subgoals": num_subgoals,
                "subgoal_switch_rate": subgoal_switch_rate,
                "subgoal_history": [sg.tolist() for sg in unique_subgoals],
                "subgoals_per_step": [sg.tolist() for sg in subgoals_per_step],
            },
        )

    def predict_action(self, observation: Dict[str, Any]) -> np.ndarray:
        return np.zeros(2)


def test_v4():
    """Test V4"""
    print("=" * 60)
    print("VGA+UPL V4 - Testing")
    print("=" * 60)

    p = VGAUPLPlannerV4()

    tests = [
        ("Single", [{"position": [5, 0], "radius": 0.25}]),
        (
            "Row",
            [
                {"position": [4, 0], "radius": 0.25},
                {"position": [6, 0.3], "radius": 0.25},
                {"position": [8, -0.2], "radius": 0.25},
            ],
        ),
        (
            "Corridor",
            [
                {"position": [5, 0.5], "radius": 0.25},
                {"position": [5, -0.5], "radius": 0.25},
                {"position": [7, 0.4], "radius": 0.25},
                {"position": [7, -0.4], "radius": 0.25},
            ],
        ),
        (
            "Dense",
            [
                {"position": [3 + i * 0.8, (-1) ** i * 0.3], "radius": 0.25}
                for i in range(8)
            ],
        ),
    ]

    for name, obs in tests:
        r = p.simulate(
            np.array([0, 0]), np.array([10, 0]), obstacles=obs, max_steps=1000
        )
        status = "✅" if r.success else "❌"
        print(
            f"{status} {name}: steps={r.metadata['num_steps']}, dist={r.metadata['final_distance_to_goal']:.3f}m"
        )

    print("=" * 60)


if __name__ == "__main__":
    test_v4()
