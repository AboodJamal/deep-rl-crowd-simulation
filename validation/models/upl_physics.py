"""
Universal Power Law (UPL) Physics Model
========================================

Implements the physics-based movement model from the VGA paper (arXiv:2501.05100).
This module calculates forces and velocities using the Universal Power Law equations
to simulate realistic pedestrian dynamics.

Physics Equations (from paper):
-------------------------------
1. Interaction Force (F_int): Repulsion from obstacles/agents
   F_int = A * exp((r - d) / B) * n_ij
   where:
   - A: strength parameter
   - B: range parameter
   - r: interaction radius
   - d: actual distance
   - n_ij: unit vector pointing away from obstacle

2. Self-Propulsion Force (F_self): Drives agent toward goal
   F_self = (v_des - v_curr) / tau
   where:
   - v_des: desired velocity toward goal
   - v_curr: current velocity
   - tau: relaxation time

3. Total Force:
   F_total = F_self + sum(F_int)

4. Velocity Update:
   v_new = v_curr + (F_total / m) * dt
   where:
   - m: mass (typically 1.0 for pedestrians)
   - dt: time step
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class UPLParameters:
    """
    Parameters for the Universal Power Law physics model.
    Values based on:
    - Helbing & Molnár (1995): Social Force Model
    - Chraibi et al. (2010): Generalized Centrifugal Force Model
    - Standard pedestrian dynamics literature
    """
    # Interaction force parameters (from Helbing's Social Force Model)
    A: float = 2000.0     # Strength of repulsion (N) - Helbing: 2000-3000N
    B: float = 0.08       # Range of repulsion (m) - Helbing: 0.08m
    r: float = 0.2        # Agent radius (m) - typical: 0.2-0.3m
    
    # Self-propulsion parameters
    v_desired: float = 1.34  # Desired walking speed (m/s) - Weidmann: 1.34 m/s
    tau: float = 0.5         # Relaxation time (s) - Helbing: 0.5s
    
    # Physical parameters
    mass: float = 80.0       # Agent mass (kg) - average adult
    max_speed: float = 2.5   # Maximum speed limit (m/s) - human sprint ~3m/s, comfortable max ~2.5m/s
    
    # Simulation parameters
    dt: float = 0.05         # Time step (s) - smaller for accuracy (Helbing uses 0.01-0.05)
    safety_margin: float = 0.1  # Extra clearance from obstacles (m)


class UPLPhysics:
    """
    Universal Power Law physics engine for pedestrian simulation.
    
    This class implements the force-based movement model used in the VGA paper.
    It calculates interaction forces from obstacles and agents, combines them with
    self-propulsion forces, and updates position/velocity accordingly.
    """
    
    def __init__(self, params: Optional[UPLParameters] = None):
        """
        Initialize the UPL physics engine.
        
        Args:
            params: UPL parameters. If None, uses default values.
        """
        self.params = params or UPLParameters()
        
    def calculate_interaction_force(
        self,
        agent_pos: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacle_radius: float = 0.0
    ) -> np.ndarray:
        """
        Calculate repulsive force from a single obstacle.
        
        Implements: F_int = A * exp((r - d) / B) * n_ij
        
        Args:
            agent_pos: Agent position [x, y]
            obstacle_pos: Obstacle position [x, y]
            obstacle_radius: Radius of the obstacle (0 for point obstacles)
            
        Returns:
            Force vector [fx, fy] in Newtons
        """
        # Vector from obstacle to agent
        diff = agent_pos - obstacle_pos
        distance = np.linalg.norm(diff)
        
        # Avoid division by zero
        if distance < 1e-6:
            return np.zeros(2)
        
        # Unit vector pointing away from obstacle
        n_ij = diff / distance
        
        # Effective distance (accounting for radii)
        d_eff = distance - obstacle_radius - self.params.r
        
        # If already beyond interaction range, no force
        # Helbing: interaction typically within 2-3 radii
        if d_eff > 2.0:  # Interaction cutoff distance
            return np.zeros(2)
        
        # Exponential repulsion (Helbing's Social Force Model)
        # F = A * exp((r - d) / B)
        exponent = (self.params.r - d_eff) / self.params.B
        # Clip exponent to prevent overflow (exp(20) ≈ 500 million)
        exponent = np.clip(exponent, -20, 15)
        magnitude = self.params.A * np.exp(exponent)
        
        return magnitude * n_ij
    
    def calculate_self_propulsion_force(
        self,
        current_velocity: np.ndarray,
        desired_direction: np.ndarray
    ) -> np.ndarray:
        """
        Calculate self-propulsion force toward the goal.
        
        Implements: F_self = (v_des - v_curr) / tau
        
        Args:
            current_velocity: Current velocity vector [vx, vy]
            desired_direction: Unit vector toward goal [dx, dy]
            
        Returns:
            Force vector [fx, fy] in Newtons
        """
        # Desired velocity vector
        v_desired = desired_direction * self.params.v_desired
        
        # Force to reach desired velocity
        force = (v_desired - current_velocity) / self.params.tau
        
        return force * self.params.mass
    
    def calculate_total_force(
        self,
        agent_pos: np.ndarray,
        current_velocity: np.ndarray,
        goal_direction: np.ndarray,
        obstacles: List[Tuple[np.ndarray, float]]
    ) -> np.ndarray:
        """
        Calculate total force acting on the agent.
        
        Args:
            agent_pos: Agent position [x, y]
            current_velocity: Current velocity [vx, vy]
            goal_direction: Unit vector toward current goal
            obstacles: List of (position, radius) tuples for obstacles
            
        Returns:
            Total force vector [fx, fy]
        """
        # Self-propulsion force
        f_self = self.calculate_self_propulsion_force(
            current_velocity, goal_direction
        )
        
        # Interaction forces from all obstacles
        f_int_total = np.zeros(2)
        for obs_pos, obs_radius in obstacles:
            f_int = self.calculate_interaction_force(
                agent_pos, obs_pos, obs_radius
            )
            f_int_total += f_int
        
        # Total force
        return f_self + f_int_total
    
    def update_velocity(
        self,
        current_velocity: np.ndarray,
        total_force: np.ndarray
    ) -> np.ndarray:
        """
        Update velocity based on force.
        
        Implements: v_new = v_curr + (F / m) * dt
        
        Args:
            current_velocity: Current velocity [vx, vy]
            total_force: Total force [fx, fy]
            
        Returns:
            New velocity vector [vx, vy]
        """
        # Acceleration = F / m
        acceleration = total_force / self.params.mass
        
        # Velocity update
        new_velocity = current_velocity + acceleration * self.params.dt
        
        # Enforce speed limit
        speed = np.linalg.norm(new_velocity)
        if speed > self.params.max_speed:
            new_velocity = new_velocity * (self.params.max_speed / speed)
        
        return new_velocity
    
    def step(
        self,
        agent_pos: np.ndarray,
        current_velocity: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: List[Tuple[np.ndarray, float]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform one physics time step.
        
        Args:
            agent_pos: Current position [x, y]
            current_velocity: Current velocity [vx, vy]
            goal_pos: Goal position [x, y]
            obstacles: List of (position, radius) tuples
            
        Returns:
            Tuple of (new_position, new_velocity, force_applied)
        """
        # Goal direction
        goal_diff = goal_pos - agent_pos
        goal_distance = np.linalg.norm(goal_diff)
        
        if goal_distance < 1e-6:
            # Already at goal
            return agent_pos, np.zeros(2), np.zeros(2)
        
        goal_direction = goal_diff / goal_distance
        
        # Calculate total force
        total_force = self.calculate_total_force(
            agent_pos, current_velocity, goal_direction, obstacles
        )
        
        # Update velocity
        new_velocity = self.update_velocity(current_velocity, total_force)
        
        # Update position
        new_pos = agent_pos + new_velocity * self.params.dt
        
        return new_pos, new_velocity, total_force
    
    def simulate_trajectory(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: List[Tuple[np.ndarray, float]],
        max_steps: int = 1000,
        goal_tolerance: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Simulate a complete trajectory from start to goal.
        
        Args:
            start_pos: Starting position [x, y]
            goal_pos: Goal position [x, y]
            obstacles: List of (position, radius) tuples
            max_steps: Maximum simulation steps
            goal_tolerance: Distance to goal considered as "reached"
            
        Returns:
            Tuple of (positions, velocities, success)
            - positions: Array of shape (N, 2) with trajectory points
            - velocities: Array of shape (N, 2) with velocity at each point
            - success: True if goal was reached
        """
        positions = [start_pos.copy()]
        velocities = [np.zeros(2)]
        
        current_pos = start_pos.copy()
        current_vel = np.zeros(2)
        
        for step in range(max_steps):
            # Check if goal reached
            if np.linalg.norm(current_pos - goal_pos) < goal_tolerance:
                return np.array(positions), np.array(velocities), True
            
            # Perform physics step
            new_pos, new_vel, _ = self.step(
                current_pos, current_vel, goal_pos, obstacles
            )
            
            # Store results
            positions.append(new_pos.copy())
            velocities.append(new_vel.copy())
            
            # Update state
            current_pos = new_pos
            current_vel = new_vel
            
            # Check for getting stuck (very low velocity)
            if np.linalg.norm(current_vel) < 0.01 and step > 50:
                # Agent is stuck
                return np.array(positions), np.array(velocities), False
        
        # Max steps reached without reaching goal
        return np.array(positions), np.array(velocities), False


def test_upl_physics():
    """
    Test the UPL physics implementation with a simple scenario.
    """
    print("Testing UPL Physics Implementation...")
    
    # Create physics engine
    upl = UPLPhysics()
    
    # Test scenario: agent moving toward goal with one obstacle
    start = np.array([0.0, 0.0])
    goal = np.array([10.0, 0.0])
    obstacles = [(np.array([5.0, 0.0]), 0.5)]  # Obstacle blocking direct path
    
    # Simulate
    positions, velocities, success = upl.simulate_trajectory(
        start, goal, obstacles, max_steps=500
    )
    
    print(f"Trajectory length: {len(positions)} steps")
    print(f"Goal reached: {success}")
    print(f"Final position: {positions[-1]}")
    print(f"Final distance to goal: {np.linalg.norm(positions[-1] - goal):.3f}m")
    
    # Check that agent avoided the obstacle
    min_dist_to_obstacle = min([
        np.linalg.norm(pos - obstacles[0][0]) 
        for pos in positions
    ])
    print(f"Minimum distance to obstacle: {min_dist_to_obstacle:.3f}m")
    print(f"Collision occurred: {min_dist_to_obstacle < obstacles[0][1] + 0.3}")


if __name__ == "__main__":
    test_upl_physics()
