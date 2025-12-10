"""
Abstract Base Class for Navigation Models

Defines common interface for DRL, VGA, and classical models

Author: [Your Name]
Date: December 2025
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """Container for simulation trajectory and metadata"""
    positions: np.ndarray  # (N, 2) - trajectory positions
    timestamps: np.ndarray  # (N,) - time points
    velocities: np.ndarray  # (N, 2) - velocities at each point
    actions: Optional[np.ndarray] = None  # (N, action_dim) - actions taken
    
    success: bool = False  # Whether goal was reached
    collision: bool = False  # Whether collision occurred
    stuck: bool = False  # Whether agent got stuck
    
    # Metrics
    travel_time: float = 0.0
    path_length: float = 0.0
    num_collisions: int = 0
    
    # Additional info
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Compute derived metrics"""
        if self.metadata is None:
            self.metadata = {}
        
        # Compute travel time
        if len(self.timestamps) > 0:
            self.travel_time = self.timestamps[-1] - self.timestamps[0]
        
        # Compute path length
        if len(self.positions) > 1:
            diffs = np.diff(self.positions, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            self.path_length = np.sum(distances)


class NavigationModel(ABC):
    """
    Abstract base class for all navigation models
    
    All models (DRL, VGA, Social Force) should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Args:
            name: Model identifier
            **kwargs: Model-specific configuration
        """
        self.name = name
        self.config = kwargs
    
    @abstractmethod
    def reset(self, start_pos: np.ndarray, goal_pos: np.ndarray, **kwargs):
        """
        Reset model for new episode
        
        Args:
            start_pos: Starting position (x, y)
            goal_pos: Goal position (x, y)
            **kwargs: Additional environment state (obstacles, etc.)
        """
        pass
    
    @abstractmethod
    def predict_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Predict next action given current observation
        
        Args:
            observation: Dictionary with:
                - 'position': current (x, y)
                - 'velocity': current (vx, vy)
                - 'goal': goal (x, y)
                - 'obstacles': list of obstacle info
                - other model-specific observations
                
        Returns:
            action: Action array (format depends on model)
                - DRL: [linear_vel, angular_vel]
                - VGA: [target_x, target_y] or [vx, vy]
                - Social Force: [fx, fy] forces
        """
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray, dt: float) -> Dict[str, Any]:
        """
        Update internal state after taking action
        
        Args:
            action: Action taken
            dt: Time step
            
        Returns:
            Dictionary with updated state
        """
        pass
    
    def simulate(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: list = None,
        max_steps: int = 1000,
        dt: float = 0.1,
        goal_threshold: float = 1.0,
        **kwargs
    ) -> SimulationResult:
        """
        Run full simulation from start to goal
        
        Args:
            start_pos: Starting position (x, y)
            goal_pos: Goal position (x, y)
            obstacles: List of obstacles (model-specific format)
            max_steps: Maximum simulation steps
            dt: Time step size
            goal_threshold: Distance threshold to consider goal reached
            **kwargs: Additional simulation parameters
            
        Returns:
            SimulationResult with full trajectory
        """
        # Initialize
        self.reset(start_pos, goal_pos, obstacles=obstacles, **kwargs)
        
        # Storage for trajectory
        positions = [start_pos.copy()]
        velocities = [np.zeros(2)]
        actions_taken = []
        timestamps = [0.0]
        
        current_pos = start_pos.copy()
        current_vel = np.zeros(2)
        t = 0.0
        
        success = False
        collision = False
        stuck = False
        num_collisions = 0
        
        for step in range(max_steps):
            # Create observation
            observation = {
                'position': current_pos,
                'velocity': current_vel,
                'goal': goal_pos,
                'obstacles': obstacles,
                'time': t,
            }
            
            # Predict action
            action = self.predict_action(observation)
            actions_taken.append(action)
            
            # Update state
            state = self.step(action, dt)
            current_pos = state['position']
            current_vel = state['velocity']
            
            # Check for collision (model-specific)
            if state.get('collision', False):
                num_collisions += 1
                if not collision:  # First collision
                    collision = True
            
            # Store trajectory
            positions.append(current_pos.copy())
            velocities.append(current_vel.copy())
            t += dt
            timestamps.append(t)
            
            # Check goal reached
            dist_to_goal = np.linalg.norm(current_pos - goal_pos)
            if dist_to_goal < goal_threshold:
                success = True
                break
            
            # Check if stuck (very low velocity for extended period)
            if step > 50:
                recent_speeds = [np.linalg.norm(v) for v in velocities[-50:]]
                if np.mean(recent_speeds) < 0.01:
                    stuck = True
                    break
        
        # Create result
        result = SimulationResult(
            positions=np.array(positions),
            timestamps=np.array(timestamps),
            velocities=np.array(velocities),
            actions=np.array(actions_taken) if actions_taken else None,
            success=success,
            collision=collision,
            stuck=stuck,
            num_collisions=num_collisions,
            metadata={
                'model': self.name,
                'num_steps': len(positions),
                'goal_threshold': goal_threshold,
                'dt': dt,
            }
        )
        
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'name': self.name,
            **self.config
        }
