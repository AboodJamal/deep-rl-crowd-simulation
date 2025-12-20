"""
DRL Policy Interface

Wrapper for the trained PPO agent to use in validation experiments

Author: [Your Name]
Date: December 2025
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch

# TODO: Add path to your core modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))

from model_base import NavigationModel, SimulationResult

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # TODO: Import your custom environment
    # from ultimate_domain_randomization_env import UltimateDomainRandomizedEnv
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Make sure stable-baselines3 and your environment are installed")


class DRLPolicyInterface(NavigationModel):
    """
    Wrapper for trained DRL agent (PPO)

    Usage:
        model = DRLPolicyInterface(
            model_path="models/ultimate_generalized_agent_stage6.zip",
            vecnorm_path="models/ultimate_generalized_agent_stage6_vecnormalize.pkl"
        )
        result = model.simulate(start_pos, goal_pos, obstacles)
    """

    def __init__(
        self,
        model_path: str,
        vecnorm_path: Optional[str] = None,
        env_class=None,  # TODO: Pass your environment class
        device: str = "auto",
        **kwargs,
    ):
        """
        Args:
            model_path: Path to trained PPO model (.zip)
            vecnorm_path: Path to VecNormalize file (.pkl)
            env_class: Environment class to use
            device: Device for inference ('auto', 'cpu', 'cuda')
            **kwargs: Additional configuration
        """
        super().__init__(name="DRL-PPO", **kwargs)

        self.model_path = model_path
        self.vecnorm_path = vecnorm_path
        self.device = device

        # TODO: Initialize your environment class
        self.env_class = env_class
        if self.env_class is None:
            print("Warning: No environment class provided. Using placeholder.")
            # self.env_class = UltimateDomainRandomizedEnv

        # Load model
        self._load_model()

        # Internal state
        self.current_obs = None
        self.env = None
        self.vec_env = None

    def _load_model(self):
        """Load PPO model and VecNormalize"""
        try:
            print(f"Loading DRL model from {self.model_path}")
            self.model = PPO.load(self.model_path, device=self.device)
            print(f"✓ Model loaded successfully")

            if self.vecnorm_path and Path(self.vecnorm_path).exists():
                print(f"Loading VecNormalize from {self.vecnorm_path}")
                # VecNormalize will be applied to env in reset()
                self.vecnorm_loaded = True
            else:
                print("Warning: No VecNormalize file provided or file not found")
                self.vecnorm_loaded = False

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def reset(self, start_pos: np.ndarray, goal_pos: np.ndarray, **kwargs):
        """
        Reset environment for new episode

        Args:
            start_pos: Starting position (x, y)
            goal_pos: Goal position (x, y)
            **kwargs: Additional environment parameters
                - obstacles: List of obstacle positions
                - corridor_type: 'standard', 'lshaped', 'tshaped'
                - density: Obstacle density level
        """
        # TODO: Create environment instance with specific configuration
        if self.env_class is not None:
            env_kwargs = {
                "start_pos": start_pos,
                "goal_pos": goal_pos,
                "obstacles": kwargs.get("obstacles", []),
                "corridor_type": kwargs.get("corridor_type", "standard"),
                **kwargs,
            }

            # Create base environment
            self.env = self.env_class(**env_kwargs)

            # Wrap in VecEnv
            self.vec_env = DummyVecEnv([lambda: self.env])

            # Apply VecNormalize if available
            if self.vecnorm_loaded and self.vecnorm_path:
                self.vec_env = VecNormalize.load(self.vecnorm_path, self.vec_env)
                self.vec_env.training = False
                self.vec_env.norm_reward = False

            # Reset environment
            self.current_obs = self.vec_env.reset()
        else:
            print("Warning: Environment class not set. Cannot reset properly.")
            self.current_obs = None

    def predict_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Predict action from current observation

        Args:
            observation: Dictionary with environment state
                (not used directly - we use self.current_obs from env)

        Returns:
            action: [linear_velocity, angular_velocity]
        """
        if self.current_obs is None:
            print("Warning: No observation available. Did you call reset()?")
            return np.zeros(2)

        # Use model to predict action (deterministic for evaluation)
        action, _ = self.model.predict(self.current_obs, deterministic=True)

        return action[0]  # Extract action from batch

    def step(self, action: np.ndarray, dt: float) -> Dict[str, Any]:
        """
        Execute action in environment

        Args:
            action: Action to execute [linear_vel, angular_vel]
            dt: Time step (not used - environment has fixed dt)

        Returns:
            Dictionary with updated state
        """
        if self.vec_env is None:
            print("Warning: Environment not initialized")
            return {
                "position": np.zeros(2),
                "velocity": np.zeros(2),
                "collision": False,
            }

        # Step environment
        self.current_obs, reward, done, info = self.vec_env.step([action])

        # Extract state from environment
        # TODO: Adjust based on your environment's info dict
        env_state = self.env.get_state() if hasattr(self.env, "get_state") else {}

        return {
            "position": env_state.get("position", np.zeros(2)),
            "velocity": env_state.get("velocity", np.zeros(2)),
            "collision": info[0].get("collision", False),
            "done": done[0],
            "reward": reward[0],
        }

    def simulate(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: list = None,
        corridor_type: str = "standard",
        **kwargs,
    ) -> SimulationResult:
        """
        Run DRL agent simulation

        Args:
            start_pos: Starting position (x, y)
            goal_pos: Goal position (x, y)
            obstacles: List of obstacle positions/info
            corridor_type: Type of corridor geometry
            **kwargs: Additional simulation parameters

        Returns:
            SimulationResult with trajectory
        """
        # Reset with specific environment configuration
        self.reset(
            start_pos,
            goal_pos,
            obstacles=obstacles,
            corridor_type=corridor_type,
            **kwargs,
        )

        # Use base class simulate method
        return super().simulate(
            start_pos=start_pos, goal_pos=goal_pos, obstacles=obstacles, **kwargs
        )


# Example usage and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test DRL Policy Interface")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model .zip file"
    )
    parser.add_argument(
        "--vecnorm", type=str, default=None, help="Path to VecNormalize .pkl file"
    )
    args = parser.parse_args()

    # Create model interface
    print("\n=== Testing DRL Policy Interface ===\n")
    model = DRLPolicyInterface(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
    )

    # Test simulation
    print("\nRunning test simulation...")
    start_pos = np.array([0.0, 5.0])
    goal_pos = np.array([50.0, 5.0])

    result = model.simulate(
        start_pos=start_pos,
        goal_pos=goal_pos,
        max_steps=500,
        dt=0.1,
    )

    print(f"\nResults:")
    print(f"  Success: {result.success}")
    print(f"  Travel time: {result.travel_time:.2f}s")
    print(f"  Path length: {result.path_length:.2f}m")
    print(f"  Collisions: {result.num_collisions}")
    print(f"  Steps: {len(result.positions)}")
