"""
VGA Dataset Trainer - Train DRL on VGA Experimental Data
=========================================================

This script trains a DRL agent on the EXACT VGA experimental setup.
The environment matches the real experiment:
- 10m x 3.5m arena
- Fixed obstacle positions per scenario
- Real trial start/goal positions

Data Split:
- Train: 70% of trials per scenario
- Validation: 15% of trials
- Test: 15% of trials

Curriculum:
1. Easy scenarios (SOSP, MOSP_A) first
2. Medium scenarios (MOSP_C)
3. Hard scenarios (MOSP_B, MOSP_D) - dense obstacles
4. Mixed training across all scenarios
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

os.environ["WANDB_SYMLINK"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Numpy compatibility fix
try:
    from numpy_compat_fix import *
except ImportError:
    pass

import numpy as np
from datetime import datetime
import json
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from vga_experimental_env import VGAExperimentalEnv, VGAMixedEnv


# Import advanced policy from core
try:
    from advanced_policy_network import AdvancedActorCriticPolicy

    USE_ADVANCED_POLICY = True
except ImportError:
    print("[WARN] Could not import AdvancedActorCriticPolicy, using default MlpPolicy")
    USE_ADVANCED_POLICY = False


class VGATrainingCallback(BaseCallback):
    """Callback for tracking VGA training progress."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_successes = []
        self.scenario_stats = {}

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None and len(dones) > 0:
            for idx, done in enumerate(dones):
                if done:
                    self.episode_count += 1
                    infos = self.locals.get("infos", [])
                    if len(infos) > idx:
                        info = infos[idx]

                        success = info.get("goal_reached", False)
                        scenario = info.get("scenario", "unknown")

                        self.episode_successes.append(1.0 if success else 0.0)

                        # Track per-scenario stats
                        if scenario not in self.scenario_stats:
                            self.scenario_stats[scenario] = {"successes": 0, "total": 0}

                        self.scenario_stats[scenario]["total"] += 1
                        if success:
                            self.scenario_stats[scenario]["successes"] += 1

                        # Log to W&B
                        try:
                            wandb.log(
                                {
                                    "vga/episode": self.episode_count,
                                    "vga/success": success,
                                    "vga/scenario": scenario,
                                    "vga/collisions": info.get("collisions", 0),
                                    "vga/steps": info.get("steps", 0),
                                    "vga/distance_traveled": info.get(
                                        "distance_traveled", 0
                                    ),
                                },
                                step=self.num_timesteps,
                            )
                        except:
                            pass

        return True

    def get_success_rate(self, last_n: int = 100) -> float:
        """Get recent success rate."""
        if not self.episode_successes:
            return 0.0
        return np.mean(self.episode_successes[-last_n:]) * 100


class ProgressCallback(BaseCallback):
    """Print progress during training."""

    def __init__(self, check_freq: int = 5000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.successes = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None and len(dones) > 0:
            for idx, done in enumerate(dones):
                if done:
                    infos = self.locals.get("infos", [])
                    if len(infos) > idx:
                        self.successes.append(
                            1.0 if infos[idx].get("goal_reached", False) else 0.0
                        )

        if self.n_calls % self.check_freq == 0 and self.successes:
            rate = np.mean(self.successes[-100:]) * 100
            print(
                f"\n[Progress] {self.num_timesteps:,} steps | Success Rate: {rate:.1f}%\n"
            )

        return True


def create_vga_env(
    scenarios: list,
    split_type: str = "train",
    data_root: str = r"D:\Abdullah Jamal\downloads\VGA-exp\Pedestrian-Experimental-Data",
    n_envs: int = 4,
    randomize: bool = True,
):
    """Create vectorized VGA environment."""

    def make_env():
        if len(scenarios) == 1:
            # Single scenario
            env = VGAExperimentalEnv(
                scenario=scenarios[0],
                data_root=data_root,
                randomize_start_goal=randomize,
            )
        else:
            # Multiple scenarios
            env = VGAMixedEnv(
                scenarios=scenarios,
                data_root=data_root,
                split_type=split_type,
                randomize_start_goal=randomize,
            )
        return Monitor(env)

    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecNormalize(
        env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0
    )

    return env


def train_vga_curriculum(
    total_timesteps: int = 2000000,
    save_dir: str = "vga_training/models",
    data_root: str = r"D:\Abdullah Jamal\downloads\VGA-exp\Pedestrian-Experimental-Data",
    wandb_project: str = "vga-drl-training",
    wandb_run_name: str = None,
    n_envs: int = 4,
    resume_from: str = None,
):
    """
    Train DRL agent on VGA experimental data with curriculum.

    Curriculum Stages:
    1. SOSP (easiest - 1 obstacle)
    2. MOSP_A (4 obstacles, spread out)
    3. MOSP_C (12 obstacles, moderate density)
    4. MOSP_B + MOSP_D (hardest - tight spacing)
    5. ALL scenarios mixed
    """

    print("=" * 80)
    print("VGA EXPERIMENTAL DATA DRL TRAINING")
    print("=" * 80)
    print(f"Environment: 10m x 3.5m arena (EXACT VGA match)")
    print(f"Agent radius: 0.2m | Obstacle radius: 0.25m")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80 + "\n")

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    # Initialize W&B
    run_name = wandb_run_name or f"vga_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "algorithm": "PPO",
            "total_timesteps": total_timesteps,
            "environment": "VGA_Experimental",
            "arena_size": "10m x 3.5m",
            "agent_radius": 0.2,
            "obstacle_radius": 0.25,
            "n_envs": n_envs,
        },
        sync_tensorboard=True,
    )

    print(f"[OK] W&B initialized: {wandb.run.name}")
    print(f"  Dashboard: {wandb.run.url}\n")

    # Curriculum stages
    stages = [
        {
            "name": "Stage 1: SOSP (1 obstacle)",
            "scenarios": ["SOSP"],
            "timesteps": int(total_timesteps * 0.10),  # 10%
            "desc": "Learn basic navigation with single obstacle",
        },
        {
            "name": "Stage 2: MOSP_A (4 obstacles)",
            "scenarios": ["MOSP_A"],
            "timesteps": int(total_timesteps * 0.15),  # 15%
            "desc": "Learn navigation with sparse obstacles",
        },
        {
            "name": "Stage 3: MOSP_C (12 obstacles)",
            "scenarios": ["MOSP_C"],
            "timesteps": int(total_timesteps * 0.15),  # 15%
            "desc": "Learn navigation with medium density",
        },
        {
            "name": "Stage 4: MOSP_B (7 tight obstacles)",
            "scenarios": ["MOSP_B"],
            "timesteps": int(total_timesteps * 0.15),  # 15%
            "desc": "Learn navigation with tight obstacle spacing",
        },
        {
            "name": "Stage 5: MOSP_D (16 obstacles)",
            "scenarios": ["MOSP_D"],
            "timesteps": int(total_timesteps * 0.15),  # 15%
            "desc": "Learn navigation with high obstacle density",
        },
        {
            "name": "Stage 6: ALL Mixed",
            "scenarios": ["SOSP", "MOSP_A", "MOSP_B", "MOSP_C", "MOSP_D"],
            "timesteps": int(total_timesteps * 0.30),  # 30%
            "desc": "Generalize across all scenarios",
        },
    ]

    model = None
    all_stats = []

    # Check for resume
    if resume_from and os.path.exists(resume_from):
        print(f"[OK] Resuming from: {resume_from}")
        # We'll load model after creating first env

    for stage_idx, stage in enumerate(stages):
        print(f"\n{'=' * 80}")
        print(f"STAGE {stage_idx + 1}/{len(stages)}: {stage['name']}")
        print(f"{'=' * 80}")
        print(f"Scenarios: {', '.join(stage['scenarios'])}")
        print(f"Timesteps: {stage['timesteps']:,}")
        print(f"Description: {stage['desc']}")
        print(f"{'=' * 80}\n")

        # Create environment for this stage
        env = create_vga_env(
            scenarios=stage["scenarios"],
            split_type="train",
            data_root=data_root,
            n_envs=n_envs,
            randomize=True,
        )

        # Create or update model
        if model is None:
            # Check if resuming
            if resume_from and os.path.exists(resume_from):
                print(f"[OK] Loading model from: {resume_from}")
                model = PPO.load(
                    resume_from,
                    env=env,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            else:
                # Create new model
                print("[OK] Creating new PPO model...")

                policy_class = (
                    AdvancedActorCriticPolicy if USE_ADVANCED_POLICY else "MlpPolicy"
                )

                model = PPO(
                    policy_class,
                    env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=256,
                    n_epochs=10,
                    gamma=0.995,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.02,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    verbose=1,
                    tensorboard_log=f"{save_dir}/logs/{run_name}",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

                print(
                    f"  Policy: {policy_class if isinstance(policy_class, str) else 'AdvancedActorCriticPolicy'}"
                )
                print(f"  Device: {model.device}")
        else:
            print("[OK] Continuing from previous stage")
            model.set_env(env)

        # Setup callbacks
        vga_callback = VGATrainingCallback()
        progress_callback = ProgressCallback(check_freq=10000)
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=f"{save_dir}/checkpoints/",
            name_prefix=f"vga_stage{stage_idx + 1}",
        )
        wandb_callback = WandbCallback(
            model_save_freq=0, gradient_save_freq=0, verbose=2
        )

        # Train
        print(f"[TRAINING] Starting stage {stage_idx + 1}...")
        start_time = datetime.now()

        try:
            model.learn(
                total_timesteps=stage["timesteps"],
                callback=[
                    vga_callback,
                    progress_callback,
                    checkpoint_callback,
                    wandb_callback,
                ],
                progress_bar=True,
                reset_num_timesteps=False,
            )
        except KeyboardInterrupt:
            print("\n[WARN] Training interrupted by user")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Stage statistics
        stage_stats = {
            "stage": stage["name"],
            "scenarios": stage["scenarios"],
            "timesteps": stage["timesteps"],
            "duration_seconds": duration,
            "success_rate": vga_callback.get_success_rate(),
            "episodes": vga_callback.episode_count,
            "scenario_breakdown": {
                k: {
                    "success_rate": (
                        (v["successes"] / v["total"] * 100) if v["total"] > 0 else 0
                    ),
                    "episodes": v["total"],
                }
                for k, v in vga_callback.scenario_stats.items()
            },
        }
        all_stats.append(stage_stats)

        # Print stage summary
        print(f"\n{'=' * 80}")
        print(f"STAGE {stage_idx + 1} COMPLETE")
        print(f"{'=' * 80}")
        print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"Episodes: {stage_stats['episodes']}")
        print(f"Overall Success Rate: {stage_stats['success_rate']:.1f}%")
        print(f"\nPer-Scenario Breakdown:")
        for scenario, stats in stage_stats["scenario_breakdown"].items():
            print(
                f"  {scenario:10s}: {stats['success_rate']:5.1f}% ({stats['episodes']} episodes)"
            )
        print(f"{'=' * 80}\n")

        # Save stage model
        stage_model_path = f"{save_dir}/vga_drl_stage{stage_idx + 1}.zip"
        model.save(stage_model_path)
        env.save(stage_model_path.replace(".zip", "_vecnormalize.pkl"))
        print(f"[OK] Stage model saved: {stage_model_path}")

        env.close()

    # Save final model
    final_model_path = f"{save_dir}/vga_drl_final.zip"
    model.save(final_model_path)
    print(f"\n[OK] Final model saved: {final_model_path}")

    # Save training summary
    summary = {
        "total_timesteps": total_timesteps,
        "stages": all_stats,
        "final_model": final_model_path,
        "environment": {
            "arena_size": "10m x 3.5m",
            "agent_radius": 0.2,
            "obstacle_radius": 0.25,
            "dt": 0.05,
        },
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = f"{save_dir}/training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Training summary saved: {summary_path}")

    # Print final summary
    print(f"\n{'=' * 80}")
    print("VGA DRL TRAINING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"\nFinal model: {final_model_path}")
    print(f"\nStage Summary:")
    print("-" * 80)
    for stage in all_stats:
        scenarios_str = ", ".join(stage["scenarios"])
        print(
            f"{stage['stage']:35s} | {stage['timesteps']:8,} steps | "
            f"Success: {stage['success_rate']:5.1f}%"
        )
    print("-" * 80)

    wandb.finish()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train DRL on VGA Experimental Data")
    parser.add_argument(
        "--timesteps", type=int, default=2000000, help="Total training timesteps"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="vga_training/models",
        help="Model save directory",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"D:\Abdullah Jamal\downloads\VGA-exp\Pedestrian-Experimental-Data",
        help="Path to VGA dataset",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="vga-drl-training", help="W&B project name"
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument(
        "--n-envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to model to resume from"
    )

    args = parser.parse_args()

    train_vga_curriculum(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        data_root=args.data_root,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        n_envs=args.n_envs,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
