"""
ULTIMATE Curriculum Learning Trainer
Trains agent on ALL possible corridor types for maximum generalization:
- Standard corridors (varied density)
- L-shaped corridors
- T-shaped corridors
- U-shaped corridors (NEW)
- Multi-room corridors (NEW)
- Narrow passages, zigzag paths, clusters
- Domain randomization within each type

This creates an EXTREMELY robust, generalized agent.
"""

import argparse
import os
os.environ["WANDB_SYMLINK"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Apply numpy compatibility fix before importing stable_baselines3
try:
    import numpy_compat_fix
except ImportError:
    pass

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import wandb
from wandb.integration.sb3 import WandbCallback
import torch
import json
from collections import deque

from ultimate_domain_randomization_env import UltimateDomainRandomizedEnv
from advanced_policy_network import AdvancedActorCriticPolicy


class UltimateCurriculumCallback(BaseCallback):
    """Track ultimate curriculum progress."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_successes = []
        self.env_type_stats = {}
        
    def _on_step(self) -> bool:
        # Handle vectorized environments (dones is an array)
        dones = self.locals.get("dones")
        if dones is not None and len(dones) > 0:
            for idx, done in enumerate(dones):
                if done:
                    self.episode_count += 1
                    infos = self.locals.get("infos", [])
                    if len(infos) > idx:
                        info = infos[idx]
                        
                        success = info.get("goal_reached", False)
                        env_type = info.get("corridor_type", "unknown")
                        
                        self.episode_successes.append(1.0 if success else 0.0)
                        
                        # Track per-environment-type stats
                        if env_type not in self.env_type_stats:
                            self.env_type_stats[env_type] = {"successes": 0, "total": 0}
                        
                        self.env_type_stats[env_type]["total"] += 1
                        if success:
                            self.env_type_stats[env_type]["successes"] += 1
                        
                        # Log to W&B
                        wandb.log({
                            "curriculum/episode": self.episode_count,
                            "curriculum/success": success,
                            "curriculum/corridor_type": hash(env_type),
                            "curriculum/collisions": info.get("collisions", 0),
                            "curriculum/time": info.get("time_elapsed", 0),
                            "curriculum/obstacle_density": info.get("obstacle_density", 0),
                        }, step=self.num_timesteps)
        
        return True


class ProgressLogCallback(BaseCallback):
    """Enhanced progress logging."""
    
    def __init__(self, check_freq: int = 10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_successes = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            success_rate = np.mean(self.episode_successes[-100:]) * 100 if self.episode_successes else 0
            
            print(f"\n{'='*70}")
            print(f"Progress: {self.num_timesteps:,} steps")
            print(f"Recent Success Rate: {success_rate:.1f}%")
            print(f"{'='*70}\n")
        
        # Handle vectorized environments (dones is an array)
        dones = self.locals.get("dones")
        if dones is not None and len(dones) > 0:
            for idx, done in enumerate(dones):
                if done:
                    infos = self.locals.get("infos", [])
                    if len(infos) > idx:
                        info = infos[idx]
                        self.episode_successes.append(1.0 if info.get("goal_reached", False) else 0.0)
        
        return True


class AdaptiveHyperparameters:
    """
    RESEARCH-BASED: Performance-based hyperparameter adaptation (HOOF-inspired)
    Based on 2024-2025 research: Hyperparameters should adapt based on training dynamics.
    
    Adjusts:
    - Learning rate (based on loss trends)
    - Entropy coefficient (based on success rate)
    - Clip range (based on policy stability)
    """
    
    def __init__(
        self,
        initial_lr: float = 3e-4,
        initial_entropy: float = 0.02,
        initial_clip: float = 0.2,
    ):
        self.lr = initial_lr
        self.entropy = initial_entropy
        self.clip_range = initial_clip
        
        # Performance tracking
        self.success_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=50)
        
        # Adaptation parameters
        self.lr_min = 1e-5
        self.lr_max = 5e-4
        self.entropy_min = 0.001
        self.entropy_max = 0.05
        self.clip_min = 0.1
        self.clip_max = 0.3
        
    def update(self, success_rate: float, mean_reward: float, policy_loss: float = None):
        """Update hyperparameters based on recent performance."""
        self.success_history.append(success_rate)
        self.reward_history.append(mean_reward)
        if policy_loss is not None:
            self.loss_history.append(policy_loss)
        
        # Need enough data before adapting
        if len(self.success_history) < 20:
            return
        
        # Calculate performance metrics
        recent_success = np.mean(list(self.success_history)[-20:])
        success_trend = np.polyfit(range(len(self.success_history)), list(self.success_history), 1)[0]
        reward_variance = np.var(list(self.reward_history))
        
        # ENTROPY ADAPTATION: High entropy if stuck, low if succeeding
        if recent_success < 0.3:  # Low success -> increase exploration
            self.entropy = min(self.entropy * 1.05, self.entropy_max)
        elif recent_success > 0.7:  # High success -> decrease exploration
            self.entropy = max(self.entropy * 0.98, self.entropy_min)
        
        # LEARNING RATE ADAPTATION: Decrease if unstable, increase if plateaued
        if reward_variance > 1000:  # High variance -> decrease LR
            self.lr = max(self.lr * 0.95, self.lr_min)
        elif abs(success_trend) < 0.001 and recent_success < 0.8:  # Plateaued but not optimal -> increase LR
            self.lr = min(self.lr * 1.02, self.lr_max)
        elif success_trend < -0.01:  # Getting worse -> decrease LR
            self.lr = max(self.lr * 0.9, self.lr_min)
        
        # CLIP RANGE ADAPTATION: Tighter if stable, looser if exploring
        if recent_success > 0.7:  # Stable -> tighter clip
            self.clip_range = max(self.clip_range * 0.99, self.clip_min)
        elif recent_success < 0.3:  # Exploring -> looser clip
            self.clip_range = min(self.clip_range * 1.02, self.clip_max)
    
    def get_hyperparameters(self) -> dict:
        """Get current hyperparameter values."""
        return {
            "learning_rate": self.lr,
            "ent_coef": self.entropy,
            "clip_range": self.clip_range,
        }


class AdaptiveCallback(BaseCallback):
    """Callback to apply adaptive hyperparameters during training."""
    
    def __init__(self, adaptive_hp: AdaptiveHyperparameters, update_freq: int = 5000):
        super().__init__()
        self.adaptive_hp = adaptive_hp
        self.update_freq = update_freq
        self.episode_successes = []
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Collect episode data (handle vectorized environments)
        dones = self.locals.get("dones")
        if dones is not None and len(dones) > 0:
            for idx, done in enumerate(dones):
                if done:
                    infos = self.locals.get("infos", [])
                    if len(infos) > idx:
                        info = infos[idx]
                        self.episode_successes.append(1.0 if info.get("goal_reached", False) else 0.0)
                        self.episode_rewards.append(info.get("episode_reward", 0.0))
        
        # Update hyperparameters periodically
        if self.n_calls % self.update_freq == 0 and len(self.episode_successes) >= 10:
            success_rate = np.mean(self.episode_successes[-50:])
            mean_reward = np.mean(self.episode_rewards[-50:])
            
            # Update adaptive hyperparameters
            self.adaptive_hp.update(success_rate, mean_reward)
            
            # Apply to model (PPO allows runtime updates for some params)
            new_hp = self.adaptive_hp.get_hyperparameters()
            
            # Update learning rate
            if hasattr(self.model, "learning_rate"):
                self.model.learning_rate = new_hp["learning_rate"]
            
            # Log adaptation
            wandb.log({
                "adaptive/learning_rate": new_hp["learning_rate"],
                "adaptive/entropy_coef": new_hp["ent_coef"],
                "adaptive/clip_range": new_hp["clip_range"],
                "adaptive/success_rate": success_rate,
            }, step=self.num_timesteps)
            
            if self.verbose > 0 and self.n_calls % (self.update_freq * 4) == 0:
                print(f"\n[Adaptive HPO] LR: {new_hp['learning_rate']:.6f}, "
                      f"Entropy: {new_hp['ent_coef']:.4f}, "
                      f"Success: {success_rate:.2%}")
        
        return True

def train_ultimate_curriculum(
    total_timesteps: int = 1000000,  # More training for more variety!
    save_path: str = "models/ultimate_generalized_agent.zip",
    wandb_project: str = "ultimate-pedestrian-nav",
    wandb_run_name: str = None,
):
    """
    Train agent with ultimate curriculum covering ALL corridor types.
    
    Stages:
    1. Standard Easy (100k) - Basic navigation
    2. Standard Medium (150k) - Moderate complexity
    3. Shaped Corridors Easy (150k) - L/T/U shapes with sparse obstacles
    4. Shaped Corridors Hard (200k) - L/T/U shapes with dense obstacles
    5. All Mixed Hard (200k) - Everything together, maximum difficulty
    6. Ultra Challenge (200k) - Multi-room, extreme scenarios
    """

    # ------------------------------------------------------------------
    # 🟢 QUICK MODE: Skip training if timesteps == 0, just show summary
    # ------------------------------------------------------------------
    if total_timesteps <= 0:
        summary_path = "curriculum_logs/ultimate_training_summary.json"
        if os.path.exists(summary_path):
            print(f"\n🟢 Skipping training (0 timesteps). Showing previous summary from {summary_path}\n")
            with open(summary_path, "r") as f:
                summary = json.load(f)

            print("\nTRAINING SUMMARY:")
            print("-" * 100)
            for stage in summary.get("stages", []):
                name = stage.get("stage", "Unknown")
                diff = stage.get("config", {}).get("difficulty", "N/A")
                shapes = ", ".join(stage.get("config", {}).get("shapes", []))
                print(f"{name:18s} | {diff:8s} | "
                      f"{stage.get('timesteps', 0):8,} steps | Success: {stage.get('success_rate', 0):5.1f}% | Shapes: {shapes}")
            print("-" * 100 + "\n")
            return None
        else:
            print("⚠ No summary file found (curriculum_logs/ultimate_training_summary.json). Nothing to show.")
            return None
    # ------------------------------------------------------------------

    print("="*80)
    print("ULTIMATE CURRICULUM LEARNING - ALL CORRIDOR TYPES")
    print("="*80)
    print(f"Total training: {total_timesteps:,} steps")
    print(f"Strategy: Progressive difficulty + shape variety")
    print(f"Model: {save_path}")
    print("="*80 + "\n")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("curriculum_logs", exist_ok=True)
    
    # Initialize W&B
    config = {
        "algorithm": "PPO",
        "total_timesteps": total_timesteps,
        "training_strategy": "ultimate_curriculum_all_shapes",
        "corridor_types": ["standard", "lshaped", "tshaped", "ushaped", "multiroom"],
    }
    
    run_name = wandb_run_name or f"ultimate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=wandb_project, name=run_name, config=config, sync_tensorboard=True)
    
    print(f"✓ W&B initialized: {wandb.run.name}")
    print(f"  Dashboard: {wandb.run.url}\n")
    
    # RESEARCH-BASED Curriculum: Zone of Proximal Development approach
    # Based on 2024-2025 research: tasks should be just beyond current capability
    # Much easier start (0.001 density) with gradual progression
    stages = [
        # PHASE 1: SUPER EASY (Build confidence)
        {"name": "Super Easy Standard", "config": {"difficulty": "super_easy", "shapes": ["standard"]}, "timesteps": 300000, "desc": "Ultra-sparse obstacles (0.001-0.005) - learn basic goal-seeking"},
        
        # PHASE 2: EASY (Reinforce fundamentals)
        {"name": "Standard Sparse", "config": {"difficulty": "easy", "shapes": ["standard"]}, "timesteps": 300000, "desc": "Sparse obstacles (0.005-0.015) - reinforce goal-reaching"},
        {"name": "Standard Dense Easy", "config": {"difficulty": "easy", "shapes": ["standard"]}, "timesteps": 300000, "desc": "More obstacles but still easy - build confidence"},
        
        # PHASE 3: MEDIUM (Add complexity gradually)
        {"name": "Standard Medium", "config": {"difficulty": "medium", "shapes": ["standard"]}, "timesteps": 300000, "desc": "Medium difficulty (0.015-0.04) - gradual complexity increase"},
        {"name": "Standard Hard", "config": {"difficulty": "hard", "shapes": ["standard"]}, "timesteps": 300000, "desc": "Hard standard (0.04-0.08) - master before shapes"},
        
        # PHASE 4: L/T SHAPES (Add turns)
        {"name": "L/T Super Easy", "config": {"difficulty": "super_easy", "shapes": ["lshaped", "tshaped"]}, "timesteps": 300000, "desc": "L/T with ultra-sparse obstacles - learn corner navigation"},
        {"name": "L/T Easy", "config": {"difficulty": "easy", "shapes": ["lshaped", "tshaped"]}, "timesteps": 300000, "desc": "L/T with sparse obstacles - reinforce corner navigation"},
        {"name": "L/T Medium", "config": {"difficulty": "medium", "shapes": ["lshaped", "tshaped"]}, "timesteps": 300000, "desc": "L/T with moderate obstacles"},
        {"name": "L/T Hard", "config": {"difficulty": "hard", "shapes": ["lshaped", "tshaped"]}, "timesteps": 300000, "desc": "L/T with dense obstacles"},
        
        # PHASE 5: GENERALIZATION (Mix everything)
        {"name": "All Mixed (Std+L/T)", "config": {"difficulty": "mixed", "shapes": ["standard", "lshaped", "tshaped"]}, "timesteps": 600000, "desc": "Generalize across all shapes and difficulties"},
        
        # PHASE 6: PATTERNS & ULTRA (Final challenges)
        {"name": "Pattern Navigation", "config": {"difficulty": "medium", "shapes": ["standard"], "patterns": ["narrow", "zigzag", "clustered"]}, "timesteps": 300000, "desc": "Learn specific obstacle patterns"},
        {"name": "Ultra Challenge", "config": {"difficulty": "ultra", "shapes": ["standard", "lshaped", "tshaped"]}, "timesteps": 300000, "desc": "Ultra-hard scenarios (0.06-0.12 density)"},
    ]
    
    # Check if we should resume from a specific stage
    # Find the most recent stage checkpoint (by modification time, not just highest number)
    resume_from_stage = None
    latest_stage_time = 0
    latest_stage_num = 0
    
    for i in range(len(stages) - 1, -1, -1):  # Check from last to first
        stage_model_path = f"models/ultimate_generalized_agent_stage{i+1}.zip"
        if os.path.exists(stage_model_path):
            # Get modification time to find the most recent checkpoint
            stage_time = os.path.getmtime(stage_model_path)
            if stage_time > latest_stage_time:
                latest_stage_time = stage_time
                latest_stage_num = i + 1
    
    if latest_stage_num > 0:
        resume_from_stage = latest_stage_num
        stage_model_path = f"models/ultimate_generalized_agent_stage{resume_from_stage}.zip"
        print(f"✓ Found most recent checkpoint at stage {resume_from_stage}: {stage_model_path}")
        print(f"  Modified: {datetime.fromtimestamp(latest_stage_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Will resume from stage {resume_from_stage + 1} (skipping stages 1-{resume_from_stage})\n")
    
    model = None
    all_stage_stats = []
    
    vec_norm_final = None  # Will hold final stage's VecNormalize
    
    for stage_idx, stage in enumerate(stages):
        # Skip stages if resuming
        if resume_from_stage and (stage_idx + 1) <= resume_from_stage:
            print(f"\n⏭️  Skipping stage {stage_idx + 1} (already completed)")
            # Load the model from the checkpoint
            if stage_idx + 1 == resume_from_stage:
                stage_model_path = f"models/ultimate_generalized_agent_stage{resume_from_stage}.zip"
                print(f"  Loading model from: {stage_model_path}")
                # Create a dummy env to load the model
                try:
                    dummy_env = DummyVecEnv([lambda: Monitor(UltimateDomainRandomizedEnv())])
                    model = PPO.load(stage_model_path, env=dummy_env, print_system_info=False)
                    dummy_env.close()
                    print(f"✓ Loaded model from stage {resume_from_stage}")
                except Exception as e:
                    print(f"⚠ Warning: Could not load model from {stage_model_path}: {e}")
                    print("  Starting fresh training instead...")
                    resume_from_stage = None  # Reset to start fresh
                    model = None
            continue
        print(f"\n{'='*80}")
        print(f"STAGE {stage_idx + 1}/{len(stages)}: {stage['name'].upper()}")
        print(f"{'='*80}")
        print(f"Configuration: {stage['config']}")
        print(f"Duration: {stage['timesteps']:,} steps")
        print(f"Description: {stage['desc']}")
        print(f"{'='*80}\n")
        
        # Create environment for this stage
        def make_env():
            env = UltimateDomainRandomizedEnv(
                difficulty_level=stage['config']['difficulty'],
                allowed_shapes=stage['config']['shapes']
            )
            # If pattern-specific training, increase pattern probability
            if 'patterns' in stage['config']:
                # Force higher pattern probability for pattern training
                env.difficulty_ranges[env.difficulty_level]['pattern_prob'] = 0.8  # 80% pattern probability
            env = Monitor(env)
            return env
        
        # TIER 1 UPGRADE: Use multiple environments for better exploration
        # Note: On Windows, SubprocVecEnv has multiprocessing issues, so we use DummyVecEnv
        # DummyVecEnv still benefits from vectorization (batch processing) even if not parallel
        n_envs = 4  # Reduced from 8 due to Windows limitations
        env = DummyVecEnv([make_env for _ in range(n_envs)])
        print(f"[TIER 1] Using {n_envs} vectorized environments (Windows-compatible)")
        
        # VecNormalize: normalize observations and rewards for stable learning
        # clip_reward=50.0 allows large success bonuses (now 50.0 after rescaling) to be properly normalized
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0)
        
        # Keep reference to VecNormalize for final save (if this is last stage)
        if stage_idx == len(stages) - 1:
            vec_norm_final = env
        
        # RESEARCH-BASED: Adaptive hyperparameters based on stage
        # Based on 2024-2025 research: hyperparameters should adapt during training
        stage_num = stage_idx + 1
        total_stages = len(stages)
        
        # Adaptive learning rate: higher early, lower later (schedule-based)
        base_lr = 3.0e-4
        lr_decay = 0.995 ** stage_num  # Gradual decay
        learning_rate = base_lr * lr_decay
        
        # Adaptive entropy: higher early (exploration), lower later (exploitation)
        base_entropy = 0.02
        entropy_decay = 0.997 ** stage_num  # Gradual decay
        entropy_coef = max(0.005, base_entropy * entropy_decay)  # Don't go below 0.005
        
        # TIER 1 UPGRADE: Fixed batch size for better LSTM sequence integrity
        # Research shows: LSTMs need consistent sequence lengths for temporal learning
        # Smaller batch sizes with shorter rollouts preserve LSTM context better
        batch_size = 256  # Fixed for optimal LSTM performance
        
        # RESEARCH-BASED: Initialize adaptive hyperparameters
        adaptive_hp = AdaptiveHyperparameters(
            initial_lr=learning_rate,
            initial_entropy=entropy_coef,
            initial_clip=0.2,
        )
        
        # Create or continue model
        if model is None:
            print(f"[OK] Creating ADVANCED 2025 POLICY (CNN+Attention+LSTM)")
            model = PPO(
                AdvancedActorCriticPolicy,  # RESEARCH-BASED: Advanced policy with CNN, Attention, LSTM
                env,
                learning_rate=learning_rate,  # Adaptive
                n_steps=2048,  # TIER 1: Shorter rollouts for better LSTM sequence context (was 4096)
                batch_size=batch_size,  # Fixed at 256 for LSTM
                n_epochs=10,
                gamma=0.995,  # Long-term planning
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=entropy_coef,  # Adaptive
                vf_coef=0.5,  # Policy focus
                max_grad_norm=0.5,
                verbose=1,
                tensorboard_log=f"runs/{run_name}",
                device="cuda" if torch.cuda.is_available() else "cpu",
                # policy_kwargs handled by AdvancedActorCriticPolicy (no need to specify)
            )
            print(f"[OK] Created new PPO model with ADVANCED ARCHITECTURE")
            print(f"  Architecture: CNN (raycasting) + Attention + LSTM (memory)")
            print(f"  Learning rate: {learning_rate:.6f} (adaptive)")
            print(f"  Entropy coef: {entropy_coef:.4f} (adaptive)")
            print(f"  Batch size: {batch_size} (adaptive)")
            print(f"  Features: 512-dim (11 base + 36 rays + 3 enhanced)")
        else:
            print("[OK] Continuing from previous stage")
            # Note: PPO hyperparameters are set at creation, but we log what they would be
            model.set_env(env)
            print(f"  Stage {stage_num} recommended: LR={learning_rate:.6f}, Entropy={entropy_coef:.4f}, Batch={batch_size}")
            print(f"  (Note: Using model's original hyperparameters - adaptive values apply to new models)")
        
        # Setup callbacks
        curriculum_callback = UltimateCurriculumCallback()
        progress_callback = ProgressLogCallback(check_freq=10000)
        adaptive_callback = AdaptiveCallback(adaptive_hp, update_freq=5000)  # RESEARCH-BASED: Adaptive HPO
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path="./checkpoints/",
            name_prefix=f"ultimate_stage{stage_idx+1}",
        )
        wandb_callback = WandbCallback(model_save_freq=0, gradient_save_freq=0, verbose=2)
        
        # Train
        print(f"Training stage {stage_idx + 1}...\n")
        start_time = datetime.now()
        
        try:
            model.learn(
                total_timesteps=stage['timesteps'],
                callback=[curriculum_callback, progress_callback, adaptive_callback, checkpoint_callback, wandb_callback],
                progress_bar=True,
                reset_num_timesteps=False,
            )
        except KeyboardInterrupt:
            print("\n⚠ Training interrupted")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Stage statistics
        stage_stats = {
            "stage": stage['name'],
            "config": stage['config'],
            "timesteps": stage['timesteps'],
            "duration_seconds": duration,
            "success_rate": np.mean(curriculum_callback.episode_successes[-100:]) * 100 if curriculum_callback.episode_successes else 0,
            "total_episodes": curriculum_callback.episode_count,
            "env_type_breakdown": {
                k: {
                    "success_rate": (v["successes"] / v["total"] * 100) if v["total"] > 0 else 0,
                    "episodes": v["total"]
                }
                for k, v in curriculum_callback.env_type_stats.items()
            }
        }
        all_stage_stats.append(stage_stats)
        
        print(f"\n{'='*80}")
        print(f"STAGE {stage_idx + 1} COMPLETE")
        print(f"{'='*80}")
        print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"Episodes: {stage_stats['total_episodes']}")
        print(f"Success Rate: {stage_stats['success_rate']:.1f}%")
        print(f"\nPer-Environment Breakdown:")
        for env_type, stats in stage_stats['env_type_breakdown'].items():
            print(f"  {env_type:15s}: {stats['success_rate']:5.1f}% ({stats['episodes']} eps)")
        print(f"{'='*80}\n")
        
        # Save intermediate model
        stage_save_path = save_path.replace(".zip", f"_stage{stage_idx+1}.zip")
        model.save(stage_save_path)
        env.save(stage_save_path.replace(".zip", "_vecnormalize.pkl"))
        print(f"✓ Stage model saved: {stage_save_path}\n")
        
        # Don't close if this is the last stage (need VecNormalize for final save)
        if stage_idx < len(stages) - 1:
            env.close()
    
    # Save final model
    model.save(save_path)
    
    # Save final VecNormalize from last stage (has all training statistics!)
    if vec_norm_final is not None:
        vec_norm_final.save(save_path.replace(".zip", "_vecnormalize.pkl"))
        print(f"✓ Saved VecNormalize with training statistics from final stage")
        vec_norm_final.close()
    else:
        # Fallback: use stage6 VecNormalize
        stage6_path = save_path.replace(".zip", "_stage6_vecnormalize.pkl")
        if os.path.exists(stage6_path):
            import shutil
            shutil.copy(stage6_path, save_path.replace(".zip", "_vecnormalize.pkl"))
            print(f"✓ Copied stage6 VecNormalize as final VecNormalize")
        else:
            print(f"⚠ Warning: Could not save VecNormalize - use stage6_vecnormalize.pkl manually")
    
    print(f"\n{'='*80}")
    print("ULTIMATE CURRICULUM TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Final model: {save_path}")
    
    # Save training summary
    summary = {
        "total_timesteps": total_timesteps,
        "stages": all_stage_stats,
        "final_model": save_path,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open("curriculum_logs/ultimate_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary: curriculum_logs/ultimate_training_summary.json")
    print(f"{'='*80}\n")
    
    # Print summary table
    print("\nTRAINING SUMMARY:")
    print("-" * 100)
    for stage in all_stage_stats:
        shapes = ", ".join(stage['config']['shapes'])
        print(f"{stage['stage']:18s} | {stage['config']['difficulty']:8s} | "
            f"{stage['timesteps']:8,} steps | Success: {stage['success_rate']:5.1f}% | Shapes: {shapes}")
    print("-" * 100 + "\n")

    wandb.finish()
    return model


def main():
    parser = argparse.ArgumentParser(description="Ultimate Curriculum Learning")
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--model", type=str, default="models/ultimate_generalized_agent.zip")
    parser.add_argument("--wandb-project", type=str, default="ultimate-pedestrian-nav")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    
    args = parser.parse_args()
    
    train_ultimate_curriculum(
        total_timesteps=args.timesteps,
        save_path=args.model,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
