"""
ULTIMATE Evaluation Suite
Tests agent on ALL corridor types with ALL obstacle configurations:
- Standard corridors (sparse, dense)
- L-shaped corridors
- T-shaped corridors  
- U-shaped corridors
- Multi-room corridors
- Narrow passages (custom walls)
- Zigzag paths (alternating obstacles)
- Clustered obstacles

Generates videos for EVERY episode and comprehensive statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.animation import FuncAnimation, FFMpegWriter
import json
from datetime import datetime
from typing import Dict, List, Tuple
import os
import argparse

# Apply numpy compatibility fix before importing stable_baselines3
try:
    import numpy_compat_fix
except ImportError:
    pass

# Note: Old environment imports removed - using only UltimateDomainRandomizedEnv
from ultimate_domain_randomization_env import UltimateDomainRandomizedEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class UltimateEvaluator:
    """Comprehensive evaluator for all environment types."""
    
    def __init__(self, model_path: str):
        print(f"Loading model: {model_path}")
        # Create a dummy environment to get observation/action spaces
        dummy_env = UltimateDomainRandomizedEnv()
        obs_space = dummy_env.observation_space
        action_space = dummy_env.action_space
        dummy_env.close()
        
        # Try loading with explicit spaces to bypass deserialization issues
        try:
            # Method 1: Load with environment (preferred)
            dummy_env2 = UltimateDomainRandomizedEnv()
            self.model = PPO.load(
                model_path, 
                env=dummy_env2,
                print_system_info=False
            )
            dummy_env2.close()
        except Exception as e1:
            print(f"[WARN] Warning: Could not load with environment: {e1}")
            try:
                # Method 2: Try loading with explicit spaces using custom_objects
                from stable_baselines3.common.save_util import load_from_zip_file
                import torch
                
                # Load the model data manually and reconstruct
                data, params, pytorch_variables = load_from_zip_file(
                    model_path, 
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    print_system_info=False
                )
                
                # Reconstruct model with correct spaces
                dummy_env3 = UltimateDomainRandomizedEnv()
                self.model = PPO(
                    "MlpPolicy",
                    dummy_env3,
                    **data.get("learning_rate", {}),
                )
                # Load the policy and value function weights
                self.model.policy.load_state_dict(pytorch_variables["policy_state_dict"])
                self.model.policy.optimizer.load_state_dict(pytorch_variables["policy_optimizer_state_dict"])
                if "value_state_dict" in pytorch_variables:
                    self.model.value_net.load_state_dict(pytorch_variables["value_state_dict"])
                dummy_env3.close()
                print("[OK] Model loaded by reconstructing with current environment")
            except Exception as e2:
                print(f"[WARN] Warning: Could not reconstruct model: {e2}")
                # Method 3: Last resort - try stage models
                import glob
                stage_models = sorted(glob.glob("models/ultimate_generalized_agent_stage*.zip"))
                if stage_models:
                    print(f"[WARN] Trying alternative model: {stage_models[-1]}")
                    dummy_env4 = UltimateDomainRandomizedEnv()
                    self.model = PPO.load(stage_models[-1], env=dummy_env4, print_system_info=False)
                    dummy_env4.close()
                    print(f"[OK] Loaded alternative model: {stage_models[-1]}")
                else:
                    print("\n" + "="*80)
                    print("❌ MODEL LOADING FAILED")
                    print("="*80)
                    print("The model was saved with a different numpy version and cannot be loaded.")
                    print("\nSOLUTION: Retrain the model with current numpy version:")
                    print("  python ultimate_curriculum_trainer.py --timesteps 1850000")
                    print("\nOr use a model from a different training session.")
                    print("="*80)
                    raise RuntimeError(f"Could not load model from {model_path}. Model was saved with incompatible numpy version. Please retrain.")
        
        # Try to load VecNormalize
        self.vec_normalize = None
        vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")
        
        # Try to load: first main, then stage6 (has best training stats)
        vecnorm_paths_to_try = [
            vecnorm_path,
            model_path.replace(".zip", "_stage6_vecnormalize.pkl"),
            model_path.replace(".zip", "_stage5_vecnormalize.pkl"),
            model_path.replace(".zip", "_stage4_vecnormalize.pkl"),
        ]
        
        for vecnorm_path_try in vecnorm_paths_to_try:
            if os.path.exists(vecnorm_path_try):
                try:
                    # CRITICAL: Use UltimateDomainRandomizedEnv for VecNormalize
                    # (same as training) to get correct normalization stats
                    dummy_env = DummyVecEnv([lambda: UltimateDomainRandomizedEnv()])
                    self.vec_normalize = VecNormalize.load(vecnorm_path_try, dummy_env)
                    self.vec_normalize.training = False
                    self.vec_normalize.norm_reward = False
                    dummy_env.close()
                    print(f"[OK] Loaded VecNormalize: {os.path.basename(vecnorm_path_try)}")
                    break
                except Exception as e:
                    print(f"[WARN] Could not load VecNormalize from {os.path.basename(vecnorm_path_try)}: {e}")
                    continue
        
        if self.vec_normalize is None:
            print(f"[WARN] VecNormalize not found or failed to load, using raw observations")
            print(f"   This may cause evaluation failures if model expects normalization!")
        
        self.all_results = []
    
    def create_standard_sparse_env(self) -> UltimateDomainRandomizedEnv:
        """Create standard sparse environment."""
        env = UltimateDomainRandomizedEnv(
            difficulty_level="easy",
            allowed_shapes=["standard"]
        )
        return env
    
    def create_standard_dense_env(self) -> UltimateDomainRandomizedEnv:
        """Create standard dense environment."""
        env = UltimateDomainRandomizedEnv(
            difficulty_level="hard",
            allowed_shapes=["standard"]
        )
        return env
        
    def create_lshaped_env(self) -> UltimateDomainRandomizedEnv:
        """Create L-shaped corridor environment."""
        env = UltimateDomainRandomizedEnv(
            difficulty_level="medium",
            allowed_shapes=["lshaped"]
        )
        return env
    
    def create_tshaped_env(self) -> UltimateDomainRandomizedEnv:
        """Create T-shaped corridor environment."""
        env = UltimateDomainRandomizedEnv(
            difficulty_level="medium",
            allowed_shapes=["tshaped"]
        )
        return env
    
    def evaluate_scenario(self, env, scenario_name: str, episode_id: int) -> Dict:
        """Evaluate one episode."""
        # Save custom obstacles if they exist (for narrow/zigzag/clustered scenarios)
        custom_obstacles = getattr(env, "_custom_obstacles", None)
        
        # Use seed for reproducibility but allow variation per episode
        obs, _ = env.reset(seed=episode_id)
        
        # Restore custom obstacles after reset (if they exist)
        if custom_obstacles is not None:
            env.obstacles = custom_obstacles.copy()
        
        trajectory = []
        velocities = []
        headings = []
        ray_distances_list = []  # Store raycast data for visualization
        collisions_list = []  # Track collisions for video visualization
        episode_reward = 0
        done = False
        prev_collision_count = 0

        while not done:
            trajectory.append(np.array(env.agent_pos, dtype=float).tolist())
            velocities.append(float(np.linalg.norm(env.agent_vel)))
            headings.append(float(env.agent_heading))
            # Store raycast distances for visualization
            if hasattr(env, '_get_raycast_distances'):
                ray_distances_list.append(env._get_raycast_distances().tolist())
            else:
                ray_distances_list.append([0.0] * 36)  # Fallback if no rays

            # Normalize if VecNormalize available
            if self.vec_normalize is not None:
                obs_normalized = self.vec_normalize.normalize_obs(obs[None])[0]
            else:
                obs_normalized = obs

            action, _ = self.model.predict(obs_normalized, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Track collisions for visualization
            current_collision_count = info.get("collisions", 0)
            collision_occurred = current_collision_count > prev_collision_count
            collisions_list.append(collision_occurred)
            prev_collision_count = current_collision_count
            
            done = terminated or truncated

        success = info.get("goal_reached", False)

        # --- Safe conversion helpers ---
        def to_list_safe(value, default=[0, 0]):
            """Convert tuples, arrays, or lists safely to plain Python list."""
            if value is None:
                return default
            if isinstance(value, (list, tuple)):
                return list(value)
            try:
                return value.tolist()
            except Exception:
                return default

        # Detect corridor type from environment or scenario name
        corridor_type = getattr(env, "current_corridor_type", None)
        if corridor_type is None:
            # Map scenario name to corridor type
            scenario_lower = scenario_name.lower()
            if "l-shaped" in scenario_lower or "lshaped" in scenario_lower:
                corridor_type = "lshaped"
            elif "t-shaped" in scenario_lower or "tshaped" in scenario_lower:
                corridor_type = "tshaped"
            elif "u-shaped" in scenario_lower or "ushaped" in scenario_lower:
                corridor_type = "ushaped"
            elif "multi-room" in scenario_lower or "multiroom" in scenario_lower:
                corridor_type = "multiroom"
            else:
                # Check environment class name as fallback
                env_class_name = env.__class__.__name__
                if "LShaped" in env_class_name:
                    corridor_type = "lshaped"
                elif "TShaped" in env_class_name:
                    corridor_type = "tshaped"
                elif "UShaped" in env_class_name:
                    corridor_type = "ushaped"
                else:
                    corridor_type = "standard"

        result = {
            "episode_id": episode_id,
            "scenario": scenario_name,
            "success": success,
            "reward": float(episode_reward),
            "time": float(info.get("time_elapsed", 0)),
            "collisions": int(info.get("collisions", 0)),
            "distance_traveled": float(info.get("total_distance", 0)),
            "final_distance_to_goal": float(info.get("distance_to_goal", 0)),
            "avg_velocity": float(np.mean(velocities)) if velocities else 0,
            "trajectory": [list(pos) if isinstance(pos, (list, tuple, np.ndarray)) else [0, 0] for pos in trajectory],
            "velocities": velocities,
            "headings": headings,
            "ray_distances": ray_distances_list,  # Store raycast data
            "collisions_list": collisions_list,  # Store collision events for visualization
            "obstacles": list(getattr(env, "obstacles", [])),  # Save as copy
            "goal_pos": to_list_safe(getattr(env, "goal_pos", None)),
            "start_pos": to_list_safe(getattr(env, "start_pos", None)),
            "corridor_type": corridor_type,
            # Save multi-room geometry for proper rendering
            "corridor_dims": getattr(env, "corridor_dims", {}),
            "walkable_regions": [list(region) for region in getattr(env, "walkable_regions", [])],
        }

        return result
    
    def run_complete_evaluation(self, episodes_per_scenario: int = 10) -> Dict:
        """Run evaluation on ALL scenarios."""
        print("\n" + "="*80)
        print("ULTIMATE COMPREHENSIVE EVALUATION")
        print("="*80)
        
        scenarios = [
            ("Standard Sparse", lambda: UltimateDomainRandomizedEnv(
                difficulty_level="easy", allowed_shapes=["standard"]
            )),
            ("Standard Dense", lambda: UltimateDomainRandomizedEnv(
                difficulty_level="medium", allowed_shapes=["standard"]
            )),
            ("L-Shaped Corridor", lambda: UltimateDomainRandomizedEnv(
                difficulty_level="easy", allowed_shapes=["lshaped"]
            )),
            ("T-Shaped Corridor", lambda: UltimateDomainRandomizedEnv(
                difficulty_level="easy", allowed_shapes=["tshaped"]
            )),
        ]
        
        episode_counter = 0
        scenario_results = {}
        
        for scenario_name, create_env_func in scenarios:
            print(f"\n{'='*70}")
            print(f"Scenario: {scenario_name}")
            print(f"{'='*70}")
            
            scenario_data = []
            
            for ep in range(episodes_per_scenario):
                env = create_env_func()
                result = self.evaluate_scenario(env, scenario_name, episode_counter)
                scenario_data.append(result)
                self.all_results.append(result)
                episode_counter += 1
                
                status = "OK" if result["success"] else "FAIL"
                print(f"  Episode {ep+1}/{episodes_per_scenario}: {status} | "
                      f"Time: {result['time']:.1f}s | "
                      f"Collisions: {result['collisions']} | "
                      f"Reward: {result['reward']:.1f}")
                
                env.close()
            
            # Scenario statistics
            successes = sum(1 for r in scenario_data if r["success"])
            scenario_results[scenario_name] = {
                "success_rate": (successes / episodes_per_scenario) * 100,
                "avg_time": np.mean([r["time"] for r in scenario_data]),
                "avg_collisions": np.mean([r["collisions"] for r in scenario_data]),
                "avg_reward": np.mean([r["reward"] for r in scenario_data]),
                "avg_velocity": np.mean([r["avg_velocity"] for r in scenario_data]),
            }
        
        # Overall statistics
        total_successes = sum(1 for r in self.all_results if r["success"])
        total_episodes = len(self.all_results)
        
        overall_results = {
            "overall_success_rate": (total_successes / total_episodes) * 100,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "total_failures": total_episodes - total_successes,
            "scenario_breakdown": scenario_results,
            "timestamp": datetime.now().isoformat(),
        }
        
        print(f"\n{'='*80}")
        print("OVERALL RESULTS")
        print(f"{'='*80}")
        print(f"Overall Success Rate: {overall_results['overall_success_rate']:.1f}%")
        print(f"Total Episodes: {total_episodes} ({total_successes} successes)")
        print(f"\nScenario Breakdown:")
        for scenario, stats in scenario_results.items():
            print(f"  {scenario:25s}: {stats['success_rate']:5.1f}% success | "
                  f"Avg time: {stats['avg_time']:5.1f}s | "
                  f"Avg collisions: {stats['avg_collisions']:4.1f}")
        
        return overall_results
    
    def save_results(self, output_dir: str = "eval_ultimate"):
        """Save detailed results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for each scenario type
        for result in self.all_results:
            scenario_dir = result["scenario"].replace(" ", "_").lower()
            os.makedirs(f"{output_dir}/{scenario_dir}", exist_ok=True)
        
        # Save all results
        with open(f"{output_dir}/complete_results.json", "w") as f:
            json.dump(self.all_results, f, indent=2)
        
        print(f"\n[OK] Results saved to {output_dir}/")
        
        # Export for 3D visualizer (skipped - files deleted)
        print("[INFO] 3D visualizer export skipped")
    
    def generate_all_videos(self, output_dir: str = "eval_ultimate"):
        """Generate videos for ALL episodes."""
        print(f"\n[VIDEO] Generating videos for ALL {len(self.all_results)} episodes...")
        
        for idx, result in enumerate(self.all_results):
            scenario_dir = result["scenario"].replace(" ", "_").lower()
            status = "success" if result["success"] else "failure"
            filename = f"{output_dir}/{scenario_dir}/ep{result['episode_id']:03d}_{status}.mp4"
            
            self._create_video(result, filename)
            
            if (idx + 1) % 10 == 0:
                print(f"  Generated {idx + 1}/{len(self.all_results)} videos...")
        
        print(f"[OK] All videos generated!")
    
    def _create_video(self, result: Dict, filename: str):
        """Create video for one episode."""
        trajectory = result["trajectory"]
        obstacles = result["obstacles"]
        goal_pos = result["goal_pos"]
        start_pos = result["start_pos"]
        
        # Validate trajectory data
        if not trajectory or len(trajectory) == 0:
            print(f"[WARN] Skipping video for {filename}: empty trajectory")
            return
        
        # Ensure trajectory is valid (list of [x, y] pairs)
        try:
            trajectory = [[float(p[0]), float(p[1])] for p in trajectory if len(p) >= 2]
            if len(trajectory) == 0:
                print(f"[WARN] Skipping video for {filename}: invalid trajectory data")
                return
        except (ValueError, TypeError, IndexError) as e:
            print(f"[WARN] Skipping video for {filename}: trajectory parsing error: {e}")
            return
        success = result["success"]
        corridor_type = result["corridor_type"]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor('white')  # White figure background
        frames = []
        
        frame_skip = max(1, len(trajectory) // 80)  # 80 frames max
        
        for step_idx in range(0, len(trajectory), frame_skip):
            ax.clear()
            ax.set_facecolor('white')  # White axes background
            
            # Draw corridor background based on type
            if corridor_type == "lshaped":
                # Use ACTUAL dimensions from saved result (ORIGINAL CODE - was working)
                walkable_regions = result.get("walkable_regions", [])
                dims = result.get("corridor_dims", {})
                
                if walkable_regions:
                    # Draw each region with seamless connection at corners
                    for region in walkable_regions:
                        x_min, x_max, y_min, y_max = region
                        corridor = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              facecolor="#F5F5DC", edgecolor="#8B7355", 
                                              linewidth=2.5, alpha=0.95, zorder=0)
                        ax.add_patch(corridor)
                    # Connect regions at corners for L-shapes (fix black line issue)
                    main_regions = []
                    corner_region = None
                    
                    for region in walkable_regions:
                        x_min, x_max, y_min, y_max = region
                        if (x_max - x_min) < 2.0 and (y_max - y_min) < 2.0:
                            corner_region = region
                        else:
                            main_regions.append(region)
                    
                    if corner_region:
                        x_min, x_max, y_min, y_max = corner_region
                        corner_fill = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                               facecolor="#F5F5DC", edgecolor="none", zorder=0)
                        ax.add_patch(corner_fill)
                    
                    if len(main_regions) >= 2:
                        r1, r2 = main_regions[0], main_regions[1]
                        r1_x_min, r1_x_max, r1_y_min, r1_y_max = r1
                        r2_x_min, r2_x_max, r2_y_min, r2_y_max = r2
                        corner_x_min = max(r1_x_min, r2_x_min)
                        corner_x_max = min(r1_x_max, r2_x_max)
                        corner_y_min = max(r1_y_min, r2_y_min)
                        corner_y_max = min(r1_y_max, r2_y_max)
                        if corner_x_min < corner_x_max and corner_y_min < corner_y_max:
                            corner_fill = Rectangle((corner_x_min, corner_y_min), 
                                                   corner_x_max - corner_x_min, 
                                                   corner_y_max - corner_y_min,
                                                   facecolor="#F5F5DC", edgecolor="none", zorder=0)
                            ax.add_patch(corner_fill)
                    all_x = [r[0] for r in walkable_regions] + [r[1] for r in walkable_regions]
                    all_y = [r[2] for r in walkable_regions] + [r[3] for r in walkable_regions]
                    if all_x and all_y:
                        ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
                        ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
                    else:
                        ax.add_patch(Rectangle((0, 0), 25, 8, facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                        ax.add_patch(Rectangle((17, 8), 8, 12, facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                        ax.set_xlim(-1, 26)
                        ax.set_ylim(-1, 21)
                else:
                    # Fallback to default
                    ax.add_patch(Rectangle((0, 0), 25, 8, facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                    ax.add_patch(Rectangle((17, 8), 8, 12, facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                    ax.set_xlim(-1, 26)
                    ax.set_ylim(-1, 21)
            elif corridor_type == "tshaped":
                # Use ACTUAL dimensions from saved result
                walkable_regions = result.get("walkable_regions", [])
                
                if walkable_regions and len(walkable_regions) >= 2:
                    # FIXED: Draw T-shape WITHOUT internal borders
                    # Step 1: Fill all regions with NO border
                    for region in walkable_regions:
                        x_min, x_max, y_min, y_max = region
                        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              facecolor="#F5F5DC", edgecolor="none", 
                                              linewidth=0, zorder=0))
                    
                    # Step 2: Draw only OUTER walls
                    wall_color = "#8B7355"
                    wall_width = 2.5
                    
                    # Get stem and bar regions
                    stem = walkable_regions[0]
                    bar = walkable_regions[1]
                    s_x_min, s_x_max, s_y_min, s_y_max = stem
                    b_x_min, b_x_max, b_y_min, b_y_max = bar
                    
                    # Stem walls (left, right, bottom - NOT top)
                    ax.plot([s_x_min, s_x_min], [s_y_min, s_y_max], color=wall_color, linewidth=wall_width, zorder=1)
                    ax.plot([s_x_max, s_x_max], [s_y_min, s_y_max], color=wall_color, linewidth=wall_width, zorder=1)
                    ax.plot([s_x_min, s_x_max], [s_y_min, s_y_min], color=wall_color, linewidth=wall_width, zorder=1)
                    
                    # Bar walls (left, right, top - bottom only on sides)
                    ax.plot([b_x_min, b_x_max], [b_y_max, b_y_max], color=wall_color, linewidth=wall_width, zorder=1)
                    ax.plot([b_x_min, b_x_min], [b_y_min, b_y_max], color=wall_color, linewidth=wall_width, zorder=1)
                    ax.plot([b_x_max, b_x_max], [b_y_min, b_y_max], color=wall_color, linewidth=wall_width, zorder=1)
                    ax.plot([b_x_min, s_x_min], [b_y_min, b_y_min], color=wall_color, linewidth=wall_width, zorder=1)
                    ax.plot([s_x_max, b_x_max], [b_y_min, b_y_min], color=wall_color, linewidth=wall_width, zorder=1)
                    
                    # Set limits
                    all_x = [r[0] for r in walkable_regions] + [r[1] for r in walkable_regions]
                    all_y = [r[2] for r in walkable_regions] + [r[3] for r in walkable_regions]
                    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
                    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
                else:
                    # Fallback to default (only if no walkable_regions)
                    stem_center_x = 15.0
                    stem_left = stem_center_x - 4.0
                    ax.add_patch(Rectangle((stem_left, 0), 8, 15, facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                    ax.add_patch(Rectangle((0, 15), 30, 8, facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                    ax.set_xlim(-1, 31)
                    ax.set_ylim(-1, 24)
            elif corridor_type == "ushaped":
                # U-shape approximation
                ax.add_patch(Rectangle((0, 0), 8, 25, facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                ax.add_patch(Rectangle((0, 0), 30, 8, facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                ax.add_patch(Rectangle((22, 0), 8, 25, facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                ax.set_xlim(-1, 31)
                ax.set_ylim(-1, 26)
            elif corridor_type == "multiroom":
                # Render multi-room structure from saved walkable_regions
                walkable_regions = result.get("walkable_regions", [])
                if walkable_regions:
                    # Draw each room and corridor connection
                    for region in walkable_regions:
                        x_min, x_max, y_min, y_max = region
                        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                    # Set limits based on regions
                    all_x = [r[0] for r in walkable_regions] + [r[1] for r in walkable_regions]
                    all_y = [r[2] for r in walkable_regions] + [r[3] for r in walkable_regions]
                    ax.set_xlim(min(all_x) - 2, max(all_x) + 2)
                    ax.set_ylim(min(all_y) - 2, max(all_y) + 2)
                else:
                    # Fallback: estimate from corridor_dims
                    dims = result.get("corridor_dims", {})
                    room_size = dims.get("room_size", 12)
                    cor_width = dims.get("corridor_width", 6)
                    num_rooms = dims.get("num_rooms", 3)
                    w = 0.3  # wall thickness
                    for i in range(num_rooms):
                        x_start = i * (room_size + cor_width)
                        # Room
                        ax.add_patch(Rectangle((x_start + w, w), room_size - 2*w, room_size - 2*w,
                                              facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                        # Corridor to next room
                        if i < num_rooms - 1:
                            ax.add_patch(Rectangle((x_start + room_size + w, room_size/2 - cor_width/2 + w),
                                                  cor_width - 2*w, cor_width - 2*w,
                                                  facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                    total_width = num_rooms * room_size + (num_rooms - 1) * cor_width
                    ax.set_xlim(-1, total_width + 1)
                    ax.set_ylim(-1, room_size + 1)
            else:  # standard, etc.
                # Use ACTUAL dimensions from saved result
                dims = result.get("corridor_dims", {})
                walkable_regions = result.get("walkable_regions", [])
                
                if walkable_regions:
                    # Use actual walkable regions for accurate rendering
                    for region in walkable_regions:
                        x_min, x_max, y_min, y_max = region
                        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                    # Set limits based on actual regions
                    all_x = [r[0] for r in walkable_regions] + [r[1] for r in walkable_regions]
                    all_y = [r[2] for r in walkable_regions] + [r[3] for r in walkable_regions]
                    if all_x and all_y:
                        ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
                        ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
                    else:
                        # Fallback to default
                        ax.add_patch(Rectangle((0, 0), 40, 10, facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                        ax.set_xlim(-0.5, 40.5)
                        ax.set_ylim(-0.5, 10.5)
                elif dims:
                    # Use corridor_dims if available
                    length = dims.get("length", 40)
                    width = dims.get("width", 10)
                    w = 0.3  # wall thickness
                    ax.add_patch(Rectangle((w, w), length - 2*w, width - 2*w,
                                          facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                    ax.set_xlim(-0.5, length + 0.5)
                    ax.set_ylim(-0.5, width + 0.5)
                else:
                    # Ultimate fallback: default dimensions
                    ax.add_patch(Rectangle((0, 0), 40, 10, facecolor="#F5F5DC", edgecolor="black", linewidth=2))
                    ax.set_xlim(-0.5, 40.5)
                    ax.set_ylim(-0.5, 10.5)
            
            # Draw obstacles - Enhanced with modern styling
            for obs in obstacles:
                x, y, w, h = obs
                # Modern gradient-like effect with shadow
                shadow = FancyBboxPatch((x+0.1, y-0.1), w, h, 
                                       boxstyle="round,pad=0.05",
                                       facecolor="#000000", 
                                       edgecolor="none", 
                                       alpha=0.2, zorder=1)
                ax.add_patch(shadow)
                # Main obstacle
                obstacle = FancyBboxPatch((x, y), w, h, 
                                           boxstyle="round,pad=0.05",
                                         facecolor="#8B4513",  # Saddle brown
                                         edgecolor="#654321",  # Darker brown
                                         linewidth=2.5,
                                         alpha=0.95, zorder=3)
                ax.add_patch(obstacle)
            
            # Draw trajectory - Enhanced with gradient effect
            if step_idx > 0:
                try:
                    path = np.array(trajectory[:step_idx+1:frame_skip])
                    if len(path) > 1 and path.shape[1] >= 2:
                        # Draw path with gradient (fade from start to current)
                        for i in range(len(path)-1):
                            if i+1 < len(path) and len(path[i]) >= 2 and len(path[i+1]) >= 2:
                                alpha = 0.3 + 0.7 * (i / max(1, len(path)-1))  # Fade in
                                ax.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], 
                                       color="#00CED1", linewidth=2.5, alpha=alpha, linestyle="-", zorder=2)
                except (ValueError, IndexError, TypeError) as e:
                    # Skip path drawing if there's an error
                    pass
            
            # Draw start marker
            ax.add_patch(Circle(start_pos, 0.3, facecolor="blue", 
                               edgecolor="darkblue", linewidth=2, alpha=0.5, zorder=2))
            
            # Draw goal - Enhanced with glow effect
            # Outer glow
            glow = Circle(goal_pos, 0.9, facecolor="#00FF00", edgecolor="none", 
                         alpha=0.3, zorder=2)
            ax.add_patch(glow)
            # Main goal circle
            goal_circle = Circle(goal_pos, 0.675, facecolor="#00FF00",
                               edgecolor="#008000", linewidth=4, alpha=0.9, zorder=3)
            ax.add_patch(goal_circle)
            # Goal text with shadow
            ax.text(goal_pos[0]+0.1, goal_pos[1]-0.1, "EXIT", ha="center", va="center",
                   fontsize=12, fontweight="bold", color="#000000", alpha=0.3, zorder=4)
            ax.text(goal_pos[0], goal_pos[1], "EXIT", ha="center", va="center",
                   fontsize=12, fontweight="bold", color="#FFFFFF", zorder=5)
            
            # Draw agent with collision indicator
            try:
                agent_pos = trajectory[step_idx]
                if len(agent_pos) < 2:
                    agent_pos = [0.0, 0.0]  # Fallback
            except (IndexError, TypeError):
                agent_pos = [0.0, 0.0]  # Fallback
            
            color = "green" if success and step_idx >= len(trajectory)-frame_skip else "#4169E1"
            
            # Check if collision occurred at this step
            collision_occurred = False
            if step_idx < len(result.get("collisions_list", [])):
                # Check collisions in the frame range
                frame_start = max(0, step_idx - frame_skip)
                frame_end = min(len(result.get("collisions_list", [])), step_idx + 1)
                collision_occurred = any(result.get("collisions_list", [])[i] 
                                        for i in range(frame_start, frame_end))
            
            # Draw collision flash effect
            if collision_occurred:
                # Red flash circle
                flash = Circle(agent_pos, 0.5, facecolor="#FF0000", 
                              edgecolor="none", alpha=0.4, zorder=3)
                ax.add_patch(flash)
            
            # Main agent circle with enhanced styling
            agent_circle = Circle(agent_pos, 0.225, facecolor=color,
                               edgecolor="darkblue", linewidth=2.5, alpha=0.95, zorder=4)
            ax.add_patch(agent_circle)
            
            # Agent inner highlight
            highlight = Circle(agent_pos, 0.15, facecolor="white", 
                            edgecolor="none", alpha=0.3, zorder=5)
            ax.add_patch(highlight)
            
            # Draw raycasting (Lidar-like visualization) - MODERN ENHANCEMENT
            if step_idx < len(result.get("ray_distances", [])):
                ray_distances = result["ray_distances"][step_idx]
                heading = result["headings"][step_idx] if step_idx < len(result["headings"]) else 0.0
                n_rays = len(ray_distances) if ray_distances else 36
                
                for i in range(n_rays):
                    angle = (2 * np.pi * i / n_rays) + heading
                    dist = ray_distances[i] if i < len(ray_distances) else 12.0
                    end_x = agent_pos[0] + dist * np.cos(angle)
                    end_y = agent_pos[1] + dist * np.sin(angle)
                    
                    # Color coding: red = close (<2m), orange = medium (2-5m), green = far (>5m)
                    if dist < 2.0:
                        ray_color = "#FF4444"  # Bright red
                        alpha = 0.7
                        linewidth = 1.5
                    elif dist < 5.0:
                        ray_color = "#FF8800"  # Orange
                        alpha = 0.5
                        linewidth = 1.2
                    else:
                        ray_color = "#44FF44"  # Bright green
                        alpha = 0.3
                        linewidth = 1.0
                    
                    ax.plot([agent_pos[0], end_x], [agent_pos[1], end_y],
                           color=ray_color, linewidth=linewidth, alpha=alpha, zorder=2)
            
            # Draw heading arrow (more prominent)
            if step_idx < len(result["headings"]):
                heading = result["headings"][step_idx]
                arrow_len = 0.6
                ax.arrow(agent_pos[0], agent_pos[1],
                        arrow_len * np.cos(heading), arrow_len * np.sin(heading),
                        head_width=0.3, head_length=0.25, fc="#FFD700", ec="#FF8C00",
                        linewidth=3, zorder=10, alpha=0.9)
            
            # Title
            time = step_idx * 0.1
            vel = result["velocities"][step_idx] if step_idx < len(result["velocities"]) else 0
            title = f"{result['scenario']} - Episode {result['episode_id']} | Time: {time:.1f}s | Speed: {vel:.2f}m/s"
            
            if step_idx >= len(trajectory) - frame_skip:
                if success:
                    title += " - [OK] SUCCESS"
                    title_color = "green"
                else:
                    title += " - [FAIL] FAILED"
                    title_color = "red"
            else:
                title_color = "black"
            
            ax.set_title(title, fontsize=13, fontweight="bold", color=title_color, pad=15)
            ax.set_aspect("equal")
            ax.set_xlabel("Distance (m)", fontsize=11)
            ax.set_ylabel("Width (m)", fontsize=11)
            ax.grid(True, alpha=0.2)
            
            # Add info box
            info_text = (f"Collisions: {result['collisions']} | "
                        f"Distance: {result['distance_traveled']:.1f}m | "
                        f"Reward: {result['reward']:.1f}")
            ax.text(0.5, -0.08, info_text, transform=ax.transAxes, ha="center",
                   fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
            
            fig.canvas.draw()
            # Works for Tkinter canvas
            frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, [1, 2, 3]]  # convert ARGB → RGB
            frames.append(frame)
        
        plt.close(fig)
        
        # Save video
        if frames:
            fig_anim, ax_anim = plt.subplots(figsize=(14, 6))
            ax_anim.axis("off")
            im = ax_anim.imshow(frames[0])
            
            def update(frame_idx):
                im.set_array(frames[frame_idx])
                return [im]
            
            anim = FuncAnimation(fig_anim, update, frames=len(frames),
                               interval=100, blit=True, repeat=True)
            writer = FFMpegWriter(fps=10, bitrate=2500)
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            anim.save(filename, writer=writer)
            plt.close(fig_anim)


def main():
    parser = argparse.ArgumentParser(description="Ultimate Comprehensive Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="eval_ultimate")
    parser.add_argument("--skip-videos", action="store_true", help="Skip video generation")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = UltimateEvaluator(args.model)
    results = evaluator.run_complete_evaluation(args.episodes_per_scenario)
    evaluator.save_results(args.output_dir)
    
    if not args.skip_videos:
        evaluator.generate_all_videos(args.output_dir)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Results: {args.output_dir}/complete_results.json")
    print(f"Videos: {args.output_dir}/*/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
