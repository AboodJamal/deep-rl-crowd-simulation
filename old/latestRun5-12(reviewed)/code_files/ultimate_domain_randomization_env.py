"""
ULTIMATE Domain Randomization Environment
Supports ALL corridor types with domain randomization:
- Standard corridors
- L-shaped corridors  
- T-shaped corridors
- U-shaped corridors (NEW)
- Multi-room corridors (NEW)

Each episode randomly selects shape + randomizes:
- Obstacle density
- Obstacle arrangement
- Goal/start positions
- Corridor dimensions (within limits)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from typing import Tuple, Optional, Dict, Any, List
import math


class UltimateDomainRandomizedEnv(gym.Env):
    """
    Ultimate environment with ALL corridor shapes and full randomization.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Base parameters
    AGENT_RADIUS = 0.225
    MAX_VELOCITY = 1.4
    MAX_ACCELERATION = 2.0
    MAX_ANGULAR_VEL = 1.8
    DT = 0.1
    WALL_THICKNESS = 0.3
    
    # Raycasting (Lidar-like) parameters
    N_RAYS = 36  # 36 rays = 10° resolution
    MAX_RAY_RANGE = 12.0  # 12m max range

    def __init__(
        self,
        difficulty_level: str = "medium",  # easy, medium, hard, mixed, ultra
        allowed_shapes: List[str] = None,  # None = all shapes
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.difficulty_level = difficulty_level
        self.render_mode = render_mode
        
        # Allowed corridor shapes
        # Note: Multi-room removed (not realistic for basic navigation)
        # U-shaped kept for future use but not in current training
        if allowed_shapes is None:
            self.allowed_shapes = ["standard", "lshaped", "tshaped"]  # Focus on core shapes
        else:
            self.allowed_shapes = allowed_shapes

        # Difficulty ranges (RESEARCH-BASED: Much easier start for Zone of Proximal Development)
        # Based on 2024-2025 research: curriculum should be just beyond current capability
        self.difficulty_ranges = {
            "super_easy": {"density": (0.001, 0.005), "cluster_prob": 0.0, "pattern_prob": 0.0},  # NEW: Super easy start
            "easy": {"density": (0.005, 0.015), "cluster_prob": 0.02, "pattern_prob": 0.0},  # Much easier: was 0.005-0.02
            "medium": {"density": (0.015, 0.04), "cluster_prob": 0.1, "pattern_prob": 0.02},  # Easier: was 0.02-0.06
            "hard": {"density": (0.04, 0.08), "cluster_prob": 0.2, "pattern_prob": 0.1},  # Easier: was 0.05-0.10
            "mixed": {"density": (0.01, 0.08), "cluster_prob": 0.25, "pattern_prob": 0.08},  # Wider range, lower max
            "ultra": {"density": (0.06, 0.12), "cluster_prob": 0.4, "pattern_prob": 0.15},  # Easier: was 0.08-0.14
        }

        # Action/observation spaces (large enough for all shapes)
        self.action_space = spaces.Box(
            low=np.array([-self.MAX_VELOCITY, -self.MAX_ANGULAR_VEL]),
            high=np.array([self.MAX_VELOCITY, self.MAX_ANGULAR_VEL]),
            dtype=np.float32,
        )

        # Observation space: base (11) + raycasting (36) + enhanced features (3) = 50 values
        # ENHANCED: Added corner awareness, goal visibility, path length estimate
        base_low = np.array([0, 0, -self.MAX_VELOCITY, -self.MAX_VELOCITY, 
                             0, 0, 0, -np.pi, -np.pi, 0, -np.pi])
        base_high = np.array([60, 60, self.MAX_VELOCITY, self.MAX_VELOCITY,
                              60, 60, 100, np.pi, np.pi, 100, np.pi])
        # Raycasting: distances from 0 to MAX_RAY_RANGE
        ray_low = np.zeros(self.N_RAYS, dtype=np.float32)
        ray_high = np.full(self.N_RAYS, self.MAX_RAY_RANGE, dtype=np.float32)
        # Enhanced features: corner awareness (0/1), goal visibility (0/1), path length estimate (0-200)
        enhanced_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        enhanced_high = np.array([1.0, 1.0, 200.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, ray_low, enhanced_low]),
            high=np.concatenate([base_high, ray_high, enhanced_high]),
            dtype=np.float32,
        )

        # Current episode configuration
        self.current_corridor_type = "standard"
        self.current_obstacle_density = 0.08
        self.corridor_dims = {}  # Stores dimensions for current shape
        
        # State
        self.agent_pos = None
        self.agent_vel = None
        self.agent_heading = None
        self.goal_pos = None
        self.start_pos = None
        self.obstacles = []
        self.walkable_regions = []  # List of (x_min, x_max, y_min, y_max) for current shape

        # Tracking
        self.steps = 0
        # RESEARCH-BASED: Scale episode length with difficulty (500-2000 steps)
        # Easy: 500-800, Medium: 1000-1500, Hard: 1500-2000
        self.max_steps = self._get_max_steps_for_difficulty()
        self.total_distance_traveled = 0.0
        self.min_distance_to_goal = float("inf")
        self.collision_count = 0
        self.previous_distance_to_goal = None
        self.last_action = None  # Store last action for reward calculation
        self.prev_ang_vel = 0.0  # low-pass filtered angular velocity
        self.prev_lin_vel = 0.0  # low-pass filtered linear velocity (NEW: for smooth movement)
        self.consecutive_collisions = 0  # Track consecutive collisions (crash patterns)
        self.steps_without_progress = 0  # Track if stuck
        self.last_positions = []  # Track recent positions for diversity penalty
        
        # State visitation tracking for intrinsic motivation (novelty bonus)
        self.state_visit_count = {}  # Track state visits for exploration bonus
        
        # RESEARCH-BASED: Action smoothing parameters (prevents jerky movement)
        self.action_smooth_alpha_lin = 0.5  # Linear velocity smoothing (0.5 = balanced)
        self.action_smooth_alpha_ang = 0.4  # Angular velocity smoothing (0.4 = smoother turns)

        # Rendering
        self.fig = None
        self.ax = None
    
    def _get_max_steps_for_difficulty(self) -> int:
        """RESEARCH-BASED: Scale episode length with difficulty (500-2000 steps).
        Based on 2024-2025 research: successful maze navigation needs appropriate episode length.
        """
        difficulty = self.difficulty_level
        if difficulty == "super_easy":
            return 500  # Short episodes for very easy tasks
        elif difficulty == "easy":
            return 800  # Medium-short for easy tasks
        elif difficulty == "medium":
            return 1200  # Medium for medium tasks
        elif difficulty == "hard":
            return 1500  # Longer for hard tasks
        elif difficulty == "ultra":
            return 2000  # Longest for ultra-hard tasks
        elif difficulty == "mixed":
            return 1500  # Average for mixed difficulty
        else:
            return 1000  # Default

    def _select_corridor_type(self):
        """Randomly select corridor type from allowed shapes."""
        self.current_corridor_type = np.random.choice(self.allowed_shapes)
        
        # Set dimensions based on type
        if self.current_corridor_type == "standard":
            self.corridor_dims = {
                "length": np.random.uniform(35, 45),
                "width": np.random.uniform(8, 12),
            }
        elif self.current_corridor_type == "lshaped":
            self.corridor_dims = {
                "h_length": np.random.uniform(20, 30),
                "v_length": np.random.uniform(15, 25),
                "width": np.random.uniform(7, 10),
            }
        elif self.current_corridor_type == "tshaped":
            self.corridor_dims = {
                "stem_length": np.random.uniform(12, 18),
                "bar_length": np.random.uniform(25, 35),
                "width": np.random.uniform(7, 10),
            }
        elif self.current_corridor_type == "ushaped":
            self.corridor_dims = {
                "width": np.random.uniform(25, 35),
                "height": np.random.uniform(20, 30),
                "corridor_width": np.random.uniform(7, 10),
            }
        elif self.current_corridor_type == "multiroom":
            self.corridor_dims = {
                "room_size": np.random.uniform(10, 15),
                "corridor_width": np.random.uniform(5, 8),
                "num_rooms": np.random.randint(2, 4),
            }

    def _define_walkable_regions(self):
        """Define walkable regions based on current corridor type.
        IMPROVED: Better corner connection for L/T shapes to avoid boundary issues.
        """
        w = self.WALL_THICKNESS
        self.walkable_regions = []
        
        if self.current_corridor_type == "standard":
            length = self.corridor_dims["length"]
            width = self.corridor_dims["width"]
            self.walkable_regions = [(w, length - w, w, width - w)]
            
        elif self.current_corridor_type == "lshaped":
            h_len = self.corridor_dims["h_length"]
            v_len = self.corridor_dims["v_length"]
            cw = self.corridor_dims["width"]
            # Horizontal segment (extended slightly into corner for smoother transition)
            self.walkable_regions.append((w, h_len - w + 0.2, w, cw - w))
            # Vertical segment (extended slightly into corner)
            self.walkable_regions.append((h_len - cw + w - 0.2, h_len - w, cw - w, v_len - w))
            # CORNER OVERLAP REGION: Explicitly define corner as walkable to avoid boundary issues
            corner_overlap = 0.5  # Overlap distance
            corner_x_min = h_len - cw + w - corner_overlap
            corner_x_max = h_len - w + corner_overlap
            corner_y_min = cw - w - corner_overlap
            corner_y_max = cw - w + corner_overlap
            self.walkable_regions.append((corner_x_min, corner_x_max, corner_y_min, corner_y_max))
            
        elif self.current_corridor_type == "tshaped":
            stem_len = self.corridor_dims["stem_length"]
            bar_len = self.corridor_dims["bar_length"]
            cw = self.corridor_dims["width"]
            stem_center = bar_len / 2
            # Vertical stem (extended slightly into junction)
            self.walkable_regions.append((stem_center - cw/2 + w - 0.2, stem_center + cw/2 - w + 0.2, w, stem_len - w + 0.2))
            # Horizontal bar (extended slightly into junction)
            self.walkable_regions.append((w - 0.2, bar_len - w + 0.2, stem_len - w - 0.2, stem_len + cw - w))
            # JUNCTION OVERLAP REGION: Explicitly define junction as walkable
            junction_overlap = 0.5
            junction_x_min = stem_center - cw/2 + w - junction_overlap
            junction_x_max = stem_center + cw/2 - w + junction_overlap
            junction_y_min = stem_len - w - junction_overlap
            junction_y_max = stem_len - w + junction_overlap
            self.walkable_regions.append((junction_x_min, junction_x_max, junction_y_min, junction_y_max))
            
        elif self.current_corridor_type == "ushaped":
            width = self.corridor_dims["width"]
            height = self.corridor_dims["height"]
            cw = self.corridor_dims["corridor_width"]
            # Left vertical
            self.walkable_regions.append((w, cw - w, w, height - w))
            # Bottom horizontal
            self.walkable_regions.append((w, width - w, w, cw - w))
            # Right vertical
            self.walkable_regions.append((width - cw + w, width - w, w, height - w))
            
        elif self.current_corridor_type == "multiroom":
            room_size = self.corridor_dims["room_size"]
            cor_width = self.corridor_dims["corridor_width"]
            num_rooms = self.corridor_dims["num_rooms"]
            
            for i in range(num_rooms):
                # Room
                x_start = i * (room_size + cor_width)
                self.walkable_regions.append((x_start + w, x_start + room_size - w, w, room_size - w))
                # Corridor connecting to next room
                if i < num_rooms - 1:
                    self.walkable_regions.append((x_start + room_size + w, 
                                                 x_start + room_size + cor_width - w,
                                                 room_size/2 - cor_width/2 + w,
                                                 room_size/2 + cor_width/2 - w))

    def _is_valid_position(self, x: float, y: float) -> bool:
        """Check if position is in any walkable region.
        IMPROVED: More lenient check for corner/junction regions to avoid boundary issues.
        """
        # Check all regions (including overlap regions for L/T shapes)
        for x_min, x_max, y_min, y_max in self.walkable_regions:
            # Use slightly lenient bounds (0.05m margin) for smoother transitions
            margin = 0.05
            if (x_min - margin <= x <= x_max + margin and 
                y_min - margin <= y <= y_max + margin):
                return True
        return False
    
    def _clip_position_to_bounds(self, x: float, y: float) -> Tuple[float, float]:
        """Clip position to nearest valid position within walkable regions."""
        # Find closest valid position
        best_x, best_y = x, y
        min_dist = float("inf")
        
        for x_min, x_max, y_min, y_max in self.walkable_regions:
            # Clip to this region's bounds
            clipped_x = np.clip(x, x_min, x_max)
            clipped_y = np.clip(y, y_min, y_max)
            
            # Check distance from original position
            dist = math.sqrt((x - clipped_x)**2 + (y - clipped_y)**2)
            if dist < min_dist:
                min_dist = dist
                best_x, best_y = clipped_x, clipped_y
        
        return best_x, best_y

    def _generate_obstacles(self) -> List:
        """Generate obstacles in walkable regions."""
        obstacles = []
        ranges = self.difficulty_ranges[self.difficulty_level]
        self.current_obstacle_density = np.random.uniform(*ranges["density"])
        
        # Calculate total walkable area
        total_area = sum((x_max - x_min) * (y_max - y_min) 
                        for x_min, x_max, y_min, y_max in self.walkable_regions)
        
        num_obstacles = int(total_area * self.current_obstacle_density)
        
        # Decide obstacle type
        rand = np.random.random()
        pattern_prob = ranges.get("pattern_prob", 0.0)

        # Prefer realistic standard layouts: in-path but navigable
        if self.current_corridor_type == "standard" and rand < 0.5:
            obstacles = self._generate_standard_realistic_obstacles(num_obstacles)
        elif rand < pattern_prob:
            pattern_type = np.random.choice(["narrow", "zigzag"])
            obstacles = self._generate_narrow_passage_pattern() if pattern_type == "narrow" else self._generate_zigzag_pattern()
        elif rand < pattern_prob + ranges["cluster_prob"]:
            # Use clustered obstacles
            obstacles = self._generate_clustered_obstacles(num_obstacles)
        else:
            # Use random obstacles
            obstacles = self._generate_random_obstacles(num_obstacles)
        
        return obstacles

    def _generate_random_obstacles(self, num_obstacles: int) -> List:
        """Generate randomly placed obstacles."""
        obstacles = []
        
        for _ in range(num_obstacles):
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                attempts += 1
                
                # Pick random walkable region
                region_idx = np.random.randint(len(self.walkable_regions))
                region = self.walkable_regions[region_idx]
                x_min, x_max, y_min, y_max = region
                
                # Avoid edges
                if x_max - x_min < 3 or y_max - y_min < 3:
                    continue
                
                size = np.random.uniform(0.8, 1.2)
                x = np.random.uniform(x_min + 1, x_max - size - 1)
                y = np.random.uniform(y_min + 1, y_max - size - 1)
                
                # Keep connection zones and entrances/exits clear for realism
                if self._in_connection_clear_zone(region_idx, x, y, size):
                    continue
                
                obstacle = [x, y, size, size]
                if not self._check_obstacle_overlap(obstacle, obstacles):
                    obstacles.append(obstacle)
                    placed = True
        
        return obstacles

    def _generate_clustered_obstacles(self, num_obstacles: int) -> List:
        """Generate clustered obstacles."""
        obstacles = []
        num_clusters = max(2, num_obstacles // 5)
        
        for _ in range(num_clusters):
            # Pick region for cluster center
            region_idx = np.random.randint(len(self.walkable_regions))
            region = self.walkable_regions[region_idx]
            x_min, x_max, y_min, y_max = region
            
            if x_max - x_min < 5 or y_max - y_min < 5:
                continue
            
            center_x = np.random.uniform(x_min + 2, x_max - 2)
            center_y = np.random.uniform(y_min + 2, y_max - 2)
            
            cluster_size = np.random.randint(2, 4)
            for i in range(cluster_size):
                for j in range(cluster_size):
                    if np.random.random() > 0.5:
                        x = center_x + i * np.random.uniform(1.0, 1.5)
                        y = center_y + j * np.random.uniform(1.0, 1.5)
                        
                        if self._is_valid_position(x + 0.5, y + 0.5) and not self._in_connection_clear_zone(region_idx, x, y, 1.0):
                            size = np.random.uniform(0.8, 1.2)
                            obstacle = [x, y, size, size]
                            if not self._check_obstacle_overlap(obstacle, obstacles, margin=0.3):
                                obstacles.append(obstacle)
        
        return obstacles

    def _in_connection_clear_zone(self, region_idx: int, x: float, y: float, size: float) -> bool:
        """Keep entrances/exits and region connections clear to avoid stupid blocks.
        Clears a 2.0m buffer at the start/end of regions so there is always a navigable approach.
        """
        if region_idx < 0 or region_idx >= len(self.walkable_regions):
            return False
        x_min, x_max, y_min, y_max = self.walkable_regions[region_idx]
        buffer_m = 2.0
        # Determine if this region is first, last, or middle in the corridor chain
        is_first = region_idx == 0
        is_last = region_idx == len(self.walkable_regions) - 1
        # Keep a clear lane near both ends for middle regions; for first keep near start; for last keep near goal
        near_start = (x - x_min) < buffer_m
        near_end = (x_max - (x + size)) < buffer_m
        if is_first and near_start:
            return True
        if is_last and near_end:
            return True
        if (not is_first and near_start) or (not is_last and near_end):
            return True
        return False
    
    def _generate_narrow_passage_pattern(self) -> List:
        """Generate narrow passage pattern (wall-like obstacles creating narrow gaps)."""
        obstacles = []
        
        if not self.walkable_regions:
            return obstacles
        
        # Use first walkable region (standard corridor)
        region = self.walkable_regions[0]
        x_min, x_max, y_min, y_max = region
        
        corridor_width = y_max - y_min
        corridor_length = x_max - x_min
        
        # Create 2-4 narrow sections
        num_sections = np.random.randint(2, 5)
        section_positions = np.linspace(x_min + 5, x_max - 5, num_sections)
        
        for x_pos in section_positions:
            # Randomly decide gap position (left, center, or right)
            gap_position = np.random.choice(["left", "center", "right"])
            gap_size = np.random.uniform(1.5, 2.5)  # Gap width
            
            if gap_position == "left":
                # Gap on left, obstacles on right
                obstacles.append([x_pos - 1.0, y_min + 0.3, 2.0, corridor_width - gap_size - 0.3])
            elif gap_position == "center":
                # Obstacles on top and bottom
                gap_y_center = (y_min + y_max) / 2
                gap_y_bottom = gap_y_center - gap_size / 2
                gap_y_top = gap_y_center + gap_size / 2
                
                obstacles.append([x_pos - 1.0, y_min + 0.3, 2.0, gap_y_bottom - y_min - 0.3])
                obstacles.append([x_pos - 1.0, gap_y_top, 2.0, y_max - gap_y_top - 0.3])
            else:  # right
                # Gap on right, obstacles on left
                obstacles.append([x_pos - 1.0, y_min + gap_size + 0.3, 2.0, corridor_width - gap_size - 0.3])
        
        return obstacles
    
    def _generate_zigzag_pattern(self) -> List:
        """Generate zigzag pattern (alternating obstacles from top and bottom)."""
        obstacles = []
        
        if not self.walkable_regions:
            return obstacles
        
        # Use first walkable region (standard corridor)
        region = self.walkable_regions[0]
        x_min, x_max, y_min, y_max = region
        
        corridor_width = y_max - y_min
        
        # Create 3-5 alternating obstacles
        num_obstacles = np.random.randint(3, 6)
        x_positions = np.linspace(x_min + 4, x_max - 4, num_obstacles)
        
        alternate_top = np.random.random() > 0.5  # Start from top or bottom
        
        for i, x_pos in enumerate(x_positions):
            if (i % 2 == 0) == alternate_top:
                # Obstacle from top
                height = np.random.uniform(corridor_width * 0.5, corridor_width * 0.7)
                obstacles.append([x_pos - 1.0, y_max - height, 2.0, height - 0.3])
            else:
                # Obstacle from bottom
                height = np.random.uniform(corridor_width * 0.5, corridor_width * 0.7)
                obstacles.append([x_pos - 1.0, y_min + 0.3, 2.0, height - 0.3])
        
        return obstacles

    def _generate_standard_realistic_obstacles(self, num_obstacles: int) -> List:
        """For standard corridors: place some obstacles along the path centerline to force decisions,
        but always keep a lateral gap; also keep start/end buffers clear.
        """
        obstacles = []
        if not self.walkable_regions:
            return obstacles
        x_min, x_max, y_min, y_max = self.walkable_regions[0]
        corridor_center_y = (y_min + y_max) / 2
        width = y_max - y_min
        length = x_max - x_min

        # Target obstacle count: moderate, ensure navigability
        target = max(6, min(14, num_obstacles))
        xs = np.linspace(x_min + 4.0, x_max - 4.0, target)
        np.random.shuffle(xs)

        for i, x_pos in enumerate(xs[:target]):
            pattern = i % 4
            # Ensure a minimum lateral gap
            min_gap = max(1.6, width * 0.22)
            block_height = np.random.uniform(min_gap + 0.6, width * 0.65)
            if pattern == 0:
                # Block center-left
                y0 = corridor_center_y - block_height
                obstacles.append([x_pos - 0.2, max(y_min + 0.4, y0), 1.2, block_height])
            elif pattern == 1:
                # Block center-right
                y0 = corridor_center_y
                obstacles.append([x_pos - 0.2, y0, 1.2, min(y_max - 0.4 - y0, block_height)])
            elif pattern == 2:
                # Narrow pair leaving a center gap
                pair_h = max(0.8, (block_height - min_gap) / 2)
                obstacles.append([x_pos - 0.2, y_min + 0.4, 1.1, pair_h])
                obstacles.append([x_pos - 0.2, y_max - 0.4 - pair_h, 1.1, pair_h])
            else:
                # Small offset block to force slight lateral move
                offset = np.random.uniform(-width * 0.2, width * 0.2)
                y0 = np.clip(corridor_center_y + offset - 0.9, y_min + 0.4, y_max - 1.8)
                obstacles.append([x_pos - 0.2, y0, 1.0, 1.8])

        # Filter out anything violating connection clear zones
        filtered = []
        for ob in obstacles:
            if not self._in_connection_clear_zone(0, ob[0], ob[1], ob[2]):
                filtered.append(ob)
        return filtered

    def _check_obstacle_overlap(self, new_obs: List, existing_obs: List, margin: float = 0.5) -> bool:
        """Check obstacle overlap."""
        for obs in existing_obs:
            if (new_obs[0] < obs[0] + obs[2] + margin and
                new_obs[0] + new_obs[2] + margin > obs[0] and
                new_obs[1] < obs[1] + obs[3] + margin and
                new_obs[1] + new_obs[3] + margin > obs[1]):
                return True
        return False

    def _find_valid_position(self, region_idx: int = None) -> Tuple[float, float]:
        """Find valid position in specified region or any region."""
        if region_idx is None:
            # Try all regions
            for _ in range(100):
                region = self.walkable_regions[np.random.randint(len(self.walkable_regions))]
                x_min, x_max, y_min, y_max = region
                x = np.random.uniform(x_min + 1, x_max - 1)
                y = np.random.uniform(y_min + 1, y_max - 1)
                if not self._check_collision_at_position(x, y):
                    return (x, y)
        else:
            region = self.walkable_regions[region_idx]
            x_min, x_max, y_min, y_max = region
            for _ in range(50):
                x = np.random.uniform(x_min + 1, x_max - 1)
                y = np.random.uniform(y_min + 1, y_max - 1)
                if not self._check_collision_at_position(x, y):
                    return (x, y)
        
        # Fallback
        region = self.walkable_regions[0]
        return ((region[0] + region[1]) / 2, (region[2] + region[3]) / 2)

    def _check_collision_at_position(self, x: float, y: float) -> bool:
        """Check collision."""
        if not self._is_valid_position(x, y):
            return True
        
        for obs in self.obstacles:
            obs_x, obs_y, obs_w, obs_h = obs
            closest_x = max(obs_x, min(x, obs_x + obs_w))
            closest_y = max(obs_y, min(y, obs_y + obs_h))
            distance = math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
            if distance < self.AGENT_RADIUS:
                return True
        return False

    def _get_nearest_obstacle_info(self) -> Tuple[float, float]:
        """Get nearest obstacle info."""
        min_dist = 100.0
        nearest_angle = 0.0
        x, y = self.agent_pos
        
        for obs in self.obstacles:
            obs_x, obs_y, obs_w, obs_h = obs
            obs_center_x = obs_x + obs_w / 2
            obs_center_y = obs_y + obs_h / 2
            dist = math.sqrt((x - obs_center_x)**2 + (y - obs_center_y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_angle = math.atan2(obs_center_y - y, obs_center_x - x)
        
        return min_dist, nearest_angle
    
    def _raycast_to_obstacle(self, start_x: float, start_y: float, angle: float, ray_range: float = None) -> float:
        """Cast a ray from start position at given angle, return distance to nearest obstacle/wall.
        
        CRITICAL FIX: Skips internal junction walls for L/T shapes to allow seeing around corners.
        
        Args:
            start_x, start_y: Ray start position
            angle: Ray angle in radians
            ray_range: Maximum ray range (defaults to MAX_RAY_RANGE, increased for L/T shapes)
        
        Returns distance in meters, or ray_range if nothing hit.
        """
        # Adaptive ray range: longer for L/T shapes to see around corners
        if ray_range is None:
            if self.current_corridor_type in ["lshaped", "tshaped"]:
                ray_range = 20.0  # Increased from 12.0 for better corner visibility
            else:
                ray_range = self.MAX_RAY_RANGE
        
        # Ray endpoint
        end_x = start_x + ray_range * math.cos(angle)
        end_y = start_y + ray_range * math.sin(angle)
        
        min_dist = ray_range
        
        # CRITICAL FIX: Identify internal junction walls to skip them
        # For L/T shapes, we need to skip walls that are between regions (not external)
        internal_walls = self._get_internal_junction_walls()
        
        # Check intersection with walls (walkable region boundaries)
        for region_idx, (x_min, x_max, y_min, y_max) in enumerate(self.walkable_regions):
            # Skip small overlap regions (corner/junction) - these are not walls!
            if (x_max - x_min) < 2.0 and (y_max - y_min) < 2.0:
                continue  # This is a junction overlap region, not a wall boundary
            
            # Define walls for this region
            walls = [
                ("left", x_min, y_min, x_min, y_max),
                ("right", x_max, y_min, x_max, y_max),
                ("bottom", x_min, y_min, x_max, y_min),
                ("top", x_min, y_max, x_max, y_max),
            ]
            
            # Check each wall
            for wall_name, wx1, wy1, wx2, wy2 in walls:
                # CRITICAL FIX: Skip internal junction walls
                wall_key = (wx1, wy1, wx2, wy2)
                if wall_key in internal_walls:
                    continue  # Skip this internal wall - allow rays to pass through
                
                # Check intersection
                if self._ray_line_intersection(start_x, start_y, end_x, end_y, wx1, wy1, wx2, wy2):
                    t = self._ray_wall_intersection_t(start_x, start_y, end_x, end_y, wx1, wy1, wx2, wy2)
                    if 0 <= t <= 1:
                        hit_x = start_x + t * (end_x - start_x)
                        hit_y = start_y + t * (end_y - start_y)
                        dist = math.sqrt((hit_x - start_x)**2 + (hit_y - start_y)**2)
                        if dist < min_dist:
                            min_dist = dist
        
        # Check intersection with obstacles
        for obs in self.obstacles:
            obs_x, obs_y, obs_w, obs_h = obs
            # Check each edge of obstacle rectangle
            edges = [
                (obs_x, obs_y, obs_x + obs_w, obs_y),  # Bottom
                (obs_x + obs_w, obs_y, obs_x + obs_w, obs_y + obs_h),  # Right
                (obs_x + obs_w, obs_y + obs_h, obs_x, obs_y + obs_h),  # Top
                (obs_x, obs_y + obs_h, obs_x, obs_y),  # Left
            ]
            for edge in edges:
                if self._ray_line_intersection(start_x, start_y, end_x, end_y, edge[0], edge[1], edge[2], edge[3]):
                    t = self._ray_wall_intersection_t(start_x, start_y, end_x, end_y, edge[0], edge[1], edge[2], edge[3])
                    if 0 <= t <= 1:
                        hit_x = start_x + t * (end_x - start_x)
                        hit_y = start_y + t * (end_y - start_y)
                        dist = math.sqrt((hit_x - start_x)**2 + (hit_y - start_y)**2)
                        if dist < min_dist:
                            min_dist = dist
        
        return min_dist
    
    def _get_internal_junction_walls(self) -> set:
        """Identify internal junction walls that should NOT block rays.
        
        CRITICAL FIX: For L/T shapes, rays should pass through internal junction walls
        to allow seeing around corners. This function identifies which walls are internal.
        
        Returns:
            Set of wall tuples (x1, y1, x2, y2) that are internal junction walls.
        """
        internal_walls = set()
        
        if self.current_corridor_type == "lshaped":
            # L-shape: horizontal segment connects to vertical segment at corner
            # The "right wall" of horizontal segment and "bottom wall" of vertical segment
            # are internal (junction) walls - rays should pass through
            
            if len(self.walkable_regions) >= 2:
                # Horizontal segment (region 0)
                h_x_min, h_x_max, h_y_min, h_y_max = self.walkable_regions[0]
                # Vertical segment (region 1)
                v_x_min, v_x_max, v_y_min, v_y_max = self.walkable_regions[1]
                
                # Right wall of horizontal segment (internal junction)
                # This is where horizontal connects to vertical
                internal_walls.add((h_x_max, h_y_min, h_x_max, h_y_max))
                
                # Bottom wall of vertical segment (internal junction)
                # This is where vertical connects to horizontal
                internal_walls.add((v_x_min, v_y_min, v_x_max, v_y_min))
        
        elif self.current_corridor_type == "tshaped":
            # T-shape: vertical stem connects to horizontal bar at junction
            # The "top wall" of stem and "bottom wall" of bar are internal (junction) walls
            
            if len(self.walkable_regions) >= 2:
                # Find stem (vertical) and bar (horizontal) regions
                stem_region = None
                bar_region = None
                
                for region in self.walkable_regions:
                    x_min, x_max, y_min, y_max = region
                    # Skip small overlap regions
                    if (x_max - x_min) < 2.0 and (y_max - y_min) < 2.0:
                        continue
                    # Stem is narrower (vertical)
                    if (x_max - x_min) < (y_max - y_min):
                        stem_region = region
                    # Bar is wider (horizontal)
                    elif (x_max - x_min) > (y_max - y_min):
                        bar_region = region
                
                if stem_region and bar_region:
                    s_x_min, s_x_max, s_y_min, s_y_max = stem_region
                    b_x_min, b_x_max, b_y_min, b_y_max = bar_region
                    
                    # Top wall of stem (internal junction)
                    internal_walls.add((s_x_min, s_y_max, s_x_max, s_y_max))
                    
                    # Bottom wall of bar (internal junction) - the entire bottom wall
                    internal_walls.add((b_x_min, b_y_min, b_x_max, b_y_min))
        
        # For standard corridors, no internal walls
        # (all walls are external)
        
        return internal_walls
    
    def _ray_line_intersection(self, r1_x, r1_y, r2_x, r2_y, l1_x, l1_y, l2_x, l2_y) -> bool:
        """Check if ray (r1->r2) intersects line segment (l1->l2)."""
        # Ray direction
        dr_x = r2_x - r1_x
        dr_y = r2_y - r1_y
        # Line segment direction
        ds_x = l2_x - l1_x
        ds_y = l2_y - l1_y
        
        # Cross product
        cross = dr_x * ds_y - dr_y * ds_x
        if abs(cross) < 1e-6:
            return False  # Parallel
        
        # Vector from ray start to line start
        diff_x = l1_x - r1_x
        diff_y = l1_y - r1_y
        
        t = (diff_x * ds_y - diff_y * ds_x) / cross
        s = (diff_x * dr_y - diff_y * dr_x) / cross
        
        return t >= 0 and 0 <= s <= 1
    
    def _ray_wall_intersection_t(self, r1_x, r1_y, r2_x, r2_y, l1_x, l1_y, l2_x, l2_y) -> float:
        """Get intersection parameter t (0-1) for ray hitting line segment."""
        dr_x = r2_x - r1_x
        dr_y = r2_y - r1_y
        ds_x = l2_x - l1_x
        ds_y = l2_y - l1_y
        
        cross = dr_x * ds_y - dr_y * ds_x
        if abs(cross) < 1e-6:
            return 1.0
        
        diff_x = l1_x - r1_x
        diff_y = l1_y - r1_y
        
        t = (diff_x * ds_y - diff_y * ds_x) / cross
        return t
    
    def _get_raycast_distances(self) -> np.ndarray:
        """Get distance measurements from 36 rays (Lidar-like).
        IMPROVED: Better handling for L/T corner regions + goal direction detection.
        """
        ray_distances = np.zeros(self.N_RAYS, dtype=np.float32)
        x, y = self.agent_pos
        
        # Calculate goal direction for enhanced raycasting
        goal_direction = self.goal_pos - self.agent_pos
        goal_angle = math.atan2(goal_direction[1], goal_direction[0])
        goal_dist = np.linalg.norm(goal_direction)
        
        # Adaptive ray range: longer for L/T shapes to see around corners
        if self.current_corridor_type in ["lshaped", "tshaped"]:
            ray_range = 20.0  # Increased from 12.0 for better corner visibility
        else:
            ray_range = self.MAX_RAY_RANGE
        
        # Check if agent is in corner/junction region (for L/T shapes)
        in_corner_region = False
        if self.current_corridor_type in ["lshaped", "tshaped"]:
            # Check if agent is near corner/junction (within 2m)
            for region in self.walkable_regions:
                x_min, x_max, y_min, y_max = region
                # Check if this is a small overlap region (corner/junction)
                if (x_max - x_min) < 2.0 and (y_max - y_min) < 2.0:
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        in_corner_region = True
                        break
        
        # Cast rays in all directions (0 to 2π)
        for i in range(self.N_RAYS):
            angle = (2 * math.pi * i / self.N_RAYS) + self.agent_heading
            dist = self._raycast_to_obstacle(x, y, angle, ray_range)
            
            # ENHANCEMENT: If ray points toward goal, show "clear path" distance
            angle_to_goal_relative = angle - goal_angle
            angle_to_goal_relative = math.atan2(math.sin(angle_to_goal_relative), math.cos(angle_to_goal_relative))
            angle_diff = abs(angle_to_goal_relative)
            
            # If ray is pointing toward goal (within 10 degrees), enhance it
            if angle_diff < 0.17:  # ~10 degrees
                # Show distance to goal if no obstacle blocks it
                if dist > goal_dist * 0.8:  # If obstacle is far (80% of goal distance)
                    # Ray can "see" toward goal - use goal distance as reference
                    dist = min(dist, goal_dist + 2.0)  # Extend ray to show goal direction
            
            # If in corner region, slightly extend rays that might hit corner boundaries
            if in_corner_region and dist < 1.0:
                # If ray is close to goal direction, extend it slightly
                if angle_diff < 0.5:  # Within ~28 degrees of goal
                    dist = min(dist + 0.5, ray_range)  # Extend by 0.5m
            
            ray_distances[i] = dist
        
        return ray_distances

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with random corridor type and configuration."""
        super().reset(seed=seed)
        
        # Select corridor type and define regions
        self._select_corridor_type()
        self._define_walkable_regions()
        
        # Generate obstacles
        self.obstacles = self._generate_obstacles()
        
        # Place start and goal at corridor entrance/exit for realism
        if len(self.walkable_regions) >= 1:
            # Start near beginning of first region, centered in width
            sx, sy = self._find_region_end_position(0, end="start")
            self.start_pos = np.array((sx, sy), dtype=np.float32)
            # Goal near end of last region, centered in width
            gx, gy = self._find_region_end_position(len(self.walkable_regions) - 1, end="end")
            self.goal_pos = np.array((gx, gy), dtype=np.float32)
        else:
            self.start_pos = np.array(self._find_valid_position(), dtype=np.float32)
            self.goal_pos = np.array(self._find_valid_position(), dtype=np.float32)
        
        self.agent_pos = self.start_pos.copy()
        self.agent_vel = np.zeros(2, dtype=np.float32)
        
        dx = self.goal_pos[0] - self.agent_pos[0]
        dy = self.goal_pos[1] - self.agent_pos[1]
        self.agent_heading = math.atan2(dy, dx)
        
        self.steps = 0
        # Update max_steps based on current difficulty (may have changed)
        self.max_steps = self._get_max_steps_for_difficulty()
        self.total_distance_traveled = 0.0
        self.min_distance_to_goal = self._distance_to_goal()
        self.previous_distance_to_goal = self.min_distance_to_goal
        self.collision_count = 0
        self.consecutive_collisions = 0
        self.steps_without_progress = 0
        self.last_positions = []  # Reset position tracking
        self.state_visit_count = {}  # Reset state visitation tracking
        
        # RESEARCH-BASED: Reset action smoothing filters (prevents momentum from previous episode)
        self.prev_lin_vel = 0.0
        self.prev_ang_vel = 0.0
        
        return self._get_observation(), self._get_info()

    def _find_region_end_position(self, region_idx: int, end: str = "start") -> Tuple[float, float]:
        """Pick a position near the start/end of a region, centered in width, avoiding collisions."""
        region = self.walkable_regions[region_idx]
        x_min, x_max, y_min, y_max = region
        x_buffer = 1.0
        y_center = (y_min + y_max) / 2
        # Try a few candidates near the chosen end
        for _ in range(50):
            if end == "start":
                x = np.random.uniform(x_min + 0.5, x_min + x_buffer + 0.5)
            else:
                x = np.random.uniform(x_max - x_buffer - 0.5, x_max - 0.5)
            # small vertical jitter
            y = np.random.uniform(max(y_min + 0.5, y_center - 0.8), min(y_max - 0.5, y_center + 0.8))
            if not self._check_collision_at_position(x, y):
                return (x, y)
        # Fallback: use generic finder
        return self._find_valid_position(region_idx)

    def _get_observation(self) -> np.ndarray:
        """Get observation with raycasting (Lidar-like) + enhanced features.
        
        ENHANCED: Added corner awareness, goal visibility, and path length estimate
        to help agent navigate L/T shapes better.
        """
        dist_to_goal = self._distance_to_goal()
        angle_to_goal = self._angle_to_goal()
        nearest_obs_dist, nearest_obs_angle = self._get_nearest_obstacle_info()
        
        # Get raycast distances (36 rays) - NOW FIXED to see around corners
        ray_distances = self._get_raycast_distances()
        
        # ENHANCED FEATURES (2025 approach): Add semantic features
        corner_awareness = self._compute_corner_awareness()
        goal_visible = self._is_goal_visible()
        path_length_estimate = self._estimate_path_length()
        
        # Base observation (11 values) + Raycasting (36 values) + Enhanced (3 values) = 50 values
        base_obs = np.array([
            self.agent_pos[0], self.agent_pos[1],
            self.agent_vel[0], self.agent_vel[1],
            self.goal_pos[0], self.goal_pos[1],
            dist_to_goal, angle_to_goal, self.agent_heading,
            nearest_obs_dist, nearest_obs_angle,
        ], dtype=np.float32)
        
        enhanced_features = np.array([
            corner_awareness,
            goal_visible,
            path_length_estimate,
        ], dtype=np.float32)
        
        return np.concatenate([base_obs, ray_distances, enhanced_features])

    def _distance_to_goal(self) -> float:
        return np.linalg.norm(self.agent_pos - self.goal_pos)

    def _angle_to_goal(self) -> float:
        dx, dy = self.goal_pos - self.agent_pos
        goal_angle = math.atan2(dy, dx)
        angle_diff = goal_angle - self.agent_heading
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        return angle_diff
    
    def _compute_corner_awareness(self) -> float:
        """Detect if agent is near a corner/junction (L/T shapes).
        
        Returns 1.0 if agent is in corner/junction region, 0.0 otherwise.
        """
        if self.current_corridor_type not in ["lshaped", "tshaped"]:
            return 0.0
        
        # Check if agent is in small overlap region (corner/junction)
        for x_min, x_max, y_min, y_max in self.walkable_regions:
            # Small regions (< 2m) are corner/junction overlap regions
            if (x_max - x_min) < 2.0 and (y_max - y_min) < 2.0:
                if (x_min <= self.agent_pos[0] <= x_max and
                    y_min <= self.agent_pos[1] <= y_max):
                    return 1.0  # At corner/junction
        
        return 0.0  # Not at corner
    
    def _is_goal_visible(self) -> float:
        """Check if goal is directly visible (no walls/obstacles blocking line of sight).
        
        Returns 1.0 if goal is visible, 0.0 if blocked.
        """
        dist_to_goal = self._distance_to_goal()
        if dist_to_goal < 0.1:
            return 1.0  # Already at goal
        
        # Calculate angle to goal
        angle_to_goal = math.atan2(
            self.goal_pos[1] - self.agent_pos[1],
            self.goal_pos[0] - self.agent_pos[0]
        )
        
        # Cast ray toward goal (with small margin)
        ray_dist = self._raycast_to_obstacle(
            self.agent_pos[0], self.agent_pos[1],
            angle_to_goal, ray_range=dist_to_goal + 2.0
        )
        
        # If ray reaches goal distance (within 5% margin), goal is visible
        return 1.0 if ray_dist >= dist_to_goal * 0.95 else 0.0
    
    def _estimate_path_length(self) -> float:
        """Estimate path length to goal accounting for corridor structure.
        
        For L/T shapes, uses Manhattan distance with corner awareness.
        For standard, uses Euclidean distance.
        """
        dist_to_goal = self._distance_to_goal()
        
        if self.current_corridor_type == "standard":
            # Straight corridor: Euclidean distance is accurate
            return dist_to_goal
        
        elif self.current_corridor_type == "lshaped":
            # L-shape: Estimate path through corner
            # If goal is in different region, add corner penalty
            agent_region = self._get_agent_region()
            goal_region = self._get_goal_region()
            
            if agent_region != goal_region and agent_region is not None and goal_region is not None:
                # Goal is in different section, must go through corner
                # Use Manhattan distance as estimate
                dx = abs(self.goal_pos[0] - self.agent_pos[0])
                dy = abs(self.goal_pos[1] - self.agent_pos[1])
                return dx + dy
            else:
                # Same region or unknown: use Euclidean
                return dist_to_goal
        
        elif self.current_corridor_type == "tshaped":
            # T-shape: Similar to L-shape
            agent_region = self._get_agent_region()
            goal_region = self._get_goal_region()
            
            if agent_region != goal_region and agent_region is not None and goal_region is not None:
                dx = abs(self.goal_pos[0] - self.agent_pos[0])
                dy = abs(self.goal_pos[1] - self.agent_pos[1])
                return dx + dy
            else:
                return dist_to_goal
        
        # Default: Euclidean distance
        return dist_to_goal
    
    def _get_agent_region(self) -> int:
        """Get which walkable region the agent is currently in.
        
        Returns region index (0, 1, 2, ...) or None if not in any region.
        """
        for idx, (x_min, x_max, y_min, y_max) in enumerate(self.walkable_regions):
            if (x_min <= self.agent_pos[0] <= x_max and
                y_min <= self.agent_pos[1] <= y_max):
                return idx
        return None
    
    def _get_goal_region(self) -> int:
        """Get which walkable region the goal is in.
        
        Returns region index (0, 1, 2, ...) or None if not in any region.
        """
        for idx, (x_min, x_max, y_min, y_max) in enumerate(self.walkable_regions):
            if (x_min <= self.goal_pos[0] <= x_max and
                y_min <= self.goal_pos[1] <= y_max):
                return idx
        return None

    def _get_info(self) -> Dict[str, Any]:
        return {
            "distance_to_goal": self._distance_to_goal(),
            "total_distance": self.total_distance_traveled,
            "collisions": self.collision_count,
            "time_elapsed": self.steps * self.DT,
            "velocity_magnitude": np.linalg.norm(self.agent_vel),
            "corridor_type": self.current_corridor_type,
            "obstacle_density": self.current_obstacle_density,
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step."""
        self.steps += 1
        
        # Store action for reward calculation
        self.last_action = action.copy()
        
        # RESEARCH-BASED: Smooth action processing to prevent jerky movement
        # Based on 2025 research: smooth transitions improve learning and realism
        
        # Remove backward movement entirely for realistic pedestrian behavior
        raw_linear = float(np.clip(action[0], 0.0, self.MAX_VELOCITY))  # Forward only
        # Low-pass filter LINEAR velocity for smooth acceleration/deceleration
        self.prev_lin_vel = self.action_smooth_alpha_lin * raw_linear + (1 - self.action_smooth_alpha_lin) * self.prev_lin_vel
        desired_linear = self.prev_lin_vel
        
        # Low-pass filter angular velocity to avoid rapid spin reversals
        raw_ang = float(np.clip(action[1], -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL))
        self.prev_ang_vel = self.action_smooth_alpha_ang * raw_ang + (1 - self.action_smooth_alpha_ang) * self.prev_ang_vel
        desired_angular_vel = self.prev_ang_vel
        
        self.agent_heading += desired_angular_vel * self.DT
        self.agent_heading = math.atan2(math.sin(self.agent_heading), math.cos(self.agent_heading))
        
        # compute translational velocity along current heading
        desired_vel = np.array([
            desired_linear * math.cos(self.agent_heading),
            desired_linear * math.sin(self.agent_heading),
        ])
        
        vel_diff = desired_vel - self.agent_vel
        max_vel_change = self.MAX_ACCELERATION * self.DT
        vel_diff_mag = np.linalg.norm(vel_diff)
        if vel_diff_mag > max_vel_change:
            vel_diff *= max_vel_change / vel_diff_mag
        self.agent_vel += vel_diff
        
        old_pos = self.agent_pos.copy()
        new_pos = self.agent_pos + self.agent_vel * self.DT
        
        # Check boundary violations FIRST (before obstacle collision)
        if not self._is_valid_position(new_pos[0], new_pos[1]):
            # Agent would go outside corridor - clip to boundary
            new_pos = np.array(self._clip_position_to_bounds(new_pos[0], new_pos[1]))
            # Moderate damping to prevent escape but allow movement along boundary
            self.agent_vel *= 0.6  # Less aggressive damping
        
        collision = False
        if self._check_collision_at_position(new_pos[0], new_pos[1]):
            collision = True
            self.collision_count += 1
            self.consecutive_collisions += 1
            
            # IMPROVED COLLISION RESPONSE: Better handling for edge cases
            self.agent_pos = old_pos  # Move back to previous position
            
            # Adaptive damping based on consecutive collisions
            if self.consecutive_collisions >= 3:
                # Heavy damping for repeated collisions (stuck at edge)
                self.agent_vel *= 0.2
                # Strong angular nudge to escape
                self.agent_heading += np.random.uniform(-0.8, 0.8)
            elif self.consecutive_collisions >= 2:
                # Moderate damping and nudge
                self.agent_vel *= 0.3
                self.agent_heading += np.random.uniform(-0.5, 0.5)
            else:
                # Light damping for first collision
                self.agent_vel *= 0.5
            
            # Normalize heading
            self.agent_heading = math.atan2(math.sin(self.agent_heading), 
                                          math.cos(self.agent_heading))
        else:
            self.agent_pos = new_pos
            self.consecutive_collisions = 0  # Reset if no collision
        
        distance_moved = np.linalg.norm(self.agent_pos - old_pos)
        self.total_distance_traveled += distance_moved
        
        current_dist = self._distance_to_goal()
        
        # Track position history for diversity penalty
        self.last_positions.append(self.agent_pos.copy())
        if len(self.last_positions) > 20:
            self.last_positions.pop(0)
        
        # IMPROVED STUCK DETECTION: More lenient threshold to avoid false positives
        if self.previous_distance_to_goal is not None:
            # Check both distance progress AND position change
            dist_change = abs(current_dist - self.previous_distance_to_goal)
            pos_change = np.linalg.norm(self.agent_pos - old_pos) if hasattr(self, 'last_positions') and len(self.last_positions) > 0 else 0.0
            
            # Consider stuck if: no distance progress AND minimal position change
            if dist_change < 0.05 and pos_change < 0.1:  # More lenient: 0.05m distance, 0.1m position
                self.steps_without_progress += 1
            else:
                self.steps_without_progress = 0
        
        # IMPROVED ESCAPE MECHANISM: More aggressive when stuck, but NOT near goal
        # CRITICAL FIX: Don't apply escape mechanism when close to goal (might cause running away)
        if self.steps_without_progress > 20 and current_dist > 3.0:  # Only escape if NOT near goal
            # Check if we're near an obstacle (likely stuck at edge)
            nearest_obs_dist, _ = self._get_nearest_obstacle_info()
            if nearest_obs_dist < 1.0:  # Very close to obstacle
                # Strong escape: turn away from obstacle, but TOWARD goal if possible
                goal_direction = self.goal_pos - self.agent_pos
                goal_angle = math.atan2(goal_direction[1], goal_direction[0])
                # Prefer turning toward goal rather than random
                angle_to_goal = goal_angle - self.agent_heading
                angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal))
                exploration_nudge = angle_to_goal * 0.5 + np.random.uniform(-0.3, 0.3)  # Bias toward goal
                self.agent_heading += exploration_nudge
                # Also add small forward push
                self.agent_vel += 0.3 * np.array([math.cos(self.agent_heading), math.sin(self.agent_heading)])
            else:
                # General stuck: turn toward goal
                goal_direction = self.goal_pos - self.agent_pos
                goal_angle = math.atan2(goal_direction[1], goal_direction[0])
                angle_to_goal = goal_angle - self.agent_heading
                angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal))
                exploration_nudge = angle_to_goal * 0.3 + np.random.uniform(-0.2, 0.2)  # Bias toward goal
                self.agent_heading += exploration_nudge
            
            self.agent_heading = math.atan2(math.sin(self.agent_heading), 
                                            math.cos(self.agent_heading))
            # Reset counter to prevent continuous nudging
            self.steps_without_progress = 10  # Reset to lower value (was 15)
        # If close to goal (< 3m), DON'T apply escape mechanism - let agent reach goal naturally
        
        self.min_distance_to_goal = min(self.min_distance_to_goal, current_dist)
        
        reward = self._calculate_reward(collision, current_dist, distance_moved)
        
        terminated, truncated = False, False
        info = self._get_info()
        
        # IMPROVED GOAL DETECTION: More lenient threshold for better success rates
        goal_reach_threshold = self.AGENT_RADIUS * 3.5  # Increased to 0.7875m (more lenient)
        
        # Check if agent is aligned with goal (heading towards it)
        goal_direction = self.goal_pos - self.agent_pos
        goal_angle = math.atan2(goal_direction[1], goal_direction[0])
        heading_diff = abs(goal_angle - self.agent_heading)
        heading_diff = min(heading_diff, 2 * math.pi - heading_diff)  # Wrap to [0, π]
        is_aligned = heading_diff < 0.8  # More lenient: within ~45 degrees (was 0.5 = 28 degrees)
        
        if current_dist < goal_reach_threshold:
            # Terminate if reasonably aligned OR very close (no strict requirement)
            if is_aligned or current_dist < self.AGENT_RADIUS * 2.5:
                terminated = True
                info["goal_reached"] = True
                reward += 1000.0  # Increased success bonus (was 500.0)
            else:
                # Close but not aligned - still give strong reward to encourage reaching
                if current_dist < self.AGENT_RADIUS * 4.0:
                    reward += 200.0  # Bonus for being very close
                info["goal_reached"] = False
        else:
            info["goal_reached"] = False
        
        if self.steps >= self.max_steps:
            truncated = True
        
        if self.collision_count >= 50:
            truncated = True
            reward -= 50.0
        
        self.previous_distance_to_goal = current_dist
        
        return self._get_observation(), reward, terminated, truncated, info

    def _calculate_reward(self, collision: bool, dist_to_goal: float, distance_moved: float) -> float:
        """Calculate reward with improved penalties for realistic behavior and stagnation prevention."""
        reward = 0.0
        vel_magnitude = np.linalg.norm(self.agent_vel)
        
        # 1. PROGRESS REWARD (FIXED: Reduced to prevent oscillation)
        # CRITICAL FIX: Strong progress rewards (100.0) cause oscillation near goal
        # Research shows: distance rewards must be balanced with step penalties
        if self.previous_distance_to_goal is not None:
            progress = self.previous_distance_to_goal - dist_to_goal
            # REDUCED from 100.0 to 10.0 to prevent reward hacking via oscillation
            reward += progress * 10.0  # Much weaker to prevent local optima
            
            # PROGRESS VELOCITY BONUS - reward fast progress
            if progress > 0:
                progress_velocity = progress / self.DT
                reward += progress_velocity * 20.0  # Increased from 10.0
            
            # REDUCED PENALTY for moving away - only when far from goal
            # Don't penalize heavily when close (might be avoiding obstacle)
            if progress < -0.1:
                if dist_to_goal > 5.0:  # Only penalize when far from goal
                    reward -= abs(progress) * 50.0  # REDUCED from 200.0 to 50.0
                elif dist_to_goal > 2.0:  # Medium distance
                    reward -= abs(progress) * 25.0  # Lighter penalty
                # No penalty when very close (< 2m) - agent might be maneuvering
            
            # LIGHT penalty for moving away when very close (might be fine-tuning)
            if progress < -0.05 and dist_to_goal < 1.0:
                reward -= 5.0  # REDUCED from 30.0 to 5.0 (was too harsh)
        
        # 2. DISTANCE-BASED REWARD (REDUCED to prevent oscillation)
        # CRITICAL FIX: Strong distance rewards (20.0) contribute to oscillation
        # Reduced to prevent reward hacking
        reward += 5.0 / (1.0 + dist_to_goal)  # REDUCED from 20.0 to 5.0
        
        # GOAL PROXIMITY BONUS (FIXED: Reduced to prevent oscillation)
        # CRITICAL FIX: Strong proximity bonus (20.0) encourages staying near goal
        # Reduced to prevent "honey pot" trap
        if dist_to_goal < 3.0:  # Reduced range from 5.0 to 3.0
            proximity_bonus = (3.0 - dist_to_goal) * 5.0  # REDUCED: Up to 15 bonus (was 100)
            reward += proximity_bonus
        
        # CRITICAL FIX: Anti-oscillation near goal
        # Instead of extreme attraction (which causes oscillation), use entry bonus
        if dist_to_goal < 1.0:  # Within 1.0m of goal
            # Reward moving TOWARD goal, but penalize moving AWAY
            goal_direction = self.goal_pos - self.agent_pos
            goal_dir_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-6)
            forward_component = np.dot(self.agent_vel, goal_dir_norm)
            if forward_component > 0:
                # Reward approaching goal
                reward += forward_component * 20.0  # Moderate bonus (was 50.0)
            else:
                # HEAVY penalty for moving away when very close (prevents oscillation)
                reward -= abs(forward_component) * 50.0  # Strong penalty for backing away
        
        # GOAL ALIGNMENT BONUS (STRENGTHENED - reward facing the goal)
        if dist_to_goal < 5.0:  # Extended range from 3.0 to 5.0
            goal_direction = self.goal_pos - self.agent_pos
            goal_angle = math.atan2(goal_direction[1], goal_direction[0])
            heading_diff = abs(goal_angle - self.agent_heading)
            heading_diff = min(heading_diff, 2 * math.pi - heading_diff)
            if heading_diff < 0.5:  # More lenient: within ~28 degrees (was 0.3 = 17 degrees)
                alignment_bonus = (0.5 - heading_diff) * 40.0  # STRENGTHENED: Up to 20 bonus (was 6)
                reward += alignment_bonus
        
        # 3. FORWARD MOVEMENT BONUS (STRENGTHENED)
        goal_direction = self.goal_pos - self.agent_pos
        goal_dir_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-6)
        forward_component = np.dot(self.agent_vel, goal_dir_norm)
        if forward_component > 0:
            reward += forward_component * 10.0  # STRENGTHENED from 3.0 to 10.0
        else:
            # Penalize moving away from goal (but lighter than before)
            reward -= abs(forward_component) * 2.0  # Light penalty for backward movement
        
        # 4. VELOCITY EFFICIENCY
        if 0.8 < vel_magnitude < 1.3:
            reward += 1.0
        elif vel_magnitude < 0.2:
            reward -= 0.5  # Penalize being too slow
        
        # 5. STEP PENALTY (CRITICAL FIX: Strong penalty to prevent oscillation)
        # Research shows: Without step penalty, agent learns to oscillate near goal
        # This is the PRIMARY cause of "going away then coming back" behavior
        reward -= 0.1  # STRONG step penalty (was 0.02) - prevents reward hacking
        
        # 6. COLLISION PENALTY (RESEARCH-BASED: Much stronger for safety)
        # Research shows: collision penalties must be strong enough to learn avoidance
        if collision:
            # Base penalty: Much stronger (was 20.0, now 50.0)
            base_penalty = 50.0
            
            # Progressive penalty: more collisions = exponentially bigger penalty
            collision_multiplier = 1.0 + (self.collision_count / 5.0)  # Faster escalation (was /10.0)
            reward -= base_penalty * collision_multiplier
            
            # EXTRA PENALTY for consecutive collisions (crash pattern = unrealistic)
            if self.consecutive_collisions >= 2:  # Trigger earlier (was 3)
                reward -= 10.0 * (self.consecutive_collisions - 1)  # Stronger escalation
                
            # EXTRA PENALTY for straight-line crashes (high velocity + collision = bad)
            if vel_magnitude > 0.8:
                reward -= 15.0  # Much stronger (was 5.0)
            
            # SAFETY CONSTRAINT: Hard penalty for dangerous behavior
            if vel_magnitude > 1.2:  # Very high speed collision
                reward -= 30.0  # Additional heavy penalty
        
        # 7. SPINNING PENALTY (NEW - penalize excessive rotation)
        angular_val = self.prev_ang_vel
        angular_vel_magnitude = abs(angular_val)
        # Get angular velocity from action or heading change
        if hasattr(self, 'last_action') and self.last_action is not None:
            ang_cmd = abs(self.last_action[1]) if len(self.last_action) > 1 else 0
            if ang_cmd > 2.0:  # More than 2 rad/s (114 deg/s) is excessive
                reward -= 2.0 * (ang_cmd - 2.0)  # Penalty proportional to command
            # Penalize spinning in place (high angular vel + low forward vel)
            if ang_cmd > 1.5 and vel_magnitude < 0.3:
                reward -= 5.0  # Heavy penalty for spinning in place
        
        # 8. BACKWARD MOVEMENT PENALTY (NEW - penalize going backward)
        forward_vel = self.agent_vel[0] if len(self.agent_vel) > 0 else 0
        if forward_vel < -0.2:  # Moving backward
            reward -= 2.0  # Stronger penalty for backward movement
        
        # 9. STALLING PENALTY (INCREASED - penalize being stuck)
        if vel_magnitude < 0.3 and dist_to_goal > 5.0:  # Changed threshold from 0.1 to 0.3
            # Progressive penalty: longer stuck = bigger penalty
            stalling_multiplier = 1.0 + (self.steps_without_progress / 50.0)
            reward -= 2.0 * stalling_multiplier  # Increased from 0.5 to 2.0
        
        # 10. NO PROGRESS PENALTY (INCREASED, UNCAPPED)
        if self.steps_without_progress > 10:  # Trigger earlier (changed from 20 to 10)
            # Progressive, uncapped penalty
            penalty = 0.5 * (self.steps_without_progress - 10)  # Removed cap!
            reward -= penalty
        
        # 11. SLOW MOVEMENT PENALTY (NEW - penalize slow but not stopped)
        if 0.1 < vel_magnitude < 0.5 and dist_to_goal > 5.0:
            reward -= 1.0  # Penalize slow movement (not just stopped)
        
        # 12. EXPLORATION BONUS (NEW - reward trying new directions when stuck)
        if self.steps_without_progress > 15:
            # If agent is making angular movement (trying to turn), reward it
            if hasattr(self, 'last_action') and self.last_action is not None:
                ang_cmd = abs(self.last_action[1]) if len(self.last_action) > 1 else 0
                if ang_cmd > 0.5:  # Trying to turn
                    reward += 1.0  # Small bonus for exploration
        
        # 13. POSITION DIVERSITY PENALTY (STRENGTHENED - prevent oscillation)
        # CRITICAL FIX: Weak penalty (2.0) can't compete with strong progress rewards
        if len(self.last_positions) >= 10:
            # Calculate how much agent has moved
            position_variance = np.var([p[0] for p in self.last_positions]) + \
                               np.var([p[1] for p in self.last_positions])
            if position_variance < 0.5:  # Very little movement
                # STRONGER penalty, especially near goal where oscillation is worst
                diversity_penalty = 10.0 if dist_to_goal < 3.0 else 5.0
                reward -= diversity_penalty
            
            # DETECT OSCILLATION: Check if agent is moving back and forth
            if len(self.last_positions) >= 5:
                recent_positions = self.last_positions[-5:]
                # Check if positions are oscillating (alternating pattern)
                x_diffs = [recent_positions[i+1][0] - recent_positions[i][0] for i in range(len(recent_positions)-1)]
                y_diffs = [recent_positions[i+1][1] - recent_positions[i][1] for i in range(len(recent_positions)-1)]
                
                # If signs alternate, it's oscillating
                x_signs = [np.sign(d) for d in x_diffs if abs(d) > 0.01]  # Filter tiny movements
                y_signs = [np.sign(d) for d in y_diffs if abs(d) > 0.01]
                x_oscillating = len(set(x_signs)) > 1 and abs(sum(x_diffs)) < 0.3 if x_signs else False
                y_oscillating = len(set(y_signs)) > 1 and abs(sum(y_diffs)) < 0.3 if y_signs else False
                
                # CRITICAL FIX: Oscillation penalty must work NEAR GOAL too!
                # Original condition (dist_to_goal > 1.0) missed oscillation at goal
                if x_oscillating or y_oscillating:
                    # Stronger penalty, especially near goal (where oscillation is most harmful)
                    oscillation_penalty = 10.0 if dist_to_goal < 2.0 else 5.0
                    reward -= oscillation_penalty

        # 14. WALL PROXIMITY PENALTY (DISABLED NEAR GOAL) - discourage hugging walls
        # CRITICAL FIX: Don't penalize wall proximity when near goal (goal might be near wall)
        x, y = self.agent_pos
        if dist_to_goal > 2.0:  # Only apply penalty when NOT near goal
            for (x_min, x_max, y_min, y_max) in self.walkable_regions:
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    min_wall_dist = min(x - x_min, x_max - x, y - y_min, y_max - y)
                    # Activated at 0.6m, but reduced strength
                    if min_wall_dist < 0.6:
                        reward -= (0.6 - min_wall_dist) * 3.0  # REDUCED from 5.0 to 3.0
                    break
        # When within 2m of goal, NO wall proximity penalty (goal might be near wall)
        
        # 15. GOAL PROXIMITY COLLISION PENALTY (NEW) - heavily penalize collisions near the goal
        if collision and dist_to_goal < self.AGENT_RADIUS * 5: # Within 5 agent radii of goal
            reward -= 100.0 # Much heavier penalty (was 50.0)
        
        # 16. INTRINSIC MOTIVATION: Novelty Bonus (RESEARCH-BASED)
        # Based on 2024-2025 research: VAE-based novelty + state visitation for exploration
        # Simplified version: state visitation count for exploration bonus
        state_key = (round(self.agent_pos[0], 1), round(self.agent_pos[1], 1))
        visit_count = self.state_visit_count.get(state_key, 0)
        self.state_visit_count[state_key] = visit_count + 1
        
        # Exploration bonus: reward visiting new states (information-theoretic)
        if visit_count == 0:  # First visit to this state
            reward += 0.5  # Small exploration bonus
        elif visit_count < 5:  # Rarely visited
            reward += 0.1  # Smaller bonus
        # No bonus for frequently visited states (encourages exploration)
        
        # 17. POTENTIAL-BASED REWARD SHAPING (FIXED: Reduced to prevent oscillation)
        # CRITICAL FIX: Potential shaping (5.0 scale) adds to oscillation problem
        # Keep it but reduce weight to prevent reward hacking
        if self.previous_distance_to_goal is not None:
            # Potential function: negative distance to goal
            potential_current = -dist_to_goal
            potential_previous = -self.previous_distance_to_goal
            # Potential-based shaping: γ*φ(s') - φ(s)
            gamma = 0.99  # Discount factor
            potential_shaping = gamma * potential_current - potential_previous
            reward += potential_shaping * 1.0  # REDUCED from 5.0 to 1.0 (prevent oscillation)
        
        # TIER 1 UPGRADE: Rescale entire reward by 1/20 (0.05) for better stability
        # Research shows: Large reward ranges (1000.0 goal) cause value function explosion
        # Modern best practice: Goal ~10-50, step penalty ~0.01-0.05, progress ~0.1-1.0
        # This scaling brings: Goal 1000→50, Step -0.1→-0.005, Collision -50→-2.5
        reward = reward * 0.05
        
        return reward

    def render(self):
        """Render environment."""
        if self.render_mode is None:
            return None
        
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        self.ax.clear()
        
        # Draw corridor based on type with seamless corner connections
        if self.current_corridor_type in ["lshaped", "tshaped"]:
            # For L/T shapes: draw main regions first, then explicitly fill corner/junction
            main_regions = []
            corner_region = None
            
            for region in self.walkable_regions:
                x_min, x_max, y_min, y_max = region
                # Check if this is a small overlap region (corner/junction)
                if (x_max - x_min) < 2.0 and (y_max - y_min) < 2.0:
                    corner_region = region
                else:
                    main_regions.append(region)
            
            # Draw main regions with seamless edges (no edgecolor on inner sides)
            for region in main_regions:
                x_min, x_max, y_min, y_max = region
                # Draw with edgecolor only on outer walls
                self.ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                           facecolor="#F5F5DC", edgecolor="#8B7355", 
                                           linewidth=2.5, alpha=0.95, zorder=0))
            
            # Explicitly fill corner/junction region to ensure seamless connection
            if corner_region:
                x_min, x_max, y_min, y_max = corner_region
                self.ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                           facecolor="#F5F5DC", edgecolor="none", 
                                           zorder=0))
            
            # For L-shapes: explicitly connect the corner
            if self.current_corridor_type == "lshaped" and len(main_regions) >= 2:
                r1, r2 = main_regions[0], main_regions[1]
                r1_x_min, r1_x_max, r1_y_min, r1_y_max = r1
                r2_x_min, r2_x_max, r2_y_min, r2_y_max = r2
                # Find overlapping corner region and fill it
                corner_x_min = max(r1_x_min, r2_x_min)
                corner_x_max = min(r1_x_max, r2_x_max)
                corner_y_min = max(r1_y_min, r2_y_min)
                corner_y_max = min(r1_y_max, r2_y_max)
                if corner_x_min < corner_x_max and corner_y_min < corner_y_max:
                    corner_fill = Rectangle((corner_x_min, corner_y_min), 
                                           corner_x_max - corner_x_min, 
                                           corner_y_max - corner_y_min,
                                           facecolor="#F5F5DC", edgecolor="none", 
                                           zorder=0)
                    self.ax.add_patch(corner_fill)
            
            # For T-shapes: explicitly connect the junction
            elif self.current_corridor_type == "tshaped" and len(main_regions) >= 2:
                r1, r2 = main_regions[0], main_regions[1]
                r1_x_min, r1_x_max, r1_y_min, r1_y_max = r1
                r2_x_min, r2_x_max, r2_y_min, r2_y_max = r2
                # Find overlapping junction region and fill it
                junction_x_min = max(r1_x_min, r2_x_min)
                junction_x_max = min(r1_x_max, r2_x_max)
                junction_y_min = max(r1_y_min, r2_y_min)
                junction_y_max = min(r1_y_max, r2_y_max)
                if junction_x_min < junction_x_max and junction_y_min < junction_y_max:
                    junction_fill = Rectangle((junction_x_min, junction_y_min), 
                                              junction_x_max - junction_x_min, 
                                              junction_y_max - junction_y_min,
                                              facecolor="#F5F5DC", edgecolor="none", 
                                              zorder=0)
                    self.ax.add_patch(junction_fill)
        else:
            # Standard corridors: simple rendering
            for region in self.walkable_regions:
                x_min, x_max, y_min, y_max = region
                self.ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                           facecolor="#F5F5DC", edgecolor="#8B7355", 
                                           linewidth=2.5, alpha=0.95, zorder=0))
        
        # Draw obstacles
        for obs in self.obstacles:
            x, y, w, h = obs
            self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                            facecolor="#A0522D", edgecolor="black", alpha=0.8))
        
        # Draw goal
        self.ax.add_patch(Circle(self.goal_pos, 0.675, facecolor="#00FF00",
                                edgecolor="darkgreen", linewidth=3, alpha=0.7))
        
        # Draw agent
        self.ax.add_patch(Circle(self.agent_pos, self.AGENT_RADIUS, facecolor="#4169E1",
                                edgecolor="darkblue", linewidth=2, alpha=0.9))
        
        # Draw raycasting (Lidar-like visualization)
        ray_distances = self._get_raycast_distances()
        for i in range(self.N_RAYS):
            angle = (2 * math.pi * i / self.N_RAYS) + self.agent_heading
            dist = ray_distances[i]
            end_x = self.agent_pos[0] + dist * math.cos(angle)
            end_y = self.agent_pos[1] + dist * math.sin(angle)
            # Color based on distance (green = far, red = close)
            if dist < 2.0:
                ray_color = "red"
                alpha = 0.8
            elif dist < 5.0:
                ray_color = "orange"
                alpha = 0.5
            else:
                ray_color = "green"
                alpha = 0.3
            self.ax.plot([self.agent_pos[0], end_x], [self.agent_pos[1], end_y],
                       color=ray_color, linewidth=1, alpha=alpha, zorder=1)
        
        # Draw heading arrow (more prominent)
        hx = self.agent_pos[0] + 0.6 * math.cos(self.agent_heading)
        hy = self.agent_pos[1] + 0.6 * math.sin(self.agent_heading)
        self.ax.arrow(self.agent_pos[0], self.agent_pos[1], hx - self.agent_pos[0], hy - self.agent_pos[1],
                     head_width=0.25, head_length=0.2, fc="yellow", ec="darkorange", linewidth=3, zorder=10)
        
        # Set limits dynamically
        all_x = [r[0] for r in self.walkable_regions] + [r[1] for r in self.walkable_regions]
        all_y = [r[2] for r in self.walkable_regions] + [r[3] for r in self.walkable_regions]
        self.ax.set_xlim(min(all_x) - 2, max(all_x) + 2)
        self.ax.set_ylim(min(all_y) - 2, max(all_y) + 2)
        self.ax.set_aspect("equal")
        
        title = (f"{self.current_corridor_type.upper()} | Time: {self.steps * self.DT:.1f}s | "
                f"Density: {self.current_obstacle_density:.2f}")
        self.ax.set_title(title, fontsize=13, fontweight="bold")
        
        if self.render_mode == "human":
            plt.pause(0.01)
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
