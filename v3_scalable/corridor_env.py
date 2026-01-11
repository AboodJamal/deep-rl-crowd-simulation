"""
REALISTIC Multi-Agent Corridor Navigation Environment
=====================================================

Key Features:
1. Agents CONTINUE after collision (don't die) - realistic pedestrian behavior
2. Visual raycasting for debugging
3. Long rays (12-16m) for early obstacle detection
4. Agents detect gaps and navigate through them
5. Realistic agent-agent avoidance (move aside, not stop)
6. Generalized obstacle placement (random positions/sizes)
7. Support for 3-10 agents
8. Smooth rotation and movement
9. Metrics: AllSuccess, AnySuccess, PerAgentSuccess

Author: GitHub Copilot
Date: December 2025
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Agent state with all properties."""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0  # Radians, 0 = facing right
    prev_heading: float = 0.0  # Track previous heading for spin detection
    total_rotation: float = 0.0  # Accumulated rotation to detect spinning
    reached_goal: bool = False
    goal_time: int = -1  # Step when goal was reached (-1 if not reached)
    wall_collisions: int = 0
    obstacle_collisions: int = 0
    agent_collisions: int = 0
    total_distance_traveled: float = 0.0
    prev_x: float = 0.0
    prev_y: float = 0.0


@dataclass
class Obstacle:
    """Obstacle definition."""
    x: float  # Center x
    y: float  # Center y
    width: float
    height: float
    attached_to: str = "none"  # "top", "bottom", "none"


class RealisticCorridorEnv(gym.Env):
    """
    Realistic multi-agent corridor environment.
    
    Key Design Principles:
    1. Collisions DON'T kill agents - they learn to recover
    2. Long raycasting for early detection
    3. Smooth movement with momentum
    4. Generalized scenarios with randomization
    5. Works with 3-10 agents
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        num_agents: int = 3,
        scenario: str = "straight",
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        randomize_scenario: bool = True,  # NEW: randomize obstacle positions
    ):
        super().__init__()
        
        self.num_agents = num_agents
        self.scenario = scenario
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.randomize_scenario = randomize_scenario
        self.current_step = 0
        
        # ============ CORRIDOR DIMENSIONS ============
        self.corridor_length = 50.0  # X dimension
        self.corridor_height = 12.0  # Y dimension
        
        # ============ AGENT PROPERTIES ============
        self.agent_radius = 0.4  # Slightly larger for better visibility
        self.max_speed = 3.0  # FASTER! (was 1.5)
        self.max_angular_velocity = 4.0  # Faster turning (was 2.5)
        self.acceleration = 0.7  # Faster acceleration (was 0.4)
        self.friction = 0.05  # Less friction for smoother movement
        self.dt = 0.1  # Time step
        self.collision_substeps = 5  # For continuous collision detection
        
        # ============ RAYCASTING - LONG RANGE! ============
        self.num_rays = 32  # More rays for better coverage
        self.ray_length = 16.0  # VERY LONG rays to see obstacles early!
        self.ray_fov = 2 * np.pi  # Full 360 degree vision
        
        # ============ GOAL PROPERTIES ============
        self.goal_radius = 2.0  # Area to count as "reached"
        self.goal_x = self.corridor_length - 4.0  # Goal on right side
        
        # ============ PERSONAL SPACE (realistic crowd behavior) ============
        self.personal_space = 1.0  # Agents uncomfortable when closer than this
        
        # ============ COLLISION PENALTIES (NOT DEATH!) ============
        self.wall_collision_penalty = -10.0  # STRONG - walls should be avoided!
        self.obstacle_collision_penalty = -8.0  # STRONG - obstacles should be avoided!
        self.agent_collision_penalty = -1.0  # LIGHT - bumping is natural in crowds
        self.personal_space_penalty = -0.5  # Mild discomfort when too close
        
        # ============ OBSERVATION SPACE ============
        # Per agent:
        # - Position (2): x, y normalized
        # - Velocity (2): vx, vy normalized
        # - Heading (2): sin, cos
        # - Goal info (3): direction unit vector (2) + distance (1)
        # - Rays (32): obstacle distances
        # - Other agents (closest 4): relative pos + vel (4*4=16)
        # Total: 2+2+2+3+32+16 = 57
        self.obs_dim = 57
        
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # ============ ACTION SPACE ============
        # [linear_velocity, angular_velocity] - intuitive control
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # ============ STATE ============
        self.agents: List[AgentState] = []
        self.obstacles: List[Obstacle] = []
        self.goal_positions: List[Tuple[float, float]] = []
        
        # ============ METRICS ============
        self.episode_metrics = {}
        
    def reset(self, seed=None, options=None):
        """Reset environment with optional randomization."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        
        # Generate obstacles based on scenario (with randomization)
        self.obstacles = self._generate_obstacles()
        
        # Initialize agents
        self._initialize_agents()
        
        # Reset metrics
        self.episode_metrics = {
            "wall_collisions": 0,
            "obstacle_collisions": 0,
            "agent_collisions": 0,
            "agents_reached_goal": 0,
            "total_steps": 0,
        }
        
        observations = self._get_all_observations()
        info = {"scenario": self.scenario, "num_obstacles": len(self.obstacles)}
        
        return observations, info
    
    def _generate_obstacles(self) -> List[Obstacle]:
        """
        Generate obstacles based on scenario with randomization.
        
        IMPROVED: 
        - Obstacles are always within corridor bounds
        - Strategic placement to force navigation decisions
        - Always leave navigable paths
        """
        obstacles = []
        
        # Safety margins
        margin = 2.0  # Minimum distance from corridor edges
        min_gap = 3.5  # Minimum gap for agents to pass through
        
        if self.scenario == "straight":
            # Empty corridor - but add a small challenge obstacle sometimes
            if self.randomize_scenario and np.random.random() < 0.3:
                # Small central obstacle to force slight navigation
                x = np.random.uniform(0.35, 0.65) * self.corridor_length
                width = np.random.uniform(1.5, 2.5)
                height = np.random.uniform(2.0, 4.0)
                y = np.random.uniform(0.3, 0.7) * self.corridor_height
                y = np.clip(y, height/2 + margin, self.corridor_height - height/2 - margin)
                obstacles.append(Obstacle(x=x, y=y, width=width, height=height, attached_to="none"))
        
        elif self.scenario == "single_wall":
            # Single wall obstacle - forces agents to go around
            attached_to = np.random.choice(["top", "bottom"]) if self.randomize_scenario else "bottom"
            
            # Position in middle third of corridor (force early detection/decision)
            x_pos = np.random.uniform(0.35, 0.55) * self.corridor_length if self.randomize_scenario else 0.45 * self.corridor_length
            
            # Height leaves enough gap for agents
            max_height = self.corridor_height - min_gap
            height_ratio = np.random.uniform(0.5, 0.7) if self.randomize_scenario else 0.6
            height = min(height_ratio * self.corridor_height, max_height)
            
            # Width - thick enough to be a real obstacle
            width = np.random.uniform(2.0, 4.0) if self.randomize_scenario else 3.0
            
            if attached_to == "bottom":
                y_pos = height / 2
            else:
                y_pos = self.corridor_height - height / 2
            
            obstacles.append(Obstacle(
                x=x_pos, y=y_pos, width=width, height=height, attached_to=attached_to
            ))
        
        elif self.scenario == "double_wall":
            # Two walls creating an S-curve path - STRATEGIC PLACEMENT
            # First wall at ~35% X, attached to bottom
            x1 = np.random.uniform(0.30, 0.40) * self.corridor_length if self.randomize_scenario else 0.35 * self.corridor_length
            h1 = np.random.uniform(0.45, 0.55) * self.corridor_height if self.randomize_scenario else 0.5 * self.corridor_height
            obstacles.append(Obstacle(
                x=x1, y=h1/2, width=3.0, height=h1, attached_to="bottom"
            ))
            
            # Second wall at ~65% X, attached to top
            x2 = np.random.uniform(0.60, 0.70) * self.corridor_length if self.randomize_scenario else 0.65 * self.corridor_length
            h2 = np.random.uniform(0.45, 0.55) * self.corridor_height if self.randomize_scenario else 0.5 * self.corridor_height
            obstacles.append(Obstacle(
                x=x2, y=self.corridor_height - h2/2, width=3.0, height=h2, attached_to="top"
            ))
        
        elif self.scenario == "narrow_passage":
            # Two walls creating a narrow gap - agents must coordinate
            x_pos = np.random.uniform(0.40, 0.55) * self.corridor_length if self.randomize_scenario else 0.48 * self.corridor_length
            
            # Gap size - challenging but passable
            gap_size = np.random.uniform(3.5, 5.0) if self.randomize_scenario else 4.0
            gap_center = np.random.uniform(0.4, 0.6) * self.corridor_height if self.randomize_scenario else 0.5 * self.corridor_height
            
            width = np.random.uniform(3.0, 5.0) if self.randomize_scenario else 4.0
            
            # Bottom wall
            h_bottom = gap_center - gap_size/2
            if h_bottom > 1.0:
                obstacles.append(Obstacle(
                    x=x_pos, y=h_bottom/2, width=width, height=h_bottom, attached_to="bottom"
                ))
            
            # Top wall
            h_top = self.corridor_height - (gap_center + gap_size/2)
            if h_top > 1.0:
                obstacles.append(Obstacle(
                    x=x_pos, y=self.corridor_height - h_top/2, width=width, height=h_top, attached_to="top"
                ))
        
        elif self.scenario == "curves":
            # Curved path - staggered obstacles
            num_obs = np.random.randint(2, 4) if self.randomize_scenario else 3
            for i in range(num_obs):
                x = 0.25 * (i + 1) * self.corridor_length
                attached = "bottom" if i % 2 == 0 else "top"
                height = np.random.uniform(0.35, 0.45) * self.corridor_height
                width = np.random.uniform(2.0, 3.5)
                
                if attached == "bottom":
                    y = height / 2
                else:
                    y = self.corridor_height - height / 2
                
                obstacles.append(Obstacle(x=x, y=y, width=width, height=height, attached_to=attached))
        
        elif self.scenario == "mixed":
            # Random mix of obstacles - but ensure navigability
            num_obstacles = np.random.randint(2, 5)
            positions_used = []
            
            for _ in range(num_obstacles):
                obs = self._create_random_obstacle_safe(positions_used)
                if obs is not None:
                    obstacles.append(obs)
                    positions_used.append((obs.x, obs.y, obs.width, obs.height))
        
        return obstacles
    
    def _create_random_obstacle_safe(self, existing_positions) -> Optional[Obstacle]:
        """Create a random obstacle that doesn't overlap with existing ones."""
        for attempt in range(10):  # Try up to 10 times
            # Ensure obstacle is within corridor bounds
            width = np.random.uniform(2.0, 4.0)
            height = np.random.uniform(2.5, 5.0)
            
            attached_to = np.random.choice(["top", "bottom", "none"])
            
            # X position - middle portion of corridor
            x = np.random.uniform(0.2, 0.75) * self.corridor_length
            
            # Y position based on attachment
            margin = 1.0
            if attached_to == "bottom":
                y = height / 2
            elif attached_to == "top":
                y = self.corridor_height - height / 2
            else:
                min_y = height/2 + margin
                max_y = self.corridor_height - height/2 - margin
                if min_y < max_y:
                    y = np.random.uniform(min_y, max_y)
                else:
                    y = self.corridor_height / 2
            
            # Check for overlap with existing obstacles
            overlap = False
            for ex, ey, ew, eh in existing_positions:
                if abs(x - ex) < (width + ew) / 2 + 2.0:  # Add spacing
                    if abs(y - ey) < (height + eh) / 2 + 2.0:
                        overlap = True
                        break
            
            if not overlap:
                return Obstacle(x=x, y=y, width=width, height=height, attached_to=attached_to)
        
        return None
    
    def _create_random_obstacle(self, size="medium") -> Obstacle:
        """Create a random obstacle (legacy function for compatibility)."""
        if size == "small":
            width = np.random.uniform(1.0, 2.0)
            height = np.random.uniform(1.5, 3.0)
        else:
            width = np.random.uniform(2.0, 4.0)
            height = np.random.uniform(3.0, 6.0)
        
        attached_to = np.random.choice(["top", "bottom", "none"])
        x = np.random.uniform(0.25, 0.70) * self.corridor_length  # Keep away from spawn and goal
        
        if attached_to == "bottom":
            y = height / 2
        elif attached_to == "top":
            y = self.corridor_height - height / 2
        else:
            min_y = height/2 + 1
            max_y = self.corridor_height - height/2 - 1
            y = np.random.uniform(min_y, max_y) if min_y < max_y else self.corridor_height / 2
        
        return Obstacle(x=x, y=y, width=width, height=height, attached_to=attached_to)
    
    def _initialize_agents(self):
        """Initialize agents at starting positions."""
        self.agents = []
        self.goal_positions = []
        
        # Spawn on left side, spread vertically
        spawn_x_min = 2.0
        spawn_x_max = 6.0
        
        # Calculate vertical spacing
        margin = 1.5  # Distance from walls for agent spawn
        # Goal margin must account for goal_radius to prevent goals from being too close to walls
        goal_margin = max(margin, self.goal_radius + 0.5)  # Ensure goal_radius fits + small buffer
        available_height = self.corridor_height - 2 * margin
        goal_available_height = self.corridor_height - 2 * goal_margin
        
        for i in range(self.num_agents):
            # Spread agents vertically
            if self.num_agents > 1:
                y = margin + (available_height * i / (self.num_agents - 1))
            else:
                y = self.corridor_height / 2
            
            # Add small randomization
            y += np.random.uniform(-0.3, 0.3)
            y = np.clip(y, margin, self.corridor_height - margin)
            
            x = np.random.uniform(spawn_x_min, spawn_x_max)
            
            agent = AgentState(
                x=x, y=y, vx=0.0, vy=0.0, heading=0.0,  # Face right
                prev_x=x, prev_y=y
            )
            self.agents.append(agent)
            
            # Goal position: ensure goals are within playable area accounting for goal_radius
            if self.num_agents > 1:
                goal_y = goal_margin + (goal_available_height * i / (self.num_agents - 1))
            else:
                goal_y = self.corridor_height / 2
            # Clip to ensure goal_radius fits within corridor
            goal_y = np.clip(goal_y, goal_margin, self.corridor_height - goal_margin)
            self.goal_positions.append((self.goal_x, goal_y))
    
    def _get_all_observations(self) -> List[np.ndarray]:
        """Get observations for all agents."""
        return [self._get_observation(i) for i in range(self.num_agents)]
    
    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """Get observation for a single agent."""
        agent = self.agents[agent_idx]
        goal = self.goal_positions[agent_idx]
        
        obs = []
        
        # === Position (2) ===
        obs.append(agent.x / self.corridor_length)  # Normalized
        obs.append(agent.y / self.corridor_height)
        
        # === Velocity (2) ===
        obs.append(agent.vx / self.max_speed)
        obs.append(agent.vy / self.max_speed)
        
        # === Heading (2) ===
        obs.append(np.sin(agent.heading))
        obs.append(np.cos(agent.heading))
        
        # === Goal info (3) ===
        dx = goal[0] - agent.x
        dy = goal[1] - agent.y
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0.01:
            obs.append(dx / dist)  # Unit vector to goal
            obs.append(dy / dist)
        else:
            obs.append(1.0)
            obs.append(0.0)
        obs.append(min(dist / self.corridor_length, 1.0))  # Normalized distance
        
        # === Raycasting (32) ===
        rays = self._cast_rays(agent_idx)
        obs.extend(rays)
        
        # === Other agents - closest 4 (16) ===
        other_obs = self._get_other_agents_obs(agent_idx, num_closest=4)
        obs.extend(other_obs)
        
        return np.array(obs, dtype=np.float32)
    
    def _cast_rays(self, agent_idx: int) -> List[float]:
        """Cast rays from agent to detect obstacles, walls, and other agents."""
        agent = self.agents[agent_idx]
        rays = []
        
        for i in range(self.num_rays):
            # Ray angle (relative to agent heading, full 360 degrees)
            ray_angle = agent.heading + (i / self.num_rays) * 2 * np.pi
            
            # Ray direction
            dx = np.cos(ray_angle)
            dy = np.sin(ray_angle)
            
            # Find intersection
            min_dist = self.ray_length
            
            # Check walls
            if dx > 0:  # Right wall
                t = (self.corridor_length - agent.x) / dx
                if 0 < t < min_dist:
                    min_dist = t
            elif dx < 0:  # Left wall
                t = -agent.x / dx
                if 0 < t < min_dist:
                    min_dist = t
            
            if dy > 0:  # Top wall
                t = (self.corridor_height - agent.y) / dy
                if 0 < t < min_dist:
                    min_dist = t
            elif dy < 0:  # Bottom wall
                t = -agent.y / dy
                if 0 < t < min_dist:
                    min_dist = t
            
            # Check obstacles
            for obs in self.obstacles:
                t = self._ray_box_intersection(
                    agent.x, agent.y, dx, dy,
                    obs.x - obs.width/2, obs.y - obs.height/2,
                    obs.width, obs.height
                )
                if t is not None and 0 < t < min_dist:
                    min_dist = t
            
            # Check other agents
            for j, other in enumerate(self.agents):
                if j != agent_idx:
                    t = self._ray_circle_intersection(
                        agent.x, agent.y, dx, dy,
                        other.x, other.y, self.agent_radius
                    )
                    if t is not None and 0 < t < min_dist:
                        min_dist = t
            
            # Normalize to [0, 1]
            rays.append(min_dist / self.ray_length)
        
        return rays
    
    def _ray_box_intersection(self, ox, oy, dx, dy, bx, by, bw, bh) -> Optional[float]:
        """Ray-box intersection test."""
        tmin = -np.inf
        tmax = np.inf
        
        if abs(dx) > 1e-10:
            tx1 = (bx - ox) / dx
            tx2 = (bx + bw - ox) / dx
            tmin = max(tmin, min(tx1, tx2))
            tmax = min(tmax, max(tx1, tx2))
        elif ox < bx or ox > bx + bw:
            return None
        
        if abs(dy) > 1e-10:
            ty1 = (by - oy) / dy
            ty2 = (by + bh - oy) / dy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))
        elif oy < by or oy > by + bh:
            return None
        
        if tmax >= tmin and tmax >= 0:
            return tmin if tmin >= 0 else tmax
        return None
    
    def _ray_circle_intersection(self, ox, oy, dx, dy, cx, cy, r) -> Optional[float]:
        """Ray-circle intersection test."""
        fx = ox - cx
        fy = oy - cy
        
        a = dx*dx + dy*dy
        b = 2 * (fx*dx + fy*dy)
        c = fx*fx + fy*fy - r*r
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return None
        
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)
        
        if t1 >= 0:
            return t1
        if t2 >= 0:
            return t2
        return None
    
    def _get_other_agents_obs(self, agent_idx: int, num_closest: int = 4) -> List[float]:
        """Get observations of closest other agents."""
        agent = self.agents[agent_idx]
        
        # Calculate distances to all other agents
        others = []
        for j, other in enumerate(self.agents):
            if j != agent_idx:
                dx = other.x - agent.x
                dy = other.y - agent.y
                dist = np.sqrt(dx**2 + dy**2)
                others.append((dist, j, dx, dy))
        
        # Sort by distance
        others.sort(key=lambda x: x[0])
        
        # Take closest num_closest
        obs = []
        for i in range(num_closest):
            if i < len(others):
                dist, j, dx, dy = others[i]
                other = self.agents[j]
                
                # Relative position (normalized)
                obs.append(dx / 10.0)
                obs.append(dy / 10.0)
                
                # Relative velocity (normalized)
                obs.append((other.vx - agent.vx) / self.max_speed)
                obs.append((other.vy - agent.vy) / self.max_speed)
            else:
                # Padding for non-existent agents
                obs.extend([0.0, 0.0, 0.0, 0.0])
        
        return obs
    
    def step(self, actions: List[np.ndarray]):
        """
        Execute one simulation step.
        
        CRITICAL: Collisions DON'T kill agents - they continue!
        """
        self.current_step += 1
        
        rewards = []
        
        for i, action in enumerate(actions):
            if self.agents[i].reached_goal:
                rewards.append(0.0)  # No reward for already finished agents
                continue
            
            # Apply action and physics
            reward = self._step_agent(i, action)
            rewards.append(reward)
        
        # Get observations
        observations = self._get_all_observations()
        
        # Check termination
        all_reached = all(a.reached_goal for a in self.agents)
        truncated = self.current_step >= self.max_steps
        terminated = all_reached
        
        # Compute comprehensive metrics
        info = self._compute_info()
        
        return observations, rewards, terminated, truncated, info
    
    def _step_agent(self, agent_idx: int, action: np.ndarray) -> float:
        """
        Step a single agent with CONTINUOUS COLLISION DETECTION.
        
        Obstacles are SOLID - agents CANNOT pass through them!
        Uses sub-stepping to prevent tunneling.
        
        Returns reward for this step.
        """
        agent = self.agents[agent_idx]
        goal = self.goal_positions[agent_idx]
        
        # Store previous position for distance tracking
        agent.prev_x = agent.x
        agent.prev_y = agent.y
        
        # === Parse action ===
        # action[0]: linear velocity (-1 to 1)
        # action[1]: angular velocity (-1 to 1)
        target_speed = np.clip(action[0], -1, 1) * self.max_speed
        angular_vel = np.clip(action[1], -1, 1) * self.max_angular_velocity
        
        # === Track rotation for spin detection ===
        agent.prev_heading = agent.heading
        
        # === Update heading (smooth rotation) ===
        agent.heading += angular_vel * self.dt
        agent.heading = np.arctan2(np.sin(agent.heading), np.cos(agent.heading))  # Normalize
        
        # Track total rotation (for spinning detection)
        heading_change = abs(angular_vel * self.dt)
        agent.total_rotation += heading_change
        
        # === Update velocity (with acceleration) ===
        target_vx = target_speed * np.cos(agent.heading)
        target_vy = target_speed * np.sin(agent.heading)
        
        # Smooth acceleration
        agent.vx += (target_vx - agent.vx) * self.acceleration
        agent.vy += (target_vy - agent.vy) * self.acceleration
        
        # Apply friction
        speed = np.sqrt(agent.vx**2 + agent.vy**2)
        if speed > 0.01:
            friction_factor = max(0, 1 - self.friction * self.dt)
            agent.vx *= friction_factor
            agent.vy *= friction_factor
        
        # Clamp speed
        speed = np.sqrt(agent.vx**2 + agent.vy**2)
        if speed > self.max_speed:
            agent.vx = (agent.vx / speed) * self.max_speed
            agent.vy = (agent.vy / speed) * self.max_speed
        
        # === CONTINUOUS COLLISION DETECTION with sub-stepping ===
        # This prevents agents from "tunneling" through obstacles!
        reward = 0.0
        sub_dt = self.dt / self.collision_substeps
        
        current_x = agent.x
        current_y = agent.y
        
        for substep in range(self.collision_substeps):
            # Compute sub-step movement
            new_x = current_x + agent.vx * sub_dt
            new_y = current_y + agent.vy * sub_dt
            
            # === WALL COLLISION (hard boundary) ===
            wall_collision = False
            if new_x < self.agent_radius:
                new_x = self.agent_radius
                agent.vx = abs(agent.vx) * 0.1  # Bounce with damping
                wall_collision = True
            elif new_x > self.corridor_length - self.agent_radius:
                new_x = self.corridor_length - self.agent_radius
                agent.vx = -abs(agent.vx) * 0.1
                wall_collision = True
            
            if new_y < self.agent_radius:
                new_y = self.agent_radius
                agent.vy = abs(agent.vy) * 0.1
                wall_collision = True
            elif new_y > self.corridor_height - self.agent_radius:
                new_y = self.corridor_height - self.agent_radius
                agent.vy = -abs(agent.vy) * 0.1
                wall_collision = True
            
            if wall_collision and substep == 0:  # Only count once per step
                agent.wall_collisions += 1
                self.episode_metrics["wall_collisions"] += 1
                reward += self.wall_collision_penalty
            
            # === OBSTACLE COLLISION (SOLID - cannot pass through!) ===
            for obs in self.obstacles:
                # Check if new position intersects obstacle
                if self._circle_box_collision(
                    new_x, new_y, self.agent_radius,
                    obs.x - obs.width/2, obs.y - obs.height/2, obs.width, obs.height
                ):
                    # STOP at obstacle boundary - compute exact collision point
                    collision_result = self._resolve_obstacle_collision(
                        current_x, current_y, new_x, new_y, obs
                    )
                    new_x, new_y = collision_result['pos']
                    
                    # Reflect/stop velocity based on collision normal
                    if collision_result['normal'][0] != 0:
                        agent.vx = -agent.vx * 0.1  # Bounce with heavy damping
                    if collision_result['normal'][1] != 0:
                        agent.vy = -agent.vy * 0.1
                    
                    if substep == 0:  # Only count once per step
                        agent.obstacle_collisions += 1
                        self.episode_metrics["obstacle_collisions"] += 1
                        reward += self.obstacle_collision_penalty
            
            current_x = new_x
            current_y = new_y
        
        new_x, new_y = current_x, current_y
        
        # Agent-agent collision (soft - they push each other)
        for j, other in enumerate(self.agents):
            if j != agent_idx and not other.reached_goal:
                dist = np.sqrt((new_x - other.x)**2 + (new_y - other.y)**2)
                min_dist = 2 * self.agent_radius
                if dist < min_dist:
                    # Soft push apart (both agents)
                    overlap = min_dist - dist
                    if dist > 0.01:
                        push_x = (new_x - other.x) / dist * overlap * 0.6
                        push_y = (new_y - other.y) / dist * overlap * 0.6
                        new_x += push_x
                        new_y += push_y
                    
                    agent.agent_collisions += 1
                    self.episode_metrics["agent_collisions"] += 1
                    reward += self.agent_collision_penalty
        
        # Final position update
        agent.x = np.clip(new_x, self.agent_radius, self.corridor_length - self.agent_radius)
        agent.y = np.clip(new_y, self.agent_radius, self.corridor_height - self.agent_radius)
        
        # Final check - ensure NOT inside any obstacle
        for obs in self.obstacles:
            if self._circle_box_collision(
                agent.x, agent.y, self.agent_radius,
                obs.x - obs.width/2, obs.y - obs.height/2, obs.width, obs.height
            ):
                # Emergency push out
                agent.x, agent.y = self._emergency_push_out(agent.x, agent.y, obs)
        
        # Track distance traveled
        agent.total_distance_traveled += np.sqrt(
            (agent.x - agent.prev_x)**2 + (agent.y - agent.prev_y)**2
        )
        
        # === Goal check ===
        dist_to_goal = np.sqrt((agent.x - goal[0])**2 + (agent.y - goal[1])**2)
        
        if dist_to_goal < self.goal_radius and not agent.reached_goal:
            agent.reached_goal = True
            agent.goal_time = self.current_step
            self.episode_metrics["agents_reached_goal"] += 1
            reward += 500.0  # Big goal bonus!
        
        # === Shaped rewards ===
        reward += self._compute_shaped_reward(agent_idx, dist_to_goal)
        
        return reward
    
    def _point_in_box_expanded(self, px, py, r, bx, by, bw, bh) -> bool:
        """Check if point (with radius) intersects box."""
        return (bx - r < px < bx + bw + r and by - r < py < by + bh + r)
    
    def _circle_box_collision(self, cx, cy, r, bx, by, bw, bh) -> bool:
        """
        Check if a circle collides with an axis-aligned box.
        Uses proper circle-rectangle collision detection.
        """
        # Find the closest point on the box to the circle center
        closest_x = np.clip(cx, bx, bx + bw)
        closest_y = np.clip(cy, by, by + bh)
        
        # Calculate distance from circle center to closest point
        dist_x = cx - closest_x
        dist_y = cy - closest_y
        dist_sq = dist_x * dist_x + dist_y * dist_y
        
        return dist_sq < r * r
    
    def _resolve_obstacle_collision(self, old_x, old_y, new_x, new_y, obs) -> Dict:
        """
        Resolve collision with obstacle, returning the valid position and collision normal.
        Uses swept circle-box collision detection.
        """
        # Obstacle bounds (expanded by agent radius for Minkowski sum)
        bx = obs.x - obs.width/2 - self.agent_radius
        by = obs.y - obs.height/2 - self.agent_radius
        bw = obs.width + 2 * self.agent_radius
        bh = obs.height + 2 * self.agent_radius
        
        # Find which side was hit (compute distances from new position to each edge)
        cx = obs.x
        cy = obs.y
        hw = obs.width / 2 + self.agent_radius
        hh = obs.height / 2 + self.agent_radius
        
        dist_left = new_x - (cx - hw)
        dist_right = (cx + hw) - new_x
        dist_bottom = new_y - (cy - hh)
        dist_top = (cy + hh) - new_y
        
        # Find minimum penetration direction
        min_dist = min(abs(dist_left), abs(dist_right), abs(dist_bottom), abs(dist_top))
        
        normal = [0, 0]
        resolved_x, resolved_y = new_x, new_y
        
        if abs(dist_left) == min_dist:
            resolved_x = cx - hw - 0.02
            normal = [-1, 0]
        elif abs(dist_right) == min_dist:
            resolved_x = cx + hw + 0.02
            normal = [1, 0]
        elif abs(dist_bottom) == min_dist:
            resolved_y = cy - hh - 0.02
            normal = [0, -1]
        else:
            resolved_y = cy + hh + 0.02
            normal = [0, 1]
        
        return {'pos': (resolved_x, resolved_y), 'normal': normal}
    
    def _emergency_push_out(self, x, y, obs) -> Tuple[float, float]:
        """Emergency function to push agent outside obstacle."""
        cx = obs.x
        cy = obs.y
        hw = obs.width / 2 + self.agent_radius + 0.05
        hh = obs.height / 2 + self.agent_radius + 0.05
        
        # Find closest edge and push out
        dist_left = x - (cx - hw)
        dist_right = (cx + hw) - x
        dist_bottom = y - (cy - hh)
        dist_top = (cy + hh) - y
        
        min_dist = min(abs(dist_left), abs(dist_right), abs(dist_bottom), abs(dist_top))
        
        if abs(dist_left) == min_dist:
            return cx - hw, y
        elif abs(dist_right) == min_dist:
            return cx + hw, y
        elif abs(dist_bottom) == min_dist:
            return x, cy - hh
        else:
            return x, cy + hh
    
    def _push_out_of_obstacle(self, old_x, old_y, new_x, new_y, obs) -> Tuple[float, float]:
        """Push agent out of obstacle along the shortest path."""
        # Find closest edge
        cx = obs.x
        cy = obs.y
        hw = obs.width / 2 + self.agent_radius
        hh = obs.height / 2 + self.agent_radius
        
        # Distances to each edge
        dist_left = new_x - (cx - hw)
        dist_right = (cx + hw) - new_x
        dist_bottom = new_y - (cy - hh)
        dist_top = (cy + hh) - new_y
        
        # Find minimum
        min_dist = min(abs(dist_left), abs(dist_right), abs(dist_bottom), abs(dist_top))
        
        if abs(dist_left) == min_dist:
            return cx - hw - 0.01, new_y
        elif abs(dist_right) == min_dist:
            return cx + hw + 0.01, new_y
        elif abs(dist_bottom) == min_dist:
            return new_x, cy - hh - 0.01
        else:
            return new_x, cy + hh + 0.01
    
    def _compute_shaped_reward(self, agent_idx: int, dist_to_goal: float) -> float:
        """
        Compute shaped reward for REALISTIC pedestrian behavior.
        
        KEY IMPROVEMENTS:
        1. Ray-based proactive avoidance (turn when you SEE obstacle, not when you HIT it)
        2. Anti-spinning penalty (no rotating in place!)
        3. Clear path reward (reward for having clear path ahead)
        4. Smooth goal approach (enter goal, don't spin near it)
        """
        agent = self.agents[agent_idx]
        goal = self.goal_positions[agent_idx]
        reward = 0.0
        
        # Get current rays for this agent
        rays = self._cast_rays_for_reward(agent_idx)
        
        # === 1. PROGRESS REWARD (most important!) ===
        prev_dist = np.sqrt((agent.prev_x - goal[0])**2 + (agent.prev_y - goal[1])**2)
        progress = prev_dist - dist_to_goal
        reward += progress * 80.0  # VERY STRONG progress incentive
        
        # === 2. RAY-BASED PROACTIVE AVOIDANCE ===
        # Reward for having clear path in the direction of movement
        heading_dir = np.array([np.cos(agent.heading), np.sin(agent.heading)])
        
        # Get rays in front of agent (front 90 degrees = rays around index 0)
        front_rays = []
        num_front = self.num_rays // 4  # Front quarter
        for i in range(-num_front//2, num_front//2 + 1):
            ray_idx = i % self.num_rays
            front_rays.append(rays[ray_idx])
        
        avg_front_clearance = np.mean(front_rays)  # 0 = blocked, 1 = clear
        
        # REWARD for clear path ahead, PENALTY for blocked path
        if avg_front_clearance > 0.7:
            reward += 3.0  # Good! Clear path ahead
        elif avg_front_clearance > 0.4:
            reward += 1.0  # Acceptable
        elif avg_front_clearance < 0.2:
            reward -= 2.0  # Bad! Heading toward obstacle - should turn!
        
        # === 3. PROACTIVE TURNING REWARD ===
        # If obstacle detected ahead, reward for turning toward clear side
        if avg_front_clearance < 0.5:
            # Find which side is clearer
            left_rays = rays[self.num_rays//4 : self.num_rays//2]  # Left quarter
            right_rays = rays[3*self.num_rays//4 : self.num_rays]  # Right quarter
            
            avg_left = np.mean(left_rays) if len(left_rays) > 0 else 0
            avg_right = np.mean(right_rays) if len(right_rays) > 0 else 0
            
            # Reward turning toward the clearer side
            heading_change = agent.heading - agent.prev_heading
            heading_change = np.arctan2(np.sin(heading_change), np.cos(heading_change))
            
            if avg_left > avg_right and heading_change > 0.05:
                reward += 2.0  # Good! Turning left toward clearer path
            elif avg_right > avg_left and heading_change < -0.05:
                reward += 2.0  # Good! Turning right toward clearer path
            elif abs(heading_change) > 0.1 and avg_front_clearance < 0.3:
                reward += 1.0  # Any turning is good when blocked
        
        # === 4. VELOCITY ALIGNMENT (moving toward goal) ===
        vel = np.array([agent.vx, agent.vy])
        goal_dir = np.array([goal[0] - agent.x, goal[1] - agent.y])
        goal_dist = np.linalg.norm(goal_dir)
        speed = np.sqrt(agent.vx**2 + agent.vy**2)
        
        if goal_dist > 0.1 and speed > 0.1:
            goal_dir_norm = goal_dir / goal_dist
            vel_norm = vel / speed
            vel_alignment = np.dot(vel_norm, goal_dir_norm)
            reward += vel_alignment * 10.0  # STRONG reward for moving toward goal
        
        # === 5. ANTI-SPINNING PENALTY ===
        # Penalize excessive rotation (spinning in place is BAD!)
        heading_change = abs(agent.heading - agent.prev_heading)
        heading_change = min(heading_change, 2*np.pi - heading_change)  # Handle wrap-around
        
        if heading_change > 0.3 and speed < 0.3 * self.max_speed:
            # Rotating a lot but not moving much = SPINNING!
            reward -= 3.0  # Strong penalty for spinning
        
        # === 6. SPEED REWARD (encourage forward movement) ===
        if speed > 0.5 * self.max_speed:
            reward += 2.0  # Good speed!
        elif speed > 0.3 * self.max_speed:
            reward += 1.0  # Acceptable
        elif speed < 0.1 * self.max_speed:
            reward -= 2.0  # PENALTY for standing still or spinning in place
        
        # === 7. GOAL APPROACH BONUS ===
        # Reward for entering goal area smoothly (not spinning near it)
        if dist_to_goal < self.goal_radius * 2:
            # Extra reward for moving INTO the goal (positive velocity toward goal)
            if goal_dist > 0.1:
                vel_toward_goal = np.dot(vel, goal_dir / goal_dist)
                if vel_toward_goal > 0.3:
                    reward += 8.0  # Moving into goal!
                elif vel_toward_goal < 0:
                    reward -= 3.0  # Moving away from goal while close - bad!
        
        if dist_to_goal < self.goal_radius * 1.5:
            reward += 5.0
        if dist_to_goal < self.goal_radius * 1.1:
            reward += 10.0  # Almost there!
        
        # === 8. PERSONAL SPACE (penalize being too close to others) ===
        for j, other in enumerate(self.agents):
            if j != agent_idx and not other.reached_goal:
                dist = np.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
                if dist < self.personal_space and dist > 2 * self.agent_radius:
                    reward += self.personal_space_penalty
        
        # === 9. WALL/OBSTACLE PROXIMITY (proactive avoidance) ===
        # Check minimum ray distance - if any ray is very short, we're close to something
        min_ray = min(rays)
        if min_ray < 0.1:  # Very close to obstacle (within 1.6m)
            reward -= 1.5
        elif min_ray < 0.2:  # Close to obstacle (within 3.2m)
            reward -= 0.5
        
        # === 10. TIME PENALTY ===
        reward -= 0.05
        
        return reward
    
    def _cast_rays_for_reward(self, agent_idx: int) -> List[float]:
        """Cast rays and return normalized distances for reward computation."""
        agent = self.agents[agent_idx]
        rays = []
        
        for i in range(self.num_rays):
            ray_angle = agent.heading + (i / self.num_rays) * 2 * np.pi
            dx = np.cos(ray_angle)
            dy = np.sin(ray_angle)
            
            min_dist = self.ray_length
            
            # Walls
            if dx > 0:
                t = (self.corridor_length - agent.x) / dx
                if 0 < t < min_dist:
                    min_dist = t
            elif dx < 0:
                t = -agent.x / dx
                if 0 < t < min_dist:
                    min_dist = t
            
            if dy > 0:
                t = (self.corridor_height - agent.y) / dy
                if 0 < t < min_dist:
                    min_dist = t
            elif dy < 0:
                t = -agent.y / dy
                if 0 < t < min_dist:
                    min_dist = t
            
            # Obstacles
            for obs in self.obstacles:
                t = self._ray_box_intersection(
                    agent.x, agent.y, dx, dy,
                    obs.x - obs.width/2, obs.y - obs.height/2,
                    obs.width, obs.height
                )
                if t is not None and 0 < t < min_dist:
                    min_dist = t
            
            rays.append(min_dist / self.ray_length)
        
        return rays
    
    def _compute_info(self) -> Dict:
        """Compute comprehensive episode info."""
        num_reached = sum(1 for a in self.agents if a.reached_goal)
        
        info = {
            # Success metrics
            "all_success": num_reached == self.num_agents,  # ALL agents reached
            "any_success": num_reached > 0,  # At least one reached
            "per_agent_success": num_reached / self.num_agents,  # Percentage
            "agents_reached": num_reached,
            
            # Collision metrics
            "total_wall_collisions": self.episode_metrics["wall_collisions"],
            "total_obstacle_collisions": self.episode_metrics["obstacle_collisions"],
            "total_agent_collisions": self.episode_metrics["agent_collisions"],
            
            # Per-agent details
            "agent_reached_goal": [a.reached_goal for a in self.agents],
            "agent_goal_times": [a.goal_time for a in self.agents],
            "agent_wall_collisions": [a.wall_collisions for a in self.agents],
            "agent_distances_traveled": [a.total_distance_traveled for a in self.agents],
            
            # Episode info
            "episode_length": self.current_step,
            "scenario": self.scenario,
            "num_agents": self.num_agents,
        }
        
        return info
    
    def get_agent_positions(self) -> List[Tuple[float, float]]:
        """Get all agent positions (for visualization)."""
        return [(a.x, a.y) for a in self.agents]
    
    def get_agent_headings(self) -> List[float]:
        """Get all agent headings (for visualization)."""
        return [a.heading for a in self.agents]
    
    def get_obstacles(self) -> List[Dict]:
        """Get obstacle data (for visualization)."""
        return [
            {
                "x": obs.x, "y": obs.y,
                "width": obs.width, "height": obs.height,
                "attached_to": obs.attached_to
            }
            for obs in self.obstacles
        ]
    
    def get_rays_for_visualization(self, agent_idx: int) -> List[Tuple[float, float, float, float, float]]:
        """
        Get ray data for visualization.
        Returns list of (start_x, start_y, end_x, end_y, normalized_distance)
        """
        agent = self.agents[agent_idx]
        rays = []
        
        for i in range(self.num_rays):
            ray_angle = agent.heading + (i / self.num_rays) * 2 * np.pi
            dx = np.cos(ray_angle)
            dy = np.sin(ray_angle)
            
            # Find intersection distance
            min_dist = self.ray_length
            
            # Walls
            if dx > 0:
                t = (self.corridor_length - agent.x) / dx
                if 0 < t < min_dist:
                    min_dist = t
            elif dx < 0:
                t = -agent.x / dx
                if 0 < t < min_dist:
                    min_dist = t
            
            if dy > 0:
                t = (self.corridor_height - agent.y) / dy
                if 0 < t < min_dist:
                    min_dist = t
            elif dy < 0:
                t = -agent.y / dy
                if 0 < t < min_dist:
                    min_dist = t
            
            # Obstacles
            for obs in self.obstacles:
                t = self._ray_box_intersection(
                    agent.x, agent.y, dx, dy,
                    obs.x - obs.width/2, obs.y - obs.height/2,
                    obs.width, obs.height
                )
                if t is not None and 0 < t < min_dist:
                    min_dist = t
            
            # Other agents
            for j, other in enumerate(self.agents):
                if j != agent_idx:
                    t = self._ray_circle_intersection(
                        agent.x, agent.y, dx, dy,
                        other.x, other.y, self.agent_radius
                    )
                    if t is not None and 0 < t < min_dist:
                        min_dist = t
            
            end_x = agent.x + dx * min_dist
            end_y = agent.y + dy * min_dist
            
            rays.append((agent.x, agent.y, end_x, end_y, min_dist / self.ray_length))
        
        return rays
