"""
FIXED Multi-Agent Corridor Environment
=======================================

CRITICAL FIXES:
1. SOLID obstacles - agents CANNOT pass through
2. Faster agents with better speed
3. Simpler observation space (works with simple PPO)
4. Better goal placement and detection
5. Continuous collision detection

This is a simplified version that actually works!
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class AgentState:
    """Agent state."""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0
    prev_heading: float = 0.0
    reached_goal: bool = False
    goal_time: int = -1
    wall_collisions: int = 0
    obstacle_collisions: int = 0
    agent_collisions: int = 0
    prev_x: float = 0.0
    prev_y: float = 0.0


@dataclass
class Obstacle:
    """Obstacle definition."""
    x: float
    y: float
    width: float
    height: float
    attached_to: str = "none"


class FixedCorridorEnv(gym.Env):
    """
    Fixed Multi-Agent Corridor Environment with SOLID obstacles.
    
    Key Features:
    1. Solid obstacles - continuous collision detection
    2. Faster agents (max_speed=2.5)
    3. Simple but effective observation space
    4. Works with standard PPO training
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        num_agents: int = 3,
        scenario: str = "straight",
        max_steps: int = 800,
        render_mode: Optional[str] = None,
        randomize: bool = True,
    ):
        super().__init__()
        
        self.num_agents = num_agents
        self.scenario = scenario
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.randomize = randomize
        self.current_step = 0
        
        # Corridor dimensions
        self.corridor_length = 50.0
        self.corridor_height = 12.0
        
        # Agent properties - FASTER!
        self.agent_radius = 0.35
        self.max_speed = 2.5  # Increased from 1.4
        self.max_angular_velocity = 3.5
        self.acceleration = 0.6
        self.friction = 0.08
        self.dt = 0.1
        self.collision_substeps = 4  # For solid collision
        
        # Raycasting
        self.num_rays = 24
        self.ray_length = 15.0
        
        # Goal
        self.goal_radius = 2.5
        self.goal_x = self.corridor_length - 4.0
        
        # Observation: position(2) + velocity(2) + heading(2) + goal(3) + rays(24) + others(8) = 41
        self.obs_dim = 41
        
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        self.agents: List[AgentState] = []
        self.obstacles: List[Obstacle] = []
        self.goal_positions: List[Tuple[float, float]] = []
        self.episode_metrics = {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.obstacles = self._generate_obstacles()
        self._initialize_agents()
        
        self.episode_metrics = {
            "wall_collisions": 0,
            "obstacle_collisions": 0,
            "agent_collisions": 0,
            "agents_reached_goal": 0,
        }
        
        observations = self._get_all_observations()
        return observations, {"scenario": self.scenario}
    
    def _generate_obstacles(self) -> List[Obstacle]:
        """Generate obstacles based on scenario."""
        obstacles = []
        
        if self.scenario == "straight":
            # Empty or small random obstacle
            if self.randomize and np.random.random() < 0.3:
                x = np.random.uniform(0.35, 0.55) * self.corridor_length
                obstacles.append(Obstacle(
                    x=x, y=self.corridor_height/2,
                    width=2.0, height=3.0, attached_to="none"
                ))
        
        elif self.scenario == "single_wall_bottom":
            x = np.random.uniform(0.4, 0.5) * self.corridor_length if self.randomize else 0.45 * self.corridor_length
            h = np.random.uniform(0.5, 0.65) * self.corridor_height if self.randomize else 0.55 * self.corridor_height
            obstacles.append(Obstacle(x=x, y=h/2, width=3.0, height=h, attached_to="bottom"))
        
        elif self.scenario == "single_wall_top":
            x = np.random.uniform(0.4, 0.5) * self.corridor_length if self.randomize else 0.45 * self.corridor_length
            h = np.random.uniform(0.5, 0.65) * self.corridor_height if self.randomize else 0.55 * self.corridor_height
            obstacles.append(Obstacle(x=x, y=self.corridor_height - h/2, width=3.0, height=h, attached_to="top"))
        
        elif self.scenario == "s_curve":
            x1 = np.random.uniform(0.3, 0.38) * self.corridor_length if self.randomize else 0.35 * self.corridor_length
            h1 = 0.5 * self.corridor_height
            obstacles.append(Obstacle(x=x1, y=h1/2, width=3.0, height=h1, attached_to="bottom"))
            
            x2 = np.random.uniform(0.62, 0.70) * self.corridor_length if self.randomize else 0.65 * self.corridor_length
            h2 = 0.5 * self.corridor_height
            obstacles.append(Obstacle(x=x2, y=self.corridor_height - h2/2, width=3.0, height=h2, attached_to="top"))
        
        elif self.scenario == "narrow_passage":
            x = np.random.uniform(0.42, 0.52) * self.corridor_length if self.randomize else 0.47 * self.corridor_length
            gap = np.random.uniform(4.0, 5.0) if self.randomize else 4.5
            center = self.corridor_height / 2
            
            h_bottom = center - gap/2
            if h_bottom > 1.0:
                obstacles.append(Obstacle(x=x, y=h_bottom/2, width=4.0, height=h_bottom, attached_to="bottom"))
            
            h_top = self.corridor_height - (center + gap/2)
            if h_top > 1.0:
                obstacles.append(Obstacle(x=x, y=self.corridor_height - h_top/2, width=4.0, height=h_top, attached_to="top"))
        
        elif self.scenario == "chicane":
            positions = [0.28, 0.50, 0.72]
            for i, pos in enumerate(positions):
                x = pos * self.corridor_length
                h = 0.42 * self.corridor_height
                if i % 2 == 0:
                    obstacles.append(Obstacle(x=x, y=h/2, width=2.5, height=h, attached_to="bottom"))
                else:
                    obstacles.append(Obstacle(x=x, y=self.corridor_height - h/2, width=2.5, height=h, attached_to="top"))
        
        elif self.scenario == "mixed":
            # Wall + scattered
            x = np.random.uniform(0.35, 0.45) * self.corridor_length
            h = 0.5 * self.corridor_height
            if np.random.random() < 0.5:
                obstacles.append(Obstacle(x=x, y=h/2, width=3.0, height=h, attached_to="bottom"))
            else:
                obstacles.append(Obstacle(x=x, y=self.corridor_height - h/2, width=3.0, height=h, attached_to="top"))
            
            # Add 1-2 scattered
            for _ in range(np.random.randint(1, 3)):
                sx = np.random.uniform(x + 6, self.corridor_length - 8)
                sy = np.random.uniform(3, self.corridor_height - 3)
                obstacles.append(Obstacle(x=sx, y=sy, width=2.0, height=2.5, attached_to="none"))
        
        return obstacles
    
    def _initialize_agents(self):
        """Initialize agents at starting positions."""
        self.agents = []
        self.goal_positions = []
        
        margin = 1.5
        spawn_x = (3.0, 7.0)
        # Goal margin must account for goal_radius to prevent goals from being too close to walls
        goal_margin = max(margin, self.goal_radius + 0.5)  # Ensure goal_radius fits + small buffer
        
        for i in range(self.num_agents):
            if self.num_agents > 1:
                y = margin + (self.corridor_height - 2*margin) * i / (self.num_agents - 1)
            else:
                y = self.corridor_height / 2
            
            y += np.random.uniform(-0.3, 0.3) if self.randomize else 0
            y = np.clip(y, margin, self.corridor_height - margin)
            x = np.random.uniform(*spawn_x)
            
            self.agents.append(AgentState(x=x, y=y, heading=0.0, prev_x=x, prev_y=y))
            
            # Goal positioning: ensure goals are within playable area accounting for goal_radius
            if self.num_agents > 1:
                goal_y = goal_margin + (self.corridor_height - 2*goal_margin) * i / (self.num_agents - 1)
            else:
                goal_y = self.corridor_height / 2
            # Clip to ensure goal_radius fits within corridor
            goal_y = np.clip(goal_y, goal_margin, self.corridor_height - goal_margin)
            self.goal_positions.append((self.goal_x, goal_y))
    
    def _get_all_observations(self) -> List[np.ndarray]:
        return [self._get_observation(i) for i in range(self.num_agents)]
    
    def _get_observation(self, agent_idx: int) -> np.ndarray:
        agent = self.agents[agent_idx]
        goal = self.goal_positions[agent_idx]
        
        obs = []
        
        # Position (2)
        obs.append(agent.x / self.corridor_length)
        obs.append(agent.y / self.corridor_height)
        
        # Velocity (2)
        obs.append(agent.vx / self.max_speed)
        obs.append(agent.vy / self.max_speed)
        
        # Heading (2)
        obs.append(np.sin(agent.heading))
        obs.append(np.cos(agent.heading))
        
        # Goal (3)
        dx = goal[0] - agent.x
        dy = goal[1] - agent.y
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0.01:
            obs.append(dx / dist)
            obs.append(dy / dist)
        else:
            obs.append(1.0)
            obs.append(0.0)
        obs.append(min(dist / self.corridor_length, 1.0))
        
        # Rays (24)
        rays = self._cast_rays(agent_idx)
        obs.extend(rays)
        
        # Other agents (8) - 2 closest, 4 features each
        other_obs = self._get_other_agents_obs(agent_idx)
        obs.extend(other_obs)
        
        return np.array(obs, dtype=np.float32)
    
    def _cast_rays(self, agent_idx: int) -> List[float]:
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
            
            # Other agents
            for j, other in enumerate(self.agents):
                if j != agent_idx:
                    t = self._ray_circle_intersection(
                        agent.x, agent.y, dx, dy,
                        other.x, other.y, self.agent_radius
                    )
                    if t is not None and 0 < t < min_dist:
                        min_dist = t
            
            rays.append(min_dist / self.ray_length)
        
        return rays
    
    def _ray_box_intersection(self, ox, oy, dx, dy, bx, by, bw, bh):
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
            return tmin if tmin >= 0 else None
        return None
    
    def _ray_circle_intersection(self, ox, oy, dx, dy, cx, cy, r):
        fx = ox - cx
        fy = oy - cy
        a = dx*dx + dy*dy
        b = 2 * (fx*dx + fy*dy)
        c = fx*fx + fy*fy - r*r
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        disc = np.sqrt(disc)
        t1 = (-b - disc) / (2*a)
        if t1 >= 0:
            return t1
        t2 = (-b + disc) / (2*a)
        if t2 >= 0:
            return t2
        return None
    
    def _get_other_agents_obs(self, agent_idx: int) -> List[float]:
        agent = self.agents[agent_idx]
        
        others = []
        for j, other in enumerate(self.agents):
            if j != agent_idx:
                dx = other.x - agent.x
                dy = other.y - agent.y
                dist = np.sqrt(dx**2 + dy**2)
                others.append((dist, dx, dy, other.vx - agent.vx, other.vy - agent.vy))
        
        others.sort(key=lambda x: x[0])
        
        obs = []
        for i in range(2):  # 2 closest agents
            if i < len(others):
                dist, dx, dy, dvx, dvy = others[i]
                obs.extend([dx/10, dy/10, dvx/self.max_speed, dvy/self.max_speed])
            else:
                obs.extend([0, 0, 0, 0])
        
        return obs
    
    def step(self, actions: List[np.ndarray]):
        self.current_step += 1
        
        rewards = []
        
        for i, action in enumerate(actions):
            if self.agents[i].reached_goal:
                rewards.append(0.0)
                continue
            
            reward = self._step_agent(i, action)
            rewards.append(reward)
        
        observations = self._get_all_observations()
        
        all_reached = all(a.reached_goal for a in self.agents)
        truncated = self.current_step >= self.max_steps
        terminated = all_reached
        
        info = self._compute_info()
        
        return observations, rewards, terminated, truncated, info
    
    def _step_agent(self, agent_idx: int, action: np.ndarray) -> float:
        """Step agent with SOLID collision detection."""
        agent = self.agents[agent_idx]
        goal = self.goal_positions[agent_idx]
        
        agent.prev_x = agent.x
        agent.prev_y = agent.y
        agent.prev_heading = agent.heading
        
        # Parse action
        target_speed = np.clip(action[0], -1, 1) * self.max_speed
        angular_vel = np.clip(action[1], -1, 1) * self.max_angular_velocity
        
        # Update heading
        agent.heading += angular_vel * self.dt
        agent.heading = np.arctan2(np.sin(agent.heading), np.cos(agent.heading))
        
        # Update velocity
        target_vx = target_speed * np.cos(agent.heading)
        target_vy = target_speed * np.sin(agent.heading)
        
        agent.vx += (target_vx - agent.vx) * self.acceleration
        agent.vy += (target_vy - agent.vy) * self.acceleration
        
        # Friction
        speed = np.sqrt(agent.vx**2 + agent.vy**2)
        if speed > 0.01:
            agent.vx *= (1 - self.friction * self.dt)
            agent.vy *= (1 - self.friction * self.dt)
        
        # Clamp speed
        speed = np.sqrt(agent.vx**2 + agent.vy**2)
        if speed > self.max_speed:
            agent.vx = (agent.vx / speed) * self.max_speed
            agent.vy = (agent.vy / speed) * self.max_speed
        
        # SOLID COLLISION DETECTION with sub-stepping
        reward = 0.0
        sub_dt = self.dt / self.collision_substeps
        
        curr_x, curr_y = agent.x, agent.y
        
        for substep in range(self.collision_substeps):
            new_x = curr_x + agent.vx * sub_dt
            new_y = curr_y + agent.vy * sub_dt
            
            # Wall collision
            if new_x < self.agent_radius:
                new_x = self.agent_radius
                agent.vx = abs(agent.vx) * 0.1
                if substep == 0:
                    agent.wall_collisions += 1
                    self.episode_metrics["wall_collisions"] += 1
                    reward -= 10.0
            elif new_x > self.corridor_length - self.agent_radius:
                new_x = self.corridor_length - self.agent_radius
                agent.vx = -abs(agent.vx) * 0.1
                if substep == 0:
                    agent.wall_collisions += 1
                    self.episode_metrics["wall_collisions"] += 1
                    reward -= 10.0
            
            if new_y < self.agent_radius:
                new_y = self.agent_radius
                agent.vy = abs(agent.vy) * 0.1
                if substep == 0:
                    agent.wall_collisions += 1
                    self.episode_metrics["wall_collisions"] += 1
                    reward -= 10.0
            elif new_y > self.corridor_height - self.agent_radius:
                new_y = self.corridor_height - self.agent_radius
                agent.vy = -abs(agent.vy) * 0.1
                if substep == 0:
                    agent.wall_collisions += 1
                    self.episode_metrics["wall_collisions"] += 1
                    reward -= 10.0
            
            # SOLID obstacle collision
            for obs in self.obstacles:
                if self._circle_box_collision(new_x, new_y, self.agent_radius,
                                             obs.x - obs.width/2, obs.y - obs.height/2,
                                             obs.width, obs.height):
                    # STOP at obstacle - push out
                    new_x, new_y, nx, ny = self._push_out_of_obstacle(curr_x, curr_y, new_x, new_y, obs)
                    if nx != 0:
                        agent.vx = -agent.vx * 0.1
                    if ny != 0:
                        agent.vy = -agent.vy * 0.1
                    
                    if substep == 0:
                        agent.obstacle_collisions += 1
                        self.episode_metrics["obstacle_collisions"] += 1
                        reward -= 8.0
            
            curr_x, curr_y = new_x, new_y
        
        # Agent-agent collision
        for j, other in enumerate(self.agents):
            if j != agent_idx and not other.reached_goal:
                dist = np.sqrt((curr_x - other.x)**2 + (curr_y - other.y)**2)
                if dist < 2 * self.agent_radius:
                    overlap = 2 * self.agent_radius - dist
                    if dist > 0.01:
                        push_x = (curr_x - other.x) / dist * overlap * 0.5
                        push_y = (curr_y - other.y) / dist * overlap * 0.5
                        curr_x += push_x
                        curr_y += push_y
                    
                    agent.agent_collisions += 1
                    self.episode_metrics["agent_collisions"] += 1
                    reward -= 1.0
        
        # Final position
        agent.x = np.clip(curr_x, self.agent_radius, self.corridor_length - self.agent_radius)
        agent.y = np.clip(curr_y, self.agent_radius, self.corridor_height - self.agent_radius)
        
        # Final obstacle check
        for obs in self.obstacles:
            if self._circle_box_collision(agent.x, agent.y, self.agent_radius,
                                         obs.x - obs.width/2, obs.y - obs.height/2,
                                         obs.width, obs.height):
                agent.x, agent.y, _, _ = self._push_out_of_obstacle(
                    agent.prev_x, agent.prev_y, agent.x, agent.y, obs
                )
        
        # Goal check
        dist_to_goal = np.sqrt((agent.x - goal[0])**2 + (agent.y - goal[1])**2)
        
        if dist_to_goal < self.goal_radius and not agent.reached_goal:
            agent.reached_goal = True
            agent.goal_time = self.current_step
            self.episode_metrics["agents_reached_goal"] += 1
            reward += 500.0
        
        # Shaped rewards
        reward += self._compute_shaped_reward(agent_idx, dist_to_goal)
        
        return reward
    
    def _circle_box_collision(self, cx, cy, r, bx, by, bw, bh) -> bool:
        closest_x = np.clip(cx, bx, bx + bw)
        closest_y = np.clip(cy, by, by + bh)
        dist_sq = (cx - closest_x)**2 + (cy - closest_y)**2
        return dist_sq < r * r
    
    def _push_out_of_obstacle(self, old_x, old_y, new_x, new_y, obs):
        cx, cy = obs.x, obs.y
        hw = obs.width / 2 + self.agent_radius + 0.02
        hh = obs.height / 2 + self.agent_radius + 0.02
        
        dist_left = new_x - (cx - hw)
        dist_right = (cx + hw) - new_x
        dist_bottom = new_y - (cy - hh)
        dist_top = (cy + hh) - new_y
        
        min_dist = min(abs(dist_left), abs(dist_right), abs(dist_bottom), abs(dist_top))
        
        nx, ny = 0, 0
        if abs(dist_left) == min_dist:
            new_x = cx - hw
            nx = -1
        elif abs(dist_right) == min_dist:
            new_x = cx + hw
            nx = 1
        elif abs(dist_bottom) == min_dist:
            new_y = cy - hh
            ny = -1
        else:
            new_y = cy + hh
            ny = 1
        
        return new_x, new_y, nx, ny
    
    def _compute_shaped_reward(self, agent_idx: int, dist_to_goal: float) -> float:
        agent = self.agents[agent_idx]
        goal = self.goal_positions[agent_idx]
        reward = 0.0
        
        # Progress reward
        prev_dist = np.sqrt((agent.prev_x - goal[0])**2 + (agent.prev_y - goal[1])**2)
        progress = prev_dist - dist_to_goal
        reward += progress * 60.0
        
        # Velocity alignment
        vel = np.array([agent.vx, agent.vy])
        goal_dir = np.array([goal[0] - agent.x, goal[1] - agent.y])
        goal_dist = np.linalg.norm(goal_dir)
        speed = np.sqrt(agent.vx**2 + agent.vy**2)
        
        if goal_dist > 0.1 and speed > 0.1:
            alignment = np.dot(vel, goal_dir) / (speed * goal_dist)
            reward += alignment * 8.0
        
        # Speed reward
        if speed > 0.4 * self.max_speed:
            reward += 2.0
        elif speed < 0.1 * self.max_speed:
            reward -= 2.0
        
        # Anti-spinning
        heading_change = abs(agent.heading - agent.prev_heading)
        heading_change = min(heading_change, 2*np.pi - heading_change)
        if heading_change > 0.3 and speed < 0.3 * self.max_speed:
            reward -= 3.0
        
        # Goal proximity
        if dist_to_goal < self.goal_radius * 2:
            reward += 5.0
        if dist_to_goal < self.goal_radius * 1.2:
            reward += 10.0
        
        # Time penalty
        reward -= 0.03
        
        return reward
    
    def _compute_info(self) -> Dict:
        num_reached = sum(1 for a in self.agents if a.reached_goal)
        
        return {
            "all_success": num_reached == self.num_agents,
            "any_success": num_reached > 0,
            "per_agent_success": num_reached / self.num_agents,
            "agents_reached": num_reached,
            "total_wall_collisions": self.episode_metrics["wall_collisions"],
            "total_obstacle_collisions": self.episode_metrics["obstacle_collisions"],
            "total_agent_collisions": self.episode_metrics["agent_collisions"],
            "episode_length": self.current_step,
            "scenario": self.scenario,
        }
    
    def get_agent_positions(self):
        return [(a.x, a.y) for a in self.agents]
    
    def get_agent_headings(self):
        return [a.heading for a in self.agents]
    
    def get_obstacles(self):
        return [{"x": o.x, "y": o.y, "width": o.width, "height": o.height, "attached_to": o.attached_to}
                for o in self.obstacles]

