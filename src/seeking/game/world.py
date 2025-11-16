from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from seeking.config import GridWorldConfig

Position = Tuple[int, int]


ACTIONS: Dict[int, Position] = {
    0: (0, 1),   # up
    1: (0, -1),  # down
    2: (-1, 0),  # left
    3: (1, 0),   # right
    4: (0, 0),   # stay
}

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "STAY",
}


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, object]


class GridWorld:
    """Grid-world navigation task with deterministic transitions."""

    def __init__(self, config: Optional[GridWorldConfig] = None) -> None:
        self.config = config or GridWorldConfig()
        self.width = self.config.width
        self.height = self.config.height
        self._rng = random.Random(self.config.obstacle_seed)
        self.player_position: Position = (0, 0)
        self.goal_position: Position = (self.width - 1, self.height - 1)
        self.obstacles: set[Position] = set()
        self.steps = 0
        self.history: List[Dict[str, object]] = []

    # ------------------------------------------------------------------ #
    # Environment lifecycle
    # ------------------------------------------------------------------ #
    def seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self._rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self.seed(seed)
        self.steps = 0
        self.player_position = (0, 0)
        self.goal_position = (self.width - 1, self.height - 1)
        self.obstacles = self._generate_obstacles()
        self.history.clear()
        return self._get_observation()

    def step(self, action: int) -> StepResult:
        if action not in ACTIONS:
            raise ValueError(f"Unsupported action: {action}")

        dx, dy = ACTIONS[action]
        target = (self.player_position[0] + dx, self.player_position[1] + dy)
        reward = self.config.step_penalty
        terminated = False
        truncated = False

        if not self._in_bounds(target):
            reward += self.config.out_of_bounds_penalty
            target = self.player_position
        elif target in self.obstacles:
            reward += self.config.obstacle_penalty
            target = self.player_position
        else:
            self.player_position = target

        self.steps += 1

        if self.player_position == self.goal_position:
            terminated = True
            reward += self.config.goal_reward
        elif self.steps >= self.config.max_steps:
            truncated = True

        obs = self._get_observation()
        info = {
            "position": self.player_position,
            "steps": self.steps,
            "action_name": ACTION_NAMES[action],
        }

        if self.config.log_history:
            self.history.append(
                {
                    "step": self.steps,
                    "position": self.player_position,
                    "reward": reward,
                    "action": ACTION_NAMES[action],
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )

        return StepResult(obs, reward, terminated, truncated, info)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def observation_space_size(self) -> int:
        return 4

    @property
    def action_space_size(self) -> int:
        return len(ACTIONS)

    def sample_action(self) -> int:
        return self._rng.choice(list(ACTIONS))

    def _get_observation(self) -> np.ndarray:
        px, py = self.player_position
        gx, gy = self.goal_position
        nx = px / (self.width - 1 if self.width > 1 else 1)
        ny = py / (self.height - 1 if self.height > 1 else 1)
        dx = (gx - px) / max(self.width - 1, 1)
        dy = (gy - py) / max(self.height - 1, 1)
        return np.array([nx, ny, dx, dy], dtype=np.float32)

    def _generate_obstacles(self) -> set[Position]:
        obstacles: set[Position] = set()
        attempts = 0
        while len(obstacles) < min(self.config.num_obstacles, self.width * self.height - 2):
            attempts += 1
            if attempts > 5000:
                break
            candidate = (
                self._rng.randrange(0, self.width),
                self._rng.randrange(0, self.height),
            )
            if candidate in (self.player_position, self.goal_position):
                continue
            obstacles.add(candidate)
        return obstacles

    def _in_bounds(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def snapshot(self) -> Dict[str, object]:
        """Return lightweight description for rendering."""
        return {
            "player": self.player_position,
            "goal": self.goal_position,
            "obstacles": list(self.obstacles),
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
        }
