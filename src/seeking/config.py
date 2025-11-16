from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GridWorldConfig:
    """Simple configuration container for the gridworld environment."""

    width: int = 10
    height: int = 10
    num_obstacles: int = 10
    max_steps: int = 200
    obstacle_seed: Optional[int] = None
    goal_reward: float = 1.0
    step_penalty: float = -0.01
    obstacle_penalty: float = -1.0
    out_of_bounds_penalty: float = -1.0
    headless: bool = False
    log_history: bool = True
    tile_pixel_size: int = 64
    colors: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "background": (10, 10, 50),
            "grid": (40, 40, 70),
            "player": (0, 200, 255),
            "goal": (0, 255, 150),
            "obstacle": (255, 120, 0),
        }
    )
