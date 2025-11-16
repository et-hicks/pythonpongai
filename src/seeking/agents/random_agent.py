from __future__ import annotations

from seeking.game.world import GridWorld


class RandomAgent:
    """Baseline agent useful for smoke testing."""

    def __init__(self, env: GridWorld) -> None:
        self.env = env

    def act(self, *_args, **_kwargs) -> int:
        return self.env.sample_action()
