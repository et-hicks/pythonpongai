"""Core package for the Seeking RL + Arcade playground."""

from .config import GridWorldConfig
from .game.world import GridWorld

__all__ = ["GridWorld", "GridWorldConfig"]
