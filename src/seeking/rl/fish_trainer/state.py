from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

GRID_HEIGHT = 16
GRID_WIDTH = 24
CELL_SIZE = 50


@dataclass
class QuantizedFishState:
    matrix: np.ndarray

    def flattened(self) -> np.ndarray:
        return self.matrix.astype(np.float32).flatten()


def quantize_ascii_frame(
    fish_position: Tuple[float, float],
    drops: Iterable[Tuple[float, float, str]],
    window_width: int,
    window_height: int,
) -> QuantizedFishState:
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    fx, fy = fish_position
    fish_col = min(GRID_WIDTH - 1, max(0, int(fx // CELL_SIZE)))
    fish_row = min(GRID_HEIGHT - 1, max(0, int(fy // CELL_SIZE)))
    grid[fish_row, fish_col] = 1.0

    for x, y, label in drops:
        col = min(GRID_WIDTH - 1, max(0, int(x // CELL_SIZE)))
        row = min(GRID_HEIGHT - 1, max(0, int(y // CELL_SIZE)))
        if label == "food":
            grid[row, col] = 2.0
        elif label == "bomb":
            grid[row, col] = -1.0
        elif label == "coin":
            grid[row, col] = 3.0

    row_height = window_height / GRID_HEIGHT
    col_width = window_width / GRID_WIDTH
    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            if grid[r, c] != 0:
                continue
            center_x = (c + 0.5) * col_width
            center_y = (r + 0.5) * row_height
            distance_to_top = window_height - center_y
            grid[r, c] = max(0.0, min(0.3, distance_to_top / window_height * 0.3))

    return QuantizedFishState(matrix=grid)
