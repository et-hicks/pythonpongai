"""Neural networks for quantized Pong grid observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

ACTION_LABELS = ("up", "down", "neutral")


def _tokenize(line: str) -> list[str]:
    """Return non-space glyphs so we can reconstruct the grid."""
    tokens: list[str] = []
    stripped = line.strip()
    if not stripped:
        return tokens
    for chunk in stripped.split():
        if len(chunk) == 1:
            tokens.append(chunk)
        else:
            tokens.extend(char for char in chunk if char.strip())
    return tokens


@dataclass(slots=True)
class QuantizedStateEncoder:
    """Rebuild a (channels, rows, cols) tensor from ASCII grid snapshots."""

    device: str | torch.device = "cpu"

    def encode(self, grid: str | Sequence[str]) -> torch.Tensor:
        lines = self._prepare_lines(grid)
        if not lines:
            raise ValueError("Quantized state payload did not contain any rows.")
        tokenized = [_tokenize(line) for line in lines]
        tokenized = [line for line in tokenized if line]
        if not tokenized:
            raise ValueError("Quantized state payload did not contain recognizable tokens.")

        cols = max(len(row) for row in tokenized)
        rows = len(tokenized)

        tensor = torch.zeros((3, rows, cols), dtype=torch.float32, device=self.device)
        tensor[0] = 1.0  # default to "empty" channel

        mapping = {"0": 0, "|": 1, ".": 2}

        for r, row in enumerate(tokenized):
            for c, glyph in enumerate(row[:cols]):
                if glyph not in mapping:
                    continue
                channel = mapping[glyph]
                if channel == 0:
                    continue
                tensor[:, r, c] = 0.0
                tensor[channel, r, c] = 1.0

        return tensor

    def _prepare_lines(self, grid: str | Sequence[str]) -> list[str]:
        if isinstance(grid, str):
            return [line for line in grid.splitlines() if line.strip()]
        return [line for line in grid if line.strip()]


class _SharedEncoder(nn.Module):
    """Feature extractor shared between controllers."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 3:
            state = state.unsqueeze(0)
        return self.layers(state)


class PongDQN(nn.Module):
    """Simple convolutional DQN head for the quantized Pong board."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _SharedEncoder()
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTION_LABELS)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.encoder(state)
        q_values = self.head(features)
        return q_values.squeeze(0)

    @torch.no_grad()
    def act(self, state: torch.Tensor, epsilon: float = 0.0) -> tuple[int, torch.Tensor]:
        state = state.to(next(self.parameters()).device)
        q_values = self(state)
        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, len(ACTION_LABELS), (1,), device=q_values.device).item()
        else:
            action = torch.argmax(q_values).item()
        return action, q_values


class PongActorCritic(nn.Module):
    """Shared-encoder actor-critic model for the quantized Pong board."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _SharedEncoder()
        self.policy_head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTION_LABELS)),
        )
        self.value_head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(state)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return logits.squeeze(0), values.squeeze(0)

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        state = state.to(next(self.parameters()).device)
        logits, value = self(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), logits, value
