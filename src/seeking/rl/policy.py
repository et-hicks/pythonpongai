from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class ActionOutput:
    action: int
    log_prob: torch.Tensor
    logits: torch.Tensor


class PolicyNetwork(nn.Module):
    """Simple feed-forward policy suitable for discrete actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def act(self, obs: torch.Tensor) -> ActionOutput:
        logits = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return ActionOutput(action.item(), dist.log_prob(action), logits)

    def act_greedy(self, obs: torch.Tensor) -> Tuple[int, torch.Tensor]:
        logits = self(obs)
        action = torch.argmax(logits, dim=-1)
        return action.item(), logits

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, map_location: str | None = None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location))
