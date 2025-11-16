from __future__ import annotations

import torch
from torch import nn


class PongPolicyNetwork(nn.Module):
    """Four-layer feed-forward policy mapping a 4-dim state to 2 action logits."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
            logits = self.net(state)
            return logits.squeeze(0)
        return self.net(state)
