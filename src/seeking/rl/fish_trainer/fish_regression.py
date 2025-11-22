from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .fish_dqn import ACTION_SPACE
from .state import QuantizedFishState


class FishRegressionController:
    def __init__(self, lr: float = 1e-3, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self.model = nn.Linear(16 * 24, len(ACTION_SPACE)).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state: QuantizedFishState) -> str:
        tensor = torch.tensor(state.flattened(), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            scores = self.model(tensor)
        return ACTION_SPACE[int(torch.argmax(scores, dim=1).item())]

    def fit(self, state: QuantizedFishState, target_action: str) -> None:
        target_idx = ACTION_SPACE.index(target_action)
        target = torch.zeros((1, len(ACTION_SPACE)), device=self.device)
        target[0, target_idx] = 1.0
        tensor = torch.tensor(state.flattened(), dtype=torch.float32, device=self.device).unsqueeze(0)
        pred = self.model(tensor)
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


__all__ = ["FishRegressionController"]
