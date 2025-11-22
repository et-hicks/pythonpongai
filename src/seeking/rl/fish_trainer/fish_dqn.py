from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .state import QuantizedFishState


ACTION_SPACE = ("up", "down", "left", "right")


class _QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(ACTION_SPACE)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class FishDQN:
    def __init__(self, gamma: float = 0.9, lr: float = 1e-3, device: torch.device | None = None) -> None:
        self.gamma = gamma
        self.device = device or torch.device("cpu")
        self.epsilon = 0.1
        self.buffer: Deque[Transition] = deque(maxlen=512)
        input_dim = 16 * 24
        self.network = _QNetwork(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state: QuantizedFishState) -> str:
        if np.random.rand() < self.epsilon:
            return ACTION_SPACE[np.random.randint(len(ACTION_SPACE))]
        with torch.no_grad():
            tensor = torch.tensor(state.flattened(), dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.network(tensor)
            action_idx = int(torch.argmax(q_values, dim=1).item())
        return ACTION_SPACE[action_idx]

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def learn(self, batch_size: int = 32) -> None:
        if len(self.buffer) < batch_size:
            return
        batch = [self.buffer[np.random.randint(len(self.buffer))] for _ in range(batch_size)]
        states = torch.tensor([t.state for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([t.next_state for t in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.network(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.network(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = self.loss_fn(q_selected, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


__all__ = ["FishDQN", "Transition", "ACTION_SPACE"]
