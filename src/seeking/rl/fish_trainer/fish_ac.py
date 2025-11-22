from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .state import QuantizedFishState
from .fish_dqn import ACTION_SPACE


class _ActorCriticNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, len(ACTION_SPACE))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        return self.actor(features), self.critic(features)


class FishActorCritic:
    def __init__(self, lr: float = 1e-3, gamma: float = 0.9, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self.gamma = gamma
        self.model = _ActorCriticNet(16 * 24).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state: QuantizedFishState) -> str:
        tensor = torch.tensor(state.flattened(), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        action_idx = int(torch.multinomial(probs, num_samples=1).item())
        return ACTION_SPACE[action_idx]

    def learn(
        self,
        state: QuantizedFishState,
        action: str,
        reward: float,
        next_state: QuantizedFishState,
        done: bool,
    ) -> None:
        action_idx = ACTION_SPACE.index(action)
        state_tensor = torch.tensor(state.flattened(), dtype=torch.float32, device=self.device).unsqueeze(0)
        next_tensor = torch.tensor(next_state.flattened(), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(state_tensor)
        _, next_value = self.model(next_tensor)

        advantage = reward + self.gamma * float(next_value.item()) * (1 - int(done)) - float(value.item())
        log_probs = torch.log_softmax(logits, dim=1)
        log_prob = log_probs[0, action_idx]
        policy_loss = -log_prob * advantage
        value_target = reward + self.gamma * float(next_value.item()) * (1 - int(done))
        value_loss = (value - value_target) ** 2
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


__all__ = ["FishActorCritic"]
