from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch import optim

from seeking.game.world import GridWorld
from seeking.rl.policy import PolicyNetwork


@dataclass
class EpisodeStats:
    episode: int
    total_reward: float
    loss: float
    steps: int


class Trainer:
    """Light-weight REINFORCE style trainer."""

    def __init__(
        self,
        env: GridWorld,
        policy: PolicyNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.policy = policy.to(device)
        self.gamma = gamma
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.console = Console()

    def train(self, episodes: int = 50) -> List[EpisodeStats]:
        stats: List[EpisodeStats] = []
        for episode in range(1, episodes + 1):
            rewards, log_probs, steps = self._rollout_episode()
            returns = self._discounted_returns(rewards)
            loss = self._update_policy(log_probs, returns)
            episode_reward = sum(rewards)
            stats.append(EpisodeStats(episode, episode_reward, loss, steps))
            if episode % 10 == 0 or episode == 1:
                self._log_episode(stats[-1])
        return stats

    def _rollout_episode(self) -> Tuple[List[float], List[torch.Tensor], int]:
        obs = torch.from_numpy(self.env.reset()).float().to(self.device)
        done = False
        rewards: List[float] = []
        log_probs: List[torch.Tensor] = []
        steps = 0

        while not done:
            action_out = self.policy.act(obs.unsqueeze(0))
            result = self.env.step(action_out.action)
            rewards.append(result.reward)
            log_probs.append(action_out.log_prob)
            obs = torch.from_numpy(result.observation).float().to(self.device)
            steps += 1
            done = result.terminated or result.truncated
        return rewards, log_probs, steps

    def _discounted_returns(self, rewards: List[float]) -> torch.Tensor:
        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.gamma * running
            returns.append(running)
        returns = list(reversed(returns))
        tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        return tensor

    def _update_policy(self, log_probs: List[torch.Tensor], returns: torch.Tensor) -> float:
        loss = torch.stack([-lp * ret for lp, ret in zip(log_probs, returns)]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    def _log_episode(self, stats: EpisodeStats) -> None:
        table = Table(title=f"Episode {stats.episode}")
        table.add_column("Reward", justify="right")
        table.add_column("Loss", justify="right")
        table.add_column("Steps", justify="right")
        table.add_row(
            f"{stats.total_reward:.2f}",
            f"{stats.loss:.3f}",
            f"{stats.steps}",
        )
        self.console.print(table)
