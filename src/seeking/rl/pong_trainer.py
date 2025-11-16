from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import torch
from torch import optim

from seeking.game.pong import (
    BALL_RADIUS,
    BALL_SPEED,
    PADDLE_HEIGHT,
    PADDLE_SPEED,
    PADDLE_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from seeking.rl.pong_policy import PongPolicyNetwork


LEFT_PADDLE_X = 40
RIGHT_PADDLE_X = WINDOW_WIDTH - 40 - PADDLE_WIDTH
HIT_REWARD = 0.4
SCORE_REWARD = 1.0
SCORE_PENALTY = 1.0
UNTOUCHED_PENALTY = 0.5


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class StepResult:
    state: List[float]
    reward: float
    done: bool


class PongTrainingEnv:
    """Minimal headless Pong simulation for RL training."""

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)
        self.left_score = 0
        self.right_score = 0
        self.left_y = 0.0
        self.right_y = 0.0
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.last_hit: str | None = None
        self.reset()

    def reset(self) -> List[float]:
        self.left_score = 0
        self.right_score = 0
        self._reset_round()
        return self._state()

    def _reset_round(self) -> None:
        self.left_y = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.right_y = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.ball_x = WINDOW_WIDTH / 2
        self.ball_y = WINDOW_HEIGHT / 2
        horizontal = self.random.choice([-1, 1])
        vertical = self.random.uniform(-0.7, 0.7)
        magnitude = (vertical**2 + 1) ** 0.5
        self.ball_vx = horizontal * BALL_SPEED / magnitude
        self.ball_vy = vertical * BALL_SPEED / magnitude
        self.last_hit = None

    def step(self, action: int) -> StepResult:
        direction = -1 if action == 0 else 1
        self.left_y += direction * PADDLE_SPEED
        self.left_y = clamp(self.left_y, 0, WINDOW_HEIGHT - PADDLE_HEIGHT)

        right_center = self.right_y + PADDLE_HEIGHT / 2
        if self.ball_y < right_center - 10:
            self.right_y -= PADDLE_SPEED * 0.9
        elif self.ball_y > right_center + 10:
            self.right_y += PADDLE_SPEED * 0.9
        self.right_y = clamp(self.right_y, 0, WINDOW_HEIGHT - PADDLE_HEIGHT)

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        if self.ball_y - BALL_RADIUS <= 0:
            self.ball_y = BALL_RADIUS
            self.ball_vy *= -1
        elif self.ball_y + BALL_RADIUS >= WINDOW_HEIGHT:
            self.ball_y = WINDOW_HEIGHT - BALL_RADIUS
            self.ball_vy *= -1

        reward = -abs(self._left_center() - self.ball_y) / WINDOW_HEIGHT

        if (
            self.ball_vx < 0
            and self.ball_x - BALL_RADIUS <= LEFT_PADDLE_X + PADDLE_WIDTH
            and self.left_y <= self.ball_y <= self.left_y + PADDLE_HEIGHT
        ):
            self.ball_x = LEFT_PADDLE_X + PADDLE_WIDTH + BALL_RADIUS
            self.ball_vx = abs(self.ball_vx)
            reward += HIT_REWARD
            self.last_hit = "left"

        if (
            self.ball_vx > 0
            and self.ball_x + BALL_RADIUS >= RIGHT_PADDLE_X
            and self.right_y <= self.ball_y <= self.right_y + PADDLE_HEIGHT
        ):
            self.ball_x = RIGHT_PADDLE_X - BALL_RADIUS
            self.ball_vx = -abs(self.ball_vx)
            self.last_hit = "right"

        done = False
        if self.ball_x < -BALL_RADIUS:
            if self.last_hit == "right":
                self.right_score += 1
                reward -= SCORE_PENALTY
            elif self.last_hit is None:
                reward -= UNTOUCHED_PENALTY
            done = True
            self._reset_round()
        elif self.ball_x > WINDOW_WIDTH + BALL_RADIUS:
            if self.last_hit == "left":
                self.left_score += 1
                reward += SCORE_REWARD
            elif self.last_hit is None:
                reward -= UNTOUCHED_PENALTY
            done = True
            self._reset_round()

        return StepResult(self._state(), reward, done)

    def _state(self) -> List[float]:
        top_norm = self.left_y / WINDOW_HEIGHT
        bottom_norm = (self.left_y + PADDLE_HEIGHT) / WINDOW_HEIGHT
        score_norm = (self.left_score - self.right_score) / 10.0
        ball_norm = self.ball_y / WINDOW_HEIGHT
        return [top_norm, bottom_norm, score_norm, ball_norm]

    def _left_center(self) -> float:
        return self.left_y + PADDLE_HEIGHT / 2


class PongReinforceTrainer:
    """Simple REINFORCE loop for the Pong policy."""

    def __init__(
        self,
        device: torch.device,
        lr: float = 1e-3,
        gamma: float = 0.99,
        max_steps: int = 500,
        seed: int | None = None,
    ) -> None:
        self.device = device
        self.policy = PongPolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.max_steps = max_steps
        self.env = PongTrainingEnv(seed=seed)

    def train(self, episodes: int, checkpoint_path: str | None = None) -> None:
        for episode in range(1, episodes + 1):
            log_probs: List[torch.Tensor] = []
            rewards: List[float] = []
            state = self.env.reset()
            episode_reward = 0.0

            for _ in range(self.max_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                logits = self.policy(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                result = self.env.step(action.item())
                log_probs.append(dist.log_prob(action))
                rewards.append(result.reward)
                episode_reward += result.reward
                state = result.state
                if result.done:
                    break

            loss = self._policy_gradient_loss(log_probs, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if episode % 10 == 0:
                avg_reward = episode_reward / len(rewards) if rewards else 0.0
                print(f"[Pong Trainer] Episode {episode:04d}  Total reward: {episode_reward:.2f}  "
                      f"Avg step reward: {avg_reward:.4f}")

        if checkpoint_path:
            torch.save(self.policy.state_dict(), checkpoint_path)
            print(f"[Pong Trainer] Saved policy state_dict to {checkpoint_path}")

    def _policy_gradient_loss(self, log_probs: List[torch.Tensor], rewards: List[float]) -> torch.Tensor:
        if not log_probs:
            return torch.tensor(0.0, device=self.device)
        returns = []
        g = 0.0
        for reward in reversed(rewards):
            g = reward + self.gamma * g
            returns.insert(0, g)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        log_stack = torch.stack(log_probs)
        return -(returns_tensor * log_stack).sum()
