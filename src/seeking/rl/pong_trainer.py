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
HIT_REWARD = 0.1
SCORE_REWARD = 1.0
CONCEDE_PENALTY = -1.0
DISTANCE_PENALTY_SCALE = 5e-4
CORNER_THRESHOLD = 0.05
CORNER_PENALTY = 0.01
IDLE_DELTA = 2.0
IDLE_THRESHOLD = 45
IDLE_PENALTY = 0.005
BORING_STEP_LIMIT = 500
TERMINAL_BASE_PENALTY = 0.5
TERMINAL_DISTANCE_SCALE = 0.001


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class StepResult:
    state: List[float]
    reward: float
    done: bool


class PongTrainingEnv:
    """Minimal headless Pong simulation for RL training."""

    def __init__(self, seed: int | None = None, projectile_shape: str = "ball") -> None:
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
        self.steps_since_hit = 0
        self.left_idle_steps = 0
        self.prev_left_y = 0.0
        self.projectile_shape = projectile_shape
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
        self.steps_since_hit = 0
        self.left_idle_steps = 0
        self.prev_left_y = self.left_y

    def step(self, action: int) -> StepResult:
        direction = -1 if action == 0 else 1
        self.left_y += direction * PADDLE_SPEED
        self.left_y = clamp(self.left_y, 0, WINDOW_HEIGHT - PADDLE_HEIGHT)
        if abs(self.left_y - self.prev_left_y) < IDLE_DELTA:
            self.left_idle_steps += 1
        else:
            self.left_idle_steps = 0
        self.prev_left_y = self.left_y

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

        reward = 0.0
        left_center = self._left_center()
        reward -= DISTANCE_PENALTY_SCALE * abs(left_center - self.ball_y)
        norm_center = left_center / WINDOW_HEIGHT
        if norm_center < CORNER_THRESHOLD or norm_center > 1 - CORNER_THRESHOLD:
            reward -= CORNER_PENALTY
        if self.left_idle_steps > IDLE_THRESHOLD:
            reward -= IDLE_PENALTY

        self.steps_since_hit += 1

        if (
            self.ball_vx < 0
            and self.ball_x - BALL_RADIUS <= LEFT_PADDLE_X + PADDLE_WIDTH
            and self.left_y <= self.ball_y <= self.left_y + PADDLE_HEIGHT
        ):
            self.ball_x = LEFT_PADDLE_X + PADDLE_WIDTH + BALL_RADIUS
            self.ball_vx = abs(self.ball_vx)
            reward += HIT_REWARD
            self.last_hit = "left"
            self.steps_since_hit = 0
            self._apply_shape_spin(True)

        if (
            self.ball_vx > 0
            and self.ball_x + BALL_RADIUS >= RIGHT_PADDLE_X
            and self.right_y <= self.ball_y <= self.right_y + PADDLE_HEIGHT
        ):
            self.ball_x = RIGHT_PADDLE_X - BALL_RADIUS
            self.ball_vx = -abs(self.ball_vx)
            self.last_hit = "right"
            self.steps_since_hit = 0
            self._apply_shape_spin(False)

        done = False
        distance_for_terminal = abs(left_center - self.ball_y)
        if self.ball_x < -BALL_RADIUS:
            self.right_score += 1
            reward += CONCEDE_PENALTY
            done = True
            self.steps_since_hit = 0
            self._reset_round()
        elif self.ball_x > WINDOW_WIDTH + BALL_RADIUS:
            self.left_score += 1
            reward += SCORE_REWARD
            done = True
            self.steps_since_hit = 0
            self._reset_round()
        elif self.steps_since_hit > BORING_STEP_LIMIT:
            penalty = TERMINAL_BASE_PENALTY + TERMINAL_DISTANCE_SCALE * distance_for_terminal
            reward -= penalty
            done = True
            self.steps_since_hit = 0
            self._reset_round()

        return StepResult(self._state(), reward, done)

    def _state(self) -> List[float]:
        left_top = self.left_y / WINDOW_HEIGHT
        left_bottom = (self.left_y + PADDLE_HEIGHT) / WINDOW_HEIGHT
        right_top = self.right_y / WINDOW_HEIGHT
        right_bottom = (self.right_y + PADDLE_HEIGHT) / WINDOW_HEIGHT
        score_norm = (self.left_score - self.right_score) / 10.0
        ball_norm = self.ball_y / WINDOW_HEIGHT
        return [left_top, left_bottom, right_top, right_bottom, score_norm, ball_norm]

    def _left_center(self) -> float:
        return self.left_y + PADDLE_HEIGHT / 2

    def _apply_shape_spin(self, left_hit: bool) -> None:
        if self.projectile_shape == "ball":
            return
        if self.projectile_shape == "square":
            jitter = self.random.uniform(-2.0, 2.0)
            self.ball_vy += jitter if left_hit else -jitter
        elif self.projectile_shape == "triangle":
            self.ball_vx += self.random.uniform(-3.0, 3.0)
            self.ball_vy += self.random.uniform(-4.0, 4.0)


class PongReinforceTrainer:
    """Simple REINFORCE loop for the Pong policy."""

    def __init__(
        self,
        device: torch.device,
        lr: float = 1e-3,
        gamma: float = 0.99,
        max_steps: int = 500,
        entropy_coef: float = 0.02,
        seed: int | None = None,
        projectile_shape: str = "ball",
    ) -> None:
        self.device = device
        self.policy = PongPolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.max_steps = max_steps
        self.entropy_coef = entropy_coef
        self.projectile_shape = projectile_shape
        self.env = PongTrainingEnv(seed=seed, projectile_shape=projectile_shape)
        self.device_label = describe_device(self.device)

    def train(self, episodes: int, checkpoint_path: str | None = None) -> None:
        print(f"[Pong Trainer] Training on {self.device_label} (projectile={self.projectile_shape})")
        for episode in range(1, episodes + 1):
            log_probs: List[torch.Tensor] = []
            rewards: List[float] = []
            entropies: List[torch.Tensor] = []
            state = self.env.reset()
            episode_reward = 0.0

            for _ in range(self.max_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                logits = self.policy(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                result = self.env.step(action.item())
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                rewards.append(result.reward)
                episode_reward += result.reward
                state = result.state
                if result.done:
                    break

            loss = self._policy_gradient_loss(log_probs, rewards)
            if entropies:
                entropy_term = torch.stack(entropies).mean()
            else:
                entropy_term = torch.tensor(0.0, device=self.device)
            total_loss = loss - self.entropy_coef * entropy_term
            self.optimizer.zero_grad()
            total_loss.backward()
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
def describe_device(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(device.index or torch.cuda.current_device())
        except Exception:
            name = "CUDA"
        return f"CUDA ({name})"
    if device.type == "mps":
        return "Apple GPU (MPS)"
    return "CPU"
