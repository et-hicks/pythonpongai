from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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


@dataclass
class DualStepResult:
    left_state: List[float]
    right_state: List[float]
    left_reward: float
    right_reward: float
    done: bool


class DualPongEnv:
    def __init__(self, seed: int | None = None, projectile_shape: str = "ball") -> None:
        self.random = random.Random(seed)
        self.left_y = 0.0
        self.right_y = 0.0
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.left_score = 0
        self.right_score = 0
        self.last_hit: str | None = None
        self.steps_since_hit = 0
        self.left_idle_steps = 0
        self.right_idle_steps = 0
        self.prev_left_y = 0.0
        self.prev_right_y = 0.0
        self.shape = projectile_shape
        self.reset()

    def reset(self) -> Tuple[List[float], List[float]]:
        self.left_y = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.right_y = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.prev_left_y = self.left_y
        self.prev_right_y = self.right_y
        self.left_idle_steps = 0
        self.right_idle_steps = 0
        self.steps_since_hit = 0
        self.left_score = 0
        self.right_score = 0
        self._serve_ball()
        self.last_hit = None
        return self._state_left(), self._state_right()

    def _serve_ball(self) -> None:
        self.ball_x = WINDOW_WIDTH / 2
        self.ball_y = WINDOW_HEIGHT / 2 + self.random.randint(-100, 100)
        self.ball_y = max(BALL_RADIUS, min(WINDOW_HEIGHT - BALL_RADIUS, self.ball_y))
        horizontal = self.random.choice([-1, 1])
        vertical = self.random.uniform(-0.7, 0.7)
        magnitude = (vertical**2 + 1) ** 0.5
        self.ball_vx = horizontal * BALL_SPEED / magnitude
        self.ball_vy = vertical * BALL_SPEED / magnitude

    def step(self, left_action: int, right_action: int) -> DualStepResult:
        self._move_paddles(left_action, right_action)
        self._move_ball()
        left_reward, right_reward = self._compute_dense_rewards()
        collision = self._handle_collisions()
        if collision:
            if collision == "left":
                left_reward += HIT_REWARD
            else:
                right_reward += HIT_REWARD
            self.steps_since_hit = 0
            self._apply_shape_spin(collision)
        else:
            self.steps_since_hit += 1

        done = False
        if self.ball_x < -BALL_RADIUS:
            self.right_score += 1
            left_reward += CONCEDE_PENALTY
            right_reward += SCORE_REWARD
            done = True
        elif self.ball_x > WINDOW_WIDTH + BALL_RADIUS:
            self.left_score += 1
            right_reward += CONCEDE_PENALTY
            left_reward += SCORE_REWARD
            done = True
        elif self.steps_since_hit > BORING_STEP_LIMIT:
            left_penalty = TERMINAL_BASE_PENALTY + TERMINAL_DISTANCE_SCALE * abs(
                self.left_center - self.ball_y
            )
            right_penalty = TERMINAL_BASE_PENALTY + TERMINAL_DISTANCE_SCALE * abs(
                self.right_center - self.ball_y
            )
            left_reward -= left_penalty
            right_reward -= right_penalty
            done = True

        if done:
            self._serve_ball()
            self.steps_since_hit = 0
            self.left_idle_steps = 0
            self.right_idle_steps = 0
            self.prev_left_y = self.left_y
            self.prev_right_y = self.right_y

        return DualStepResult(self._state_left(), self._state_right(), left_reward, right_reward, done)

    def _move_paddles(self, left_action: int, right_action: int) -> None:
        left_dir = -1 if left_action == 0 else 1
        right_dir = -1 if right_action == 0 else 1

        self.left_y += left_dir * PADDLE_SPEED
        self.left_y = max(0.0, min(WINDOW_HEIGHT - PADDLE_HEIGHT, self.left_y))
        self.right_y += right_dir * PADDLE_SPEED
        self.right_y = max(0.0, min(WINDOW_HEIGHT - PADDLE_HEIGHT, self.right_y))

        if abs(self.left_y - self.prev_left_y) < IDLE_DELTA:
            self.left_idle_steps += 1
        else:
            self.left_idle_steps = 0
        if abs(self.right_y - self.prev_right_y) < IDLE_DELTA:
            self.right_idle_steps += 1
        else:
            self.right_idle_steps = 0

        self.prev_left_y = self.left_y
        self.prev_right_y = self.right_y

    def _move_ball(self) -> None:
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        if self.ball_y - BALL_RADIUS <= 0:
            self.ball_y = BALL_RADIUS
            self.ball_vy *= -1
        elif self.ball_y + BALL_RADIUS >= WINDOW_HEIGHT:
            self.ball_y = WINDOW_HEIGHT - BALL_RADIUS
            self.ball_vy *= -1

    def _handle_collisions(self) -> str | None:
        if (
            self.ball_vx < 0
            and self.ball_x - BALL_RADIUS <= LEFT_PADDLE_X + PADDLE_WIDTH
            and self.left_y <= self.ball_y <= self.left_y + PADDLE_HEIGHT
        ):
            self.ball_x = LEFT_PADDLE_X + PADDLE_WIDTH + BALL_RADIUS
            self.ball_vx = abs(self.ball_vx)
            self.last_hit = "left"
            return "left"

        if (
            self.ball_vx > 0
            and self.ball_x + BALL_RADIUS >= RIGHT_PADDLE_X
            and self.right_y <= self.ball_y <= self.right_y + PADDLE_HEIGHT
        ):
            self.ball_x = RIGHT_PADDLE_X - BALL_RADIUS
            self.ball_vx = -abs(self.ball_vx)
            self.last_hit = "right"
            return "right"
        return None

    def _apply_shape_spin(self, collision_side: str) -> None:
        if self.shape == "ball":
            return
        if self.shape == "square":
            jitter = self.random.uniform(-2.5, 2.5)
            self.ball_vy += jitter if collision_side == "left" else -jitter
        elif self.shape == "triangle":
            self.ball_vx += self.random.uniform(-3.5, 3.5)
            self.ball_vy += self.random.uniform(-4.5, 4.5)

    def _compute_dense_rewards(self) -> Tuple[float, float]:
        left_center = self.left_center
        right_center = self.right_center
        left_reward = -DISTANCE_PENALTY_SCALE * abs(left_center - self.ball_y)
        right_reward = -DISTANCE_PENALTY_SCALE * abs(right_center - self.ball_y)

        left_norm = left_center / WINDOW_HEIGHT
        right_norm = right_center / WINDOW_HEIGHT
        if left_norm < CORNER_THRESHOLD or left_norm > 1 - CORNER_THRESHOLD:
            left_reward -= CORNER_PENALTY
        if right_norm < CORNER_THRESHOLD or right_norm > 1 - CORNER_THRESHOLD:
            right_reward -= CORNER_PENALTY

        if self.left_idle_steps > IDLE_THRESHOLD:
            left_reward -= IDLE_PENALTY
        if self.right_idle_steps > IDLE_THRESHOLD:
            right_reward -= IDLE_PENALTY

        return left_reward, right_reward

    @property
    def left_center(self) -> float:
        return self.left_y + PADDLE_HEIGHT / 2

    @property
    def right_center(self) -> float:
        return self.right_y + PADDLE_HEIGHT / 2

    def _state_left(self) -> List[float]:
        return [
            self.left_y / WINDOW_HEIGHT,
            (self.left_y + PADDLE_HEIGHT) / WINDOW_HEIGHT,
            self.right_y / WINDOW_HEIGHT,
            (self.right_y + PADDLE_HEIGHT) / WINDOW_HEIGHT,
            (self.left_score - self.right_score) / 10.0,
            self.ball_y / WINDOW_HEIGHT,
        ]

    def _state_right(self) -> List[float]:
        return [
            self.right_y / WINDOW_HEIGHT,
            (self.right_y + PADDLE_HEIGHT) / WINDOW_HEIGHT,
            self.left_y / WINDOW_HEIGHT,
            (self.left_y + PADDLE_HEIGHT) / WINDOW_HEIGHT,
            (self.right_score - self.left_score) / 10.0,
            self.ball_y / WINDOW_HEIGHT,
        ]


class PongDualTrainer:
    def __init__(
        self,
        device: torch.device,
        green_checkpoint: str,
        purple_checkpoint: str,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.02,
        seed: int | None = None,
        projectile_shape: str = "ball",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.green_checkpoint = Path(green_checkpoint)
        self.purple_checkpoint = Path(purple_checkpoint)
        self.projectile_shape = projectile_shape
        self.env = DualPongEnv(seed=seed, projectile_shape=projectile_shape)
        self.device_label = describe_device(self.device)

        self.green_policy = PongPolicyNetwork().to(self.device)
        self.purple_policy = PongPolicyNetwork().to(self.device)
        self.green_opt = optim.Adam(self.green_policy.parameters(), lr=lr)
        self.purple_opt = optim.Adam(self.purple_policy.parameters(), lr=lr)

        self._maybe_load(self.green_policy, self.green_checkpoint)
        self._maybe_load(self.purple_policy, self.purple_checkpoint)

    def _maybe_load(self, policy: PongPolicyNetwork, path: Path) -> None:
        if path.exists():
            policy.load_state_dict(torch.load(path, map_location=self.device))

    def train(self, episodes: int) -> None:
        print(
            f"[Pong Dual Trainer] Training on {self.device_label} "
            f"(projectile={self.projectile_shape})"
        )
        for episode in range(1, episodes + 1):
            start = time.perf_counter()
            left_state, right_state = self.env.reset()
            left_log_probs: List[torch.Tensor] = []
            right_log_probs: List[torch.Tensor] = []
            left_rewards: List[float] = []
            right_rewards: List[float] = []
            left_entropies: List[torch.Tensor] = []
            right_entropies: List[torch.Tensor] = []

            while True:
                left_tensor = torch.tensor(left_state, dtype=torch.float32, device=self.device)
                right_tensor = torch.tensor(right_state, dtype=torch.float32, device=self.device)
                left_logits = self.green_policy(left_tensor)
                right_logits = self.purple_policy(right_tensor)
                left_dist = torch.distributions.Categorical(logits=left_logits)
                right_dist = torch.distributions.Categorical(logits=right_logits)
                left_action = left_dist.sample()
                right_action = right_dist.sample()
                result = self.env.step(left_action.item(), right_action.item())
                left_log_probs.append(left_dist.log_prob(left_action))
                right_log_probs.append(right_dist.log_prob(right_action))
                left_entropies.append(left_dist.entropy())
                right_entropies.append(right_dist.entropy())
                left_rewards.append(result.left_reward)
                right_rewards.append(result.right_reward)
                left_state = result.left_state
                right_state = result.right_state
                if result.done:
                    break

            self._update_policy(
                self.green_policy,
                self.green_opt,
                left_log_probs,
                left_rewards,
                left_entropies,
            )
            self._update_policy(
                self.purple_policy,
                self.purple_opt,
                right_log_probs,
                right_rewards,
                right_entropies,
            )

            duration = time.perf_counter() - start
            if episode % 10 == 0:
                print(f"[Pong Dual Trainer] Completed episode {episode:04d} ({duration:.2f}s)")

    def _update_policy(
        self,
        policy: PongPolicyNetwork,
        optimizer: optim.Optimizer,
        log_probs: List[torch.Tensor],
        rewards: List[float],
        entropies: List[torch.Tensor],
    ) -> None:
        if not log_probs:
            return
        returns = []
        g = 0.0
        for reward in reversed(rewards):
            g = reward + self.gamma * g
            returns.insert(0, g)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        log_stack = torch.stack(log_probs)
        loss = -(returns_tensor * log_stack).sum()
        entropy_term = torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=self.device)
        total_loss = loss - self.entropy_coef * entropy_term
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    def save(self) -> None:
        self.green_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.purple_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.green_policy.state_dict(), self.green_checkpoint)
        torch.save(self.purple_policy.state_dict(), self.purple_checkpoint)

    def promote_winner(self, winner: str) -> None:
        if winner not in {"green", "purple"}:
            return
        src = self.green_policy if winner == "green" else self.purple_policy
        state = src.state_dict()
        self.green_policy.load_state_dict(state)
        self.purple_policy.load_state_dict(state)
        self.save()
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
