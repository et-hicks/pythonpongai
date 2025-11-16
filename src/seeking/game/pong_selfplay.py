from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pygame
import torch
from torch import optim

from seeking.game.pong import (
    BALL_DEFAULT_COLOR,
    BALL_RADIUS,
    BACKGROUND_COLOR,
    LEFT_COLOR,
    NET_COLOR,
    Paddle,
    PADDLE_HEIGHT,
    PADDLE_WIDTH,
    RIGHT_COLOR,
    SCORE_COLOR,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    Ball,
)
from seeking.rl.pong_policy import PongPolicyNetwork


HIT_REWARD = 0.4
SCORE_REWARD = 1.0
SCORE_PENALTY = 1.0
UNTOUCHED_PENALTY = 0.5


@dataclass
class AgentBuffer:
    log_probs: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    def clear(self) -> None:
        self.log_probs.clear()
        self.rewards.clear()


class SelfPlayPong:
    """Runs the Pong UI while two policies train against each other."""

    def __init__(
        self,
        device: torch.device,
        lr: float,
        gamma: float,
        checkpoint_path: str | None = None,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Seeking Pong (Self-Play Training)")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        font_size = 32
        self.font = pygame.font.SysFont("monospace", font_size, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 20, bold=False)

        self.left_paddle = Paddle(
            pygame.Rect(40, WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT),
            LEFT_COLOR,
        )
        self.right_paddle = Paddle(
            pygame.Rect(
                WINDOW_WIDTH - 40 - PADDLE_WIDTH,
                WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2,
                PADDLE_WIDTH,
                PADDLE_HEIGHT,
            ),
            RIGHT_COLOR,
        )
        self.ball = Ball()

        self.left_score = 0
        self.right_score = 0
        self.left_penalties = 0
        self.right_penalties = 0
        self.device = device
        self.gamma = gamma
        self.left_policy = PongPolicyNetwork().to(self.device)
        self.right_policy = PongPolicyNetwork().to(self.device)
        self.left_opt = optim.Adam(self.left_policy.parameters(), lr=lr)
        self.right_opt = optim.Adam(self.right_policy.parameters(), lr=lr)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.left_buffer = AgentBuffer()
        self.right_buffer = AgentBuffer()

        self.running = True
        self.paused = False
        self.rounds_completed = 0
        self.last_left_return = 0.0
        self.last_right_return = 0.0
        self.last_hit: str | None = None
        self._load_checkpoint_if_available()

    def run(self) -> None:
        try:
            while self.running:
                self._handle_events()
                if not self.paused:
                    self._step_training()
                self._render()
                self.clock.tick(120)
        finally:
            self._save_checkpoint()
            pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

    def _step_training(self) -> None:
        left_state = self._build_state(self.left_paddle.rect, self.left_score - self.right_score)
        right_state = self._build_state(self.right_paddle.rect, self.right_score - self.left_score)

        left_action, left_log_prob = self._sample_action(self.left_policy, left_state)
        right_action, right_log_prob = self._sample_action(self.right_policy, right_state)

        self.left_paddle.move(-1 if left_action == 0 else 1)
        self.right_paddle.move(-1 if right_action == 0 else 1)

        self.ball.move()
        left_reward = -abs(self.left_paddle.rect.centery - self.ball.rect.centery) / WINDOW_HEIGHT
        right_reward = -abs(self.right_paddle.rect.centery - self.ball.rect.centery) / WINDOW_HEIGHT

        if self.ball.rect.colliderect(self.left_paddle.rect):
            self.ball.rect.left = self.left_paddle.rect.right
            self.ball.velocity[0] = abs(self.ball.velocity[0])
            self.ball.color = self.left_paddle.color
            left_reward += HIT_REWARD
            self.last_hit = "left"
        elif self.ball.rect.colliderect(self.right_paddle.rect):
            self.ball.rect.right = self.right_paddle.rect.left
            self.ball.velocity[0] = -abs(self.ball.velocity[0])
            self.ball.color = self.right_paddle.color
            right_reward += HIT_REWARD
            self.last_hit = "right"

        ball_active = self.ball.color != BALL_DEFAULT_COLOR
        point_over = False
        if self.ball.rect.right < 0:
            if ball_active:
                self.right_score += 1
                left_reward -= SCORE_PENALTY
                if self.last_hit == "right":
                    right_reward += SCORE_REWARD
            else:
                self.left_penalties += 1
                left_reward -= UNTOUCHED_PENALTY
            point_over = True
        elif self.ball.rect.left > WINDOW_WIDTH:
            if ball_active:
                self.left_score += 1
                right_reward -= SCORE_PENALTY
                if self.last_hit == "left":
                    left_reward += SCORE_REWARD
            else:
                self.right_penalties += 1
                right_reward -= UNTOUCHED_PENALTY
            point_over = True
        if point_over:
            self.last_hit = None

        self.left_buffer.log_probs.append(left_log_prob)
        self.left_buffer.rewards.append(left_reward)
        self.right_buffer.log_probs.append(right_log_prob)
        self.right_buffer.rewards.append(right_reward)

        if point_over:
            self._complete_round()

    def _complete_round(self) -> None:
        left_return = self._update_agent(self.left_buffer, self.left_opt)
        right_return = self._update_agent(self.right_buffer, self.right_opt)
        self.last_left_return = left_return
        self.last_right_return = right_return
        self.left_buffer.clear()
        self.right_buffer.clear()
        self.ball.reset()
        self.last_hit = None
        self.rounds_completed += 1

    def _update_agent(self, buffer: AgentBuffer, optimizer: optim.Optimizer) -> float:
        if not buffer.log_probs:
            return 0.0
        returns = []
        g = 0.0
        for reward in reversed(buffer.rewards):
            g = reward + self.gamma * g
            returns.insert(0, g)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        log_stack = torch.stack(buffer.log_probs)
        loss = -(returns_tensor * log_stack).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(returns_tensor.mean().item())

    def _sample_action(self, policy: PongPolicyNetwork, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        logits = policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def _build_state(self, paddle_rect: pygame.Rect, score_delta: int) -> torch.Tensor:
        state = torch.tensor(
            [
                paddle_rect.top / WINDOW_HEIGHT,
                paddle_rect.bottom / WINDOW_HEIGHT,
                score_delta / 10.0,
                self.ball.rect.centery / WINDOW_HEIGHT,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return state

    def _render(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)
        for y in range(0, WINDOW_HEIGHT, 40):
            pygame.draw.rect(self.screen, NET_COLOR, (WINDOW_WIDTH // 2 - 2, y, 4, 20))
        pygame.draw.rect(self.screen, self.left_paddle.color, self.left_paddle.rect)
        pygame.draw.rect(self.screen, self.right_paddle.color, self.right_paddle.rect)
        self.ball.draw(self.screen)

        score_text = self.font.render(
            f"LEFT: {self.left_score} (P{self.left_penalties})    "
            f"RIGHT: {self.right_score} (P{self.right_penalties})",
            True,
            SCORE_COLOR,
        )
        text_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 30))
        self.screen.blit(score_text, text_rect)

        info_text = self.small_font.render(
            f"Self-play rounds: {self.rounds_completed}  "
            f"Avg returns (L/R): {self.last_left_return:.3f} / {self.last_right_return:.3f}  "
            "SPACE=Pause  Q=Quit",
            True,
            SCORE_COLOR,
        )
        info_rect = info_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 20))
        self.screen.blit(info_text, info_rect)

        if self.paused:
            paused_text = self.font.render("PAUSED (SPACE TO RESUME, Q TO QUIT)", True, SCORE_COLOR)
            paused_rect = paused_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            pygame.draw.rect(self.screen, (0, 0, 0), paused_rect.inflate(40, 20))
            self.screen.blit(paused_text, paused_rect)

        pygame.display.flip()


    def _load_checkpoint_if_available(self) -> None:
        if not self.checkpoint_path:
            return
        if not self.checkpoint_path.exists():
            return
        try:
            payload = torch.load(self.checkpoint_path, map_location=self.device)
        except OSError:
            return
        left_state = payload.get("left_policy")
        right_state = payload.get("right_policy")
        if left_state:
            self.left_policy.load_state_dict(left_state)
        if right_state:
            self.right_policy.load_state_dict(right_state)
        left_opt_state = payload.get("left_opt")
        right_opt_state = payload.get("right_opt")
        if left_opt_state:
            self.left_opt.load_state_dict(left_opt_state)
        if right_opt_state:
            self.right_opt.load_state_dict(right_opt_state)
        meta = payload.get("meta", {})
        self.rounds_completed = int(meta.get("rounds_completed", 0))
        self.last_left_return = float(meta.get("last_left_return", 0.0))
        self.last_right_return = float(meta.get("last_right_return", 0.0))
        self.left_penalties = int(meta.get("left_penalties", 0))
        self.right_penalties = int(meta.get("right_penalties", 0))
        print(f"[Pong SelfPlay] Loaded checkpoint from {self.checkpoint_path}")

    def _save_checkpoint(self) -> None:
        if not self.checkpoint_path:
            return
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        payload = {
            "left_policy": self.left_policy.state_dict(),
            "right_policy": self.right_policy.state_dict(),
            "left_opt": self.left_opt.state_dict(),
            "right_opt": self.right_opt.state_dict(),
            "meta": {
                "rounds_completed": self.rounds_completed,
                "last_left_return": self.last_left_return,
                "last_right_return": self.last_right_return,
                "left_penalties": self.left_penalties,
                "right_penalties": self.right_penalties,
            },
        }
        torch.save(payload, self.checkpoint_path)
        print(f"[Pong SelfPlay] Saved checkpoint to {self.checkpoint_path}")


def run_pong_selfplay(
    device: torch.device,
    lr: float,
    gamma: float,
    checkpoint_path: str | None = None,
) -> None:
    """Entry point used by the CLI to launch the self-play UI."""
    game = SelfPlayPong(device=device, lr=lr, gamma=gamma, checkpoint_path=checkpoint_path)
    game.run()
