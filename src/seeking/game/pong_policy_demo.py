from __future__ import annotations

from pathlib import Path

import pygame
import torch

from seeking.game.pong import (
    BALL_DEFAULT_COLOR,
    BACKGROUND_COLOR,
    LEFT_COLOR,
    NET_COLOR,
    PADDLE_HEIGHT,
    PADDLE_SPEED,
    PADDLE_WIDTH,
    RIGHT_COLOR,
    SCORE_COLOR,
    TARGET_FPS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    Ball,
    Paddle,
)
from seeking.rl.pong_policy import PongPolicyNetwork


class PongPolicyDemo:
    """Render the Pong UI with a trained policy controlling the left paddle."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        sample_actions: bool = False,
        projectile_shape: str = "ball",
    ) -> None:
        self.device = device
        self.sample_actions = sample_actions
        self.ball_shape = projectile_shape
        self.policy = PongPolicyNetwork().to(self.device)
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Pong policy checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(ckpt, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

        pygame.init()
        pygame.display.set_caption("Pong Policy Demo")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        font_size = 32
        self.font = pygame.font.SysFont("monospace", font_size, bold=True)

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
        self.running = True

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(TARGET_FPS) / 1000.0
            self._handle_events()
            self._step_world(dt * TARGET_FPS)
            self._render()
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False

    def _step_world(self, scale: float) -> None:
        left_dir = self._policy_direction()
        self.left_paddle.move(left_dir, scale)
        right_dir = self._bot_direction()
        self.right_paddle.move(right_dir, scale)

        self.ball.move(scale, self.ball_shape)

        if self.ball.rect.colliderect(self.left_paddle.rect):
            self.ball.rect.left = self.left_paddle.rect.right
            self.ball.velocity[0] = abs(self.ball.velocity[0])
            self.ball.color = self.left_paddle.color
        elif self.ball.rect.colliderect(self.right_paddle.rect):
            self.ball.rect.right = self.right_paddle.rect.left
            self.ball.velocity[0] = -abs(self.ball.velocity[0])
            self.ball.color = self.right_paddle.color

        scored_ball = self.ball.color != BALL_DEFAULT_COLOR
        if self.ball.rect.right < 0:
            if scored_ball:
                self.right_score += 1
            else:
                self.left_penalties += 1
            self.ball.reset()
        elif self.ball.rect.left > WINDOW_WIDTH:
            if scored_ball:
                self.left_score += 1
            else:
                self.right_penalties += 1
            self.ball.reset()

    def _policy_direction(self) -> int:
        state = torch.tensor(
            [
                self.left_paddle.rect.top / WINDOW_HEIGHT,
                self.left_paddle.rect.bottom / WINDOW_HEIGHT,
                self.right_paddle.rect.top / WINDOW_HEIGHT,
                self.right_paddle.rect.bottom / WINDOW_HEIGHT,
                (self.left_score - self.right_score) / 10.0,
                self.ball.rect.centery / WINDOW_HEIGHT,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            logits = self.policy(state)
        if self.sample_actions:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
        else:
            action = torch.argmax(logits).item()
        return -1 if action == 0 else 1

    def _bot_direction(self) -> int:
        if self.ball.rect.centery < self.right_paddle.rect.centery - 10:
            return -1
        if self.ball.rect.centery > self.right_paddle.rect.centery + 10:
            return 1
        return 0

    def _render(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)
        for y in range(0, WINDOW_HEIGHT, 40):
            pygame.draw.rect(self.screen, NET_COLOR, (WINDOW_WIDTH // 2 - 2, y, 4, 20))
        pygame.draw.rect(self.screen, self.left_paddle.color, self.left_paddle.rect)
        pygame.draw.rect(self.screen, self.right_paddle.color, self.right_paddle.rect)
        self.ball.draw(self.screen, self.ball_shape)
        score_text = self.font.render(
            f"LEFT: {self.left_score} (B{self.left_penalties})    RIGHT: {self.right_score} (B{self.right_penalties})",
            True,
            SCORE_COLOR,
        )
        text_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 30))
        self.screen.blit(score_text, text_rect)
        pygame.display.flip()


def run_pong_policy_demo(
    checkpoint_path: str,
    device: torch.device,
    sample_actions: bool = False,
    projectile_shape: str = "ball",
) -> None:
    demo = PongPolicyDemo(
        checkpoint_path=checkpoint_path,
        device=device,
        sample_actions=sample_actions,
        projectile_shape=projectile_shape,
    )
    demo.run()
