from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pygame
import torch

from seeking.rl.pong_policy import PongPolicyNetwork

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

PADDLE_WIDTH = 20
PADDLE_HEIGHT = 180
PADDLE_SPEED = 9
BALL_RADIUS = 14
BALL_SPEED = 8
TARGET_FPS = 120

LEFT_COLOR = (0, 200, 120)
RIGHT_COLOR = (180, 0, 255)
BACKGROUND_COLOR = (15, 15, 20)
NET_COLOR = (70, 70, 90)
SCORE_COLOR = (255, 255, 255)
BALL_DEFAULT_COLOR = (255, 255, 255)


@dataclass
class Paddle:
    rect: pygame.Rect
    color: tuple[int, int, int]
    _y_float: float = 0.0

    def __post_init__(self) -> None:
        self._y_float = float(self.rect.y)

    def move(self, direction: int, scale: float = 1.0) -> None:
        self._y_float += direction * PADDLE_SPEED * scale
        self._y_float = max(0.0, min(WINDOW_HEIGHT - self.rect.height, self._y_float))
        self.rect.y = int(self._y_float)


class Ball:
    def __init__(self) -> None:
        self.rect = pygame.Rect(
            WINDOW_WIDTH // 2 - BALL_RADIUS,
            WINDOW_HEIGHT // 2 - BALL_RADIUS,
            BALL_RADIUS * 2,
            BALL_RADIUS * 2,
        )
        self.velocity = [BALL_SPEED, BALL_SPEED]
        self.color = BALL_DEFAULT_COLOR
        self._x_float = float(self.rect.x)
        self._y_float = float(self.rect.y)
        self.reset()

    def reset(self) -> None:
        offset = random.randint(-100, 100)
        center_y = WINDOW_HEIGHT // 2 + offset
        center_y = max(BALL_RADIUS, min(WINDOW_HEIGHT - BALL_RADIUS, center_y))
        self._x_float = float(WINDOW_WIDTH // 2 - BALL_RADIUS)
        self._y_float = float(center_y - BALL_RADIUS)
        self.rect.center = (WINDOW_WIDTH // 2, center_y)
        horizontal = random.choice([-1, 1])
        vertical = random.uniform(-0.7, 0.7)
        magnitude = (vertical**2 + 1) ** 0.5
        self.velocity[0] = horizontal * BALL_SPEED / magnitude
        self.velocity[1] = vertical * BALL_SPEED / magnitude
        self.color = BALL_DEFAULT_COLOR

    def move(self, scale: float = 1.0) -> None:
        self._x_float += self.velocity[0] * scale
        self._y_float += self.velocity[1] * scale
        self.rect.x = int(self._x_float)
        self.rect.y = int(self._y_float)
        if self.rect.top <= 0 or self.rect.bottom >= WINDOW_HEIGHT:
            self.velocity[1] *= -1
            self._y_float = float(self.rect.y)

    def draw(self, surface: pygame.Surface, shape: str = "ball") -> None:
        if shape == "square":
            pygame.draw.rect(surface, self.color, self.rect)
        elif shape == "triangle":
            top = (self.rect.centerx, self.rect.top)
            left = (self.rect.left, self.rect.bottom)
            right = (self.rect.right, self.rect.bottom)
            pygame.draw.polygon(surface, self.color, [top, left, right])
        else:
            pygame.draw.circle(surface, self.color, self.rect.center, BALL_RADIUS)


MODE_LABELS = {
    "two_player": "MODE: TWO PLAYER (GREEN vs PURPLE)",
    "human_vs_ai": "MODE: HUMAN vs AI (GREEN = HUMAN)",
    "ai_vs_ai": "MODE: AI vs AI (auto battle)",
}


class PongGame:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        green_checkpoint: str | None = None,
        purple_checkpoint: str | None = None,
        default_shape: str = "ball",
        green_shape: str = "ball",
        purple_shape: str = "ball",
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Seeking Pong")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        font_size = 32
        self.font = pygame.font.SysFont("monospace", font_size, bold=True)
        self.left_score = 0
        self.right_score = 0
        self.left_penalties = 0
        self.right_penalties = 0
        self.rally_hits = 0
        self.device = device or torch.device("cpu")
        self.green_checkpoint = Path(green_checkpoint) if green_checkpoint else None
        self.purple_checkpoint = Path(purple_checkpoint) if purple_checkpoint else None
        self.green_policy: PongPolicyNetwork | None = None
        self.purple_policy: PongPolicyNetwork | None = None
        self.green_shape = green_shape
        self.purple_shape = purple_shape
        self.menu_active = True
        self.mode: str | None = None
        self.mode_label = "SELECT MODE: [1] Two Player  [2] Human vs AI  [3] AI vs AI"
        self.ai_vs_ai_used = False
        self.pause_menu_active = False
        self.shape_options = ["ball", "square", "triangle"]
        self.shape_index = self.shape_options.index(default_shape) if default_shape in self.shape_options else 0
        self.ball_shape = self.shape_options[self.shape_index]

        left_rect = pygame.Rect(
            40, WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT
        )
        right_rect = pygame.Rect(
            WINDOW_WIDTH - 40 - PADDLE_WIDTH,
            WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2,
            PADDLE_WIDTH,
            PADDLE_HEIGHT,
        )
        self.left_paddle = Paddle(left_rect, LEFT_COLOR)
        self.right_paddle = Paddle(right_rect, RIGHT_COLOR)
        self.ball = Ball()
        self.running = True
        self.paused = False

    def run(self) -> None:
        while self.running:
            self._handle_events()
            dt = self.clock.tick(TARGET_FPS) / 1000.0
            if self.menu_active:
                self._render_menu()
                continue
            if not self.paused:
                scale = dt * TARGET_FPS
                self._handle_input(scale)
                self._update_world(scale)
            self._render()
        self._finalize_ai_battle()
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self.menu_active:
                    if event.key == pygame.K_1:
                        self._set_mode("two_player")
                    elif event.key == pygame.K_2:
                        self._set_mode("human_vs_ai")
                    elif event.key == pygame.K_3:
                        self._set_mode("ai_vs_ai")
                    elif event.key == pygame.K_LEFT:
                        self._shift_shape(-1)
                    elif event.key == pygame.K_RIGHT:
                        self._shift_shape(1)
                    continue
                if self.pause_menu_active:
                    if event.key == pygame.K_r:
                        self.paused = False
                        self.pause_menu_active = False
                    elif event.key == pygame.K_m:
                        self._return_to_menu()
                    continue
                if event.key == pygame.K_ESCAPE:
                    self.paused = True
                    self.pause_menu_active = True
                    continue
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

    def _handle_input(self, scale: float) -> None:
        if self.mode == "ai_vs_ai":
            left_dir = self._policy_direction(
                self.green_policy, self.left_paddle.rect, self.right_paddle.rect, self.left_score - self.right_score
            )
            right_dir = self._policy_direction(
                self.purple_policy, self.right_paddle.rect, self.left_paddle.rect, self.right_score - self.left_score
            )
            self.left_paddle.move(left_dir, scale)
            self.right_paddle.move(right_dir, scale)
            return

        keys = pygame.key.get_pressed()
        left_direction = 0
        if keys[pygame.K_w] or keys[pygame.K_a]:
            left_direction -= 1
        if keys[pygame.K_s] or keys[pygame.K_d]:
            left_direction += 1

        if self.mode == "human_vs_ai":
            right_direction = self._policy_direction(
                self.purple_policy, self.right_paddle.rect, self.left_paddle.rect, self.right_score - self.left_score
            )
        else:
            right_direction = 0
            if keys[pygame.K_UP] or keys[pygame.K_LEFT]:
                right_direction -= 1
            if keys[pygame.K_DOWN] or keys[pygame.K_RIGHT]:
                right_direction += 1

        self.left_paddle.move(left_direction, scale)
        self.right_paddle.move(right_direction, scale)

    def _update_world(self, scale: float) -> None:
        self.ball.move(scale)
        if self.ball.rect.colliderect(self.left_paddle.rect):
            self.ball.rect.left = self.left_paddle.rect.right
            self.ball.velocity[0] = abs(self.ball.velocity[0])
            self.ball.color = self.left_paddle.color
            self.rally_hits += 1
        elif self.ball.rect.colliderect(self.right_paddle.rect):
            self.ball.rect.right = self.right_paddle.rect.left
            self.ball.velocity[0] = -abs(self.ball.velocity[0])
            self.ball.color = self.right_paddle.color
            self.rally_hits += 1

        scored_ball = self.ball.color != BALL_DEFAULT_COLOR
        if self.ball.rect.right < 0:
            if scored_ball:
                self.right_score += 1
            else:
                self.left_penalties += 1
            self.ball.reset()
            self.rally_hits = 0
        elif self.ball.rect.left > WINDOW_WIDTH:
            if scored_ball:
                self.left_score += 1
            else:
                self.right_penalties += 1
            self.ball.reset()
            self.rally_hits = 0

    def _render(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)
        for y in range(0, WINDOW_HEIGHT, 40):
            pygame.draw.rect(self.screen, NET_COLOR, (WINDOW_WIDTH // 2 - 2, y, 4, 20))
        pygame.draw.rect(self.screen, self.left_paddle.color, self.left_paddle.rect)
        pygame.draw.rect(self.screen, self.right_paddle.color, self.right_paddle.rect)
        self.ball.draw(self.screen, self.ball_shape)
        score_text = self.font.render(
            f"LEFT: {self.left_score} (B{self.left_penalties})    "
            f"RIGHT: {self.right_score} (B{self.right_penalties})",
            True,
            SCORE_COLOR,
        )
        text_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 30))
        self.screen.blit(score_text, text_rect)
        rally_text = self.font.render(f"RALLY HITS: {self.rally_hits}", True, SCORE_COLOR)
        rally_rect = rally_text.get_rect(center=(WINDOW_WIDTH // 2, 70))
        self.screen.blit(rally_text, rally_rect)
        mode_text = self.font.render(f"{self.mode_label} | Projectile: {self.ball_shape.upper()}", True, SCORE_COLOR)
        mode_rect = mode_text.get_rect(center=(WINDOW_WIDTH // 2, 110))
        self.screen.blit(mode_text, mode_rect)
        if self.paused:
            if self.pause_menu_active:
                options = [
                    "PAUSED",
                    "[R] Resume",
                    "[M] Return to Main Menu",
                    "SPACE also toggles pause",
                ]
            else:
                options = ["PAUSED (SPACE TO RESUME, Q TO QUIT)"]
            for idx, text in enumerate(options):
                paused_text = self.font.render(text, True, SCORE_COLOR)
                paused_rect = paused_text.get_rect(
                    center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + idx * 40)
                )
                pygame.draw.rect(
                    self.screen,
                    (0, 0, 0),
                    paused_rect.inflate(40, 20),
                )
                self.screen.blit(paused_text, paused_rect)
        pygame.display.flip()

    def _render_menu(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)
        title = self.font.render("Seeking Pong - Select Mode", True, SCORE_COLOR)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3))
        self.screen.blit(title, title_rect)
        option1 = self.font.render("[1] Two Player", True, LEFT_COLOR)
        option2 = self.font.render("[2] Human vs AI", True, SCORE_COLOR)
        option3 = self.font.render("[3] AI vs AI (battle)", True, RIGHT_COLOR)
        self.screen.blit(option1, option1.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3 + 60)))
        self.screen.blit(option2, option2.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3 + 120)))
        self.screen.blit(option3, option3.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3 + 180)))
        shape_text = self.font.render(
            f"Projectile Shape: {self.ball_shape.upper()}  (←/→ to change)", True, SCORE_COLOR
        )
        self.screen.blit(shape_text, shape_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3 + 240)))
        hint = self.font.render("Press Q to quit", True, SCORE_COLOR)
        self.screen.blit(hint, hint.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 80)))
        pygame.display.flip()

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        base_label = MODE_LABELS.get(mode, "MODE")
        if mode in {"human_vs_ai", "ai_vs_ai"}:
            base_label += f" | GREEN AI: {self.green_shape.upper()}  PURPLE AI: {self.purple_shape.upper()}"
        self.mode_label = base_label
        self.menu_active = False
        self._reset_game_state()
        if mode in {"human_vs_ai", "ai_vs_ai"}:
            self._load_policies()
        self.ai_vs_ai_used = mode == "ai_vs_ai"
        self.pause_menu_active = False

    def _shift_shape(self, delta: int) -> None:
        self.shape_index = (self.shape_index + delta) % len(self.shape_options)
        self.ball_shape = self.shape_options[self.shape_index]

    def _reset_game_state(self) -> None:
        self.paused = False
        self.pause_menu_active = False
        self.left_score = 0
        self.right_score = 0
        self.left_penalties = 0
        self.right_penalties = 0
        self.rally_hits = 0
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
        self.ball.reset()

    def _load_policies(self) -> None:
        self.green_policy = self._load_policy(self.green_checkpoint)
        self.purple_policy = self._load_policy(self.purple_checkpoint)

    def _load_policy(self, path: Optional[Path]) -> PongPolicyNetwork | None:
        if not path:
            return None
        if not path.exists():
            print(f"[Pong] Policy checkpoint not found: {path}")
            return None
        policy = PongPolicyNetwork().to(self.device)
        policy.load_state_dict(torch.load(path, map_location=self.device))
        policy.eval()
        return policy

    def _policy_direction(
        self,
        policy: PongPolicyNetwork | None,
        paddle_rect: pygame.Rect,
        opponent_rect: pygame.Rect,
        score_delta: int,
    ) -> int:
        if policy is None:
            return self._tracking_direction(paddle_rect)
        state = torch.tensor(
            [
                paddle_rect.top / WINDOW_HEIGHT,
                paddle_rect.bottom / WINDOW_HEIGHT,
                opponent_rect.top / WINDOW_HEIGHT,
                opponent_rect.bottom / WINDOW_HEIGHT,
                score_delta / 10.0,
                self.ball.rect.centery / WINDOW_HEIGHT,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            logits = policy(state)
        action = torch.argmax(logits).item()
        return -1 if action == 0 else 1

    def _tracking_direction(self, paddle_rect: pygame.Rect) -> int:
        if self.ball.rect.centery < paddle_rect.centery - 10:
            return -1
        if self.ball.rect.centery > paddle_rect.centery + 10:
            return 1
        return 0

    def _finalize_ai_battle(self) -> None:
        if not self.ai_vs_ai_used:
            return
        if self.left_score == self.right_score:
            print("[Pong] AI vs AI ended in a tie. Checkpoints unchanged.")
            self.ai_vs_ai_used = False
            return
        winner_policy = self.green_policy if self.left_score > self.right_score else self.purple_policy
        if winner_policy is None or not self.green_checkpoint or not self.purple_checkpoint:
            print("[Pong] AI vs AI completed, but checkpoints missing; cannot promote winner.")
            self.ai_vs_ai_used = False
            return
        winner_name = "GREEN" if self.left_score > self.right_score else "PURPLE"
        payload = winner_policy.state_dict()
        self.green_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.purple_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.green_checkpoint)
        torch.save(payload, self.purple_checkpoint)
        print(f"[Pong] AI vs AI winner: {winner_name}. Checkpoints updated.")
        self.ai_vs_ai_used = False

    def _return_to_menu(self) -> None:
        if self.mode == "ai_vs_ai":
            self._finalize_ai_battle()
        self.mode = None
        self.mode_label = "SELECT MODE: [1] Two Player  [2] Human vs AI  [3] AI vs AI"
        self.menu_active = True
        self.ai_vs_ai_used = False
        self._reset_game_state()


def run_pong(
    device: Optional[torch.device] = None,
    green_checkpoint: str | None = None,
    purple_checkpoint: str | None = None,
    default_shape: str = "ball",
    green_shape: str = "ball",
    purple_shape: str = "ball",
) -> None:
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = PongGame(
        device=resolved_device,
        green_checkpoint=green_checkpoint,
        purple_checkpoint=purple_checkpoint,
        default_shape=default_shape,
        green_shape=green_shape,
        purple_shape=purple_shape,
    )
    game.run()
