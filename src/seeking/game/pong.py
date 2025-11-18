from __future__ import annotations

import random
from dataclasses import dataclass
import math
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
HIGHLIGHT_COLOR = (255, 220, 0)
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
        self.angle = random.uniform(0, math.tau)
        self.angular_velocity = random.uniform(-3.0, 3.0)
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
        self.angle = random.uniform(0, math.tau)
        self.angular_velocity = random.uniform(-3.0, 3.0)

    def move(self, scale: float = 1.0, shape: str = "ball") -> None:
        self._x_float += self.velocity[0] * scale
        self._y_float += self.velocity[1] * scale
        self.rect.x = int(self._x_float)
        self.rect.y = int(self._y_float)
        if self.rect.top <= 0 or self.rect.bottom >= WINDOW_HEIGHT:
            self.velocity[1] *= -1
            self._y_float = float(self.rect.y)
            if shape != "ball":
                self._apply_wall_spin(shape)
        if shape in {"triangle", "square"}:
            self.angle = (self.angle + math.radians(self.angular_velocity) * scale) % (2 * math.pi)

    def draw(self, surface: pygame.Surface, shape: str = "ball") -> None:
        if shape == "square":
            vertices = self.square_vertices()
            if vertices:
                pygame.draw.polygon(surface, self.color, vertices)
            else:
                pygame.draw.rect(surface, self.color, self.rect)
        elif shape == "triangle":
            vertices = self.triangle_vertices()
            if vertices:
                pygame.draw.polygon(surface, self.color, vertices)
            else:
                pygame.draw.polygon(
                    surface,
                    self.color,
                    [
                        (self.rect.centerx, self.rect.top),
                        (self.rect.left, self.rect.bottom),
                        (self.rect.right, self.rect.bottom),
                    ],
                )
        else:
            pygame.draw.circle(surface, self.color, self.rect.center, BALL_RADIUS)

    def triangle_vertices(self) -> list[tuple[float, float]]:
        if not hasattr(self, "angle"):
            return []
        center = (self._x_float + BALL_RADIUS, self._y_float + BALL_RADIUS)
        vertices = []
        for offset in (0, 2 * math.pi / 3, 4 * math.pi / 3):
            ang = self.angle + offset
            x = center[0] + BALL_RADIUS * math.cos(ang)
            y = center[1] + BALL_RADIUS * math.sin(ang)
            vertices.append((x, y))
        return vertices

    def square_vertices(self) -> list[tuple[float, float]]:
        if not hasattr(self, "angle"):
            return []
        center = (self._x_float + BALL_RADIUS, self._y_float + BALL_RADIUS)
        half = BALL_RADIUS
        vertices = []
        for offset in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4):
            ang = self.angle + offset
            x = center[0] + half * math.sqrt(2) * math.cos(ang)
            y = center[1] + half * math.sqrt(2) * math.sin(ang)
            vertices.append((x, y))
        return vertices

    def add_spin(self, strength: float = 5.0) -> None:
        self.angular_velocity += random.uniform(-strength, strength)
        self.angular_velocity = max(-12.0, min(12.0, self.angular_velocity))

    def _apply_wall_spin(self, shape: str) -> None:
        if shape == "square":
            self.velocity[0] += random.uniform(-0.5, 0.5)
            self.add_spin(3.0)
        elif shape == "triangle":
            self.add_spin(4.0)
        self._normalize_speed()

    def _normalize_speed(self) -> None:
        speed = (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5
        if speed == 0:
            speed = 1.0
        scale = BALL_SPEED / speed
        self.velocity[0] *= scale
        self.velocity[1] *= scale


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
                if event.key == pygame.K_q:
                    self.running = False
                    continue
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
                if event.key == pygame.K_SPACE:
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
        self.ball.move(scale, self.ball_shape)
        if self.ball.rect.colliderect(self.left_paddle.rect):
            self.ball.rect.left = self.left_paddle.rect.right
            self.ball._x_float = float(self.ball.rect.x)
            self.ball.color = self.left_paddle.color
            self.rally_hits += 1
            if self.ball_shape == "triangle":
                normal = self._triangle_edge_normal((1.0, 0.0))
                self._reflect_velocity(normal, chaotic=True)
                self.ball.add_spin(6.0)
            else:
                self.ball.velocity[0] = abs(self.ball.velocity[0])
                self._apply_shape_bounce()
        elif self.ball.rect.colliderect(self.right_paddle.rect):
            self.ball.rect.right = self.right_paddle.rect.left
            self.ball._x_float = float(self.ball.rect.x)
            self.ball.color = self.right_paddle.color
            self.rally_hits += 1
            if self.ball_shape == "triangle":
                normal = self._triangle_edge_normal((-1.0, 0.0))
                self._reflect_velocity(normal, chaotic=True)
                self.ball.add_spin(6.0)
            else:
                self.ball.velocity[0] = -abs(self.ball.velocity[0])
                self._apply_shape_bounce()

        scored_ball = self.ball.color != BALL_DEFAULT_COLOR
        if self.ball.rect.right < 0:
            if scored_ball:
                self.right_score += 1
            else:
                self.left_penalties += 1
            self.ball.reset()
            self.rally_hits = 0
            self.ball._normalize_speed()
        elif self.ball.rect.left > WINDOW_WIDTH:
            if scored_ball:
                self.left_score += 1
            else:
                self.right_penalties += 1
            self.ball.reset()
            self.rally_hits = 0
            self.ball._normalize_speed()

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
        shape_symbols = {"ball": "BALL", "square": "SQUARE", "triangle": "TRI"}
        base_y = WINDOW_HEIGHT // 3 + 240
        offsets = [-120, 0, 120]
        for idx, shape in enumerate(self.shape_options):
            symbol = shape_symbols.get(shape, shape.upper())
            display = f"({symbol})" if idx == self.shape_index else symbol
            color = HIGHLIGHT_COLOR if idx == self.shape_index else SCORE_COLOR
            shape_text = self.font.render(display, True, color)
            self.screen.blit(shape_text, shape_text.get_rect(center=(WINDOW_WIDTH // 2 + offsets[idx], base_y)))
        hint_shape = self.font.render("< / > to change projectile shape", True, SCORE_COLOR)
        self.screen.blit(hint_shape, hint_shape.get_rect(center=(WINDOW_WIDTH // 2, base_y + 50)))
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

    def _apply_shape_bounce(self) -> None:
        if self.ball_shape == "ball":
            return
        if self.ball_shape == "square":
            self.ball.velocity[1] += random.uniform(-2.0, 2.0)
            self.ball.add_spin(4.0)
            self.ball._normalize_speed()
        elif self.ball_shape == "triangle":
            # handled by reflection logic
            pass

    def _triangle_edge_normal(self, target_normal: tuple[float, float]) -> tuple[float, float]:
        verts = self.ball.triangle_vertices()
        if not verts:
            return target_normal
        normals = []
        center = self.ball.rect.center
        for idx in range(3):
            p1 = verts[idx]
            p2 = verts[(idx + 1) % 3]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = self._normalize_vector((edge[1], -edge[0]))
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            to_center = (center[0] - mid[0], center[1] - mid[1])
            if normal[0] * to_center[0] + normal[1] * to_center[1] > 0:
                normal = (-normal[0], -normal[1])
            normals.append(normal)
        target_normal = self._normalize_vector(target_normal)
        normals.sort(key=lambda n: n[0] * target_normal[0] + n[1] * target_normal[1], reverse=True)
        return normals[0]

    def _reflect_velocity(self, normal: tuple[float, float], chaotic: bool = False) -> None:
        normal = self._normalize_vector(normal)
        vx, vy = self.ball.velocity
        dot = vx * normal[0] + vy * normal[1]
        vx = vx - 2 * dot * normal[0]
        vy = vy - 2 * dot * normal[1]
        if chaotic:
            angle = random.uniform(-0.12, 0.12)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            vx, vy = vx * cos_a - vy * sin_a, vx * sin_a + vy * cos_a
        self.ball.velocity[0] = vx
        self.ball.velocity[1] = vy
        self.ball._normalize_speed()

    def _normalize_vector(self, vec: tuple[float, float]) -> tuple[float, float]:
        length = (vec[0] ** 2 + vec[1] ** 2) ** 0.5
        if length == 0:
            return (0.0, 0.0)
        return (vec[0] / length, vec[1] / length)

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
