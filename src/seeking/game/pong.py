from __future__ import annotations

import random
from dataclasses import dataclass

import pygame

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

PADDLE_WIDTH = 20
PADDLE_HEIGHT = 180
PADDLE_SPEED = 9
BALL_RADIUS = 14
BALL_SPEED = 8

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

    def move(self, direction: int) -> None:
        self.rect.y += direction * PADDLE_SPEED
        self.rect.y = max(0, min(WINDOW_HEIGHT - self.rect.height, self.rect.y))


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
        self.reset()

    def reset(self) -> None:
        offset = random.randint(-100, 100)
        center_y = WINDOW_HEIGHT // 2 + offset
        center_y = max(BALL_RADIUS, min(WINDOW_HEIGHT - BALL_RADIUS, center_y))
        self.rect.center = (WINDOW_WIDTH // 2, center_y)
        horizontal = random.choice([-1, 1])
        vertical = random.uniform(-0.7, 0.7)
        magnitude = (vertical**2 + 1) ** 0.5
        self.velocity[0] = horizontal * BALL_SPEED / magnitude
        self.velocity[1] = vertical * BALL_SPEED / magnitude
        self.color = BALL_DEFAULT_COLOR

    def move(self) -> None:
        self.rect.x += int(self.velocity[0])
        self.rect.y += int(self.velocity[1])
        if self.rect.top <= 0 or self.rect.bottom >= WINDOW_HEIGHT:
            self.velocity[1] *= -1

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, self.color, self.rect.center, BALL_RADIUS)


class PongGame:
    def __init__(self) -> None:
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
            if not self.paused:
                self._handle_input()
                self._update_world()
            self._render()
            self.clock.tick(120)
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

    def _handle_input(self) -> None:
        keys = pygame.key.get_pressed()
        left_direction = 0
        right_direction = 0

        if keys[pygame.K_w] or keys[pygame.K_a]:
            left_direction -= 1
        if keys[pygame.K_s] or keys[pygame.K_d]:
            left_direction += 1
        if keys[pygame.K_UP] or keys[pygame.K_LEFT]:
            right_direction -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_RIGHT]:
            right_direction += 1

        self.left_paddle.move(left_direction)
        self.right_paddle.move(right_direction)

    def _update_world(self) -> None:
        self.ball.move()
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
        if self.paused:
            paused_text = self.font.render("PAUSED (SPACE TO RESUME, Q TO QUIT)", True, SCORE_COLOR)
            paused_rect = paused_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                paused_rect.inflate(40, 20),
            )
            self.screen.blit(paused_text, paused_rect)
        pygame.display.flip()


def run_pong() -> None:
    game = PongGame()
    game.run()
