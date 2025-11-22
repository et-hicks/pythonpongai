from __future__ import annotations

import random

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import arcade

from seeking.rl.fish_trainer import (
    ACTION_SPACE,
    FishActorCritic,
    FishDQN,
    FishRegressionController,
    QuantizedFishState,
    quantize_ascii_frame,
    Transition,
)

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

FOOD_PROB = 0.7
BOMB_PROB = 0.2
COIN_PROB = 0.1
TOTAL_DROPS = 40
DROP_INTERVAL = 0.55
DROP_SPEED = 220.0

FISH_SPEED = 280.0
ENERGY_DECAY_SECONDS = 1.0
ENERGY_MAX = 100
FOOD_BOOST = 15
BOMB_PENALTY = 20

WAVE_FRAMES = ["^~~~", "~^~~", "~~^~", "~~~^"]

ASCII_COLOR = arcade.color.WHITE
BACKGROUND_COLOR = arcade.color.BLACK


@dataclass
class Drop:
    x: float
    y: float
    label: str

    def sprite(self) -> str:
        if self.label == "food":
            return "."
        if self.label == "bomb":
            return "@"
        return "$"


@dataclass
class Fish:
    x: float
    y: float
    art: str = "o=-<"
    size: float = 24

    def move(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy
        self.x = max(self.size, min(WINDOW_WIDTH - self.size, self.x))
        self.y = max(self.size, min(WINDOW_HEIGHT - self.size * 2, self.y))

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        half_w = self.size * 1.5
        half_h = self.size * 0.6
        return (self.x - half_w, self.x + half_w, self.y - half_h, self.y + half_h)


class FishFoodGame(arcade.Window):
    def __init__(self) -> None:
        update_rate = 1 / 60
        super().__init__(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            "ASCII Fish Food",
            update_rate=update_rate,
            draw_rate=update_rate,
        )
        self.set_location(150, 80)
        arcade.set_background_color(BACKGROUND_COLOR)
        self.state = "mode_select"
        self.wave_frame = 0
        self.wave_timer = 0.0
        self.energy = ENERGY_MAX
        self.energy_timer = 0.0
        self.drops: List[Drop] = []
        self.spawned = 0
        self.spawn_timer = 0.0
        self.fish = Fish(x=WINDOW_WIDTH / 2, y=120)
        self.dollars = 0
        self.active_controller: Optional[Callable[[QuantizedFishState], str]] = None
        self.controller_name = "Player"
        self.dqn: Optional[FishDQN] = None
        self.actor_critic: Optional[FishActorCritic] = None
        self.regression: Optional[FishRegressionController] = None
        self.last_state: Optional[QuantizedFishState] = None
        self.last_action: Optional[str] = None
        self.font_size = 18
        self._move_vector = [0.0, 0.0]

    # arcade hooks ---------------------------------------------------- #
    def on_draw(self) -> None:
        arcade.start_render()
        if self.state == "mode_select":
            self._draw_mode_select()
            return
        if self.state == "ai_select":
            self._draw_ai_select()
            return
        self._draw_gameplay()
        if self.state == "game_over":
            self._draw_game_over()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if self.state == "mode_select":
            if symbol in (arcade.key.KEY_1, arcade.key.NUM_1):
                self.state = "running"
                self.controller_name = "Player"
                self.reset_game()
            elif symbol in (arcade.key.KEY_2, arcade.key.NUM_2):
                self.state = "ai_select"
            return
        if self.state == "ai_select":
            if symbol in (arcade.key.KEY_1, arcade.key.NUM_1):
                self._select_ai("dqn")
            elif symbol in (arcade.key.KEY_2, arcade.key.NUM_2):
                self._select_ai("ac")
            elif symbol in (arcade.key.KEY_3, arcade.key.NUM_3):
                self._select_ai("regression")
            return
        if self.state == "game_over":
            if symbol == arcade.key.Y:
                self.state = "running"
                self.reset_game()
            elif symbol == arcade.key.N:
                arcade.close_window()
            return
        if self.active_controller is None:
            self._handle_player_input(symbol, modifiers)

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        if self.active_controller is None:
            if symbol in (arcade.key.UP, arcade.key.W):
                self._move_vector[1] = 0
            elif symbol in (arcade.key.DOWN, arcade.key.S):
                self._move_vector[1] = 0
            elif symbol in (arcade.key.LEFT, arcade.key.A):
                self._move_vector[0] = 0
            elif symbol in (arcade.key.RIGHT, arcade.key.D):
                self._move_vector[0] = 0

    def setup(self) -> None:
        self.reset_game()
        self.state = "mode_select"
        self.drops = []
        self.spawned = 0
        self.spawn_timer = 0.0

    # game logic ------------------------------------------------------ #
    def reset_game(self) -> None:
        self.energy = ENERGY_MAX
        self.energy_timer = 0.0
        self.drops = []
        self.spawned = 0
        self.spawn_timer = 0.0
        self.fish = Fish(x=WINDOW_WIDTH / 2, y=120)
        self.dollars = 0
        self.wave_frame = 0
        self.wave_timer = 0.0
        self.state = "running"
        self._move_vector = [0.0, 0.0]
        self.last_state = None
        self.last_action = None
        if self.active_controller is None:
            self.set_update_rate(1 / 60)

    def _select_ai(self, kind: str) -> None:
        device = None
        if kind == "dqn":
            self.dqn = FishDQN(device=device)
            self.active_controller = self.dqn.act
            self.controller_name = "DQN"
        elif kind == "ac":
            self.actor_critic = FishActorCritic(device=device)
            self.active_controller = self.actor_critic.act
            self.controller_name = "Actor-Critic"
        else:
            self.regression = FishRegressionController(device=device)
            self.active_controller = self.regression.act
            self.controller_name = "Regression"
        self.set_update_rate(0.1)
        self.reset_game()

    def _handle_player_input(self, symbol: int, modifiers: int) -> None:
        if symbol in (arcade.key.UP, arcade.key.W):
            self._move_vector[1] = 1
        elif symbol in (arcade.key.DOWN, arcade.key.S):
            self._move_vector[1] = -1
        elif symbol in (arcade.key.LEFT, arcade.key.A):
            self._move_vector[0] = -1
        elif symbol in (arcade.key.RIGHT, arcade.key.D):
            self._move_vector[0] = 1

    def _spawn_drop(self) -> None:
        if self.spawned >= TOTAL_DROPS:
            return
        pick = random.random()
        if pick < FOOD_PROB:
            label = "food"
        elif pick < FOOD_PROB + BOMB_PROB:
            label = "bomb"
        else:
            label = "coin"
        x = random.uniform(20, WINDOW_WIDTH - 20)
        y = WINDOW_HEIGHT + random.uniform(10, 50)
        self.drops.append(Drop(x=x, y=y, label=label))
        self.spawned += 1

    def _apply_energy_decay(self) -> None:
        roll = random.random()
        if roll < 0.2:
            delta = -2
        elif roll < 0.7:
            delta = -1
        else:
            delta = -3
        self.energy = max(0, self.energy + delta)

    def _move_fish(self, dt: float) -> None:
        dx, dy = self._move_vector
        self.fish.move(dx * FISH_SPEED * dt, dy * FISH_SPEED * dt)

    def _advance_drops(self, dt: float) -> None:
        remaining = []
        fx1, fx2, fy1, fy2 = self.fish.bounds
        for drop in self.drops:
            drop.y -= DROP_SPEED * dt
            if drop.y < -10:
                continue
            if fx1 <= drop.x <= fx2 and fy1 <= drop.y <= fy2:
                self._resolve_collision(drop.label)
                continue
            remaining.append(drop)
        self.drops = remaining

    def _resolve_collision(self, label: str) -> None:
        if label == "food":
            self.energy = min(ENERGY_MAX, self.energy + FOOD_BOOST)
        elif label == "bomb":
            self.energy = max(0, self.energy - BOMB_PENALTY)
        elif label == "coin":
            self.dollars += 1

    def _wave_text(self) -> str:
        pattern = WAVE_FRAMES[self.wave_frame % len(WAVE_FRAMES)]
        repetitions = WINDOW_WIDTH // len(pattern) + 2
        return pattern * repetitions

    def update(self, delta_time: float) -> None:
        if self.state != "running":
            return
        self.wave_timer += delta_time
        if self.wave_timer >= 0.15:
            self.wave_timer = 0.0
            self.wave_frame = (self.wave_frame + 1) % len(WAVE_FRAMES)

        self.spawn_timer += delta_time
        if self.spawn_timer >= DROP_INTERVAL:
            self.spawn_timer = 0.0
            self._spawn_drop()

        self.energy_timer += delta_time
        if self.energy_timer >= ENERGY_DECAY_SECONDS:
            self.energy_timer = 0.0
            self._apply_energy_decay()

        self._drive_controller(delta_time)
        self._advance_drops(delta_time)

        if self.spawned >= TOTAL_DROPS and not self.drops:
            self.spawned = 0
            self.spawn_timer = 0.0

        if self.energy <= 0:
            self.state = "game_over"

    def _drive_controller(self, delta_time: float) -> None:
        if self.active_controller is None:
            self._move_fish(delta_time)
            return
        drops_state = [(d.x, d.y, d.label) for d in self.drops]
        quantized = quantize_ascii_frame((self.fish.x, self.fish.y), drops_state, WINDOW_WIDTH, WINDOW_HEIGHT)
        action = self.active_controller(quantized)
        dx, dy = 0.0, 0.0
        if action == "up":
            dy = 1.0
        elif action == "down":
            dy = -1.0
        elif action == "left":
            dx = -1.0
        elif action == "right":
            dx = 1.0
        self.fish.move(dx * FISH_SPEED * delta_time, dy * FISH_SPEED * delta_time)
        self._train_active_model(quantized, action)

    def _train_active_model(self, new_state: QuantizedFishState, action: str) -> None:
        if self.last_state is None or self.last_action is None:
            self.last_state = new_state
            self.last_action = action
            return
        reward = self._estimate_reward()
        done = self.energy <= 0
        if self.dqn:
            transition = Transition(
                state=self.last_state.flattened(),
                action=ACTION_SPACE.index(self.last_action),
                reward=reward,
                next_state=new_state.flattened(),
                done=done,
            )
            self.dqn.push(transition)
            self.dqn.learn()
        if self.actor_critic:
            self.actor_critic.learn(self.last_state, self.last_action, reward, new_state, done)
        if self.regression:
            target = self._heuristic_action()
            self.regression.fit(new_state, target)
        self.last_state = new_state
        self.last_action = action

    def _heuristic_action(self) -> str:
        if not self.drops:
            return "up"
        nearest = min(self.drops, key=lambda d: (d.y - self.fish.y) ** 2 + (d.x - self.fish.x) ** 2)
        dx = nearest.x - self.fish.x
        dy = nearest.y - self.fish.y
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        return "up" if dy > 0 else "down"

    def _estimate_reward(self) -> float:
        reward = 0.0
        if self.energy > 0:
            reward += 0.01
        reward += self.dollars * 0.05
        return reward

    # drawing --------------------------------------------------------- #
    def _draw_mode_select(self) -> None:
        arcade.draw_text(
            "ASCII Fish Food",
            WINDOW_WIDTH / 2,
            WINDOW_HEIGHT / 2 + 80,
            ASCII_COLOR,
            32,
            anchor_x="center",
            font_name="monospace",
        )
        arcade.draw_text(
            "Press 1 for Player control",
            WINDOW_WIDTH / 2,
            WINDOW_HEIGHT / 2 + 20,
            ASCII_COLOR,
            20,
            anchor_x="center",
            font_name="monospace",
        )
        arcade.draw_text(
            "Press 2 for AI control",
            WINDOW_WIDTH / 2,
            WINDOW_HEIGHT / 2 - 20,
            ASCII_COLOR,
            20,
            anchor_x="center",
            font_name="monospace",
        )

    def _draw_ai_select(self) -> None:
        arcade.draw_text(
            "Pick an AI: 1=DQN  2=AC  3=Regression",
            WINDOW_WIDTH / 2,
            WINDOW_HEIGHT / 2,
            ASCII_COLOR,
            22,
            anchor_x="center",
            font_name="monospace",
        )

    def _draw_gameplay(self) -> None:
        wave_y = WINDOW_HEIGHT - 24
        wave_text = self._wave_text()
        arcade.draw_text(wave_text, 0, wave_y, ASCII_COLOR, 16, font_name="monospace")
        arcade.draw_text(
            self.fish.art,
            self.fish.x,
            self.fish.y,
            ASCII_COLOR,
            self.font_size,
            anchor_x="center",
            anchor_y="center",
            font_name="monospace",
        )
        for drop in self.drops:
            arcade.draw_text(
                drop.sprite(),
                drop.x,
                drop.y,
                ASCII_COLOR,
                self.font_size,
                anchor_x="center",
                anchor_y="center",
                font_name="monospace",
            )

        bar_width = 320
        bar_height = 18
        ratio = self.energy / ENERGY_MAX
        arcade.draw_rectangle_outline(20 + bar_width / 2, 24, bar_width, bar_height, ASCII_COLOR, 2)
        arcade.draw_rectangle_filled(
            20 + (bar_width * ratio) / 2,
            24,
            bar_width * ratio,
            bar_height - 4,
            ASCII_COLOR,
        )
        arcade.draw_text(f"Energy: {self.energy}", 24, 42, ASCII_COLOR, 14, font_name="monospace")
        arcade.draw_text(f"Cash: ${self.dollars}", WINDOW_WIDTH - 180, 24, ASCII_COLOR, 16, font_name="monospace")
        arcade.draw_text(f"Controller: {self.controller_name}", WINDOW_WIDTH / 2 - 80, 24, ASCII_COLOR, 14, font_name="monospace")

    def _draw_game_over(self) -> None:
        overlay_text = f"you collected ${self.dollars} dollars."
        arcade.draw_lrtb_rectangle_filled(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, (0, 0, 0, 200))
        arcade.draw_text(
            "Game Over",
            WINDOW_WIDTH / 2,
            WINDOW_HEIGHT / 2 + 60,
            ASCII_COLOR,
            28,
            anchor_x="center",
            font_name="monospace",
        )
        arcade.draw_text(
            overlay_text,
            WINDOW_WIDTH / 2,
            WINDOW_HEIGHT / 2 + 20,
            ASCII_COLOR,
            22,
            anchor_x="center",
            font_name="monospace",
        )
        arcade.draw_text(
            "play again? y/n",
            WINDOW_WIDTH / 2,
            WINDOW_HEIGHT / 2 - 20,
            ASCII_COLOR,
            20,
            anchor_x="center",
            font_name="monospace",
        )


def run_fish_game() -> None:
    window = FishFoodGame()
    window.setup()
    arcade.run()
