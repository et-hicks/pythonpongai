from __future__ import annotations

from typing import Dict, Optional

import arcade

from seeking.game.world import GridWorld, ACTIONS


KEY_BINDINGS: Dict[int, int] = {
    arcade.key.UP: 0,
    arcade.key.W: 0,
    arcade.key.DOWN: 1,
    arcade.key.S: 1,
    arcade.key.LEFT: 2,
    arcade.key.A: 2,
    arcade.key.RIGHT: 3,
    arcade.key.D: 3,
    arcade.key.SPACE: 4,
}


class ArcadeGridWorldApp(arcade.Window):
    """Interactive Arcade app for controlling the gridworld."""

    def __init__(self, env: GridWorld) -> None:
        self.env = env
        tile_size = self.env.config.tile_pixel_size
        width = env.width * tile_size
        height = env.height * tile_size
        super().__init__(width, height, "Seeking GridWorld", update_rate=1 / 30)
        self.set_location(100, 100)
        arcade.set_background_color(env.config.colors["background"])
        self.last_reward: float = 0.0
        self.status_text: str = "Press arrow keys or WASD to move."
        self._latest_info: Optional[dict] = None
        self.env.reset()

    # Arcade hooks ----------------------------------------------------- #
    def on_draw(self) -> None:
        arcade.start_render()
        snapshot = self.env.snapshot()
        colors = self.env.config.colors
        tile = self.env.config.tile_pixel_size
        for x in range(snapshot["width"]):
            for y in range(snapshot["height"]):
                screen_x = x * tile + tile / 2
                screen_y = y * tile + tile / 2
                arcade.draw_rectangle_outline(
                    screen_x,
                    screen_y,
                    tile,
                    tile,
                    colors["grid"],
                    border_width=1,
                )
        for (ox, oy) in snapshot["obstacles"]:
            arcade.draw_rectangle_filled(
                ox * tile + tile / 2,
                oy * tile + tile / 2,
                tile,
                tile,
                colors["obstacle"],
            )
        gx, gy = snapshot["goal"]
        arcade.draw_rectangle_filled(
            gx * tile + tile / 2,
            gy * tile + tile / 2,
            tile,
            tile,
            colors["goal"],
        )
        px, py = snapshot["player"]
        arcade.draw_circle_filled(
            px * tile + tile / 2,
            py * tile + tile / 2,
            tile * 0.35,
            colors["player"],
        )

        text = f"{self.status_text}  Reward: {self.last_reward:.2f}  Steps: {snapshot['steps']}"
        arcade.draw_text(text, 10, 10, arcade.color.WHITE, 14)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == arcade.key.R:
            self.env.reset()
            self.status_text = "Environment reset."
            self.last_reward = 0.0
            return
        action = KEY_BINDINGS.get(symbol)
        if action is None:
            return
        result = self.env.step(action)
        self.last_reward = result.reward
        self._latest_info = result.info
        if result.terminated:
            self.status_text = "Goal reached! Press R to reset."
        elif result.truncated:
            self.status_text = "Step limit reached. Press R to reset."
        else:
            self.status_text = f"Last action: {result.info['action_name']}"


def run_interactive(env: Optional[GridWorld] = None) -> None:
    """Utility helper for quick manual runs."""
    env = env or GridWorld()
    app = ArcadeGridWorldApp(env)
    arcade.run()
