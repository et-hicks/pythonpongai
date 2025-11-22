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
        update_rate = 1 / 30
        super().__init__(width, height, "Seeking GridWorld", update_rate=update_rate, draw_rate=update_rate)
        self.set_location(100, 100)
        arcade.set_background_color(env.config.colors["background"])
        self.last_reward: float = 0.0
        self.status_text: str = "Press arrow keys or WASD to move."
        self._latest_info: Optional[dict] = None
        self.env.reset()

    @staticmethod
    def draw_rectangular_outline(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        color: arcade.Color,
        border_width: float = 1,
    ) -> None:
        """Compatibility wrapper for rectangle outlines across Arcade versions."""

        if hasattr(arcade, "draw_rectangle_outline"):
            arcade.draw_rectangle_outline(center_x, center_y, width, height, color, border_width=border_width)
            return

        # Arcade 3.x renamed the primitive to left/bottom/width/height variants.
        if hasattr(arcade, "draw_lbwh_rectangle_outline"):
            left = center_x - width / 2
            bottom = center_y - height / 2
            arcade.draw_lbwh_rectangle_outline(left, bottom, width, height, color, border_width=border_width)
            return

        raise AttributeError("No rectangle outline primitive available in this version of arcade.")

    @staticmethod
    def draw_rectangular_filled(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        color: arcade.Color,
    ) -> None:
        """Compatibility wrapper for rectangle fills across Arcade versions."""

        if hasattr(arcade, "draw_rectangle_filled"):
            arcade.draw_rectangle_filled(center_x, center_y, width, height, color)
            return

        if hasattr(arcade, "draw_lbwh_rectangle_filled"):
            left = center_x - width / 2
            bottom = center_y - height / 2
            arcade.draw_lbwh_rectangle_filled(left, bottom, width, height, color)
            return

        raise AttributeError("No rectangle filled primitive available in this version of arcade.")

    # Arcade hooks ----------------------------------------------------- #
    def on_draw(self) -> None:
        # In Arcade 3.x, start_render is for module-level scripts; Window.draw should clear instead.
        self.clear()
        snapshot = self.env.snapshot()
        colors = self.env.config.colors
        tile = self.env.config.tile_pixel_size
        for x in range(snapshot["width"]):
            for y in range(snapshot["height"]):
                screen_x = x * tile + tile / 2
                screen_y = y * tile + tile / 2
                self.draw_rectangular_outline(
                    screen_x,
                    screen_y,
                    tile,
                    tile,
                    colors["grid"],
                    border_width=1,
                )
        for (ox, oy) in snapshot["obstacles"]:
            self.draw_rectangular_filled(
                ox * tile + tile / 2,
                oy * tile + tile / 2,
                tile,
                tile,
                colors["obstacle"],
            )
        gx, gy = snapshot["goal"]
        self.draw_rectangular_filled(
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
