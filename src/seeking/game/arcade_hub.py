from __future__ import annotations

import arcade

from seeking.game.fish_ascii import run_fish_game
from seeking.game.pong import run_pong

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800


class ArcadeLauncher(arcade.Window):
    def __init__(self) -> None:
        update_rate = 1 / 60
        super().__init__(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            "Arcade Hub",
            update_rate=update_rate,
            draw_rate=update_rate,
        )
        arcade.set_background_color(arcade.color.BLACK)
        self.selection: str | None = None

    def on_draw(self) -> None:
        arcade.start_render()
        arcade.draw_text(
            "Press 1 for Pong",
            WINDOW_WIDTH / 2,
            WINDOW_HEIGHT / 2 + 20,
            arcade.color.WHITE,
            28,
            anchor_x="center",
            font_name="monospace",
        )
        arcade.draw_text(
            "Press 2 for ASCII Fish Food",
            WINDOW_WIDTH / 2,
            WINDOW_HEIGHT / 2 - 20,
            arcade.color.WHITE,
            24,
            anchor_x="center",
            font_name="monospace",
        )

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol in (arcade.key.KEY_1, arcade.key.NUM_1):
            self.selection = "pong"
            arcade.close_window()
        elif symbol in (arcade.key.KEY_2, arcade.key.NUM_2):
            self.selection = "fish"
            arcade.close_window()


def launch_arcade_hub(pong_kwargs: dict | None = None) -> None:
    window = ArcadeLauncher()
    arcade.run()
    if window.selection == "pong":
        run_pong(**(pong_kwargs or {}))
    elif window.selection == "fish":
        run_fish_game()
