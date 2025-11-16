# Seeking Playground

Arcade + PyTorch experimentation environment for building and testing 2D reinforcement learning ideas. The project keeps the game/simulation code separated from the learning stack so that you can prototype new policies or drop in alternative algorithms without rewriting the world logic.

## Features

- **Arcade visualization** for interactive play (WASD/arrow keys + reset hotkey).
- **Headless PyTorch trainer** that can run as fast as possible without rendering.
- **Modular layout** that separates environment definition, agents, and RL utilities.
- **Standalone Pygame Pong** mini-game for quick keyboard battles.
- **Configurable gridworld** (size, obstacles, penalties, rewards, seeds).
- **Packaging ready** with `pyproject.toml` + optional editable install for IDE support.

## Project layout

```
seeking/
├── pyproject.toml          # package metadata + dependencies
├── requirements.txt        # quick installation list
└── src/seeking
    ├── config.py           # dataclass configuring the gridworld
    ├── game/
    │   ├── world.py        # environment + Gym-style API
    │   └── arcade_runner.py# Arcade window for human control
    ├── rl/
    │   ├── policy.py       # PyTorch policy network helpers
    │   └── trainer.py      # REINFORCE-style training loop
    ├── agents/             # reference agents (random baseline)
    └── main.py             # CLI entry point
```

## Prerequisites

- Python 3.10+
- macOS / Linux windowing stack for Arcade (headless training does not require a GUI).

## Environment setup

All commands run from the repository root.

1. Create and activate a virtual environment.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell: `.\.venv\Scripts\Activate.ps1`

2. Upgrade pip and install dependencies (editable install keeps imports tidy).

   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

   Alternatively, `pip install -r requirements.txt` if you do not need an editable install.

3. Verify Arcade can open a window (skip this if you only care about headless training):

   ```bash
   python -c "import arcade; print(arcade.__version__)"
   ```

## Usage

### Interactive play

Open the Arcade window and control the agent with arrow keys / WASD. Press `R` to reset the level.

```bash
python -m seeking.main --mode play --width 12 --height 12 --obstacles 15
```

### Headless training

Runs a simple REINFORCE trainer implemented in `seeking.rl.trainer.Trainer`. All Torch operations happen without Arcade, so this mode is suitable for running on servers without display support.

```bash
python -m seeking.main --mode train --episodes 200 --lr 5e-4 --gamma 0.95 --seed 123
```

Logs are printed through `rich` every few episodes. Edit `Trainer` to plug in advanced algorithms, value heads, replay buffers, etc. The environment exposes normalized observations (`np.float32`), discrete actions, and convenience helpers for sampling and rendering.

### Pygame Pong

Launch a local Pong battle powered by Pygame. The left paddle uses `W`/`A` to move up and `S`/`D` to move down. The right paddle uses the arrow keys (Up/Left = up, Down/Right = down). Tap `SPACE` to pause/resume, and press `Q` (or close the window) to exit. The ball starts white, takes on the color of the last paddle hit, and scores are shown as `LEFT: N    RIGHT: M`.

```bash
python -m seeking.main --mode pong
```

#### Training the Pong agent

Flip on the `--pong-train` flag to launch a lightweight REINFORCE trainer that builds a four-layer feed-forward network with two hidden layers (64 and 32 units with ReLU). The model ingests a normalized `(4,)` state vector containing the paddle's top and bottom positions, the current score differential, and the ball's vertical position, and emits unnormalized logits for the two discrete actions (`move_up`, `move_down`). The trainer can run on CPU or GPU (set via `--device`) and saves a standard `state_dict` when `--pong-checkpoint` is provided.

```bash
python -m seeking.main --mode pong --pong-train --episodes 500 --lr 5e-4 --pong-checkpoint pong_policy.pt
```

The saved checkpoint can later be loaded with `torch.load` or plugged into `PongPolicyNetwork.load_state_dict` for experimentation.

#### Self-play training inside the UI

To watch two reinforcement learning agents learn directly in the rendered Pong match, start the new self-play mode. Both paddles run identical four-layer feed-forward policies (4 inputs → 64 → 32 → 2) and receive symmetric on-screen rewards (track ball alignment, bounces, and scoring). Parameters such as `--lr`, `--gamma`, `--device`, and `--episodes` still apply, though the self-play loop trains continuously until you quit.

```bash
python -m seeking.main --mode pong --pong-selfplay --lr 1e-3 --gamma 0.99
```

Use `SPACE` to pause/resume and `Q` (or closing the window) to stop the session.

Each time you exit self-play, the latest weights and optimizer state are saved (by default to `pong_selfplay_state.pt`) and automatically reloaded the next time you launch the command. Customize the location with `--pong-selfplay-checkpoint /path/to/file.pt`.

### Running with custom code

Because the project is installed in editable mode, you can import the modules directly:

```python
from seeking.config import GridWorldConfig
from seeking.game.world import GridWor
from seeking.rl.policy import PolicyNetwork
```

This makes it straightforward to embed the environment inside Jupyter notebooks or separate research scripts.

## Extending the playground

- Swap out `GridWorld` for another environment by adding a new module under `seeking/game`.
- Plug in different action or observation spaces by adjusting the dataclass in `config.py`.
- Add agents (scripted policies, planners, etc.) under `seeking/agents` so you can benchmark RL results.
- When training in the cloud, start the CLI with `--mode train --device cuda` to leverage GPUs if available.

## Troubleshooting

- **Arcade cannot open a window**: ensure you run it locally with display access. For SSH sessions, forward X11 or rely on headless training mode.
- **Torch missing CUDA**: specify `--device cpu` if your Python build does not ship GPU support.
- **Dependencies in global Python**: double-check that the virtual environment is activated before installing packages.
