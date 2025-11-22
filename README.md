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

### WebSocket streaming backend

Frontend experiments can stream gameplay data to a lightweight FastAPI server. The backend logs each payload, feeds quantized grid snapshots through two neural controllers (Deep Q for green, actor-critic for purple), and trains them online while returning their predicted actions to the frontend every frame.

1. Install the dependencies (if you have already run `pip install -e .`, FastAPI and Uvicorn are included).
2. Start the server:

   ```bash
   uvicorn seeking.server.websocket_server:app --reload --host 0.0.0.0 --port 8000
   ```

3. Connect from the frontend via `ws://localhost:8000/ws/game` and send the ASCII grid shown earlier (columns of `0`, paddles rendered as `|`, ball rendered as `.`). Every payload is printed, then decoded into a `(channels, rows, cols)` tensor and passed through both controllers.
4. The server responds with a JSON blob describing what each network decided to do (`up`, `down`, or `neutral`). Responses are validated through a dataclass so the payload is stable:

```json
{
  "type": "model_actions",
  "green": "up",
  "purple": "neutral"
}
```

Send structured payloads when you want to annotate events or share richer state:

```json
{
  "matrix": [
    ["0", "0", "0", "|", "...", "."],
    ["0", "0", "0", "0", "0", "0"]
  ],
  "score": {"green": 3, "purple": 5},
  "scored": "purple",
  "debug": false
}
```

The backend validates this shape via dataclasses. The `matrix` must be a `string[][]`, `score` holds the integer totals, `scored` indicates the last scorer (`"green"`/`"purple"`/`null`), and `debug` toggles any experimental UI diagnostics. The server flattens the matrix, converts `scored` into `green_scored`/`purple_scored` events, and keeps training the models off the derived rewards.

Each round the server rewards the paddle that scores (+1) and punishes the paddle that gets scored on (-1). Staying neutral for too many consecutive frames incurs a small penalty so the agents are encouraged to move. Those rewards update a tiny DQN replay buffer for the green paddle and a one-step actor-critic optimizer for the purple paddle, so playing more frames directly improves the checkpointed models. Checkpoints named `green_dqn.pt` and `purple_ac.pt` are loaded when a client connects (logging “green loaded”/“purple loaded”) and saved on disconnect so progress persists across runs.

Validation-only runs: start uvicorn with `SKIP_PREDICTIONS=1` (alias `SEEKING_SKIP_PREDICTIONS=1`) to disable inference/training entirely:

```bash
SKIP_PREDICTIONS=1 uvicorn seeking.server.websocket_server:app --reload --host 0.0.0.0 --port 8000
```

You can also connect via `ws://localhost:8000/ws/game?skip_predictions=1` to toggle it per WebSocket connection and simply sanity-check the incoming data stream.

### Quantized Pong controllers

`seeking.rl.pong_quantized_models` introduces:

- `QuantizedStateEncoder` – converts the ASCII board (0/|/.) into a three-channel tensor (empty/paddle/ball).
- `PongDQN` – convolutional encoder + linear head (3 actions) intended for the green paddle.
- `PongActorCritic` – shared encoder with separate policy/value heads for the purple paddle.

Both expose `.act(...)` helpers so they can be slotted into streaming or offline trainers. Swap in your trained weights, then extend the FastAPI server to load checkpoints on boot to drive the live paddle decisions.

### Pygame Pong

Launch a local Pong battle powered by Pygame. The left paddle uses `W`/`A` to move up and `S`/`D` to move down. The right paddle uses the arrow keys (Up/Left = up, Down/Right = down). Tap `SPACE` to pause/resume, and press `Q` (or close the window) to exit. The ball starts white, takes on the color of the last paddle hit, and scores are shown as `LEFT: N    RIGHT: M`.

```bash
python -m seeking.main --mode pong
```

On boot you'll see a simple in-game menu: press `1` for classic two-player, `2` for human vs AI (right paddle tracks with the current checkpoint), or `3` for AI vs AI. Use the left/right arrows on the menu to cycle the projectile between a ball, square, or triangle—your choice is shown at the top during gameplay. When you choose AI vs AI and close the window, the paddle with the higher score overwrites both checkpoints so the reigning champion is used next time. Press `ESC` to open the pause overlay where you can resume or jump back to the main menu (returning from AI vs AI through this menu will also promote the current winner before showing the menu). By default the human/AI modes look for checkpoints under `runs/green_<shape>.pt` and `runs/purple_<shape>.pt` where `<shape>` is `circle`, `square`, or `triangle`; create those files via `--pong-train`/`--pong-competition` or point the CLI flags elsewhere (`--pong-green-shape`, `--pong-purple-shape`).

#### Training the Pong agent

Flip on the `--pong-train` flag to launch a lightweight REINFORCE trainer that builds a four-layer feed-forward network with two hidden layers (64 and 32 units with ReLU). The model ingests a normalized `(6,)` state vector containing your paddle's top/bottom, the opponent paddle's top/bottom, the current score differential, and the ball's vertical position, and emits unnormalized logits for the two discrete actions (`move_up`, `move_down`). The trainer can run on CPU or GPU (set via `--device`) and saves a standard `state_dict` when `--pong-checkpoint` is provided.

Quick start:

```bash
python -m seeking.main --mode pong --pong-train
```

```bash
python -m seeking.main --mode pong --pong-train --episodes 500 --lr 5e-4 --pong-train-shape square --pong-checkpoint runs/policy_square.pt
```

The saved checkpoint can later be loaded with `torch.load` or plugged into `PongPolicyNetwork.load_state_dict` for experimentation.

To watch the trained policy inside the UI, load the checkpoint with the new demo flag (use `--pong-run-shape` to pick the projectile used during playback):

```bash
python -m seeking.main --mode pong --pong-demo runs/policy_square.pt --pong-run-shape triangle
```

You can also chain training and playback in one go:

```bash
python -m seeking.main --mode pong --pong-train --episodes 500 --pong-train-shape square --pong-checkpoint runs/policy_square.pt --pong-demo runs/policy_square.pt --pong-run-shape triangle
```

Shape-specific AIs: pass `--pong-train-shape square` (or `triangle`) to produce a checkpoint like `runs/policy_square.pt`, and use `--pong-green-shape` / `--pong-purple-shape` to point the UI or competition flow at different shapes when two AIs battle.

#### Competitive loop (train → battle → promote)

If you want both paddles to keep improving through sparring, use the competition flag. Each invocation trains separate green/purple policies, launches a battle UI so they can fight it out, and then copies the winner into both checkpoints before exiting. The next time you run the command, training resumes from the reigning champion.

```bash
python -m seeking.main --mode pong --pong-competition --episodes 500 --pong-green-shape ball --pong-purple-shape square --pong-run-shape triangle
```

Close the UI when you are satisfied with the battle; the score at that moment determines the winner.

Pass `--pong-green-shape` / `--pong-purple-shape` (and rely on the default `runs/green_<shape>.pt` naming, e.g., `green_circle.pt`) to pit differently trained agents against one another, e.g., triangle-vs-square checkpoints.

## Apple Silicon (M1/M2) GPU usage

PyTorch can leverage the Apple GPU via Metal Performance Shaders (MPS). If you installed a recent `torch` wheel, just pass `--device mps` (or set `TORCH_DEVICE=mps`) when running any training command:

```bash
python -m seeking.main --mode pong --pong-train --device mps
```

If you want training/battle commands to automatically attempt the MPS device, we include a helper flag: add `--device auto` and the CLI will try `cuda`, then `mps`, and finally fall back to `cpu`. Use it like so:

```bash
python -m seeking.main --mode pong --device auto --pong-train --pong-train-shape triangle
```

Ensure you have at least Python 3.10 and the official PyTorch builds for macOS arm64 installed.。

E.g.

```bash
python -m seeking.main --mode pong --pong-competition --episodes 500 --pong-green-checkpoint runs/green_circle.pt --pong-purple-checkpoint runs/purple_circle.pt --pong-purple-shape triangle --pong-green-shape triangle --pong-run-shape triangle
```

#### Self-play training inside the UI

To watch two reinforcement learning agents learn directly in the rendered Pong match, start the new self-play mode. Both paddles run identical four-layer feed-forward policies (6 inputs → 64 → 32 → 2) and receive symmetric on-screen rewards (track ball alignment, bounces, and scoring). Parameters such as `--lr`, `--gamma`, `--device`, and `--episodes` still apply, though the self-play loop trains continuously until you quit.

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

## Codex Prompts

a couple of things
1 - there is currently a pong game. what I want is a simplistic game of fish food in python arcade, where the are randomly spawning drops of food, bombs, or coins. the first round should have 40 drops total - a random mix of coins bombs and food, leaning 70% food, 20% bombs, and 10% chance of coins dropping. the fish should have an energy bar at the bottom. the bar goes to 100, and the food should recharge it up 15 points. every second, the fish has a 20% chance to lose 2 points, 50% chance to lose 1 point, and 30% chance to lose 3 points of energy. when the bar reaches zero, the game ends. make the art in an ascii style, with the ability to upgrade the art over time. the fish should look like "o=-<" the food should look like "." the money should look like "$" and the bomb should look like "@". make the background black, the text white. the items should spawn above the waves, which are at the top and should alternate between "^~~~", "~^~~","~~^~","~~~^", of course repeat this pattern across the page, which should be 1200px wide, and 800px tall. i think the waves with four animation frames will look really really cool, and animate really nicely. when the game ends, make a pause screen that says "you collected $x dollars." beneath it saying "play again? y/n" where pressing y on the keyboard plays the game again.

2 - when booting up the arcade game, there should be a start screen. the user should press 1 for pong, which would boot up that normally, and 2 for the ascii fish game. if someone selects the fish game, present them with an option. 1 for player, 2 for AI. if 1 is pressed, the fish should be controlled with either wasd or the arrow keys. these should be interchangeable.

3 - if someone presses ai, I want another screen that asks them DQN, AC, or Regression. in the folder rl/fish_trainer/ i want the following files. fish_dqn.py, fish_ac.py, and fish_regression.py. these will be models that take in the game state, and outputs up/down/left/right for the fish game. this is an ascii game, which means the input should be simple BUT we still need to quantize everything. the input of the models should be downgraded by 50px, such that the input is only a 16x24 matrix. instead of the multi sized fish, make the fish "f", but keep everything else the same. when the model is selected, drop the target frame rate to 10fps, and train the models based on what is on the screen. the regression model should use linear regression, the DQN should use deep q learning, the ac should use the actor-critic paradigm to control the fish

