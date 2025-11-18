"""FastAPI WebSocket backend that streams and trains game controllers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
from seeking.rl import (
    ACTION_LABELS,
    ActorCriticTrainer,
    DQNTrainer,
    QuantizedStateEncoder,
    create_actor_critic_controller,
    create_dqn_controller,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("seeking.websocket")

app = FastAPI(title="Seeking WebSocket Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
GREEN_CHECKPOINT = RUNS_DIR / "green_dqn.pt"
PURPLE_CHECKPOINT = RUNS_DIR / "purple_ac.pt"

STATE_ENCODER = QuantizedStateEncoder()
SCORE_REWARD = 1.0
IDLE_PENALTY = 0.05


def _env_flag(*names: str, default: str = "0") -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value.lower() in {"1", "true", "yes"}
    return default.lower() in {"1", "true", "yes"}


GLOBAL_SKIP_PREDICTIONS = _env_flag("SEEKING_SKIP_PREDICTIONS", "SEEKING_VALIDATE_ONLY", default="0")


GREEN_TRAINER = DQNTrainer(create_dqn_controller())
PURPLE_TRAINER = ActorCriticTrainer(create_actor_critic_controller())


def _format_payload(message: dict[str, Any]) -> str:
    """Return a readable representation of the received websocket payload."""
    if message.get("text") is not None:
        text = message["text"]
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            return text
    if message.get("bytes") is not None:
        preview = message["bytes"][:128]
        return f"<{len(message['bytes'])} raw bytes> {preview!r}"
    return f"<unknown payload> {message}"


def _parse_payload(text: str) -> tuple[str | None, dict[str, bool]]:
    if not text:
        return None, {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text, {}
    if isinstance(data, dict):
        grid = data.get("grid")
        events = data.get("events") or {}
        if isinstance(grid, str):
            return grid, _normalize_events(events)
        return None, _normalize_events(events)
    return text, {}


def _normalize_events(raw_events: Any) -> dict[str, bool]:
    if not isinstance(raw_events, dict):
        return {}
    normalized = {}
    for key, value in raw_events.items():
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, (int, float)):
            normalized[key] = bool(value)
        elif isinstance(value, str):
            normalized[key] = value.lower() in {"1", "true", "yes"}
    return normalized


def _base_rewards(events: dict[str, bool]) -> dict[str, float]:
    rewards = {"green": 0.0, "purple": 0.0}
    if events.get("green_scored"):
        rewards["green"] += SCORE_REWARD
        rewards["purple"] -= SCORE_REWARD
    if events.get("purple_scored"):
        rewards["purple"] += SCORE_REWARD
        rewards["green"] -= SCORE_REWARD
    return rewards


def _should_skip_predictions(websocket: WebSocket) -> bool:
    if GLOBAL_SKIP_PREDICTIONS:
        return True
    query_bytes = websocket.scope.get("query_string", b"")
    if not query_bytes:
        return False
    query = parse_qs(query_bytes.decode())
    for key in ("validate_only", "skipPrediction", "skip_prediction", "skip_predictions"):
        values = query.get(key)
        if values and values[-1].lower() in {"1", "true", "yes"}:
            return True
    return False


def _load_controller_weights() -> None:
    if GREEN_CHECKPOINT.exists():
        try:
            state = torch.load(GREEN_CHECKPOINT, map_location="cpu")
            GREEN_TRAINER.model.load_state_dict(state)
            GREEN_TRAINER.model.eval()
            GREEN_TRAINER.reset_state()
            logger.info("Loaded green DQN weights from %s", GREEN_CHECKPOINT)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load %s: %s", GREEN_CHECKPOINT, exc)
            GREEN_TRAINER.replace_model(create_dqn_controller())
            torch.save(GREEN_TRAINER.model.state_dict(), GREEN_CHECKPOINT)
            logger.info("Initialized fresh green DQN weights at %s", GREEN_CHECKPOINT)
    else:
        GREEN_TRAINER.replace_model(create_dqn_controller())
        torch.save(GREEN_TRAINER.model.state_dict(), GREEN_CHECKPOINT)
        logger.info("Created initial green DQN checkpoint at %s", GREEN_CHECKPOINT)

    if PURPLE_CHECKPOINT.exists():
        try:
            state = torch.load(PURPLE_CHECKPOINT, map_location="cpu")
            PURPLE_TRAINER.model.load_state_dict(state)
            PURPLE_TRAINER.model.eval()
            PURPLE_TRAINER.reset_state()
            logger.info("Loaded purple actor-critic weights from %s", PURPLE_CHECKPOINT)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load %s: %s", PURPLE_CHECKPOINT, exc)
            PURPLE_TRAINER.replace_model(create_actor_critic_controller())
            torch.save(PURPLE_TRAINER.model.state_dict(), PURPLE_CHECKPOINT)
            logger.info("Initialized fresh purple actor-critic weights at %s", PURPLE_CHECKPOINT)
    else:
        PURPLE_TRAINER.replace_model(create_actor_critic_controller())
        torch.save(PURPLE_TRAINER.model.state_dict(), PURPLE_CHECKPOINT)
        logger.info("Created initial purple actor-critic checkpoint at %s", PURPLE_CHECKPOINT)


def _save_controller_weights() -> None:
    try:
        torch.save(GREEN_TRAINER.model.state_dict(), GREEN_CHECKPOINT)
        logger.info("Saved green DQN weights to %s", GREEN_CHECKPOINT)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to save green DQN weights: %s", exc)
    try:
        torch.save(PURPLE_TRAINER.model.state_dict(), PURPLE_CHECKPOINT)
        logger.info("Saved purple actor-critic weights to %s", PURPLE_CHECKPOINT)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to save purple actor-critic weights: %s", exc)


@app.websocket("/ws/game")
async def game_stream(websocket: WebSocket) -> None:
    """Accept a game WebSocket and stream predictions + continual learning."""
    await websocket.accept()
    _load_controller_weights()
    connection_skip = _should_skip_predictions(websocket)
    client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    logger.info(
        "WebSocket client connected: %s (validation_mode=%s)",
        client_info,
        connection_skip,
    )

    try:
        while True:
            message = await websocket.receive()
            formatted = _format_payload(message)
            logger.debug("Game payload from %s:\n%s", client_info, formatted)
            text_payload = message.get("text") or ""
            grid_text, events = _parse_payload(text_payload)
            skip_predictions = connection_skip
            if not grid_text:
                continue
            try:
                state = STATE_ENCODER.encode(grid_text)
            except ValueError:
                logger.debug("Failed to parse grid payload from %s", client_info)
                continue

            if skip_predictions:
                continue

            done = bool(events.get("round_over") or events.get("reset"))
            rewards = _base_rewards(events)
            if GREEN_TRAINER.pending_idle_penalty:
                rewards["green"] -= IDLE_PENALTY
            if PURPLE_TRAINER.pending_idle_penalty:
                rewards["purple"] -= IDLE_PENALTY

            GREEN_TRAINER.observe(state, rewards["green"], done)
            PURPLE_TRAINER.observe(state, rewards["purple"], done)

            green_action, green_q_values = GREEN_TRAINER.act(state)
            purple_action, purple_logits, purple_value = PURPLE_TRAINER.act(state)

            GREEN_TRAINER.register_action(green_action)
            PURPLE_TRAINER.register_action(purple_action)

            prediction = {
                "type": "model_actions",
                "green": {
                    "action": ACTION_LABELS[green_action],
                    "q_values": green_q_values.detach().cpu().tolist(),
                },
                "purple": {
                    "action": ACTION_LABELS[purple_action],
                    "logits": purple_logits.detach().cpu().tolist(),
                    "value": float(purple_value.detach().cpu().item()),
                },
            }
            await websocket.send_json(prediction)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", client_info)
        _save_controller_weights()
