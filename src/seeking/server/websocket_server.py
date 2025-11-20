"""FastAPI WebSocket backend that streams and trains game controllers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
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
ALLOWED_DIRECTIONS = {"up", "down", "neutral"}


def _env_flag(*names: str, default: str = "0") -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value.lower() in {"1", "true", "yes"}
    return default.lower() in {"1", "true", "yes"}


GLOBAL_SKIP_PREDICTIONS = _env_flag("SEEKING_SKIP_PREDICTIONS", "SEEKING_VALIDATE_ONLY", default="0")
DEBUG_MODE = _env_flag("SEEKING_DEBUG", "DEBUG")


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


@dataclass(slots=True)
class ScorePayload:
    green: int
    purple: int

    @staticmethod
    def from_obj(obj: Any) -> "ScorePayload | None":
        if not isinstance(obj, dict):
            return None
        try:
            green = int(obj["green"])
            purple = int(obj["purple"])
        except (KeyError, TypeError, ValueError):
            return None
        return ScorePayload(green=green, purple=purple)


@dataclass(slots=True)
class ClientGamePayload:
    matrix: list[list[str]]
    score: ScorePayload
    scored: str | None
    debug: bool

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ClientGamePayload | None":
        matrix = data.get("matrix")
        score_obj = data.get("score")
        scored = data.get("scored")
        debug = data.get("debug")

        if not isinstance(matrix, list) or any(not isinstance(row, list) for row in matrix):
            return None

        if not isinstance(scored, (str, type(None))):
            return None

        if not isinstance(debug, bool):
            debug = bool(debug)

        score = ScorePayload.from_obj(score_obj)
        if score is None:
            return None

        if not matrix or any(not all(isinstance(cell, str) for cell in row) for row in matrix):
            return None

        return ClientGamePayload(matrix=matrix, score=score, scored=scored, debug=debug)


@dataclass(slots=True)
class ModelPredictionPayload:
    type: str
    green: str
    purple: str

    def __post_init__(self) -> None:
        if self.type != "model_actions":
            raise ValueError("Prediction payload type must be 'model_actions'")
        if self.green not in ALLOWED_DIRECTIONS:
            raise ValueError(f"Unsupported green action: {self.green!r}")
        if self.purple not in ALLOWED_DIRECTIONS:
            raise ValueError(f"Unsupported purple action: {self.purple!r}")


def _flatten_matrix(matrix: list[list[str]]) -> str:
    return "\n".join(" ".join(row) for row in matrix)


def _apply_structured_events(
    base_events: dict[str, bool], structured: ClientGamePayload | None
) -> dict[str, bool]:
    if structured is None:
        return base_events
    events = dict(base_events)
    if structured.scored:
        scored = structured.scored.lower()
        if scored == "green":
            events["green_scored"] = True
        elif scored == "purple":
            events["purple_scored"] = True
    return events


def _parse_payload(text: str) -> tuple[str | None, dict[str, bool], ClientGamePayload | None]:
    if not text:
        return None, {}, None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text, {}, None
    if isinstance(data, dict):
        structured = ClientGamePayload.from_dict(data)
        events = _normalize_events(data.get("events") or {})
        events = _apply_structured_events(events, structured)
        if structured:
            grid_str = _flatten_matrix(structured.matrix)
            return grid_str, events, structured
        return None, events, None
    return text, {}, None


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
            logger.info("green loaded")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load %s: %s", GREEN_CHECKPOINT, exc)
            GREEN_TRAINER.replace_model(create_dqn_controller())
            torch.save(GREEN_TRAINER.model.state_dict(), GREEN_CHECKPOINT)
            logger.info("Initialized fresh green DQN weights at %s", GREEN_CHECKPOINT)
            logger.info("green loaded")
    else:
        GREEN_TRAINER.replace_model(create_dqn_controller())
        torch.save(GREEN_TRAINER.model.state_dict(), GREEN_CHECKPOINT)
        logger.info("Created initial green DQN checkpoint at %s", GREEN_CHECKPOINT)
        logger.info("green loaded")

    if PURPLE_CHECKPOINT.exists():
        try:
            state = torch.load(PURPLE_CHECKPOINT, map_location="cpu")
            PURPLE_TRAINER.model.load_state_dict(state)
            PURPLE_TRAINER.model.eval()
            PURPLE_TRAINER.reset_state()
            logger.info("Loaded purple actor-critic weights from %s", PURPLE_CHECKPOINT)
            logger.info("purple loaded")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load %s: %s", PURPLE_CHECKPOINT, exc)
            PURPLE_TRAINER.replace_model(create_actor_critic_controller())
            torch.save(PURPLE_TRAINER.model.state_dict(), PURPLE_CHECKPOINT)
            logger.info("Initialized fresh purple actor-critic weights at %s", PURPLE_CHECKPOINT)
            logger.info("purple loaded")
    else:
        PURPLE_TRAINER.replace_model(create_actor_critic_controller())
        torch.save(PURPLE_TRAINER.model.state_dict(), PURPLE_CHECKPOINT)
        logger.info("Created initial purple actor-critic checkpoint at %s", PURPLE_CHECKPOINT)
        logger.info("purple loaded")


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
            grid_text, events, structured_payload = _parse_payload(text_payload)
            if structured_payload:
                logger.debug(
                    "Client scores green=%s purple=%s scored=%s debug=%s",
                    structured_payload.score.green,
                    structured_payload.score.purple,
                    structured_payload.scored,
                    structured_payload.debug,
                )
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

            green_action_idx, _ = GREEN_TRAINER.act(state)
            purple_action_idx, _, _ = PURPLE_TRAINER.act(state)

            GREEN_TRAINER.register_action(green_action_idx)
            PURPLE_TRAINER.register_action(purple_action_idx)

            green_action = ACTION_LABELS[green_action_idx]
            purple_action = ACTION_LABELS[purple_action_idx]

            payload = ModelPredictionPayload(
                type="model_actions",
                green=green_action,
                purple=purple_action,
            )
            print(payload)
            await websocket.send_json(asdict(payload))
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", client_info)
        _save_controller_weights()
