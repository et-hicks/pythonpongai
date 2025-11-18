"""FastAPI WebSocket backend that streams messages from the frontend game."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
from seeking.rl import ACTION_LABELS, PongActorCritic, PongDQN, QuantizedStateEncoder

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


def _new_green_controller() -> PongDQN:
    model = PongDQN()
    model.eval()
    return model


def _new_purple_controller() -> PongActorCritic:
    model = PongActorCritic()
    model.eval()
    return model


GREEN_CONTROLLER = _new_green_controller()
PURPLE_CONTROLLER = _new_purple_controller()


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
        # Limit output size to avoid flooding the console.
        preview = message["bytes"][:128]
        return f"<{len(message['bytes'])} raw bytes> {preview!r}"
    return f"<unknown payload> {message}"


def _looks_like_quantized_grid(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    return any(("|" in line or "." in line) for line in lines)


def _build_prediction(text: str) -> dict[str, Any] | None:
    if not _looks_like_quantized_grid(text):
        return None
    try:
        state = STATE_ENCODER.encode(text)
    except ValueError:
        return None

    green_action, green_q_values = GREEN_CONTROLLER.act(state)
    purple_action, purple_logits, purple_value = PURPLE_CONTROLLER.act(state)

    return {
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


def _load_controller_weights() -> None:
    """Load controller parameters from disk if checkpoints exist."""
    global GREEN_CONTROLLER, PURPLE_CONTROLLER
    if GREEN_CHECKPOINT.exists():
        try:
            GREEN_CONTROLLER.load_state_dict(torch.load(GREEN_CHECKPOINT, map_location="cpu"))
            GREEN_CONTROLLER.eval()
            logger.info("Loaded green DQN weights from %s", GREEN_CHECKPOINT)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load %s: %s", GREEN_CHECKPOINT, exc)
            GREEN_CONTROLLER = _new_green_controller()
            torch.save(GREEN_CONTROLLER.state_dict(), GREEN_CHECKPOINT)
            logger.info("Initialized fresh green DQN weights at %s", GREEN_CHECKPOINT)
    else:
        GREEN_CONTROLLER = _new_green_controller()
        torch.save(GREEN_CONTROLLER.state_dict(), GREEN_CHECKPOINT)
        logger.info("Created initial green DQN checkpoint at %s", GREEN_CHECKPOINT)

    if PURPLE_CHECKPOINT.exists():
        try:
            PURPLE_CONTROLLER.load_state_dict(torch.load(PURPLE_CHECKPOINT, map_location="cpu"))
            PURPLE_CONTROLLER.eval()
            logger.info("Loaded purple actor-critic weights from %s", PURPLE_CHECKPOINT)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load %s: %s", PURPLE_CHECKPOINT, exc)
            PURPLE_CONTROLLER = _new_purple_controller()
            torch.save(PURPLE_CONTROLLER.state_dict(), PURPLE_CHECKPOINT)
            logger.info("Initialized fresh purple actor-critic weights at %s", PURPLE_CHECKPOINT)
    else:
        PURPLE_CONTROLLER = _new_purple_controller()
        torch.save(PURPLE_CONTROLLER.state_dict(), PURPLE_CHECKPOINT)
        logger.info("Created initial purple actor-critic checkpoint at %s", PURPLE_CHECKPOINT)


def _save_controller_weights() -> None:
    """Persist controller parameters to disk."""
    global GREEN_CONTROLLER, PURPLE_CONTROLLER
    try:
        torch.save(GREEN_CONTROLLER.state_dict(), GREEN_CHECKPOINT)
        logger.info("Saved green DQN weights to %s", GREEN_CHECKPOINT)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to save green DQN weights: %s", exc)
    try:
        torch.save(PURPLE_CONTROLLER.state_dict(), PURPLE_CHECKPOINT)
        logger.info("Saved purple actor-critic weights to %s", PURPLE_CHECKPOINT)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to save purple actor-critic weights: %s", exc)


@app.websocket("/ws/game")
async def game_stream(websocket: WebSocket) -> None:
    """Accept a game WebSocket and log every incoming payload."""
    await websocket.accept()
    _load_controller_weights()
    client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    logger.info("WebSocket client connected: %s", client_info)

    try:
        while True:
            message = await websocket.receive()
            formatted = _format_payload(message)
            logger.debug("Game payload from %s:\n%s", client_info, formatted)
            prediction = _build_prediction(message.get("text") or "")
            if prediction:
                logger.debug(
                    "Controllers -> green(DQN)=%s, purple(AC)=%s",
                    prediction["green"]["action"],
                    prediction["purple"]["action"],
                )
                await websocket.send_json(prediction)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", client_info)
        _save_controller_weights()
