"""FastAPI WebSocket backend that streams messages from the frontend game."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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


STATE_ENCODER = QuantizedStateEncoder()
GREEN_CONTROLLER = PongDQN()
PURPLE_CONTROLLER = PongActorCritic()
GREEN_CONTROLLER.eval()
PURPLE_CONTROLLER.eval()


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


@app.websocket("/ws/game")
async def game_stream(websocket: WebSocket) -> None:
    """Accept a game WebSocket and log every incoming payload."""
    await websocket.accept()
    client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    logger.info("WebSocket client connected: %s", client_info)

    try:
        while True:
            message = await websocket.receive()
            formatted = _format_payload(message)
            logger.info("Game payload from %s:\n%s", client_info, formatted)
            prediction = _build_prediction(message.get("text") or "")
            if prediction:
                logger.info(
                    "Controllers -> green(DQN)=%s, purple(AC)=%s",
                    prediction["green"]["action"],
                    prediction["purple"]["action"],
                )
                await websocket.send_json(prediction)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", client_info)
