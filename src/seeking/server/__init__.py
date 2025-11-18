"""Server utilities and FastAPI apps for streaming game data."""

__all__ = ["app"]

from .websocket_server import app  # re-export for convenience
