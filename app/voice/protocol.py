"""WebSocket control-message protocol for /voice/stream/local_test.

Binary frames from the browser are always a complete recorded utterance (unchanged).
Text frames are JSON control messages. A browser that never sends any keeps the legacy
behaviour: one binary MP3 reply per turn.

Browser -> server (text):
  {"event": "hello", "caps": ["tts_stream"]}        opt in to chunked TTS replies
  {"event": "client_timing", "turn": 1, "upload_ms": 120, "playback_start_ms": 2300, ...}

Server -> browser (only after "tts_stream" was announced):
  text  {"event": "tts_start", "format": "audio/mpeg", "turn": n}
  binary <MP3 chunk> ... (repeated as Edge-TTS produces audio)
  text  {"event": "tts_end", "turn": n}
"""
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

CAP_TTS_STREAM = "tts_stream"


def parse_control(text: str) -> Optional[dict]:
    """Parse a text frame. Returns a dict with an 'event' key, or None if malformed."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        logger.warning("Ignoring non-JSON control message: %r", str(text)[:100])
        return None
    if not isinstance(data, dict) or not isinstance(data.get("event"), str):
        logger.warning("Ignoring control message without an 'event': %r", str(text)[:100])
        return None
    return data


def wants_tts_stream(hello: dict) -> bool:
    caps = hello.get("caps")
    return isinstance(caps, list) and CAP_TTS_STREAM in caps
