"""Edge-TTS synthesis as an async chunk stream, with timing and a small reply cache."""
import time
from collections import OrderedDict
from typing import AsyncIterator, Callable, Optional

from app.voice.timing import TurnTimer

TTS_VOICE = "hi-IN-SwaraNeural"
TTS_RATE = "+20%"   # was +12%; check intelligibility by ear (see report)
TTS_PITCH = "+0Hz"
CACHE_SIZE = 64


class LRUAudioCache:
    def __init__(self, size: int = CACHE_SIZE):
        self.size = size
        self._data: "OrderedDict[tuple, bytes]" = OrderedDict()

    def get(self, key) -> Optional[bytes]:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key, audio: bytes) -> None:
        self._data[key] = audio
        self._data.move_to_end(key)
        while len(self._data) > self.size:
            self._data.popitem(last=False)


_cache = LRUAudioCache()


def _default_factory(text: str):
    import edge_tts  # lazy: keeps this module importable without edge-tts installed
    return edge_tts.Communicate(text=text, voice=TTS_VOICE, rate=TTS_RATE, pitch=TTS_PITCH)


async def tts_chunks(
    text: str,
    timer: Optional[TurnTimer] = None,
    factory: Callable = _default_factory,
    cache: LRUAudioCache = _cache,
) -> AsyncIterator[bytes]:
    """Yield MP3 chunks as Edge-TTS produces them.

    Records `tts_first_chunk` (time until the first audio byte) and `tts_total`
    (time until generation finished). In streaming mode tts_total also includes the time the
    consumer spends sending each chunk, because the generator is suspended while it does.
    Repeated identical replies are served from a small LRU cache.
    """
    key = (text, TTS_VOICE, TTS_RATE, TTS_PITCH)
    cached = cache.get(key)
    if cached is not None:
        if timer:
            timer.note("tts_cache", "hit")
        yield cached
        return

    start = time.perf_counter()
    first = True
    buf = []
    async for chunk in factory(text).stream():
        if chunk.get("type") != "audio":
            continue
        if first and timer:
            timer.add("tts_first_chunk", time.perf_counter() - start)
            first = False
        buf.append(chunk["data"])
        yield chunk["data"]
    if not buf:
        raise RuntimeError("Edge-TTS returned empty audio.")
    if timer:
        timer.add("tts_total", time.perf_counter() - start)
    cache.put(key, b"".join(buf))
