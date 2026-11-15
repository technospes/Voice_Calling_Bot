import asyncio
import time

import pytest

from app.voice import timing
from app.voice.protocol import parse_control, wants_tts_stream
from app.voice.tts_stream import LRUAudioCache, tts_chunks


# ---------------- timing ----------------
def test_span_records_even_on_exception_and_noop_without_timer():
    with timing.span("x"):          # no active timer: must not raise
        pass
    t = timing.start_turn()
    with pytest.raises(RuntimeError):
        with t.span("stt_deepgram"):
            time.sleep(0.01)
            raise RuntimeError("timeout")
    assert t.spans["stt_deepgram"] >= 0.01
    with timing.span("llm"):
        pass
    assert "llm" in t.spans and "turn_total" in t.summary()


# ---------------- control protocol ----------------
def test_parse_control_and_capabilities():
    assert parse_control('{"event": "hello", "caps": ["tts_stream"]}')["event"] == "hello"
    assert parse_control("not json") is None
    assert parse_control('{"no_event": 1}') is None
    assert parse_control("[1, 2]") is None
    assert wants_tts_stream({"event": "hello", "caps": ["tts_stream"]})
    assert not wants_tts_stream({"event": "hello"})
    assert not wants_tts_stream({"event": "hello", "caps": "tts_stream"})


# ---------------- TTS chunk stream ----------------
class FakeComm:
    def __init__(self, chunks, delay=0.0):
        self.chunks, self.delay = chunks, delay

    async def stream(self):
        for c in self.chunks:
            await asyncio.sleep(self.delay)
            yield c


def collect(text, timer, factory, cache):
    async def run():
        return [c async for c in tts_chunks(text, timer, factory=factory, cache=cache)]
    return asyncio.run(run())


def test_first_chunk_is_measured_separately_from_total():
    t = timing.TurnTimer()
    chunks = [{"type": "WordBoundary"}, {"type": "audio", "data": b"a"}, {"type": "audio", "data": b"b"}]
    out = collect("hi", t, lambda _: FakeComm(chunks, delay=0.02), LRUAudioCache())
    assert out == [b"a", b"b"]
    assert 0 < t.spans["tts_first_chunk"] < t.spans["tts_total"]


def test_cache_hit_skips_synthesis_and_empty_audio_raises():
    cache, calls = LRUAudioCache(), []

    def factory(text):
        calls.append(text)
        return FakeComm([{"type": "audio", "data": b"xy"}])

    t = timing.TurnTimer()
    assert collect("same", t, factory, cache) == [b"xy"]
    assert collect("same", t, factory, cache) == [b"xy"]
    assert calls == ["same"] and t.notes["tts_cache"] == "hit"
    with pytest.raises(RuntimeError):
        collect("empty", t, lambda _: FakeComm([{"type": "WordBoundary"}]), cache)


def test_lru_evicts_oldest():
    c = LRUAudioCache(size=2)
    c.put("a", b"1"); c.put("b", b"2"); c.get("a"); c.put("c", b"3")
    assert c.get("b") is None and c.get("a") == b"1" and c.get("c") == b"3"
