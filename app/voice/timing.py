"""Per-turn latency profiling.

A TurnTimer is created at the start of a turn and stored in a ContextVar, so deep code
(LLM call, menu load) can record spans with `span("llm")` without being passed the timer.
When no timer is active, `span()` is a no-op, so this is safe in tests and telephony paths.
"""
import contextvars
import time
from contextlib import contextmanager
from typing import Dict, Optional

_current: contextvars.ContextVar = contextvars.ContextVar("turn_timer", default=None)


class TurnTimer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.spans: Dict[str, float] = {}
        self.notes: Dict[str, object] = {}

    def add(self, name: str, seconds: float) -> None:
        self.spans[name] = self.spans.get(name, 0.0) + seconds

    def note(self, name: str, value) -> None:
        self.notes[name] = value

    @contextmanager
    def span(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:  # recorded even if the wrapped call raises (e.g. Deepgram timeout)
            self.add(name, time.perf_counter() - start)

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def summary(self) -> str:
        parts = [f"{k} {v:.2f}s" for k, v in self.spans.items()]
        parts += [f"{k}={v}" for k, v in self.notes.items()]
        parts.append(f"turn_total {self.elapsed():.2f}s")
        return " | ".join(parts)


def start_turn() -> TurnTimer:
    timer = TurnTimer()
    _current.set(timer)
    return timer


def current_timer() -> Optional[TurnTimer]:
    return _current.get()


@contextmanager
def span(name: str):
    timer = _current.get()
    if timer is None:
        yield
    else:
        with timer.span(name):
            yield
