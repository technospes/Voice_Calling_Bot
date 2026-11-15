"""Speech guard and timing helpers for live telephony audio.

Provides:
- InactivityClock: Tracks pure silence vs bounded credit for VAD bursts, resetting
  fully only when an utterance is accepted as relevant by the relevance gate.
- IdlePromptScheduler: Manages race-safe delayed re-prompts after an unhandled barge-in.
- GateFailurePrompter: Triggers gentle clarification prompts when the relevance gate
  itself fails (e.g. timeout or LLM error), without cutting active playback.
"""
import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

logger = logging.getLogger("speech-guard")


class InactivityClock:
    """Manages call inactivity and silence watchdog timeouts.
    
    Prevents background noise or side-talk from holding the line open indefinitely
    by requiring gate-accepted speech to grant full resets, while granting bounded
    credit for unhandled VAD speech bursts.
    """

    def __init__(
        self,
        start_time: float,
        inactivity_timeout_sec: float = 35.0,
        max_speech_burst_sec: float = 15.0,
    ) -> None:
        self.inactivity_timeout_sec = inactivity_timeout_sec
        self.max_speech_burst_sec = max_speech_burst_sec
        self.last_speech_time = start_time
        self.burst_start_time: Optional[float] = None

    def speech_started(self, now: float) -> None:
        """Marked by VAD START_OF_SPEECH."""
        if self.burst_start_time is None:
            self.burst_start_time = now

    def speech_ended(self, now: float) -> None:
        """Marked by VAD END_OF_SPEECH or STT transcript arrival."""
        if self.burst_start_time is not None:
            # Grant bounded credit: max of max_speech_burst_sec
            burst_duration = min(now - self.burst_start_time, self.max_speech_burst_sec)
            self.last_speech_time = max(self.last_speech_time, self.burst_start_time + burst_duration)
            self.burst_start_time = None

    def note_relevant(self, now: float) -> None:
        """Called only when the gate accepts an utterance as relevant order speech."""
        self.last_speech_time = now
        self.burst_start_time = None

    def clear_stuck_burst(self, now: float) -> bool:
        """Watchdog check: resets is_customer_speaking if end-of-speech was lost."""
        if self.burst_start_time is not None and (now - self.burst_start_time > self.max_speech_burst_sec):
            self.burst_start_time = None
            return True
        return False

    def is_inactive(self, now: float, turn_busy: bool = False) -> bool:
        """Checks if silence exceeds threshold (exempt while turn processing is active)."""
        if turn_busy or self.burst_start_time is not None:
            return False
        return (now - self.last_speech_time) >= self.inactivity_timeout_sec


class IdlePromptScheduler:
    """Schedules a fallback prompt if customer barged in but transcript was lost or dropped."""

    def __init__(
        self,
        delay_sec: float = 4.0,
        should_prompt: Optional[Callable[[], bool]] = None,
        speak_if_idle: Optional[Callable[[str], Awaitable[bool]]] = None,
        prompt_text: str = "हाँजी, कहिए?",
    ) -> None:
        self.delay_sec = delay_sec
        self.should_prompt = should_prompt or (lambda: True)
        self.speak_if_idle = speak_if_idle
        self.prompt_text = prompt_text
        self._task: Optional[asyncio.Task] = None

    def speech_started(self) -> None:
        """Cancel task synchronously on barge-in so playback cancellation is immediate."""
        if self._task and not self._task.done():
            self._task.cancel()

    async def speech_ended(self) -> None:
        """Schedules delayed re-prompt on end of speech."""
        await self.cancel()
        self._task = asyncio.create_task(self._run_timer())

    def transcript_handled(self) -> None:
        """Prevents late END_OF_SPEECH from scheduling a prompt after transcript arrival."""
        if self._task and not self._task.done():
            self._task.cancel()

    async def cancel(self) -> None:
        """Cancels and awaits the prompt task."""
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _run_timer(self) -> None:
        try:
            await asyncio.sleep(self.delay_sec)
            if self.should_prompt() and self.speak_if_idle:
                logger.info("Idle prompt triggered: prompting customer.")
                await self.speak_if_idle(self.prompt_text)
        except asyncio.CancelledError:
            pass


class GateFailurePrompter:
    """Prompts caller to repeat only when the gate classifier itself encounters an error/timeout."""

    def __init__(
        self,
        speak_if_idle: Callable[[str], Awaitable[bool]],
        prompt_text: str = "माफ़ कीजिए, आपकी आवाज़ साफ़ नहीं आई। कृपया दोबारा बोलिए?",
    ) -> None:
        self.speak_if_idle = speak_if_idle
        self.prompt_text = prompt_text
        self.has_prompted = False

    def reset(self) -> None:
        self.has_prompted = False

    async def prompt(self, can_prompt: bool = True) -> bool:
        if not can_prompt or self.has_prompted:
            return False
        self.has_prompted = True
        logger.info("Relevance gate failure prompter triggered.")
        return await self.speak_if_idle(self.prompt_text)