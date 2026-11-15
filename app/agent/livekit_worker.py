import asyncio
import logging
import os
import time
import uuid
from typing import Set, Optional

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.plugins import silero, deepgram
import edge_tts
from sqlalchemy.exc import IntegrityError
from sqlalchemy.future import select

from app.agent.graph import order_agent, load_menu
from app.agent.relevance_gate import run_gated_turn
from app.agent.state import OrderState
from app.db.models import Call
from app.db.session import AsyncSessionLocal
from app.services.order_service import create_order_in_db
from app.services.outbox_dispatcher import process_outbox_batch
from app.voice.speech_guard import GateFailurePrompter, IdlePromptScheduler, InactivityClock

load_dotenv()
logger = logging.getLogger("voice-agent")

# Acoustic shielding: 0.25s threshold rejects breath, coughs, and speaker echo
VAD_MODEL = silero.VAD.load(
    min_silence_duration=0.35,
    min_speech_duration=0.25,
    prefix_padding_duration=0.10,
)

SAMPLE_RATE = 24000
FRAME_DURATION_MS = 20
SAMPLES_PER_FRAME = (SAMPLE_RATE * FRAME_DURATION_MS) // 1000  # 480
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2                       # 960 (16-bit mono)

MAX_CALL_DURATION_SEC = 240.0   # 4 minutes monotonic hard ceiling
INACTIVITY_TIMEOUT_SEC = 35.0   # 35 seconds of true silence
MAX_SPEECH_BURST_SEC = 15.0     # 15 seconds guard against lost VAD end-of-speech packets

_BACKGROUND_TASKS: Set[asyncio.Task] = set()


def _on_dispatch_done(task: asyncio.Task) -> None:
    """Safely cleans up the task set and logs background dispatch failures."""
    _BACKGROUND_TASKS.discard(task)
    if task.cancelled():
        logger.warning("Outbox dispatch background task was cancelled.")
        return
    error = task.exception()
    if error:
        logger.error(
            "Outbox dispatch background task failed.",
            exc_info=(type(error), error, error.__traceback__),
        )


def prewarm(proc: agents.JobProcess):
    """Executes synchronously inside the worker subprocess initialization."""
    import groq
    import httpcore
    import langchain_groq
    from app.agent.nodes import _get_structured_llm
    
    try:
        _get_structured_llm()
        logger.info("✅ Schemas and LLM modules prewarmed.")
    except Exception as e:
        logger.warning(f"Prewarm notice: {e}")


async def stream_edge_tts_pipe(text: str, source: rtc.AudioSource):
    """Pipes Edge-TTS into FFmpeg stdin with resilient task and process termination."""
    if not text:
        return

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "mp3",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        # stderr was PIPE but never read: a full pipe buffer can stall ffmpeg mid-reply.
        stderr=asyncio.subprocess.DEVNULL,
    )

    communicate = edge_tts.Communicate(text, "hi-IN-SwaraNeural", rate="+18%")

    async def feed_stdin():
        try:
            async for chunk in communicate.stream():
                if proc.returncode is not None:
                    break
                if chunk["type"] == "audio" and proc.stdin:
                    proc.stdin.write(chunk["data"])
                    await proc.stdin.drain()
        except asyncio.CancelledError:
            raise  # let cancellation propagate; `finally` still closes stdin
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            logger.debug(f"Feed stdin exception: {e}")
        finally:
            if proc.stdin and not proc.stdin.is_closing():
                try:
                    proc.stdin.close()
                except Exception:
                    pass

    async def read_stdout():
        start_time = time.perf_counter()
        frame_idx = 0
        carry = b""

        while True:
            needed = BYTES_PER_FRAME - len(carry)
            try:
                raw = await proc.stdout.read(needed)
            except Exception:
                break

            if not raw:
                break

            carry += raw
            if len(carry) == BYTES_PER_FRAME:
                frame = rtc.AudioFrame(
                    data=carry,
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=SAMPLES_PER_FRAME,
                )
                await source.capture_frame(frame)
                carry = b""
                frame_idx += 1

                expected_time = start_time + (frame_idx * (FRAME_DURATION_MS / 1000.0))
                delay = expected_time - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)

        if carry:
            carry += b"\x00" * (BYTES_PER_FRAME - len(carry))
            await source.capture_frame(
                rtc.AudioFrame(
                    data=carry,
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=SAMPLES_PER_FRAME,
                )
            )

    writer = asyncio.create_task(feed_stdin())
    reader = asyncio.create_task(read_stdout())

    try:
        await asyncio.gather(writer, reader)
    except asyncio.CancelledError:
        writer.cancel()
        reader.cancel()
        await asyncio.gather(writer, reader, return_exceptions=True)
        raise
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            except Exception:
                pass
        try:
            await proc.wait()
        except Exception:
            pass


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    call_sid = ctx.room.name
    agent_mode = os.getenv("AGENT_MODE", "restaurant").strip().lower()
    is_clinic = (agent_mode == "clinic")
    logger.info(f"Connected to LiveKit room: {call_sid} [mode={agent_mode}]")

    restaurant_id = "9480b5c5-343a-48e2-b2a4-059ae20e90a9"
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")

    if is_clinic:
        from app.clinic.service import load_booked_slots
        raw_menu = []
        initial_booked_slots = await load_booked_slots(restaurant_id)
    else:
        raw_menu = await load_menu(restaurant_id)
        initial_booked_slots = []

    if deepgram_key:
        stt = deepgram.STT(
            model="nova-2",
            language="hi",
            api_key=deepgram_key,
            interim_results=False,
            smart_format=True,
            endpointing_ms=300,
        )
    else:
        from livekit.plugins import openai as lk_openai
        stt = lk_openai.STT(
            model="whisper-large-v3",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )

    source = rtc.AudioSource(SAMPLE_RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    await ctx.room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    playback_handle = None
    playback_lock = asyncio.Lock()
    turn_lock = asyncio.Lock()

    def _on_playback_done(task: asyncio.Task) -> None:
        """Retrieve/log playback failures so a failed prompt never leaves an unretrieved exception."""
        if not task.cancelled() and task.exception() is not None:
            logger.error("TTS playback failed: %s", type(task.exception()).__name__)

    async def cancel_active_speech():
        nonlocal playback_handle
        async with playback_lock:
            if playback_handle and not playback_handle.done():
                playback_handle.cancel()
                # gather() (not a bare await + `except CancelledError: pass`) so that if THIS
                # task is cancelled while waiting, the cancellation still propagates.
                await asyncio.gather(playback_handle, return_exceptions=True)
            source.clear_queue()

    async def speak(text: str, wait: bool = False, timeout: float = 8.0):
        nonlocal playback_handle
        await cancel_active_speech()
        async with playback_lock:
            playback_handle = asyncio.create_task(stream_edge_tts_pipe(text, source))
            playback_handle.add_done_callback(_on_playback_done)
        if wait:
            try:
                await asyncio.wait_for(asyncio.shield(playback_handle), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Playback timed out after {timeout}s.")
                await cancel_active_speech()

    async def speak_if_idle(text: str) -> bool:
        """Speak only if the assistant is NOT already talking. Never cancels playback.

        Used by the idle and gate-failure prompts so they cannot cut off a real reply.
        """
        nonlocal playback_handle
        async with playback_lock:
            if playback_handle is not None and not playback_handle.done():
                return False
            playback_handle = asyncio.create_task(stream_edge_tts_pipe(text, source))
            playback_handle.add_done_callback(_on_playback_done)
            return True

    caller_phone = None
    for _, participant in ctx.room.remote_participants.items():
        if participant.attributes.get("sip.caller"):
            caller_phone = participant.attributes.get("sip.caller")
            break
        elif participant.identity and participant.identity.startswith("+"):
            caller_phone = participant.identity
            break

    async with AsyncSessionLocal() as session:
        try:
            new_call = Call(
                restaurant_id=uuid.UUID(restaurant_id),
                provider_call_sid=call_sid,
                from_number=caller_phone or "UNAVAILABLE",
                to_number="LIVEKIT_SIP_TRUNK",
            )
            session.add(new_call)
            await session.commit()
            call_db_id = new_call.id
        except IntegrityError:
            await session.rollback()
            res = await session.execute(select(Call).where(Call.provider_call_sid == call_sid))
            existing_call = res.scalars().first()
            if existing_call is None:
                raise RuntimeError(
                    f"Call record conflict encountered, but no existing call found with SID: {call_sid}"
                )
            call_db_id = existing_call.id

    current_state: OrderState = {
        "call_id": call_db_id,
        "call_sid": call_sid,
        "restaurant_id": restaurant_id,
        "step": "ordering",
        "cart": [],
        "customer_phone": caller_phone,
        "customer_name": None,
        "delivery_address": None,
        "latest_speech": "",
        "reply": "",
        "confirmation_status": "pending",
        "retries": 0,
        "details_retries": 0,
        "conversation_history": [],
        "menu_items": raw_menu,
    }
    if is_clinic:
        current_state["booked_slots"] = initial_booked_slots

    call_start_monotonic = time.monotonic()
    # Inactivity policy lives in speech_guard.InactivityClock (unit-tested without LiveKit).
    # Only gate-accepted speech fully resets it; VAD bursts bank bounded credit (see its docstring).
    clock = InactivityClock(call_start_monotonic, INACTIVITY_TIMEOUT_SEC, MAX_SPEECH_BURST_SEC)
    gate_failure_prompter = GateFailurePrompter(speak_if_idle)
    is_customer_speaking = False
    call_active = True
    processed_tracks = set()
    _track_tasks: Set[asyncio.Task] = set()

    async def guardrail_watchdog():
        nonlocal call_active, is_customer_speaking
        while call_active:
            await asyncio.sleep(1.0)
            now = time.monotonic()

            if clock.clear_stuck_burst(now):
                logger.warning("VAD end-of-speech event missed; auto-clearing is_customer_speaking.")
                is_customer_speaking = False

            if now - call_start_monotonic >= MAX_CALL_DURATION_SEC:
                logger.warning(f"[{call_sid}] 240s hard limit reached.")
                call_active = False
                await speak("माफ़ कीजिए, कॉल की समय सीमा समाप्त हो गई है। धन्यवाद!", wait=True, timeout=3.0)
                await ctx.room.disconnect()
                break

            if clock.is_inactive(now, turn_busy=turn_lock.locked()):
                logger.info(f"[{call_sid}] 35s silence watchdog triggered.")
                call_active = False
                await speak("कोई जवाब न मिलने के कारण कॉल समाप्त की जा रही है। धन्यवाद!", wait=True, timeout=3.0)
                await ctx.room.disconnect()
                break

    watchdog_task = asyncio.create_task(guardrail_watchdog())

    async def process_customer_audio(audio_track: rtc.Track):
        nonlocal is_customer_speaking, call_active

        if audio_track.sid in processed_tracks:
            return
        processed_tracks.add(audio_track.sid)

        audio_stream = rtc.AudioStream(audio_track)
        stt_stream = stt.stream()
        vad_stream = VAD_MODEL.stream()

        async def push_audio():
            try:
                async for event in audio_stream:
                    if not call_active:
                        break
                    stt_stream.push_frame(event.frame)
                    vad_stream.push_frame(event.frame)
            except Exception as e:
                logger.error(f"Audio stream push error: {e}")

        # Re-prompt after a barge-in that never produced a transcript. Race-safe against a late
        # END_OF_SPEECH and uses speak_if_idle so it can never cut off an assistant reply.
        idle_prompt = IdlePromptScheduler(
            4.0,
            should_prompt=lambda: (
                call_active
                and not is_customer_speaking
                and not turn_lock.locked()  # a gate/agent turn is in flight; don't talk over it
                and current_state.get("step") != "closing"
                and current_state.get("confirmation_status") == "pending"
            ),
            speak_if_idle=speak_if_idle,
        )

        async def monitor_vad():
            nonlocal is_customer_speaking
            async for vad_event in vad_stream:
                if not call_active:
                    break
                now = time.monotonic()
                if vad_event.type == agents.vad.VADEventType.START_OF_SPEECH:
                    is_customer_speaking = True
                    # Inactivity is NOT fully reset by VAD (TV/side-talk also trigger it); the clock
                    # only banks bounded credit for the burst. Only gate-accepted speech resets it.
                    clock.speech_started(now)
                    idle_prompt.speech_started()  # sync cancel: barge-in must not wait on this
                    logger.info("🟢 Barge-in: caller speaking.")
                    await cancel_active_speech()
                    await idle_prompt.cancel()
                elif vad_event.type == agents.vad.VADEventType.END_OF_SPEECH:
                    is_customer_speaking = False
                    clock.speech_ended(now)
                    if current_state.get("step") != "closing":
                        await idle_prompt.speech_ended()

        push_task = asyncio.create_task(push_audio())
        vad_task = asyncio.create_task(monitor_vad())

        try:
            async for stt_event in stt_stream:
                if not call_active:
                    break
                if stt_event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
                    # FIX 1: Immediately clear speaking state on transcript arrival.
                    # The inactivity clock is reset only AFTER the relevance gate accepts the
                    # utterance (see on_relevant below). Marking the transcript handled now stops a
                    # late END_OF_SPEECH from arming an idle prompt for a burst already processed.
                    is_customer_speaking = False
                    clock.speech_ended(time.monotonic())
                    idle_prompt.transcript_handled()

                    user_text = stt_event.alternatives[0].text.strip()
                    if not user_text:
                        continue

                    async with turn_lock:
                        if not call_active or current_state.get("step") == "closing":
                            logger.info("Transcript ignored: call closing or inactive.")
                            continue

                        t_start = time.perf_counter()
                        logger.info(f"🗣️ Customer: {user_text}")

                        async def on_relevant():
                            clock.note_relevant(time.monotonic())
                            gate_failure_prompter.reset()
                            await idle_prompt.cancel()  # cancel AND await the obsolete idle prompt

                        if is_clinic:
                            from app.clinic.gate import run_clinic_gated_turn
                            from app.clinic.nodes import process_clinic_turn
                            outcome = await run_clinic_gated_turn(
                                user_text,
                                current_state,
                                process_clinic_turn,
                                on_relevant=on_relevant,
                            )
                        else:
                            # Gate -> (accepted cleaned_text only) -> existing LangGraph pipeline
                            outcome = await run_gated_turn(
                                user_text,
                                current_state,
                                current_state.get("menu_items") or raw_menu,
                                order_agent.ainvoke,
                                on_relevant=on_relevant,
                            )

                        gate = outcome.gate
                        if outcome.new_state is None:
                            logger.info(
                                f"🚫 Gate rejected [{gate.category}/{gate.stage}] "
                                f"({gate.reason}); no agent turn, cart and watchdog untouched."
                            )
                            if gate.failed:
                                # Gate FAILURE (Groq timeout/error/malformed), not unrelated speech:
                                # ask the caller to repeat. Runs under turn_lock, never cancels
                                # playback, and touches no cart/retries/clock state.
                                await gate_failure_prompter.prompt(
                                    can_prompt=(
                                        call_active
                                        and not is_customer_speaking
                                        and current_state.get("step") != "closing"
                                    )
                                )
                            continue
                        logger.info(f"✅ Gate accepted [{gate.category}/{gate.stage}] conf={gate.confidence:.2f}")
                        current_state.update(outcome.new_state)

                        ai_text = current_state.get("reply", "")
                        step = current_state.get("step")
                        status = current_state.get("confirmation_status")
                        retries = current_state.get("retries", 0)
                        delivery_address = current_state.get("delivery_address")
                        customer_name = current_state.get("customer_name")
                        cart = current_state.get("cart", [])

                        logger.info(
                            f"🤖 Decision ({time.perf_counter() - t_start:.2f}s | Retries: {retries}): {ai_text} "
                            f"[step={step}, status={status}, addr={bool(delivery_address)}]"
                        )

                        # Fail-safe state synchronization
                        if delivery_address and ("ऑर्डर दर्ज किया जा रहा है" in ai_text or "दर्ज" in ai_text):
                            step = "closing"
                            status = "pending"
                            current_state["step"] = "closing"
                            current_state["confirmation_status"] = "pending"

                        if retries >= 3:
                            call_active = False
                            await speak(ai_text, wait=True, timeout=4.0)
                            await ctx.room.disconnect()
                            break

                        if step == "closing":
                            await idle_prompt.cancel()
                            logger.info("Closing flow: fallback prompts disarmed.")

                            if status == "cancelled":
                                call_active = False
                                logger.info(f"Order cancelled by customer for call {call_sid}")
                                await speak(ai_text, wait=True, timeout=4.0)
                                await asyncio.sleep(0.5)
                                await ctx.room.disconnect()
                                break

                            if status != "pending":
                                call_active = False
                                logger.error(f"Unexpected closing status '{status}' for call {call_sid}")
                                await ctx.room.disconnect()
                                break

                            if not cart or not delivery_address:
                                call_active = False
                                logger.error(
                                    f"Refusing to persist incomplete order for call {call_sid}: "
                                    f"cart={bool(cart)}, address={bool(delivery_address)}"
                                )
                                incomplete_msg = (
                                    "माफ़ कीजिए, आपकी अपॉइंटमेंट जानकारी अधूरी है। कृपया दोबारा प्रयास करें।"
                                    if is_clinic
                                    else "माफ़ कीजिए, आपकी ऑर्डर जानकारी अधूरी है। कृपया दोबारा प्रयास करें।"
                                )
                                await speak(
                                    incomplete_msg,
                                    wait=True,
                                    timeout=4.0,
                                )
                                await ctx.room.disconnect()
                                break

                            try:
                                logger.info("Writing order and outbox notification to database...")
                                async def _db_save():
                                    async with AsyncSessionLocal() as session:
                                        if is_clinic:
                                            from app.clinic.service import create_appointment_in_db
                                            return await create_appointment_in_db(current_state, session)
                                        return await create_order_in_db(current_state, session)

                                created_order = await asyncio.wait_for(_db_save(), timeout=5.0)

                                if not created_order or not getattr(created_order, "order_number", None):
                                    raise RuntimeError("Database did not return a valid order object.")

                                logger.info(f"✅ Order {created_order.order_number} committed successfully.")
                                current_state["confirmation_status"] = "confirmed"

                                try:
                                    dispatch_task = asyncio.create_task(process_outbox_batch())
                                    _BACKGROUND_TASKS.add(dispatch_task)
                                    dispatch_task.add_done_callback(_on_dispatch_done)
                                except Exception as dispatch_err:
                                    logger.warning(f"Immediate dispatch enqueue warning: {dispatch_err}")

                                if is_clinic:
                                    final_speech = (
                                        f"धन्यवाद {customer_name}! आपका अपॉइंटमेंट कन्फर्म हो गया है।"
                                        if customer_name
                                        else "धन्यवाद! आपका अपॉइंटमेंट कन्फर्म हो गया है।"
                                    )
                                else:
                                    final_speech = (
                                        f"धन्यवाद {customer_name}! आपका ऑर्डर कन्फर्म हो गया है।"
                                        if customer_name
                                        else "धन्यवाद! आपका ऑर्डर कन्फर्म हो गया है।"
                                    )
                                logger.info(f"Speaking confirmation: {final_speech}")
                                await speak(final_speech, wait=True, timeout=6.0)
                                await asyncio.sleep(0.5)

                                call_active = False
                                await ctx.room.disconnect()
                                break

                            except Exception as save_err:
                                if is_clinic and type(save_err).__name__ == "SlotUnavailableError":
                                    from app.clinic.catalog import get_open_slots
                                    from app.clinic.service import load_booked_slots
                                    logger.warning(f"Slot taken mid-call ({save_err}); re-offering open slots.")
                                    taken_id = cart[0].get("slot_id") if cart else None
                                    booked = set(await load_booked_slots(restaurant_id))
                                    if taken_id:
                                        booked.add(taken_id)
                                    current_state["booked_slots"] = list(booked)
                                    if cart:
                                        cart[0]["slot"] = None
                                        cart[0]["slot_id"] = None
                                    current_state["step"] = "ordering"
                                    current_state["confirmation_status"] = "pending"
                                    open_slots = get_open_slots(booked=booked)
                                    if open_slots:
                                        slots_txt = " या ".join(s["label"] for s in open_slots[:3])
                                        recovery_msg = f"माफ़ कीजिए, वह समय अभी बुक हो गया है। उपलब्ध समय: {slots_txt}। कौन सा समय सही रहेगा?"
                                        await speak(recovery_msg, wait=False)
                                        continue

                                call_active = False
                                logger.exception(f"Order commit failed for call {call_sid}: {save_err}")
                                current_state["confirmation_status"] = "failed"

                                fail_msg = (
                                    "माफ़ कीजिए, सिस्टम में समस्या के कारण आपका अपॉइंटमेंट दर्ज नहीं हो पाया। कृपया कुछ समय बाद पुनः प्रयास करें।"
                                    if is_clinic
                                    else "माफ़ कीजिए, सिस्टम में समस्या के कारण आपका ऑर्डर दर्ज नहीं हो पाया। कृपया कुछ समय बाद पुनः प्रयास करें।"
                                )
                                await speak(
                                    fail_msg,
                                    wait=True,
                                    timeout=4.0,
                                )
                                await ctx.room.disconnect()
                                break

                        # In-flight conversational turn: speak response asynchronously
                        await speak(ai_text, wait=False)

        except Exception as e:
            logger.exception(f"STT stream loop exception: {e}")
        finally:
            push_task.cancel()
            vad_task.cancel()
            await asyncio.gather(push_task, vad_task, return_exceptions=True)
            # Awaited AFTER vad_task is gone so nothing can re-arm it behind our back.
            await idle_prompt.cancel()

            for stream_obj in (audio_stream, stt_stream, vad_stream):
                try:
                    await stream_obj.aclose()
                except Exception:
                    pass

    def _start_track_task(track_obj: rtc.Track):
        t = asyncio.create_task(process_customer_audio(track_obj))
        _track_tasks.add(t)
        t.add_done_callback(_track_tasks.discard)

    try:
        @ctx.room.on("track_subscribed")
        def on_track_subscribed(audio_track: rtc.Track, publication, participant):
            if audio_track.kind == rtc.TrackKind.KIND_AUDIO:
                _start_track_task(audio_track)

        for _, participant in ctx.room.remote_participants.items():
            for _, publication in participant.track_publications.items():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_AUDIO:
                    _start_track_task(publication.track)

        if is_clinic:
            await speak("नमस्ते! स्माइल डेंटल क्लिनिक में आपका स्वागत है। आप किस समस्या के लिए अपॉइंटमेंट बुक करना चाहेंगे?", wait=False)
        else:
            await speak("Namaste! Spice Villa mein swagat hai. Kya order करेंगे?", wait=False)

        while call_active and ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(0.5)

    finally:
        call_active = False

        watchdog_task.cancel()
        await asyncio.gather(watchdog_task, return_exceptions=True)

        if _track_tasks:
            for t in _track_tasks:
                t.cancel()
            await asyncio.gather(*_track_tasks, return_exceptions=True)
            _track_tasks.clear()

        await cancel_active_speech()


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=os.getenv("AGENT_NAME", "order-agent"),
        )
    )