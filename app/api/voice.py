from fastapi import APIRouter, Request, WebSocket, HTTPException, WebSocketDisconnect
from sqlalchemy.future import select
from app.telephony.simulator import SimulatorAdapter
from app.telephony.twilio_adapter import TwilioAdapter
from app.config import settings
from app.db.session import AsyncSessionLocal
from app.db.models import Call, Restaurant, MenuItem
from app.voice.stream_handler import handle_audio_stream
from app.agent.graph import order_agent, menu_item_to_dict
from app.agent.state import initial_order_state
from app.services.order_service import create_order_in_db
import asyncio
import logging
import time
import httpx
from app.voice.protocol import parse_control, wants_tts_stream
from app.voice.timing import start_turn
from app.voice.tts_stream import tts_chunks
from groq import AsyncGroq

router = APIRouter(prefix="/voice", tags=["voice"])
adapter = SimulatorAdapter()  # Using simulator for Phase 1 testing
twilio_adapter = TwilioAdapter()
groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
logger = logging.getLogger(__name__)
SAVE_FAILED_REPLY = "माफ़ कीजिए, आपका ऑर्डर सेव नहीं हो पाया। कृपया दोबारा कोशिश करें।"

@router.post("/incoming")
async def incoming_call(request: Request):
    """Triggered when the restaurant number rings."""
    payload = await request.json()
    call_info = adapter.parse_incoming(payload)
    
    async with AsyncSessionLocal() as session:
        # Find the restaurant by the dialed number
        res = await session.execute(select(Restaurant).where(Restaurant.inbound_number == call_info.to_number))
        restaurant = res.scalars().first()
        
        if not restaurant:
            raise HTTPException(status_code=404, detail="Restaurant not found for this number")
            
        # Log the call in the database
        new_call = Call(
            restaurant_id=restaurant.id,
            provider_call_sid=call_info.call_sid,
            from_number=call_info.from_number,
            to_number=call_info.to_number
        )
        session.add(new_call)
        await session.commit()

    # Construct the WebSocket URL for live audio streaming
    ws_protocol = "wss" if settings.PUBLIC_BASE_URL.startswith("https") else "ws"
    base_domain = settings.PUBLIC_BASE_URL.split("://")[-1]
    wss_url = f"{ws_protocol}://{base_domain}/voice/stream/{call_info.call_sid}"
    
    return adapter.answer_with_stream(call_info, wss_url)

@router.post("/status")
async def call_status(request: Request):
    """Triggered when a call ends or fails."""
    return {"status": "logged"}

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen?model=nova-2&language=hi"


async def _warm_connection(client: httpx.AsyncClient) -> None:
    """Open the TLS connection to Deepgram in the background so the first real request reuses it."""
    try:
        await client.head("https://api.deepgram.com/", timeout=5.0)
    except Exception as e:
        logger.debug("Deepgram warm-up failed (non-fatal): %s", e)


async def _transcribe(client: httpx.AsyncClient, audio_data: bytes, menu_hint: str, timer) -> str:
    """Deepgram Nova-2 (primary) -> Groq Whisper (fallback). Behaviour unchanged; now timed."""
    try:
        with timer.span("stt_deepgram"):
            dg_res = await client.post(
                DEEPGRAM_URL,
                headers={"Authorization": f"Token {settings.DEEPGRAM_API_KEY}"},
                content=audio_data,
                timeout=5.0,
            )
            dg_res.raise_for_status()
        body = dg_res.json()
        timer.note("audio_sec", round(float(body.get("metadata", {}).get("duration", 0)), 2))
        user_text = body["results"]["channels"][0]["alternatives"][0]["transcript"]
        if not user_text.strip():
            raise ValueError("Deepgram returned empty string.")
        return user_text
    except Exception as e:
        print(f"⚠️ Deepgram STT failed ({e}). Falling back to Groq...")
        with timer.span("stt_whisper"):
            transcription = await groq_client.audio.transcriptions.create(
                file=("audio.webm", audio_data),
                model="whisper-large-v3",
                prompt=(
                    f"Restaurant order containing these items: {menu_hint}. "
                    "Common phrases: ek, do, naan, paneer, masala, nahi bas, confirm kar do."
                ),
            )
        return transcription.text


@router.websocket("/stream/local_test")
async def local_audio_stream(websocket: WebSocket, context_id: str = "test"):
    await websocket.accept()
    print(f"[Local Test] WebSocket connected for {context_id}")

    # Prevent DB collisions across sessions
    unique_context_id = f"test_{int(time.time())}"

    # Create a Call record so the state has a valid call_id FK root
    restaurant_id = "9480b5c5-343a-48e2-b2a4-059ae20e90a9"
    async with AsyncSessionLocal() as session:
        new_call = Call(
            restaurant_id=restaurant_id,
            provider_call_sid=unique_context_id,
            from_number="+919876543210",
            to_number="LOCAL_TEST"
        )
        session.add(new_call)
        await session.commit()
        call_db_id = new_call.id

        # Fetch menu once so we can hint Whisper and know current items
        menu_res = await session.execute(
            select(MenuItem).where(MenuItem.restaurant_id == restaurant_id)
        )
        menu_items_db = menu_res.scalars().all()

    current_state = initial_order_state(call_db_id, unique_context_id, restaurant_id, "+919876543210")
    # Menu was already fetched above: seed the cache so the first turn skips a DB round trip.
    current_state["menu_items"] = [menu_item_to_dict(m) for m in menu_items_db if m.is_available]
    menu_hint = ", ".join(m.name for m in menu_items_db)
    order_saved = False
    stream_tts = False   # becomes True only if the browser sends {"event":"hello","caps":["tts_stream"]}
    turn_no = 0

    # One HTTP client (and its TLS connection) for the whole WebSocket session.
    async with httpx.AsyncClient() as client:
        warm_task = asyncio.create_task(_warm_connection(client))
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code", 1000))

                # ---- control (text) frames: capability hello + client-side timings ----
                if message.get("text") is not None:
                    ctrl = parse_control(message["text"])
                    if ctrl is None:
                        continue
                    if ctrl["event"] == "hello":
                        stream_tts = wants_tts_stream(ctrl)
                        print(f"[Local Test] Client hello: tts_stream={stream_tts}")
                    elif ctrl["event"] == "client_timing":
                        print(f"📲 CLIENT TIMING: {ctrl}")
                    continue

                audio_data = message.get("bytes")
                if not audio_data:
                    continue

                turn_no += 1
                timer = start_turn()
                timer.note("audio_kb", round(len(audio_data) / 1024, 1))

                # 1. STT
                user_text = await _transcribe(client, audio_data, menu_hint, timer)
                print(f"\n🗣️ User said: {user_text}")

                # 2. LangGraph agent (llm / menu_load spans are recorded inside; the rest is cart logic)
                current_state["latest_speech"] = user_text
                with timer.span("agent_total"):
                    new_state = await order_agent.ainvoke(current_state)
                current_state.update(new_state)
                timer.add("cart_processing", max(
                    timer.spans["agent_total"] - timer.spans.get("llm", 0.0) - timer.spans.get("menu_load", 0.0), 0.0))

                ai_text_response = current_state.get("reply") or "Theek hai."

                # Save ONLY an explicitly confirmed order, before telling the caller it is done.
                if (current_state.get("step") == "closing"
                        and current_state.get("confirmation_status") == "confirmed"
                        and not order_saved):
                    try:
                        with timer.span("db_save"):
                            async with AsyncSessionLocal() as save_session:
                                await create_order_in_db(current_state, save_session)
                        order_saved = True
                    except Exception:
                        logger.exception("Failed to save confirmed order")
                        ai_text_response = SAVE_FAILED_REPLY
                print(f"🤖 AI replying: {ai_text_response}")
                print(f"🛒 Current Cart: {current_state.get('cart')}")

                # 3. TTS (Edge-TTS, hi-IN-SwaraNeural) + send
                try:
                    if stream_tts:
                        # Incremental: forward each chunk as Edge-TTS produces it.
                        await websocket.send_json({"event": "tts_start", "format": "audio/mpeg", "turn": turn_no})
                        with timer.span("tts_and_send"):
                            async for chunk in tts_chunks(ai_text_response, timer):
                                await websocket.send_bytes(chunk)
                        await websocket.send_json({"event": "tts_end", "turn": turn_no})
                    else:
                        # Legacy: generate everything, then send one binary message.
                        tts_bytes = b"".join([c async for c in tts_chunks(ai_text_response, timer)])
                        with timer.span("ws_send"):
                            await websocket.send_bytes(tts_bytes)
                except Exception as e:
                    print(f"❌ Edge-TTS failed: {e}")
                    raise

                print(f"⏱️ TURN {turn_no}: {timer.summary()}")

                # 4. Conversation finished (confirmed or cancelled; only confirmed was saved above)
                if current_state.get("step") == "closing":
                    print(f"✅ Call finished: {current_state.get('confirmation_status')} (saved: {order_saved})")
                    await websocket.close()
                    break

        except WebSocketDisconnect:
            print("[Local Test] Client disconnected.")
        except Exception as e:
            print(f"[Local Test] Stream error: {e}")
            await websocket.close()
        finally:
            warm_task.cancel()


@router.websocket("/stream/{call_sid}")
async def audio_stream(websocket: WebSocket, call_sid: str):
    """The live two-way audio WebSocket connection."""
    await handle_audio_stream(websocket, call_sid)

@router.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    """Secure webhook for Twilio calls."""
    await twilio_adapter.verify_request(request)
    
    form_data = await request.form()
    call_info = twilio_adapter.parse_incoming(dict(form_data))
    
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(Restaurant).where(Restaurant.inbound_number == call_info.to_number))
        restaurant = res.scalars().first()
        
        if not restaurant:
            print(f"Call rejected: Number {call_info.to_number} not found in DB.")
            return twilio_adapter.reject_call()
            
        new_call = Call(
            restaurant_id=restaurant.id, provider_call_sid=call_info.call_sid,
            from_number=call_info.from_number, to_number=call_info.to_number
        )
        session.add(new_call)
        await session.commit()

    ws_protocol = "wss" if settings.PUBLIC_BASE_URL.startswith("https") else "ws"
    base_domain = settings.PUBLIC_BASE_URL.split("://")[-1]
    wss_url = f"{ws_protocol}://{base_domain}/voice/twilio/stream/{call_info.call_sid}"
    
    return twilio_adapter.answer_with_stream(call_info, wss_url)

@router.websocket("/twilio/stream/{call_sid}")
async def twilio_audio_stream(websocket: WebSocket, call_sid: str):
    await websocket.accept()
    print(f"\n[Twilio {call_sid}] Stream connected successfully.")
    
    # Phase 5B will handle the base64 media stream here
    try:
        while True:
            data = await websocket.receive_json()
            event_type = data.get("event")
            print(f"Twilio Event: {event_type}")
            if event_type == "stop":
                break
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass