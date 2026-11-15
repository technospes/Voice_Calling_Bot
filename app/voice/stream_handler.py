import logging

from fastapi import WebSocket
from sqlalchemy.future import select
from starlette.websockets import WebSocketDisconnect

from app.agent.graph import order_agent
from app.agent.state import initial_order_state
from app.db.models import Call, CallTurn, Restaurant, Speaker
from app.db.session import AsyncSessionLocal
from app.services.order_service import create_order_in_db
from app.voice.timing import start_turn

logger = logging.getLogger(__name__)

SAVE_FAILED_REPLY = "माफ़ कीजिए, आपका ऑर्डर सेव नहीं हो पाया। कृपया दोबारा कोशिश करें।"


async def _owner_number(restaurant_id) -> str | None:
    """Real forwarding number from the restaurants table (no hardcoded number)."""
    async with AsyncSessionLocal() as s:
        res = await s.execute(select(Restaurant.owner_forward_number).where(Restaurant.id == restaurant_id))
        return res.scalars().first()


async def handle_audio_stream(websocket: WebSocket, call_sid: str):
    await websocket.accept()
    print(f"\n[{call_sid}] Call connected. Loading context...")

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Call).where(Call.provider_call_sid == call_sid))
        call = result.scalars().first()

        if not call:
            print(f"[{call_sid}] Call not found in DB.")
            await websocket.close()
            return

        print(f"[{call_sid}] Playing greeting...")
        await websocket.send_json({
            "event": "play_tts",
            "text": "Spice Villa mein aapka swagat hai. Order ke liye 1, owner se baat karne ke liye 3 dabaiye."
        })

        state = initial_order_state(call.id, call_sid, call.restaurant_id, call.from_number)
        turn_count = 1
        order_saved = False
        try:
            while True:
                data = await websocket.receive_json()
                event = data.get("event")

                if event == "dtmf":
                    digit = data.get("digit")
                    print(f"[{call_sid}] Caller pressed: {digit}")

                    session.add(CallTurn(call_id=call.id, turn_no=turn_count, speaker=Speaker.CUSTOMER, dtmf=digit))
                    await session.commit()
                    turn_count += 1

                    if digit == "3":
                        owner = await _owner_number(call.restaurant_id)
                        if owner:
                            print(f"[{call_sid}] Forwarding to owner...")
                            await websocket.send_json({"event": "forward", "to_number": owner})
                            break
                        logger.error("[%s] No owner_forward_number configured for restaurant %s", call_sid, call.restaurant_id)
                        await websocket.send_json({"event": "play_tts", "text": "Owner abhi available nahi hain. Aap order bol sakte hain."})
                    else:
                        await websocket.send_json({"event": "play_tts", "text": "Aapki aawaz record ki ja rahi hai. Boliye."})

                elif event == "speech":
                    spoken_text = data.get("text") or ""
                    print(f"[{call_sid}] Caller said: {spoken_text}")
                    session.add(CallTurn(call_id=call.id, turn_no=turn_count, speaker=Speaker.CUSTOMER, text=spoken_text))

                    state["latest_speech"] = spoken_text
                    timer = start_turn()
                    with timer.span("agent_total"):
                        new_state = await order_agent.ainvoke(state)
                    print(f"[{call_sid}] ⏱️ {timer.summary()}")
                    state.update(new_state)

                    agent_reply = state.get("reply") or "Theek hai."
                    step = state.get("step")

                    # Persist ONLY an explicitly confirmed order, and before telling the caller it is done.
                    if step == "closing" and state.get("confirmation_status") == "confirmed" and not order_saved:
                        try:
                            async with AsyncSessionLocal() as save_session:
                                await create_order_in_db(state, save_session)
                            order_saved = True
                        except Exception:
                            logger.exception("[%s] Failed to save confirmed order", call_sid)
                            agent_reply = SAVE_FAILED_REPLY

                    print(f"[{call_sid}] Agent replying: {agent_reply}")
                    print(f"[{call_sid}] Current Cart: {state.get('cart')}")

                    session.add(CallTurn(call_id=call.id, turn_no=turn_count + 1, speaker=Speaker.BOT, text=agent_reply))
                    await session.commit()
                    turn_count += 2

                    await websocket.send_json({"event": "play_tts", "text": agent_reply})

                    if step == "closing":  # confirmed or cancelled
                        await websocket.send_json({"event": "stop"})
                        break

                    if step == "forward":
                        owner = await _owner_number(call.restaurant_id)
                        if owner:
                            await websocket.send_json({"event": "forward", "to_number": owner})
                            break
                        logger.error("[%s] No owner_forward_number configured for restaurant %s", call_sid, call.restaurant_id)
                        state["step"] = "ordering"

        except WebSocketDisconnect:
            print(f"[{call_sid}] Call disconnected by user (order saved: {order_saved}).")
        except Exception:
            logger.exception("[%s] Stream error", call_sid)