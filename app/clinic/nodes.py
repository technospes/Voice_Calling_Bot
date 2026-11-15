"""Conversational state machine for Smile Dental Clinic appointment booking."""
import asyncio
import logging
import re
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from app.agent.nodes import (
    _get_llm,
    _fast_classify_confirmation,
    _clean_and_validate_name,
    _regex_extract_name,
)
from app.clinic.catalog import CLINIC_SERVICES, get_open_slots, get_slot, is_emergency, match_service

logger = logging.getLogger("clinic-nodes")

LLM_TIMEOUT_SEC = 8.0
MAX_RETRIES = 3

EMERGENCY_REPLY = (
    "यह आपातकालीन स्थिति लग रही है। कृपया तुरंत नज़दीकी अस्पताल जाएँ या क्लिनिक को सीधे कॉल करें। "
    "फिर भी अपॉइंटमेंट चाहिए तो बताइए।"
)

DEFAULT_ENTRY = {
    "item_id": "srv-checkup",
    "name": "General Dental Checkup",
    "quantity": 1,
    "unit_price": 300,
    "slot": None,      # spoken label, e.g. "कल सुबह 11:00 बजे"
    "slot_id": None,   # canonical id, "YYYY-MM-DDTHH:MM" in clinic timezone
}

# Maps common Devanagari confirmation words to Roman so mixed-script Deepgram output
# like "हां confirm कर दो" or "कर दो confirm" matches deterministically in 0ms.
_CONFIRM_DEVANAGARI_MAP = {
    "हाँ": "haan",
    "हां": "haan",
    "जी": "ji",
    "कन्फर्म": "confirm",
    "कंफर्म": "confirm",
    "कर": "kar",
    "करो": "karo",
    "दो": "do",
    "दीजिए": "dijiye",
    "पक्का": "pakka",
    "बिल्कुल": "bilkul",
    "ठीक": "theek",
    "है": "hai",
    "सही": "sahi",
    "डन": "done",
    "ओके": "ok",
    "यस": "yes",
    "नहीं": "nahi",
    "ना": "na",
    "मत": "mat",
    "कैंसिल": "cancel",
    "पूरा": "poora",
    "रुको": "ruko",
}

_MIXED_CONFIRM_REGEX = re.compile(
    r"\b(confirm|pakka|done|book\s+kar|kar\s+do|kar\s+dijiye)\b",
    re.IGNORECASE,
)


def classify_clinic_confirmation(spoken: str) -> Optional[str]:
    """Handles pure Hindi, pure English, and Deepgram mixed-script ('हां confirm कर दो')."""
    direct = _fast_classify_confirmation(spoken)
    if direct:
        return direct

    clean = re.sub(r"[^\w\s\u0900-\u097F]", " ", spoken or "").strip()
    if not clean:
        return None

    romanized_tokens = [_CONFIRM_DEVANAGARI_MAP.get(tok, tok.lower()) for tok in clean.split()]
    romanized = " ".join(romanized_tokens)

    mapped = _fast_classify_confirmation(romanized)
    if mapped:
        return mapped

    # Catch mixed patterns like "kar do confirm", "haan book kar do", "yes kar do"
    has_neg = any(w in romanized_tokens for w in ("nahi", "na", "no", "mat", "cancel", "ruko"))
    if not has_neg and _MIXED_CONFIRM_REGEX.search(romanized):
        return "confirm"

    return None


class ClinicTurnExtraction(BaseModel):
    intent: Literal["book_or_update", "ask_info", "confirm_slot", "cancel", "unclear"] = "book_or_update"
    service_query: Optional[str] = Field(default=None, description="Dental treatment or symptom mentioned (e.g. Checkup, Cleaning, Root Canal, Tooth Pain)")
    slot_id: Optional[str] = Field(default=None, description="EXACT id from AVAILABLE SLOTS matching the caller's preferred day/time. null if the caller gave none or asked for a time not in the list.")
    requested_slot_text: Optional[str] = Field(default=None, description="Caller's own words if they asked for a day/time that is NOT in AVAILABLE SLOTS")


class PatientDetailsExtraction(BaseModel):
    patient_name: Optional[str] = Field(default=None, description="Patient's personal name only. Null if not stated.")
    age_or_symptom: Optional[str] = Field(default=None, description="Patient's age or dental issue only (e.g. '24 years, tooth pain'). Null if caller only stated their name.")


def _out(history: List[dict], reply: str, cart: List[dict], step: str, status: str = "pending", retries: int = 0) -> dict:
    history = (history + [{"role": "agent", "content": reply}])[-12:]
    return {
        "cart": cart,
        "reply": reply,
        "step": step,
        "confirmation_status": status,
        "retries": retries,
        "conversation_history": history,
    }


def _retry_or_close(history: List[dict], reply: str, cart: List[dict], step: str, new_retries: int) -> dict:
    """Bounded retries: after MAX_RETRIES end politely WITHOUT saving (status=cancelled)."""
    if new_retries >= MAX_RETRIES:
        return _out(history, "माफ़ कीजिए, आवाज़ साफ़ नहीं आ रही। कृपया क्लिनिक को दोबारा कॉल करें।",
                    cart, "closing", "cancelled", 0)
    return _out(history, reply, cart, step, "pending", new_retries)


def _slots_speech(slots: List[dict], limit: int = 3) -> str:
    return " या ".join(s["label"] for s in slots[:limit])


def _readback(entry: dict) -> str:
    return f"{entry['name']}, समय: {entry['slot']}, परामर्श शुल्क {entry['unit_price']} रुपये। कन्फर्म कर दूँ?"


async def _extract(schema, prompt: str):
    extractor = _get_llm().with_structured_output(schema, method="json_schema")
    return await asyncio.wait_for(extractor.ainvoke(prompt), LLM_TIMEOUT_SEC)


async def _collect_details(state: dict, history: List[dict], cart: List[dict], spoken: str, retries: int) -> dict:
    quick_name = _regex_extract_name(spoken)
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = f"""Extract patient details from this dental clinic phone call.
CONVERSATION:
{convo}
CALLER SAID: "{spoken}"
Rules:
1. `patient_name`: Personal name only (e.g. 'Ayush'). Null if not stated.
2. `age_or_symptom`: Age in years or dental symptom/note (e.g. '22 years', 'daant mein dard'). Null if caller only gave their name.
Output JSON with `patient_name` and `age_or_symptom`."""
    try:
        ext: PatientDetailsExtraction = await _extract(PatientDetailsExtraction, prompt)
    except Exception as exc:
        logger.warning("Patient detail extraction failed: %s", type(exc).__name__)
        ext = PatientDetailsExtraction(
            patient_name=quick_name or (spoken if len(spoken.split()) <= 3 and not any(c.isdigit() for c in spoken) else None),
            age_or_symptom=spoken if any(c.isdigit() for c in spoken) else None,
        )

    extracted_name = (_clean_and_validate_name(ext.patient_name) if ext.patient_name else None) or quick_name
    extracted_note = (ext.age_or_symptom or "").strip() or None

    # Guard: Prevent LLM from copying a bare name into age_or_symptom
    if extracted_name and extracted_note and extracted_note.lower() == extracted_name.lower():
        extracted_note = None

    name = extracted_name or state.get("customer_name")
    note = extracted_note or state.get("delivery_address")

    if not name and 1 <= len(spoken.split()) <= 3 and not extracted_note and not any(c.isdigit() for c in spoken):
        name = _clean_and_validate_name(spoken)

    if name and not note:
        res = _out(history, f"धन्यवाद {name}! कृपया पेशेंट की उम्र या दांत की समस्या बता दीजिए।", cart, "collecting_details", "pending", 0)
        res["customer_name"] = name
        return res

    if note and not name:
        res = _out(history, "नोट कर लिया। कृपया पेशेंट का नाम भी बता दीजिए।", cart, "collecting_details", "pending", 0)
        res["delivery_address"] = note
        return res

    if not name and not note:
        return _retry_or_close(history, "कृपया पेशेंट का नाम और उम्र बताइए।", cart, "collecting_details", retries + 1)

    res = _out(history, "धन्यवाद! आपका अपॉइंटमेंट दर्ज किया जा रहा है।", cart, "closing", "pending", 0)
    res["customer_name"] = name
    res["delivery_address"] = note
    return res


async def process_clinic_turn(state: dict) -> dict:
    spoken = (state.get("latest_speech") or "").strip()
    cart = [dict(c) for c in (state.get("cart") or [])]
    history = list(state.get("conversation_history") or [])
    step = state.get("step", "ordering")
    retries = state.get("retries", 0)

    if not spoken:
        return _retry_or_close(history, "आवाज़ साफ़ नहीं आई। कृपया फिर से बताइए?", cart, step, retries + 1)

    history.append({"role": "user", "content": spoken})

    if is_emergency(spoken):
        return _out(history, EMERGENCY_REPLY, cart, step, "pending", retries)

    # 1. AWAITING CONFIRMATION
    changing = False
    declined = False
    if step == "awaiting_confirmation":
        label = classify_clinic_confirmation(spoken)
        if label == "confirm" and cart and cart[0].get("slot_id"):
            reply = "स्लॉट नोट हो गया! अपॉइंटमेंट बुक करने के लिए पेशेंट का नाम और उम्र बता दीजिए।"
            return _out(history, reply, cart, "collecting_details", "pending", 0)
        if label == "cancel":
            return _out(history, "ठीक है, अपॉइंटमेंट रिक्वेस्ट कैंसिल कर दी गई है।", cart, "closing", "cancelled", 0)
        changing, declined, step = True, label == "decline", "ordering"

    # 2. COLLECTING PATIENT DETAILS
    if step == "collecting_details":
        return await _collect_details(state, history, cart, spoken, retries)

    if step == "closing":
        return _out(history, "धन्यवाद!", cart, "closing", state.get("confirmation_status", "pending"), 0)

    # 3. SERVICE + SLOT SELECTION ('ordering')
    slots = get_open_slots(booked=state.get("booked_slots"))
    if not slots:
        return _out(history, "माफ़ कीजिए, अभी कोई स्लॉट खाली नहीं है। कृपया क्लिनिक को बाद में कॉल करें।",
                    cart, "closing", "cancelled", 0)

    entry = cart[0] if cart else dict(DEFAULT_ENTRY)
    prev_item_id = entry.get("item_id")
    prev_slot_id = entry.get("slot_id")

    if entry.get("slot_id") and not get_slot(entry["slot_id"], slots):
        entry["slot"] = entry["slot_id"] = None

    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    services_str = ", ".join(f"{s['name']} (₹{s['base_price']})" for s in CLINIC_SERVICES)
    slots_str = "\n".join(f"- {s['id']} = {s['label']}" for s in slots)

    prompt = f"""You are a dental clinic receptionist extractor.
AVAILABLE SERVICES: {services_str}
AVAILABLE SLOTS (id = spoken label):
{slots_str}
CURRENT BOOKING STATE: {entry}
RECENT CONVERSATION:
{convo}
CALLER SAID: "{spoken}"

Rules:
- If the caller is confirming/agreeing to the current booking (e.g. 'haan confirm kar do', 'kar do confirm', 'yes', 'book kar do'), set intent='confirm_slot'.
- Extract `service_query` ONLY if caller mentions a dental treatment or symptom in this turn.
- Set `slot_id` ONLY if the caller chooses a specific time slot from AVAILABLE SLOTS.
- If the caller asks for a day/time that is not in AVAILABLE SLOTS, leave `slot_id` null and put their words in `requested_slot_text`.
- If caller asks about fees or available timings, set intent='ask_info'. If they want to stop, intent='cancel'.
"""
    try:
        ext: ClinicTurnExtraction = await _extract(ClinicTurnExtraction, prompt)
    except Exception as exc:
        logger.warning("Clinic extraction error: %s", type(exc).__name__)
        return _retry_or_close(history, "माफ़ कीजिए, फिर से बताइए—आपको किस समय अपॉइंटमेंट चाहिए?",
                               cart, "awaiting_confirmation" if changing and entry.get("slot_id") else "ordering", retries + 1)

    if ext.intent == "cancel":
        return _out(history, "ठीक है, अपॉइंटमेंट कैंसिल कर दिया।", cart, "closing", "cancelled", 0)

    # Handle LLM-classified confirmation fallback when in awaiting_confirmation
    if ext.intent == "confirm_slot" and entry.get("slot_id"):
        cart = [entry]
        reply = "स्लॉट नोट हो गया! अपॉइंटमेंट बुक करने के लिए पेशेंट का नाम और उम्र बता दीजिए।"
        return _out(history, reply, cart, "collecting_details", "pending", 0)

    if ext.intent == "ask_info":
        fees = ", ".join(f"{s['name']} {s['base_price']} रुपये" for s in CLINIC_SERVICES)
        reply = f"हमारे यहाँ {fees}। उपलब्ध समय: {_slots_speech(slots)}। कौन सा बुक करूँ?"
        return _out(history, reply, cart, "ordering", "pending", 0)

    new_info = False

    if ext.service_query:
        matched = match_service(ext.service_query)
        if not matched:
            names = ", ".join(s["name"] for s in CLINIC_SERVICES)
            return _out(history, f"माफ़ कीजिए, यह सेवा उपलब्ध नहीं है। हमारे यहाँ {names} उपलब्ध हैं। कौन सी चाहिए?",
                        cart, "ordering", "pending", 0)
        if matched["id"] != prev_item_id or not changing:
            new_info = True
        entry.update(item_id=matched["id"], name=matched["name"], unit_price=matched["base_price"])

    slot = get_slot(ext.slot_id, slots) if ext.slot_id else None
    if slot:
        if slot["id"] != prev_slot_id or not changing:
            new_info = True
        entry.update(slot=slot["label"], slot_id=slot["id"])
    elif ext.requested_slot_text:
        return _out(history, f"माफ़ कीजिए, उस समय स्लॉट खाली नहीं है। उपलब्ध समय: {_slots_speech(slots)}। कौन सा चाहिए?",
                    cart, "ordering", "pending", 0)

    cart = [entry]

    if not entry.get("slot_id"):
        if changing and declined and not new_info:
            return _out(history, "ठीक है, आप कौन सा समय या इलाज बदलना चाहेंगे?", cart, "ordering", "pending", 0)
        reply = f"{entry['name']} नोट कर लिया। उपलब्ध समय: {_slots_speech(slots)}। कौन सा समय सही रहेगा?"
        if new_info:
            return _out(history, reply, cart, "ordering", "pending", 0)
        return _retry_or_close(history, reply, cart, "ordering", retries + 1)

    if changing and not new_info:
        if declined:
            return _out(history, "ठीक है, आप कौन सा समय या इलाज बदलना चाहेंगे?", cart, "ordering", "pending", 0)
        return _retry_or_close(history, "कृपया हाँ या ना में बताइए। " + _readback(entry),
                               cart, "awaiting_confirmation", retries + 1)

    return _out(history, _readback(entry), cart, "awaiting_confirmation", "pending", 0)