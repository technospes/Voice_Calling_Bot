"""Semantic relevance gate for the Dental Clinic mode."""
import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Optional

from app.agent.relevance_gate import (
    GateResult,
    LLMGateOutput,
    TurnOutcome,
    _ambiguous,
    _coerce,
    _default_llm_call,
    _failure,
    _reject,
    _tokens,
    is_grounded,
    GATE_LLM_TIMEOUT_SEC,
    MIN_CONFIDENCE,
)
from app.clinic.catalog import is_emergency
from app.clinic.nodes import classify_clinic_confirmation

logger = logging.getLogger("clinic-gate")

AGENT_TURN_TIMEOUT_SEC = float(os.getenv("CLINIC_AGENT_TIMEOUT_SEC", "20"))
AGENT_ERROR_REPLY = "माफ़ कीजिए, कुछ तकनीकी दिक्कत आई। कृपया दोबारा बताइए।"

_FAST_CLINIC_WORDS = {
    "appointment", "booking", "book", "doctor", "dentist", "daant", "dard", "pain",
    "checkup", "cleaning", "safai", "canal", "rct", "kal", "aaj", "subah",
    "shaam", "baje", "slot", "fee", "fees", "kitni", "kitna",
    "अपॉइंटमेंट", "बुकिंग", "डॉक्टर", "दांत", "दर्द", "सफाई", "चेकअप", "कल", "आज",
    "सुबह", "शाम", "बजे", "स्लॉट", "फीस",
}


def _build_clinic_gate_prompt(transcript: str, state: dict) -> str:
    step = state.get("step", "ordering")
    history = (state.get("conversation_history") or [])[-6:]
    convo = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in history) or "(start of call)"

    return f"""You filter speech heard on a dental clinic appointment booking phone line (Hindi / Hinglish / English).
Drop unrelated background TV/family chatter.

CURRENT STEP: {step}
  - ordering: Selecting dental treatment/issue and preferred date/time slot.
  - awaiting_confirmation: Bot read back treatment, time slot, and fee, and asked to confirm.
  - collecting_details: Bot asked for the patient's name and age/issue.

RECENT CONVERSATION:
{convo}

TRANSCRIPT: "{transcript}"

CATEGORY RULES:
1. ORDER_ACTION:
   - Mentioning dental issues ("daant mein dard hai"), treatments ("cleaning", "checkup", "root canal"), preferred dates/times ("kal shaam 5 baje", "aaj"), or asking to book/confirm/cancel.
2. ORDER_DETAIL:
   - Providing patient name, age, phone number, or symptom description (especially in 'collecting_details').
3. ORDER_QUESTION:
   - Asking about clinic timings, doctor availability, consultation fees, or treatments offered.
4. CONFIRMATION:
   - Answering the confirmation question ("haan confirm kar do", "yes", "nahi") during 'awaiting_confirmation'.
5. UNRELATED:
   - Background speech completely unrelated to a clinic visit.
6. AMBIGUOUS:
   - Unclear fragments with no context.

CLEANED_TEXT RULES:
- For ORDER_ACTION, ORDER_DETAIL, ORDER_QUESTION, CONFIRMATION: You MUST output the verbatim relevant words in `cleaned_text`. Never leave `cleaned_text` empty for relevant categories.
"""


async def evaluate_clinic(
    transcript: Optional[str],
    state: dict,
    llm_call: Optional[Callable[[str], Awaitable[Any]]] = None,
) -> GateResult:
    text = (transcript or "").strip()
    toks = _tokens(text)
    if not text or not toks:
        return _reject("UNRELATED", "empty_transcript")

    step = state.get("step", "ordering")
    if step == "closing":
        return _ambiguous("closing_state")

    if is_emergency(text):
        return GateResult("ORDER_ACTION", text, 0.97, "fast:emergency", "fast")

    if step == "awaiting_confirmation":
        label = classify_clinic_confirmation(text)
        if label:
            return GateResult("CONFIRMATION", text, 0.96, f"fast:{label}", "fast")

    if any(t in _FAST_CLINIC_WORDS for t in toks):
        return GateResult("ORDER_ACTION", text, 0.96, "fast:clinic_keyword", "fast")

    if step == "collecting_details" and 1 <= len(toks) <= 8:
        return GateResult("ORDER_DETAIL", text, 0.95, "fast:patient_detail", "fast")

    call = llm_call or _default_llm_call
    try:
        raw = await asyncio.wait_for(call(_build_clinic_gate_prompt(text, state)), GATE_LLM_TIMEOUT_SEC)
        out: LLMGateOutput = _coerce(raw)
    except Exception as exc:
        logger.warning("Clinic gate LLM error: %s", type(exc).__name__)
        return _failure("llm_error")

    cleaned = (out.cleaned_text or "").strip()
    reason = (out.reason or "")[:120]
    if out.category in ("UNRELATED", "AMBIGUOUS") or not cleaned or out.confidence < MIN_CONFIDENCE:
        return _ambiguous(reason or "filtered", "llm")

    if not is_grounded(cleaned, text):
        return GateResult(out.category, text, out.confidence, "fallback_full_transcript", "llm")

    return GateResult(out.category, cleaned, out.confidence, reason, "llm")


async def run_clinic_gated_turn(
    user_text: str,
    state: dict,
    invoke_agent: Callable[[dict], Awaitable[dict]],
    on_relevant: Optional[Callable[[], Awaitable[None]]] = None,
) -> TurnOutcome:
    gate = await evaluate_clinic(user_text, state)
    if state.get("step") == "closing" or not gate.is_relevant:
        return TurnOutcome(gate, None)
    if on_relevant is not None:
        await on_relevant()
    state["latest_speech"] = gate.cleaned_text
    try:
        new_state = await asyncio.wait_for(invoke_agent(state), AGENT_TURN_TIMEOUT_SEC)
    except Exception:
        logger.exception("Clinic agent turn failed")
        new_state = {**state, "reply": AGENT_ERROR_REPLY}
    return TurnOutcome(gate, new_state)