"""Unit tests for the semantic relevance gate. No network, no API keys: the LLM is a stub."""
import asyncio
import copy

import pytest

from app.agent import relevance_gate as rg
from app.agent.relevance_gate import GateResult, LLMGateOutput, evaluate, run_gated_turn


def run(coro):
    return asyncio.run(coro)


MENU = [
    {"id": "1", "name": "Paneer Tikka", "aliases": ["पनीर टिक्का", "paneer tikka"], "base_price": 220.0, "is_available": True},
    {"id": "2", "name": "Butter Naan", "aliases": ["बटर नान"], "base_price": 50.0, "is_available": True},
    {"id": "3", "name": "Garlic Naan", "aliases": [], "base_price": 60.0, "is_available": True},
    {"id": "4", "name": "Cold Coffee", "aliases": ["कोल्ड कॉफी"], "base_price": 120.0, "is_available": True},
    {"id": "5", "name": "Butter Chicken", "aliases": [], "base_price": 320.0, "is_available": True},
    {"id": "6", "name": "Roti", "aliases": ["रोटी"], "base_price": 15.0, "is_available": True},
    {"id": "7", "name": "Coke", "aliases": [], "base_price": 40.0, "is_available": True},
]
CART = [{"item_id": "5", "name": "Butter Chicken", "quantity": 1, "unit_price": 320.0}]


def state(step="ordering", cart=None):
    return {"step": step, "cart": copy.deepcopy(CART if cart is None else cart),
            "conversation_history": [], "retries": 0, "confirmation_status": "pending"}


def stub(category, cleaned=None, conf=0.9, reason="t"):
    calls = []

    async def _call(prompt):
        calls.append(prompt)
        return LLMGateOutput(category=category, cleaned_text=cleaned, confidence=conf, reason=reason)

    _call.calls = calls
    return _call


async def must_not_call(prompt):
    raise AssertionError("LLM must not be called on this path")


def ev(text, st=None, llm=None, menu=MENU):
    return run(evaluate(text, st or state(), menu, llm or must_not_call))


# 1-2 ----------------------------------------------------------------------
def test_clean_single_item_fast_path():
    r = ev("Ek paneer tikka dena")
    assert r.is_relevant and r.stage == "fast" and r.cleaned_text == "Ek paneer tikka dena"


def test_multiple_items_and_quantities_fast_path():
    r = ev("Ek cold coffee aur ek naan")
    assert r.is_relevant and r.stage == "fast"
    assert ev("Do butter naan").is_relevant


# 3-4 ----------------------------------------------------------------------
def test_pure_office_conversation():
    r = ev("Kal office jaana hai subah jaldi", llm=stub("UNRELATED"))
    assert r.category == "UNRELATED" and not r.is_relevant and r.cleaned_text is None


def test_pure_weather_conversation():
    r = ev("Aaj bahut garmi hai baarish nahi hui", llm=stub("UNRELATED"))
    assert not r.is_relevant


# 5-6 ----------------------------------------------------------------------
def test_mixed_food_and_unrelated_goes_to_llm_and_is_trimmed():
    text = "Ek butter chicken dena aur do roti... haan kal office chalenge... aur ek coke"
    llm = stub("ORDER_ACTION", "Ek butter chicken dena aur do roti aur ek coke")
    r = ev(text, llm=llm)
    assert len(llm.calls) == 1, "mixed speech must not take the fast path"
    assert r.is_relevant and "office" not in r.cleaned_text


def test_menu_word_plus_chatter_is_not_fast_accepted():
    llm = stub("UNRELATED")
    r = ev("Ek naan kal office mein bhi milta hai", llm=llm)
    assert len(llm.calls) == 1 and not r.is_relevant


def test_mixed_address_and_unrelated():
    llm = stub("ORDER_DETAIL", "Delivery address Knowledge Park 2 hai")
    r = ev("Delivery address Knowledge Park 2 hai, kal office jaana hai", state("collecting_details"), llm)
    assert r.is_relevant and "office" not in r.cleaned_text


# 7-8 ----------------------------------------------------------------------
def test_ek_aur_with_cart_context():
    r = ev("Ek aur kar do", llm=stub("ORDER_ACTION", "Ek aur kar do"))
    assert r.is_relevant


def test_ek_aur_without_cart_is_ambiguous():
    r = ev("Ek aur kar do", state(cart=[]), stub("ORDER_ACTION", "Ek aur kar do"))
    assert r.category == "AMBIGUOUS" and not r.is_relevant


def test_contextual_removal():
    assert ev("woh hata do", llm=stub("ORDER_ACTION", "woh hata do")).is_relevant
    assert not ev("woh hata do", state(cart=[]), stub("ORDER_ACTION", "woh hata do")).is_relevant


# 9-11 ---------------------------------------------------------------------
def test_haan_during_ordering_never_confirms():
    # Even if the model claims CONFIRMATION, ordering step must reject it.
    r = ev("Haan", llm=stub("CONFIRMATION", "Haan", 0.95))
    assert not r.is_relevant and r.category == "AMBIGUOUS"
    assert not ev("Haan", llm=stub("AMBIGUOUS")).is_relevant


def test_haan_during_awaiting_confirmation_fast_accept():
    r = ev("Haan", state("awaiting_confirmation"))
    assert r.is_relevant and r.category == "CONFIRMATION" and r.stage == "fast"


def test_negative_confirmation():
    r = ev("Nahi", state("awaiting_confirmation"))
    assert r.is_relevant and r.category == "CONFIRMATION"


def test_haan_during_collecting_details_is_not_passed():
    assert not ev("Haan", state("collecting_details"), stub("CONFIRMATION", "Haan")).is_relevant


# 12 -----------------------------------------------------------------------
@pytest.mark.parametrize("text", ["", "   ", None, "...", "।"])
def test_empty_transcripts(text):
    r = run(evaluate(text, state(), MENU, must_not_call))
    assert not r.is_relevant and r.cleaned_text is None


# 13-15 --------------------------------------------------------------------
def _assert_failsafe(r):
    assert r.category == "AMBIGUOUS" and r.cleaned_text is None and r.confidence == 0.0 and not r.is_relevant


def test_invalid_structured_output():
    async def bad(prompt):
        return {"category": "BOGUS", "confidence": 5}
    _assert_failsafe(ev("kuch bhi bol raha hai", llm=bad))

    async def junk(prompt):
        return "not json at all"
    _assert_failsafe(ev("kuch bhi bol raha hai", llm=junk))


def test_groq_timeout(monkeypatch):
    monkeypatch.setattr(rg, "GATE_LLM_TIMEOUT_SEC", 0.05)

    async def slow(prompt):
        await asyncio.sleep(1)
    _assert_failsafe(ev("kuch bhi bol raha hai", llm=slow))


def test_groq_provider_failure():
    async def boom(prompt):
        raise RuntimeError("503 from provider key=gsk_SECRET")
    r = ev("kuch bhi bol raha hai", llm=boom)
    _assert_failsafe(r)
    assert "gsk_SECRET" not in r.reason


# 16-17 --------------------------------------------------------------------
def test_hallucinated_menu_item_rejected():
    llm = stub("ORDER_ACTION", "ek butter chicken aur do naan")
    r = ev("ek butter chicken dena kal office jaana hai", llm=llm)
    assert r.category == "AMBIGUOUS" and not r.is_relevant


def test_hallucinated_quantity_rejected():
    llm = stub("ORDER_ACTION", "teen butter chicken")
    r = ev("ek butter chicken dena kal office jaana hai", llm=llm)
    assert not r.is_relevant


def test_low_confidence_rejected():
    assert not ev("kuch order karna hai shayad", llm=stub("ORDER_ACTION", "order karna hai", 0.4)).is_relevant


# 18-19 --------------------------------------------------------------------
def test_hindi_transcripts():
    r = ev("एक पनीर टिक्का देना")
    assert r.is_relevant and r.stage == "fast"
    llm = stub("ORDER_ACTION", "दो रोटी और एक कोल्ड कॉफी")
    r = ev("दो रोटी और एक कोल्ड कॉफी कल ऑफिस जाना है", llm=llm)
    assert r.is_relevant and "ऑफिस" not in r.cleaned_text


def test_hinglish_transcripts_and_stt_spelling_drift():
    assert ev("Mujhe do roti aur ek coke chahiye").is_relevant
    # Cleaned text may differ slightly from STT spelling; digits vs words are equivalent.
    assert rg.is_grounded("2 roti", "do roti kal office")
    assert rg.is_grounded("paneer tikkaa", "ek paneer tikka dena")   # minor spelling drift ok
    assert not rg.is_grounded("panir tikka", "ek paneer tikka dena")  # larger drift is rejected (safe side)
    assert not rg.is_grounded("3 roti", "do roti kal office")


def test_is_relevant_is_explicit():
    for cat in ("UNRELATED", "AMBIGUOUS"):
        assert not GateResult(cat, "ek naan", 0.99, "").is_relevant
    assert not GateResult("ORDER_ACTION", "  ", 0.99, "").is_relevant
    assert not GateResult("ORDER_ACTION", "ek naan", 0.1, "").is_relevant
    assert GateResult("ORDER_QUESTION", "menu kya hai", 0.9, "").is_relevant


# 20-22: worker-level behaviour via run_gated_turn ---------------------------
class Harness:
    def __init__(self):
        self.refreshes = 0
        self.agent_calls = 0
        self.inflight = 0
        self.max_inflight = 0

    async def on_relevant(self):
        self.refreshes += 1

    async def agent(self, st):
        self.agent_calls += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.01)
        self.inflight -= 1
        return {"cart": st["cart"] + [{"item_id": "6", "name": "Roti", "quantity": 1, "unit_price": 15.0}],
                "reply": "ok", "step": st["step"], "retries": 0}


def test_unrelated_speech_does_not_refresh_watchdog_or_touch_state():
    h, st = Harness(), state()
    before = copy.deepcopy(st)
    out = run(run_gated_turn("Kal office jaana hai", st, MENU, h.agent, h.on_relevant, stub("UNRELATED")))
    assert out.new_state is None and h.refreshes == 0 and h.agent_calls == 0
    assert st == before  # cart, retries, latest_speech untouched
    out = run(run_gated_turn("Haan", st, MENU, h.agent, h.on_relevant, stub("AMBIGUOUS")))
    assert out.new_state is None and h.agent_calls == 0 and st == before


def test_relevant_speech_refreshes_and_reaches_agent_with_cleaned_text_only():
    h, st = Harness(), state()
    llm = stub("ORDER_ACTION", "ek roti")
    out = run(run_gated_turn("ek roti haan kal office", st, MENU, h.agent, h.on_relevant, llm))
    assert out.new_state is not None and h.refreshes == 1 and h.agent_calls == 1
    assert st["latest_speech"] == "ek roti"


def test_closing_state_blocks_everything():
    h, st = Harness(), state("closing")
    out = run(run_gated_turn("Do butter naan", st, MENU, h.agent, h.on_relevant, must_not_call))
    assert out.new_state is None and h.agent_calls == 0 and h.refreshes == 0
    assert st["step"] == "closing"


def test_concurrent_transcripts_are_serialized_under_turn_lock():
    async def scenario():
        h, st, lock = Harness(), state(cart=[]), asyncio.Lock()
        llm = stub("UNRELATED")

        async def one(text):
            async with lock:  # same pattern as the worker's turn_lock
                return await run_gated_turn(text, st, MENU, h.agent, h.on_relevant, llm)

        outs = await asyncio.gather(
            one("Ek roti dena"), one("Kal office jaana hai"), one("Do coke"), one("Mausam accha hai"))
        return h, st, outs

    h, st, outs = run(scenario())
    assert h.max_inflight == 1
    assert h.agent_calls == 2 and h.refreshes == 2
    assert [o.new_state is not None for o in outs] == [True, False, True, False]
