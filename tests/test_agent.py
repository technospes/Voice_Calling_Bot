import asyncio

import pytest
from pydantic import ValidationError

from app.agent import nodes
from app.agent.matching import resolve_item
from app.agent.nodes import (
    ExtractedItem,
    IntentExtraction,
    _handle_ordering,
    process_order_turn,
)
from app.agent.state import initial_order_state
from tests.conftest import BUTTER, GARLIC, LASSI, MENU, PBM, ROTI, TIKKA


def item(name, qty=1, remove_all=False):
    return ExtractedItem(item_name=name, quantity=qty, remove_all=remove_all)


def ext(intent, *items):
    return IntentExtraction(intent=intent, items=list(items))


class Bot:
    """Drives process_order_turn with a scripted (mocked) LLM."""

    def __init__(self, monkeypatch, menu=MENU):
        self.state = initial_order_state("call-1", "sid", "rest-1", "+911234567890")
        self.menu = menu
        self.prompts = []
        self.extractions = []
        self.labels = []
        self.mp = monkeypatch

        async def fake_extract(prompt):
            self.prompts.append(prompt)
            r = self.extractions.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        async def fake_classify(prompt):
            return self.labels.pop(0)

        monkeypatch.setattr(nodes, "extract_intent", fake_extract)
        monkeypatch.setattr(nodes, "classify_confirmation", fake_classify)

    def say(self, text, extraction=None, label=None):
        if extraction is not None:
            self.extractions.append(extraction)
        if label is not None:
            self.labels.append(label)
        out = asyncio.run(process_order_turn(self.state, text, self.menu))
        self.state.update(out)
        return out

    @property
    def cart(self):
        return {c["item_id"]: c["quantity"] for c in self.state["cart"]}


@pytest.fixture
def bot(monkeypatch):
    return Bot(monkeypatch)


# ---------------- schema validation ----------------
def test_negative_quantity_rejected():
    with pytest.raises(ValidationError):
        ExtractedItem(item_name="Lassi", quantity=-1)
    with pytest.raises(ValidationError):
        ExtractedItem(item_name="   ", quantity=1)
    with pytest.raises(ValidationError):
        IntentExtraction(intent="dance", items=[])


def test_malformed_extraction_is_handled_not_crashing(bot):
    out = bot.say("kuch bhi", ValueError("bad structured output"))
    assert out["cart"] == [] and out["retries"] == 1


# ---------------- matching ----------------
def test_matching_order():
    assert resolve_item("garlic naan", MENU).item["id"] == GARLIC          # exact
    assert resolve_item("PBM", MENU).item["id"] == PBM                       # alias
    assert resolve_item("लहसुन नान", MENU).item["id"] == GARLIC             # Devanagari alias
    assert resolve_item("tikka", MENU).item["id"] == TIKKA                   # unique token subset
    r = resolve_item("naan", MENU)
    assert r.status == "ambiguous" and {c["id"] for c in r.candidates} == {GARLIC, BUTTER}
    assert resolve_item("galyak naan", MENU).item["id"] == GARLIC            # unique fuzzy
    assert resolve_item("sushi", MENU).status == "none"
    assert resolve_item("", MENU).status == "none"


# ---------------- add ----------------
def test_add_multiple_items(bot):
    out = bot.say("Ek paneer butter masala aur do garlic naan",
                  ext("add", item("Paneer Butter Masala", 1), item("Garlic Naan", 2)))
    assert bot.cart == {PBM: 1, GARLIC: 2}
    assert out["conversation_history"][-1]["role"] == "agent"


def test_add_three_items(bot):
    bot.say("x", ext("add", item("Paneer Tikka"), item("Tandoori Roti", 2), item("Lassi")))
    assert bot.cart == {TIKKA: 1, ROTI: 2, LASSI: 1}


def test_repeated_and_duplicate_adds_aggregate(bot):
    bot.say("a", ext("add", item("Lassi"), item("lassi", 2)))   # same item twice in one sentence
    assert bot.cart == {LASSI: 3}
    bot.say("b", ext("add", item("Lassi")))
    assert bot.cart == {LASSI: 4} and len(bot.state["cart"]) == 1


def test_ambiguous_naan_asks_which_and_changes_nothing(bot):
    out = bot.say("naan add kar do", ext("add", item("naan")))
    assert bot.cart == {}
    # Garlic Naan has a Devanagari alias in the test menu, so it is spoken in Hindi
    assert "लहसुन नान" in out["reply"] and "Butter Naan" in out["reply"]


def test_unavailable_item_allows_partial_fulfillment(bot):
    """If one item is valid and another is unavailable, the valid one should be added."""
    out = bot.say("paneer tikka aur sushi", ext("add", item("Paneer Tikka"), item("Sushi")))
    
    # Cart should now contain exactly 1 Paneer Tikka (partial fulfillment)
    assert len(bot.cart) == 1
    assert list(bot.cart.values())[0] == 1
    
    # Reply should confirm the Paneer Tikka and reject the Sushi
    assert "Paneer Tikka जोड़ दिया" in out["reply"]
    assert "Sushi नहीं मिला" in out["reply"]


def test_add_with_zero_quantity_is_not_applied(bot):
    bot.say("x", ext("add", item("Lassi", 0)))
    assert bot.cart == {}


def test_add_without_identifiable_item_clarifies(bot):
    out = bot.say("ek aur add kar do", ext("add"))
    assert bot.cart == {} and "नाम" in out["reply"]


# ---------------- context ----------------
def test_context_add_one_more_then_set_absolute(bot):
    bot.say("Ek paneer tikka", ext("add", item("Paneer Tikka")))
    bot.say("Ek aur add kar do", ext("add", item("Paneer Tikka")))
    assert bot.cart == {TIKKA: 2}
    bot.say("Ab teen kar do", ext("change_qty", item("Paneer Tikka", 3)))
    assert bot.cart == {TIKKA: 3}          # set to 3, not 5
    # history + cart are given to the LLM
    last_prompt = bot.prompts[-1]
    assert "Ek aur add kar do" in last_prompt and "Paneer Tikka" in last_prompt and "3" in last_prompt


def test_user_message_not_duplicated_and_history_trimmed(bot):
    for _ in range(12):
        bot.say("lassi", ext("add", item("Lassi")))
    hist = bot.state["conversation_history"]
    assert len(hist) <= nodes.MAX_HISTORY
    users = [m for m in hist if m["role"] == "user"]
    agents = [m for m in hist if m["role"] == "agent"]
    assert abs(len(users) - len(agents)) <= 1       # exactly one user msg per turn


# ---------------- change / remove ----------------
def test_change_qty_absolute_and_zero_removes(bot):
    bot.say("a", ext("add", item("Garlic Naan", 2)))
    bot.say("do nahi, ek kar do", ext("change_qty", item("Garlic Naan", 1)))
    assert bot.cart == {GARLIC: 1}
    bot.say("zero", ext("change_qty", item("Garlic Naan", 0)))
    assert bot.cart == {}


def test_change_qty_item_not_in_cart(bot):
    bot.say("a", ext("add", item("Lassi")))
    out = bot.say("b", ext("change_qty", item("Paneer Tikka", 3)))
    assert bot.cart == {LASSI: 1} and "नहीं मिला" in out["reply"]


def test_remove_reduces_by_amount_using_cart_context(bot):
    bot.say("a", ext("add", item("Garlic Naan", 3), item("Butter Naan", 0 + 1)))
    # "naan" is ambiguous on the menu AND in the cart -> clarify
    out = bot.say("ek naan kam", ext("remove", item("naan", 1)))
    assert bot.cart == {GARLIC: 3, BUTTER: 1} and "लहसुन नान" in out["reply"]
    bot.say("ek garlic naan kam kar do", ext("remove", item("Garlic Naan", 1)))
    assert bot.cart == {GARLIC: 2, BUTTER: 1}


def test_remove_naan_resolves_against_cart_when_unique(bot):
    bot.say("a", ext("add", item("Garlic Naan", 2)))
    bot.say("ek naan kam kar do", ext("remove", item("naan", 1)))   # only one naan in the cart
    assert bot.cart == {GARLIC: 1}


def test_remove_entire_item_and_over_removal(bot):
    bot.say("a", ext("add", item("Garlic Naan", 2), item("Lassi", 1)))
    bot.say("garlic naan hata do", ext("remove", item("Garlic Naan", 1, remove_all=True)))
    assert bot.cart == {LASSI: 1}
    bot.say("paanch lassi kam", ext("remove", item("Lassi", 5)))
    assert bot.cart == {}                                           # never negative


def test_remove_from_empty_cart(bot):
    out = bot.say("hata do", ext("remove", item("Lassi", 1, remove_all=True)))
    assert "खाली" in out["reply"]


# ---------------- menu / repeat / total ----------------
def test_ask_menu_lists_real_items_only(bot):
    out = bot.say("menu mein kya hai", ext("ask_menu"))
    for m in MENU:
        assert (m["aliases"][-1] if m["aliases"] and any("ऀ" <= ch <= "ॿ" for ch in m["aliases"][-1]) else m["name"]) in out["reply"]
    assert "Sushi" not in out["reply"]


def test_ask_menu_empty_menu(monkeypatch):
    b = Bot(monkeypatch, menu=[])
    assert "उपलब्ध नहीं" in b.say("menu", ext("ask_menu"))["reply"]


def test_repeat_order_reads_back_and_does_not_confirm(bot):
    bot.say("a", ext("add", item("Paneer Butter Masala", 1), item("Garlic Naan", 2)))
    out = bot.say("mera order repeat karo", ext("repeat_order"))
    assert out["step"] == "ordering" and out["confirmation_status"] == "pending"
    assert "350" in out["reply"]            # 250 + 2*50 subtotal


def test_total_query_gives_accurate_subtotal(bot):
    bot.say("a", ext("add", item("Lassi", 3)))
    assert "180" in bot.say("total bata do", ext("repeat_order"))["reply"]


# ---------------- checkout / confirmation ----------------
def test_empty_cart_checkout_rejected(bot):
    out = bot.say("bas itna hi", ext("checkout"))
    assert out["step"] == "ordering" and "खाली" in out["reply"]


def test_checkout_then_confirm(bot):
    bot.say("a", ext("add", item("Lassi", 2)))
    out = bot.say("bas order confirm kar do", ext("checkout"))
    assert out["step"] == "awaiting_confirmation" and out["confirmation_status"] == "pending"
    assert "120" in out["reply"]
    out = bot.say("Haan confirm kar do")           # exact phrase -> no LLM call needed
    assert out["step"] == "closing" and out["confirmation_status"] == "confirmed"


def test_cancel_during_confirmation(bot):
    bot.say("a", ext("add", item("Lassi")))
    bot.say("done", ext("checkout"))
    out = bot.say("Nahi cancel kar do")
    assert out["step"] == "closing" and out["confirmation_status"] == "cancelled"


def test_unclear_confirmation_asks_again_and_never_confirms(bot):
    bot.say("a", ext("add", item("Lassi")))
    bot.say("done", ext("checkout"))
    out = bot.say("hmm pata nahi", label="unclear")
    assert out["step"] == "awaiting_confirmation" and out["confirmation_status"] == "pending"


def test_llm_label_must_be_exact_allowlist_value(bot):
    bot.say("a", ext("add", item("Lassi")))
    bot.say("done", ext("checkout"))
    # substring-style answers must NOT count as confirm
    out = bot.say("achha dekhte hain", label="I do not confirm this")
    assert out["confirmation_status"] == "pending" and out["step"] == "awaiting_confirmation"
    out = bot.say("achha dekhte hain", label="Confirm.")   # exact label after normalization -> ok
    assert out["confirmation_status"] == "confirmed"


def test_nahi_plus_modification_is_not_discarded(bot):
    bot.say("a", ext("add", item("Garlic Naan", 2), item("Lassi")))
    bot.say("done", ext("checkout"))
    out = bot.say("Nahi, garlic naan hata do", ext("remove", item("Garlic Naan", 1, remove_all=True)), label="modify")
    assert bot.cart == {LASSI: 1}
    assert out["step"] == "ordering" and out["confirmation_status"] == "pending"


def test_modify_without_details_returns_to_ordering(bot):
    bot.say("a", ext("add", item("Lassi")))
    bot.say("done", ext("checkout"))
    out = bot.say("thoda change karna hai", ext("unclear"), label="modify")
    assert out["step"] == "ordering" and bot.cart == {LASSI: 1}


def test_cancel_in_ordering_and_forward(bot):
    out = bot.say("cancel", ext("cancel"))
    assert out["step"] == "closing" and out["confirmation_status"] == "cancelled"
    b2 = Bot(bot.mp)
    out = b2.say("owner se baat karni hai", ext("forward_to_owner"))
    assert out["step"] == "forward" and "ओनर" in out["reply"]
    assert "आपको जोड़" not in out["reply"]


def test_every_branch_returns_history(bot):
    for e in [ext("ask_menu"), ext("repeat_order"), ext("checkout"), ext("unclear"), ext("cancel")]:
        out = bot.say("x", e)
        assert out["conversation_history"][-1]["content"] == out["reply"]
        bot.state["step"] = "ordering"


# ---------------- awaiting_confirmation: the four safety cases ----------------
def _at_confirmation(bot):
    bot.say("a", ext("add", item("Garlic Naan", 2), item("Lassi")))
    bot.say("done", ext("checkout"))
    assert bot.state["step"] == "awaiting_confirmation"
    return list(bot.state["cart"]), len(bot.state["conversation_history"])


def test_confirm_case_haan_confirm_kar_do(bot):
    _at_confirmation(bot)
    out = bot.say("Haan confirm kar do")
    assert out["step"] == "closing" and out["confirmation_status"] == "confirmed"


def test_bare_nahi_does_not_cancel_and_asks_modify_or_cancel(bot):
    cart_before, hist_before = _at_confirmation(bot)
    for word in ["Nahi", "nahi.", "नहीं"]:
        out = bot.say(word)                      # deterministic: no LLM label/extraction queued
        assert out["step"] == "awaiting_confirmation"
        assert out["confirmation_status"] == "pending"
        assert "कैंसिल" in out["reply"] and "बदल" in out["reply"]
        assert out["cart"] == cart_before        # cart preserved
    hist = bot.state["conversation_history"]
    assert len(hist) == hist_before + 6          # user+agent per attempt; earlier history kept
    assert hist[hist_before]["content"] == "Nahi"


def test_nahi_with_modification_returns_to_ordering_and_applies_it(bot):
    _at_confirmation(bot)
    out = bot.say("Nahi, garlic naan hata do",
                  ext("remove", item("Garlic Naan", 1, remove_all=True)), label="modify")
    assert out["step"] == "ordering" and out["confirmation_status"] == "pending"
    assert bot.cart == {LASSI: 1}


@pytest.mark.parametrize("phrase", ["Poora order cancel kar do", "Cancel kar do"])
def test_explicit_cancel_cancels_pending_order(bot, phrase):
    _at_confirmation(bot)
    out = bot.say(phrase)
    assert out["step"] == "closing" and out["confirmation_status"] == "cancelled"


def test_declined_then_unclear_preserve_cart_and_history(bot):
    cart_before, _ = _at_confirmation(bot)
    bot.say("nahi")
    out = bot.say("hmm", label="unclear")
    assert out["step"] == "awaiting_confirmation" and out["confirmation_status"] == "pending"
    assert out["cart"] == cart_before
    roles = [m["role"] for m in out["conversation_history"]]
    assert roles.count("user") >= 4 and roles.count("agent") >= 4


# ---------------- Groq tool_use_failed recovery ----------------
class _FakeGroqError(Exception):
    def __init__(self, body):
        super().__init__("400")
        self.body = body


def test_recover_failed_generation_from_groq_400():
    raw = '{\n  "intent": "add",\n  "items": [\n    {"item_name": "Paneer Butter Masala", "quantity": 1}\n  ]\n}'
    for body in ({"error": {"failed_generation": raw}}, {"failed_generation": raw}):
        rec = nodes._recover_failed_generation(_FakeGroqError(body))
        assert rec.intent == "add" and rec.items[0].item_name == "Paneer Butter Masala"


def test_recover_failed_generation_rejects_garbage():
    assert nodes._recover_failed_generation(_FakeGroqError({"error": {"failed_generation": "no json"}})) is None
    assert nodes._recover_failed_generation(_FakeGroqError({"error": {"failed_generation": '{"intent": "dance"}'}})) is None
    assert nodes._recover_failed_generation(ValueError("x")) is None


def test_extract_intent_uses_recovery_instead_of_failing(monkeypatch):
    raw = '{"intent": "checkout", "items": []}'

    class Boom:
        async def ainvoke(self, prompt):
            raise _FakeGroqError({"error": {"failed_generation": raw}})

    monkeypatch.setattr(nodes, "_get_structured_llm", lambda: Boom())
    assert asyncio.run(nodes.extract_intent("p")).intent == "checkout"


# ---------------- concise replies keep summary + explicit confirmation ----------------
def test_concise_checkout_reply_has_items_total_and_confirm_question(bot):
    bot.say("a", ext("add", item("Butter Naan", 1)))
    out = bot.say("bas", ext("checkout"))
    assert out["reply"] == "1 Butter Naan, 40 रुपये। कन्फर्म कर दूँ?"
    assert out["step"] == "awaiting_confirmation" and out["confirmation_status"] == "pending"


# ---------------- compound intent safety (new) ----------------
@pytest.mark.asyncio
async def test_compound_add_and_checkout(mocker):
    """Additions in a checkout intent must land in the cart BEFORE asking for confirmation."""
    mocker.patch("app.agent.nodes.extract_intent", return_value=ext(
        "checkout",
        item("Butter Naan", 2, remove_all=False),
    ))

    result = await _handle_ordering(
        "add 2 butter naan and confirm",
        cart=[],
        menu=MENU,
        history=[],
        retries=0,
    )

    assert result["step"] == "awaiting_confirmation"
    assert result["confirmation_status"] == "pending"
    assert len(result["cart"]) == 1
    assert result["cart"][0]["quantity"] == 2
    assert "80 रुपये" in result["reply"]


@pytest.mark.asyncio
async def test_removal_does_not_add_to_nonempty_cart_during_checkout(mocker):
    """A remove_all item mistakenly packed into a checkout must NOT be added."""
    mocker.patch("app.agent.nodes.extract_intent", return_value=ext(
        "checkout",
        item("Butter Naan", 1, remove_all=True),
    ))

    cart = [{"item_id": BUTTER, "name": "Butter Naan", "quantity": 1, "unit_price": 40}]
    result = await _handle_ordering(
        "remove naan and checkout",
        cart=cart,
        menu=MENU,
        history=[],
        retries=0,
    )

    # The remove_all item is filtered out of the checkout path, so the cart is
    # unchanged and the bot proceeds to ask for confirmation of what's already there.
    assert len(result["cart"]) == 1                     # cart unchanged
    assert result["cart"][0]["quantity"] == 1           # nothing added, nothing removed
    assert result["step"] == "awaiting_confirmation"    # current behavior
    assert "कार्ट खाली है" not in result["reply"]       # not the empty-cart branch


@pytest.mark.asyncio
async def test_removal_with_checkout_on_empty_cart_returns_to_ordering(mocker):
    """A remove_all item in checkout against an empty cart falls back to the empty-cart reply."""
    mocker.patch("app.agent.nodes.extract_intent", return_value=ext(
        "checkout",
        item("Butter Naan", 1, remove_all=True),
    ))

    result = await _handle_ordering(
        "remove naan and checkout",
        cart=[],
        menu=MENU,
        history=[],
        retries=0,
    )

    assert result["step"] == "ordering"
    assert result["cart"] == []
    assert "कार्ट खाली है" in result["reply"]