import pytest
from app.agent.relevance_gate import evaluate, is_grounded
from app.agent.nodes import process_order_turn

MENU_FIXTURE = [
    {"id": "1", "name": "Paneer Butter Masala", "aliases": ["paneer butter"], "base_price": 280.0, "is_available": True},
    {"id": "2", "name": "Butter Naan", "aliases": ["naan"], "base_price": 40.0, "is_available": True},
    {"id": "3", "name": "Cold Coffee", "aliases": ["coffee"], "base_price": 120.0, "is_available": True},
    {"id": "4", "name": "Paneer Tikka", "aliases": ["tikka"], "base_price": 240.0, "is_available": True},
]


# =========================================================================== #
# Lexical Grounding Unit Tests
# =========================================================================== #
def test_grounding_exact_and_cross_script():
    assert is_grounded("Order confirm kar do", "Order confirm कर दो") is True
    assert is_grounded("ek butter naan", "एक बटर नान") is True


def test_grounding_rejects_duplicates():
    # Hallucinated duplicate tokens
    assert is_grounded("paneer paneer", "paneer naan") is False


def test_grounding_rejects_inverted_order():
    # Order inversion
    assert is_grounded("naan paneer", "paneer naan") is False


def test_grounding_rejects_quantity_hallucination():
    # Altered quantities
    assert is_grounded("teen naan", "do naan") is False
    assert is_grounded("3 butter naan", "2 butter naan") is False


# =========================================================================== #
# Relevance Gate Classification Tests
# =========================================================================== #
@pytest.mark.asyncio
async def test_gate_checkout_in_ordering_classified_as_order_action():
    state = {"step": "ordering", "cart": [{"item_id": "1", "name": "Paneer Butter Masala", "quantity": 1, "unit_price": 280.0}]}
    
    for phrase in ["Order confirm kar do", "Bas itna hi", "Nahi aur kuch nahi lunga"]:
        res = await evaluate(phrase, state, MENU_FIXTURE)
        assert res.is_relevant is True
        assert res.category == "ORDER_ACTION"
        assert res.cleaned_text is not None


@pytest.mark.asyncio
async def test_gate_bill_query_is_order_question():
    state = {"step": "ordering", "cart": [{"item_id": "1", "name": "Paneer Butter Masala", "quantity": 1, "unit_price": 280.0}]}
    res = await evaluate("Bill kitna hua?", state, MENU_FIXTURE)
    assert res.is_relevant is True
    assert res.category == "ORDER_QUESTION"


@pytest.mark.asyncio
async def test_gate_affirmation_strictly_step_dependent():
    # Bare affirmation during ordering is rejected as AMBIGUOUS
    ordering_state = {"step": "ordering", "cart": []}
    res_ordering = await evaluate("Haan", ordering_state, MENU_FIXTURE)
    assert res_ordering.is_relevant is False or res_ordering.category == "AMBIGUOUS"

    # Bare affirmation during awaiting_confirmation is accepted
    confirm_state = {"step": "awaiting_confirmation", "cart": [{"item_id": "1", "name": "Paneer Butter Masala", "quantity": 1, "unit_price": 280.0}]}
    res_confirm = await evaluate("Haan", confirm_state, MENU_FIXTURE)
    assert res_confirm.is_relevant is True
    assert res_confirm.category == "CONFIRMATION"


@pytest.mark.asyncio
async def test_gate_unrelated_rejection():
    state = {"step": "ordering", "cart": []}
    res = await evaluate("Kal office mein meeting hai subah jaldi", state, MENU_FIXTURE)
    assert res.is_relevant is False
    assert res.category == "UNRELATED"


# =========================================================================== #
# LangGraph State Transition & Compound Checkout Tests
# =========================================================================== #
@pytest.mark.asyncio
async def test_transition_negative_completion_triggers_checkout():
    state = {
        "step": "ordering",
        "cart": [{"item_id": "1", "name": "Paneer Butter Masala", "quantity": 1, "unit_price": 280.0}],
        "conversation_history": [
            {"role": "assistant", "content": "1 Paneer Butter Masala जोड़ दिया। और कुछ?"}
        ],
        "retries": 0,
        "confirmation_status": "pending"
    }
    turn = await process_order_turn(state, "Nahi aur kuch nahi lunga", MENU_FIXTURE)
    assert turn["step"] == "awaiting_confirmation"
    assert "280" in turn["reply"]


@pytest.mark.asyncio
async def test_transition_compound_checkout_adds_item_and_checks_out():
    state = {
        "step": "ordering",
        "cart": [{"item_id": "1", "name": "Paneer Butter Masala", "quantity": 1, "unit_price": 280.0}],
        "conversation_history": [],
        "retries": 0,
        "confirmation_status": "pending"
    }
    turn = await process_order_turn(state, "Ek cold coffee add kar do aur bas order confirm kar do", MENU_FIXTURE)
    assert turn["step"] == "awaiting_confirmation"
    cart_items = [c["name"] for c in turn["cart"]]
    assert "Cold Coffee" in cart_items
    assert len(turn["cart"]) == 2
    # Total: 280 + 120 = 400
    assert "400" in turn["reply"]


@pytest.mark.asyncio
async def test_transition_bill_inquiry_does_not_checkout():
    state = {
        "step": "ordering",
        "cart": [{"item_id": "1", "name": "Paneer Butter Masala", "quantity": 1, "unit_price": 280.0}],
        "conversation_history": [],
        "retries": 0,
        "confirmation_status": "pending"
    }
    turn = await process_order_turn(state, "Bill kitna hua?", MENU_FIXTURE)
    assert turn["step"] == "ordering"  # Must remain in ordering step
    assert "280" in turn["reply"]

@pytest.mark.asyncio
async def test_gate_name_in_collecting_details():
    state = {
        "step": "collecting_details",
        "cart": [{"item_id": "1", "name": "Paneer Butter Masala", "quantity": 1, "unit_price": 280.0}],
        "conversation_history": [
            {"role": "assistant", "content": "ऑर्डर नोट हो गया! डिलीवरी के लिए अपना नाम और पता बता दीजिए।"}
        ]
    }
    for phrase in ["मेरा नाम आयुष है.", "Mera naam Ayush hai", "Ayush Shukla"]:
        res = await evaluate(phrase, state, MENU_FIXTURE)
        assert res.is_relevant is True
        assert res.category == "ORDER_DETAIL"
        assert res.cleaned_text is not None

@pytest.mark.asyncio
async def test_bare_name_in_collecting_details():
    state = {
        "step": "collecting_details",
        "customer_name": None,
        "cart": [{"item_id": "1", "name": "Paneer Butter Masala", "quantity": 1, "unit_price": 280.0}],
        "conversation_history": [
            {"role": "assistant", "content": "ऑर्डर नोट हो गया! डिलीवरी के लिए अपना नाम और पता बता दीजिए।"}
        ]
    }
    
    # Single name
    res1 = await evaluate("Ayush", state, MENU_FIXTURE)
    assert res1.is_relevant is True
    assert res1.category == "ORDER_DETAIL"
    assert res1.cleaned_text == "Ayush"

    # Full name in Devanagari
    res2 = await evaluate("आयुष शुक्ला", state, MENU_FIXTURE)
    assert res2.is_relevant is True
    assert res2.category == "ORDER_DETAIL"

    # Turn processing extracts customer_name
    turn = await process_order_turn(state, "Ayush", MENU_FIXTURE)
    assert turn["customer_name"] == "Ayush"
    assert turn["step"] == "collecting_details"
    assert "डिलीवरी पता बता दीजिए" in turn["reply"]

@pytest.mark.asyncio
async def test_bare_name_in_collecting_details():
    state = {
        "step": "collecting_details",
        "customer_name": None,
        "cart": [{"item_id": "1", "name": "Paneer Butter Masala", "quantity": 1, "unit_price": 280.0}],
        "conversation_history": [
            {"role": "assistant", "content": "ऑर्डर नोट हो गया! डिलीवरी के लिए अपना नाम और पता बता दीजिए।"}
        ]
    }
    
    # Single name
    res1 = await evaluate("Ayush", state, MENU_FIXTURE)
    assert res1.is_relevant is True
    assert res1.category == "ORDER_DETAIL"
    assert res1.cleaned_text == "Ayush"

    # Full name in Devanagari
    res2 = await evaluate("आयुष शुक्ला", state, MENU_FIXTURE)
    assert res2.is_relevant is True
    assert res2.category == "ORDER_DETAIL"

    # Turn processing extracts customer_name
    turn = await process_order_turn(state, "Ayush", MENU_FIXTURE)
    assert turn["customer_name"] == "Ayush"
    assert turn["step"] == "collecting_details"
    assert "डिलीवरी पता बता दीजिए" in turn["reply"]