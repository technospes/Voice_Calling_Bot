"""State schema for the Spice Villa conversational agent."""
import uuid
from typing import List, Optional, TypedDict


class CartItem(TypedDict):
    item_id: str
    name: str
    quantity: int
    unit_price: float


class OrderState(TypedDict, total=False):
    # Core session & call identity
    call_id: uuid.UUID
    call_sid: str
    restaurant_id: str

    # State machine control
    step: str  # ordering -> awaiting_confirmation -> collecting_details -> closing
    confirmation_status: str  # pending -> confirmed / cancelled

    # Order details
    cart: List[CartItem]
    customer_phone: Optional[str]
    customer_name: Optional[str]
    delivery_address: Optional[str]

    # Interaction & retry accounting
    latest_speech: str
    reply: str
    retries: int
    details_retries: int
    conversation_history: List[dict]

    # Menu cache
    menu_items: Optional[List[dict]]


def initial_order_state(call_id, call_sid: str, restaurant_id, customer_phone: Optional[str]) -> OrderState:
    """Single source of truth for a fresh call's state (used by every call path)."""
    return {
        "call_id": call_id,
        "call_sid": call_sid,
        "restaurant_id": str(restaurant_id),
        "step": "ordering",
        "cart": [],
        "customer_phone": customer_phone,
        "customer_name": None,
        "delivery_address": None,
        "latest_speech": "",
        "reply": "",
        "retries": 0,
        "details_retries": 0,
        "conversation_history": [],
        "menu_items": None,
        "confirmation_status": "pending",
    }