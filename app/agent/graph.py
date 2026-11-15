import uuid
from typing import List, Optional

from langgraph.graph import StateGraph, END
from sqlalchemy.future import select

from app.agent.nodes import process_order_turn
from app.agent.state import OrderState
from app.db.models import MenuItem
from app.db.session import AsyncSessionLocal
from app.voice.timing import span


def menu_item_to_dict(m: MenuItem) -> dict:
    return {
        "id": str(m.id),
        "name": m.name,
        "aliases": list(m.aliases or []),
        "base_price": float(m.base_price),
        "is_available": bool(m.is_available),
    }


async def load_menu(restaurant_id: str, session_factory=AsyncSessionLocal) -> List[dict]:
    """Fetch available menu items as plain, serializable dicts (no ORM objects)."""
    async with session_factory() as session:
        result = await session.execute(
            select(MenuItem)
            .where(MenuItem.restaurant_id == uuid.UUID(str(restaurant_id)))
            .where(MenuItem.is_available == True)  # noqa: E712
        )
        return [menu_item_to_dict(m) for m in result.scalars().all()]


async def ordering_node(state: OrderState) -> dict:
    """Invokes conversational state-machine and propagates all updated slots."""
    menu: Optional[List[dict]] = state.get("menu_items")
    if menu is None:
        with span("menu_load"):
            menu = await load_menu(state["restaurant_id"])

    spoken = state.get("latest_speech", "")
    
    # Process turn with full domain logic
    turn_result = await process_order_turn(state, spoken, menu)
    
    # Ensure menu_items remains attached in state
    turn_result["menu_items"] = menu
    return turn_result


# Compile deterministic single-node cycle workflow
workflow = StateGraph(OrderState)
workflow.add_node("take_items", ordering_node)
workflow.set_entry_point("take_items")
workflow.add_edge("take_items", END)
order_agent = workflow.compile()