import logging
import uuid
import json
import random
from decimal import Decimal
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text
from app.db.models import Order, OrderItem, MenuItem, OrderStatus

logger = logging.getLogger(__name__)


def generate_order_number() -> str:
    return f"SV-{random.randint(100000, 999999)}"


async def create_order_in_db(order_state: Dict[str, Any], session: AsyncSession) -> Order:
    """
    Saves confirmed order and stages an outbox notification event atomically.
    Guarantees no external notification fires if DB save rolls back.
    """
    cart = order_state.get("cart", [])
    if not cart:
        raise ValueError("Cannot persist an empty cart.")

    restaurant_id = uuid.UUID(str(order_state["restaurant_id"]))
    customer_phone = order_state.get("customer_phone") or "UNAVAILABLE"
    call_sid = order_state.get("call_sid") or f"MANUAL_{uuid.uuid4().hex[:8]}"
    call_id = order_state.get("call_id")
    if call_id and isinstance(call_id, str):
        call_id = uuid.UUID(call_id)

    idempotency_key = f"order_{call_sid}_{call_id}"

    # 1. Authoritative price and availability verification against DB
    item_ids = [uuid.UUID(str(c["item_id"])) for c in cart]
    res = await session.execute(
        select(MenuItem).where(MenuItem.id.in_(item_ids))
    )
    db_items = {str(item.id): item for item in res.scalars().all()}

    total_amount = Decimal("0")
    validated_lines = []

    for c in cart:
        item_id_str = str(c["item_id"])
        db_item = db_items.get(item_id_str)
        if not db_item or not db_item.is_available:
            raise ValueError(f"Item {c.get('name')} is no longer available.")

        qty = int(c["quantity"])
        max_allowed = getattr(db_item, "max_order_qty", 20) or 20
        if qty <= 0 or qty > max_allowed:
            raise ValueError(f"Invalid quantity {qty} for item {db_item.name}.")

        unit_price = Decimal(str(db_item.base_price))
        line_total = unit_price * qty
        total_amount += line_total
        validated_lines.append({
            "item_id": db_item.id,
            "name": db_item.name,
            "quantity": qty,
            "unit_price": unit_price,
            "line_total": line_total
        })

    # 2. Insert Order record matching models.py schema (customer_name is stored in outbox/audit)
    customer_name = order_state.get("customer_name") or "Valued Guest"
    delivery_address = order_state.get("delivery_address") or "Call for address"

    new_order = Order(
        order_number=generate_order_number(),
        restaurant_id=restaurant_id,
        call_id=call_id,
        customer_phone=customer_phone,
        delivery_address=delivery_address,
        total_amount=total_amount,
        status=OrderStatus.CONFIRMED,
        idempotency_key=idempotency_key,
        call_sid=call_sid
    )
    session.add(new_order)
    await session.flush()

    # 3. Insert Order Items matching models.py schema
    for line in validated_lines:
        order_item = OrderItem(
            order_id=new_order.id,
            item_id=line["item_id"],
            item_name=line["name"],
            quantity=line["quantity"],
            unit_price=line["unit_price"],
            line_total=line["line_total"]
        )
        session.add(order_item)

    # 4. Insert Staged Notification Outbox Record
    outbox_payload = {
        "order_id": str(new_order.id),
        "order_number": new_order.order_number,
        "restaurant_id": str(restaurant_id),
        "customer_phone": customer_phone,
        "call_sid": call_sid,
        "customer_name": customer_name,
        "delivery_address": delivery_address,
        "items": [
            {
                "name": l["name"],
                "quantity": l["quantity"],
                "price": float(l["line_total"])
            }
            for l in validated_lines
        ],
        "total_amount": float(total_amount),
        "status": "CONFIRMED"
    }

    await session.execute(
        text("""
            INSERT INTO notification_outbox (order_id, event_type, payload, status)
            VALUES (:order_id, 'order_confirmed', CAST(:payload AS jsonb), 'pending')
        """),
        {
            "order_id": new_order.id,
            "payload": json.dumps(outbox_payload)
        }
    )

    # 5. Insert Initial Audit Log (Cast enum explicitly to match audit table)
    await session.execute(
        text("""
            INSERT INTO order_status_audit (order_id, previous_status, new_status, changed_by, reason)
            VALUES (:order_id, NULL, 'confirmed'::order_lifecycle_status, 'voice_agent', 'Voice call successfully placed order')
        """),
        {"order_id": new_order.id}
    )

    await session.commit()
    logger.info(f"✅ Order {new_order.order_number} ({new_order.id}) committed with outbox event.")
    return new_order