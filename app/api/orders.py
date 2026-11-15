import uuid
from typing import Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Header, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text

from app.db.session import AsyncSessionLocal
from app.db.models import Order, OrderStatus

router = APIRouter(prefix="/orders", tags=["Order Management"])

VALID_TRANSITIONS = {
    "CONFIRMED": ["PREPARING", "CANCELLED"],
    "PREPARING": ["OUT_FOR_DELIVERY", "CANCELLED"],
    "OUT_FOR_DELIVERY": ["COMPLETED", "CANCELLED"],
    "COMPLETED": [],
    "CANCELLED": []
}

class StatusUpdateSchema(BaseModel):
    status: Literal["PREPARING", "OUT_FOR_DELIVERY", "COMPLETED", "CANCELLED"]
    reason: Optional[str] = None
    staff_notes: Optional[str] = None

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

def verify_staff_token(authorization: Optional[str] = Header(None)) -> str:
    """Verifies Bearer token for restaurant staff."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authentication token."
        )
    token = authorization.split(" ")[1]
    if token != "spice_villa_kitchen_secret_2026":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized staff member."
        )
    return "kitchen_staff_1"

@router.patch("/{order_id}/status")
async def update_order_status(
    order_id: uuid.UUID,
    payload: StatusUpdateSchema,
    staff_id: str = Depends(verify_staff_token),
    idempotency_key: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
):
    res = await db.execute(select(Order).where(Order.id == order_id))
    order = res.scalars().first()

    if not order:
        raise HTTPException(status_code=404, detail="Order not found.")

    current_status = order.status.value if hasattr(order.status, "value") else str(order.status).upper()
    target_status = payload.status.upper()

    if current_status == target_status:
        return {"message": "Status already set to target.", "order_id": str(order_id), "status": current_status}

    allowed = VALID_TRANSITIONS.get(current_status, [])
    if target_status not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Illegal transition from '{current_status}' to '{target_status}'. Allowed transitions: {allowed}"
        )

    # 1. Update Order model
    order.status = getattr(OrderStatus, target_status, target_status)
    if payload.staff_notes:
        order.staff_notes = payload.staff_notes

    # 2. Write Audit Log using standard CAST syntax
    await db.execute(
        text("""
            INSERT INTO order_status_audit (order_id, previous_status, new_status, changed_by, reason)
            VALUES (
                :order_id, 
                CAST(:prev AS order_lifecycle_status), 
                CAST(:new AS order_lifecycle_status), 
                :staff, 
                :reason
            )
        """),
        {
            "order_id": order_id,
            "prev": current_status.lower(),
            "new": target_status.lower(),
            "staff": staff_id,
            "reason": payload.reason or "Staff status update"
        }
    )

    await db.commit()
    return {
        "success": True,
        "order_id": str(order_id),
        "previous_status": current_status,
        "new_status": target_status
    }