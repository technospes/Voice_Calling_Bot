from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from app.db.session import AsyncSessionLocal
from app.db.models import Order, Call

router = APIRouter(prefix="/owner", tags=["owner"])

# Initialize templates (pointing to the new folder)
templates = Jinja2Templates(directory="templates")

@router.get("/dashboard")
async def dashboard(request: Request):
    """Renders the HTML dashboard for the owner."""
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={}
    )

@router.get("/orders")
async def get_orders():
    """Fetches all confirmed orders for the dashboard."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Order).order_by(Order.created_at.desc())
        )
        orders = result.scalars().all()
        return {"orders": orders}

@router.get("/calls")
async def get_calls():
    """Fetches the call log history."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Call).order_by(Call.started_at.desc())
        )
        calls = result.scalars().all()
        return {"calls": calls}