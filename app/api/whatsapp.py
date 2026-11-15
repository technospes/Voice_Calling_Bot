from fastapi import APIRouter, Request, Query, HTTPException
from app.config import settings

router = APIRouter(prefix="/webhooks/whatsapp", tags=["whatsapp"])

@router.get("/")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    """Meta webhook verification endpoint."""
    # In production, add WHATSAPP_VERIFY_TOKEN to your config
    verify_token = getattr(settings, "WHATSAPP_VERIFY_TOKEN", "mock_verify_token")
    
    if hub_mode == "subscribe" and hub_verify_token == verify_token:
        print("✅ WhatsApp Webhook verified by Meta.")
        return int(hub_challenge)
    raise HTTPException(status_code=403, detail="Invalid verification token")

@router.post("/")
async def receive_status_update(request: Request):
    """Receives delivery statuses (sent, delivered, read) from Meta."""
    payload = await request.json()
    
    # Placeholder: In Phase 4, we will parse this payload and update the 
    # whatsapp_messages.status column in the database based on provider_message_id.
    print(f"📩 [WHATSAPP WEBHOOK] Received update: {payload}")
    
    return {"status": "received"}