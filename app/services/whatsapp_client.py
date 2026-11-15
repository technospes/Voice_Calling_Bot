import httpx
from app.config import settings

async def send_whatsapp_receipt(order, items: list[dict], recipient_phone: str, role: str) -> dict:
    if role not in {"CUSTOMER", "OWNER"}:
        raise ValueError("role must be CUSTOMER or OWNER")
        
    items_text = "\n".join(f"{item['quantity']} x {item['name']} (₹{item['unit_price']})" for item in items)
    template_name = "order_receipt_customer" if role == "CUSTOMER" else "new_order_owner"
    
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": "en"},
            "components": [
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": str(order.order_number)},
                        {"type": "text", "text": items_text},
                        {"type": "text", "text": str(order.total_amount)}
                    ]
                }
            ]
        }
    }

    if settings.WHATSAPP_MOCK:
        print(f"\n📱 [MOCK WHATSAPP] {role} receipt to {recipient_phone}")
        print(f"Template: {template_name} | Order: {order.order_number} | Total: ₹{order.total_amount}")
        return {"messaging_product": "whatsapp", "contacts": [{"input": recipient_phone}], "messages": [{"id": f"mock_wamid_{order.order_number}"}]}

    url = f"https://graph.facebook.com/{settings.WHATSAPP_API_VERSION}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()