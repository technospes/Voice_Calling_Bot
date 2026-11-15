import asyncio
from sqlalchemy.future import select
from app.db.session import AsyncSessionLocal
from app.db.models import WhatsAppMessage, MsgStatus
from app.services.whatsapp_client import send_whatsapp_receipt

async def process_outbox():
    while True:
        async with AsyncSessionLocal() as session:
            # Fetch pending messages
            result = await session.execute(
                select(WhatsAppMessage).where(WhatsAppMessage.status == MsgStatus.QUEUED)
            )
            messages = result.scalars().all()
            
            for msg in messages:
                try:
                    # Execute network request
                    await send_whatsapp_receipt(
                        # Reconstruct basic order payload or alter client to accept raw IDs
                        order_id=msg.order_id, 
                        recipient_phone=msg.to_number, 
                        template_name=msg.template_name
                    )
                    msg.status = MsgStatus.SENT
                except Exception as e:
                    print(f"Failed to send Msg {msg.id}: {e}")
                    msg.status = MsgStatus.FAILED
                    
            if messages:
                await session.commit()
                
        await asyncio.sleep(5) # Poll every 5 seconds

if __name__ == "__main__":
    asyncio.run(process_outbox())