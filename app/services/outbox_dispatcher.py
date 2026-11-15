import datetime
import logging
import os
from typing import Optional
import httpx
from sqlalchemy import select, update

from app.config import settings
from app.db.models import NotificationOutbox
from app.db.session import AsyncSessionLocal

logger = logging.getLogger("outbox-dispatcher")

BATCH_SIZE = 10
MAX_ATTEMPTS = 5


async def process_outbox_batch() -> int:
    """Claims pending outbox events and dispatches them to the configured n8n webhook."""
    webhook_url = getattr(settings, "N8N_ORDER_WEBHOOK_URL", None) or os.getenv("N8N_ORDER_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("N8N_ORDER_WEBHOOK_URL not configured. Skipping outbox dispatch batch.")
        return 0

    webhook_secret = getattr(settings, "WEBHOOK_SHARED_SECRET", "") or os.getenv("WEBHOOK_SHARED_SECRET", "")

    now_utc = datetime.datetime.now(datetime.timezone.utc)

    async with AsyncSessionLocal() as session:
        # Simple, robust query that works across schema iterations
        stmt = (
            select(NotificationOutbox)
            .where(
                NotificationOutbox.status.in_(["pending", "failed"]),
                NotificationOutbox.attempts < MAX_ATTEMPTS,
            )
            .order_by(NotificationOutbox.created_at.asc())
            .limit(BATCH_SIZE)
            .with_for_update(skip_locked=True)
        )
        result = await session.execute(stmt)
        events = result.scalars().all()

        if not events:
            return 0

        event_ids = [e.id for e in events]
        await session.execute(
            update(NotificationOutbox)
            .where(NotificationOutbox.id.in_(event_ids))
            .values(status="processing")
        )
        await session.commit()

    processed_count = 0

    async with httpx.AsyncClient(timeout=10.0) as client:
        for event in events:
            event_id = event.id
            payload = event.payload
            headers = {
                "Content-Type": "application/json",
                "X-Outbox-Event-Id": str(event_id),
                "X-Webhook-Secret": webhook_secret,
            }

            success = False
            last_error: Optional[str] = None

            try:
                response = await client.post(webhook_url, json=payload, headers=headers)
                response.raise_for_status()
                success = True
                logger.info(f"✅ Dispatched outbox event {event_id} to n8n (HTTP {response.status_code}).")
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)}"
                logger.error(f"❌ Failed to dispatch outbox event {event_id}: {last_error}")

            async with AsyncSessionLocal() as session:
                if success:
                    update_vals = {
                        "status": "sent",
                        "last_error": None,
                    }
                    if hasattr(NotificationOutbox, "sent_at"):
                        update_vals["sent_at"] = now_utc
                    if hasattr(NotificationOutbox, "updated_at"):
                        update_vals["updated_at"] = now_utc

                    await session.execute(
                        update(NotificationOutbox)
                        .where(NotificationOutbox.id == event_id)
                        .values(**update_vals)
                    )
                    processed_count += 1
                else:
                    new_attempts = event.attempts + 1
                    new_status = "failed" if new_attempts < MAX_ATTEMPTS else "dead_letter"
                    update_vals = {
                        "status": new_status,
                        "attempts": new_attempts,
                        "last_error": last_error,
                    }
                    if hasattr(NotificationOutbox, "updated_at"):
                        update_vals["updated_at"] = now_utc

                    await session.execute(
                        update(NotificationOutbox)
                        .where(NotificationOutbox.id == event_id)
                        .values(**update_vals)
                    )
                await session.commit()

    return processed_count