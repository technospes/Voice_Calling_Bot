"""Persists dental appointments and stages outbox notifications atomically."""
import json
import logging
import os
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.clinic.catalog import CLINIC_TZ
from app.db.session import AsyncSessionLocal

logger = logging.getLogger("clinic-service")

SLOT_CAPACITY = int(os.getenv("CLINIC_SLOT_CAPACITY", "1"))  # patients per slot (= number of doctors)


class SlotUnavailableError(Exception):
    """Slot was taken (or expired) between offering it and saving. Worker must re-offer slots."""


class CreatedAppointment:
    def __init__(self, appointment_id: uuid.UUID, order_number: str, total_amount: Decimal):
        self.id = appointment_id
        self.order_number = order_number  # named order_number for livekit_worker.py compatibility
        self.total_amount = total_amount


def generate_appointment_number() -> str:
    return f"APT-{uuid.uuid4().hex[:8].upper()}"


async def load_booked_slots(clinic_id: str) -> List[str]:
    """Returns list of canonical slot_ids ('YYYY-MM-DDTHH:MM' in CLINIC_TZ) that are at full capacity."""
    try:
        cid = uuid.UUID(str(clinic_id))
        now = datetime.now(CLINIC_TZ)
        async with AsyncSessionLocal() as session:
            rows = (await session.execute(
                text("""
                    SELECT slot_start
                    FROM public.appointments
                    WHERE clinic_id = :c
                      AND status = 'CONFIRMED'
                      AND slot_start >= :now
                    GROUP BY slot_start
                    HAVING count(*) >= :cap
                """),
                {"c": cid, "now": now, "cap": SLOT_CAPACITY},
            )).scalars().all()
            return [dt.astimezone(CLINIC_TZ).strftime("%Y-%m-%dT%H:%M") for dt in rows if dt]
    except Exception as exc:
        logger.warning("Could not pre-load booked slots: %s", exc)
        return []


async def create_appointment_in_db(state: Dict[str, Any], session: AsyncSession) -> CreatedAppointment:
    cart = state.get("cart") or []
    if not cart:
        raise ValueError("Cannot book an appointment without a selected service.")
    booking = cart[0]

    slot_id = booking.get("slot_id")
    if not slot_id:
        raise ValueError("Cannot book an appointment without a selected slot.")
    slot_start = datetime.fromisoformat(slot_id).replace(tzinfo=CLINIC_TZ)
    if slot_start <= datetime.now(CLINIC_TZ):
        raise SlotUnavailableError(f"Slot {slot_id} is in the past")

    raw_clinic = state.get("restaurant_id") or state.get("clinic_id")
    if not raw_clinic:
        raise ValueError("Missing clinic id (restaurant_id) in call state.")
    clinic_id = uuid.UUID(str(raw_clinic))

    patient_phone = state.get("customer_phone") or "UNAVAILABLE"
    patient_name = state.get("customer_name") or "Patient"
    patient_note = state.get("delivery_address") or "General Consultation"
    call_sid = state.get("call_sid") or f"MANUAL_{uuid.uuid4().hex[:8]}"
    call_id = state.get("call_id")
    if call_id and isinstance(call_id, str):
        call_id = uuid.UUID(call_id)

    service_name = booking.get("name", "General Dental Checkup")
    slot_label = booking.get("slot") or slot_id
    fee = Decimal(str(booking.get("unit_price", 300)))
    idempotency_key = f"apt_{call_sid}_{call_id}"

    try:
        # Serialise concurrent bookings of the same clinic+slot (released on commit/rollback).
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"),
            {"k": f"{clinic_id}:{slot_id}"},
        )

        # Idempotent retry: same call already booked -> return it, no duplicate row/outbox event.
        existing = (await session.execute(
            text("""SELECT id, appointment_number, consultation_fee
                    FROM public.appointments WHERE idempotency_key = :k"""),
            {"k": idempotency_key},
        )).first()
        if existing:
            await session.rollback()
            logger.info("Appointment %s already exists for this call (idempotent replay)", existing.appointment_number)
            return CreatedAppointment(existing.id, existing.appointment_number, Decimal(str(existing.consultation_fee)))

        taken = (await session.execute(
            text("""SELECT count(*) FROM public.appointments
                    WHERE clinic_id = :c AND slot_start = :s AND status = 'CONFIRMED'"""),
            {"c": clinic_id, "s": slot_start},
        )).scalar_one()
        if taken >= SLOT_CAPACITY:
            raise SlotUnavailableError(f"Slot {slot_id} is full")

        apt_id = uuid.uuid4()
        apt_number = generate_appointment_number()

        await session.execute(
            text("""
                INSERT INTO public.appointments (
                    id, appointment_number, clinic_id, call_id, call_sid,
                    patient_name, patient_phone, patient_age_or_note,
                    service_name, appointment_slot, slot_start, consultation_fee, status, idempotency_key
                ) VALUES (
                    :id, :apt_number, :clinic_id, :call_id, :call_sid,
                    :patient_name, :patient_phone, :patient_note,
                    :service_name, :slot_label, :slot_start, :fee, 'CONFIRMED', :idem_key
                )
            """),
            {
                "id": apt_id, "apt_number": apt_number, "clinic_id": clinic_id,
                "call_id": call_id, "call_sid": call_sid,
                "patient_name": patient_name, "patient_phone": patient_phone,
                "patient_note": patient_note, "service_name": service_name,
                "slot_label": slot_label, "slot_start": slot_start,
                "fee": fee, "idem_key": idempotency_key,
            },
        )

        outbox_payload = {
            "mode": "CLINIC",
            "order_id": str(apt_id),
            "order_number": apt_number,
            "restaurant_id": str(clinic_id),
            "customer_phone": patient_phone,
            "call_sid": call_sid,
            "customer_name": patient_name,
            "delivery_address": f"Slot: {slot_label} | Note: {patient_note}",
            "service_name": service_name,
            "appointment_slot": slot_label,
            "slot_start": slot_start.isoformat(),
            "patient_note": patient_note,
            "items": [{"name": f"{service_name} ({slot_label})", "quantity": 1, "price": float(fee)}],
            "total_amount": float(fee),
            "status": "CONFIRMED",
        }

        await session.execute(
            text("""
                INSERT INTO public.notification_outbox (order_id, event_type, payload, status)
                VALUES (:order_id, 'appointment_confirmed', CAST(:payload AS jsonb), 'pending')
            """),
            {"order_id": apt_id, "payload": json.dumps(outbox_payload, ensure_ascii=False)},
        )

        await session.commit()
    except SlotUnavailableError:
        await session.rollback()
        raise
    except Exception:
        await session.rollback()
        logger.exception("Failed to create appointment for call %s", call_sid)
        raise

    logger.info("Appointment %s (%s) committed with outbox event.", apt_number, apt_id)
    return CreatedAppointment(apt_id, apt_number, fee)