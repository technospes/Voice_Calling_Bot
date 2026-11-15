"""Dental clinic services catalog and slot generator (real datetimes, clinic timezone)."""
import os
from datetime import datetime, timedelta
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo

from app.agent.matching import resolve_item, normalize

CLINIC_TZ = ZoneInfo(os.getenv("CLINIC_TZ", "Asia/Kolkata"))
MIN_LEAD_MINUTES = int(os.getenv("CLINIC_MIN_LEAD_MIN", "30"))

CLINIC_SERVICES: List[dict] = [
    {
        "id": "srv-checkup",
        "name": "General Dental Checkup",
        "base_price": 300,
        "is_available": True,
        "aliases": [
            "checkup", "general checkup", "consultation", "daant dikhana",
            "दांत दिखाना", "चेकअप", "परामर्श", "दांत में दर्द", "tooth pain", "dard",
        ],
    },
    {
        "id": "srv-cleaning",
        "name": "Teeth Cleaning & Scaling",
        "base_price": 800,
        "is_available": True,
        "aliases": [
            "cleaning", "scaling", "teeth cleaning", "daant ki safai", "safai",
            "दांत की सफाई", "सफाई", "स्केलिंग", "पीलापन",
        ],
    },
    {
        "id": "srv-rct",
        "name": "Root Canal Consultation",
        "base_price": 500,
        "is_available": True,
        "aliases": [
            "root canal", "rct", "naso ka ilaj", "आरसीटी", "रूट कैनाल", "नसों का इलाज",
        ],
    },
    {
        "id": "srv-extraction",
        "name": "Tooth Extraction",
        "base_price": 600,
        "is_available": True,
        "aliases": [
            "extraction", "tooth removal", "daant nikalwana", "दांत निकलवाना", "दाढ़ निकलवाना",
        ],
    },
]

# (day_offset, hour24, minute). Replace with a DB-backed doctor schedule later.
SLOT_TEMPLATE = [(0, 17, 0), (0, 18, 30), (1, 11, 0), (1, 17, 0)]

EMERGENCY_PHRASES = (
    "emergency", "इमरजेंसी", "accident", "एक्सीडेंट", "behosh", "बेहोश",
    "saans lene", "सांस लेने", "difficulty breathing", "khoon nahi ruk", "खून नहीं रुक",
)


def is_emergency(text: Optional[str]) -> bool:
    t = (text or "").lower()
    return any(p in t for p in EMERGENCY_PHRASES)


def _period(hour: int) -> str:
    if hour < 12:
        return "सुबह"
    if hour < 16:
        return "दोपहर"
    if hour < 19:
        return "शाम"
    return "रात"


def _label(start: datetime, now: datetime) -> str:
    days = (start.date() - now.date()).days
    day = "आज" if days == 0 else "कल" if days == 1 else start.strftime("%d %b")
    h12 = start.hour % 12 or 12
    return f"{day} {_period(start.hour)} {h12}:{start.minute:02d} बजे"


def get_open_slots(booked: Optional[Iterable[str]] = None, now: Optional[datetime] = None) -> List[dict]:
    """Future slots only. `booked` = optional iterable of slot ids already taken."""
    now = now or datetime.now(CLINIC_TZ)
    taken = set(booked or ())
    out: List[dict] = []
    for day, hour, minute in SLOT_TEMPLATE:
        start = (now + timedelta(days=day)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        sid = start.strftime("%Y-%m-%dT%H:%M")
        if start < now + timedelta(minutes=MIN_LEAD_MINUTES) or sid in taken:
            continue
        out.append({"id": sid, "start": start, "label": _label(start, now)})
    return out


def get_slot(slot_id: Optional[str], slots: List[dict]) -> Optional[dict]:
    return next((s for s in slots if s["id"] == slot_id), None)


def match_service(query: Optional[str]) -> Optional[dict]:
    if not query or not query.strip():
        return None
    res = resolve_item(query, CLINIC_SERVICES)
    if res.is_match:
        return res.item
    norm = normalize(query)
    low = query.lower()
    keys = ("dard", "pain", "check", "dikhana", "doctor", "दर्द", "दिखाना")
    if any(k in norm or k in low for k in keys):
        return CLINIC_SERVICES[0]
    return None