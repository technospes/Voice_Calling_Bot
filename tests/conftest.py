import sys
import types

# Use the real app.config when available; otherwise a stub so the pure-logic tests run anywhere.
try:
    import app.config  # noqa: F401
except Exception:
    stub = types.ModuleType("app.config")
    stub.settings = types.SimpleNamespace(DATABASE_URL="sqlite+aiosqlite://", GROQ_API_KEY="test")
    sys.modules["app.config"] = stub

MENU = [
    {"id": "11111111-1111-1111-1111-111111111111", "name": "Paneer Butter Masala", "aliases": ["pbm", "पनीर बटर मसाला"], "base_price": 250.0, "is_available": True},
    {"id": "22222222-2222-2222-2222-222222222222", "name": "Garlic Naan", "aliases": ["लहसुन नान"], "base_price": 50.0, "is_available": True},
    {"id": "33333333-3333-3333-3333-333333333333", "name": "Butter Naan", "aliases": [], "base_price": 40.0, "is_available": True},
    {"id": "44444444-4444-4444-4444-444444444444", "name": "Paneer Tikka", "aliases": [], "base_price": 220.0, "is_available": True},
    {"id": "55555555-5555-5555-5555-555555555555", "name": "Tandoori Roti", "aliases": [], "base_price": 20.0, "is_available": True},
    {"id": "66666666-6666-6666-6666-666666666666", "name": "Lassi", "aliases": [], "base_price": 60.0, "is_available": True},
]
PBM, GARLIC, BUTTER, TIKKA, ROTI, LASSI = (m["id"] for m in MENU)
