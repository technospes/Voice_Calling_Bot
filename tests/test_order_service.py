import asyncio
import uuid
from types import SimpleNamespace

import pytest

from app.services import order_service as svc
from tests.conftest import GARLIC, LASSI, PBM

REST = "99999999-9999-9999-9999-999999999999"
CALL = uuid.uuid4()


class FakeSession:
    def __init__(self):
        self.added, self.commits = [], 0

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        for o in self.added:
            if getattr(o, "id", None) is None:
                o.id = uuid.uuid4()

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass


def db_item(item_id, name, price, available=True):
    return SimpleNamespace(id=uuid.UUID(item_id), name=name, base_price=price, is_available=available)


@pytest.fixture
def db(monkeypatch):
    items = {
        uuid.UUID(PBM): db_item(PBM, "Paneer Butter Masala", "250.00"),
        uuid.UUID(GARLIC): db_item(GARLIC, "Garlic Naan", "50.00"),
        uuid.UUID(LASSI): db_item(LASSI, "Lassi", "60.00", available=False),
    }
    existing = {"order": None}

    async def fake_fetch(session, ids, restaurant_id):
        return {k: v for k, v in items.items() if k in set(ids)}

    async def fake_find(session, call_id):
        return existing["order"]

    monkeypatch.setattr(svc, "_fetch_menu_items", fake_fetch)
    monkeypatch.setattr(svc, "_find_order_for_call", fake_find)
    return SimpleNamespace(items=items, existing=existing)


def state(cart, status="confirmed"):
    return {"call_id": CALL, "restaurant_id": REST, "customer_phone": "+911234567890",
            "confirmation_status": status, "cart": cart}


def run(st, session=None):
    session = session or FakeSession()
    return asyncio.run(svc.create_order_in_db(st, session)), session


def line(item_id, qty, price=1.0):
    return {"item_id": item_id, "name": "x", "quantity": qty, "unit_price": price}


@pytest.mark.parametrize("status", ["pending", "cancelled", None])
def test_not_confirmed_is_rejected(db, status):
    s = FakeSession()
    with pytest.raises(ValueError):
        run(state([line(PBM, 1)], status), s)
    assert s.added == [] and s.commits == 0


def test_empty_cart_rejected(db):
    with pytest.raises(ValueError):
        run(state([]))


def test_total_uses_db_prices_not_cart_prices(db):
    order, s = run(state([line(PBM, 1, price=1.0), line(GARLIC, 2, price=0.01)]))
    assert str(order.total_amount) == "350.00"
    assert s.commits == 1


def test_duplicate_lines_are_merged(db):
    order, s = run(state([line(GARLIC, 1), line(GARLIC, 2)]))
    items = [o for o in s.added if o.__class__.__name__ == "OrderItem"]
    assert len(items) == 1 and items[0].quantity == 3


@pytest.mark.parametrize("qty", [0, -1, 1.5, "abc", None, True])
def test_bad_quantities_rejected(db, qty):
    with pytest.raises(ValueError):
        run(state([line(PBM, qty)]))


def test_invalid_id_unknown_item_and_unavailable_rejected(db):
    with pytest.raises(ValueError):
        run(state([line("unknown", 1)]))
    with pytest.raises(ValueError):
        run(state([line(str(uuid.uuid4()), 1)]))      # not in this restaurant / not in DB
    with pytest.raises(ValueError):
        run(state([line(LASSI, 1)]))                  # unavailable


def test_duplicate_save_for_same_call_returns_existing(db):
    first, s1 = run(state([line(PBM, 1)]))
    db.existing["order"] = first
    second, s2 = run(state([line(PBM, 1)]))
    assert second is first and s2.added == [] and s2.commits == 0
