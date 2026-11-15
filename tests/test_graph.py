import asyncio

from app.agent import graph, nodes
from app.agent.state import initial_order_state
from tests.conftest import MENU


def test_menu_loaded_once_and_empty_menu_not_refetched(monkeypatch):
    calls = []

    async def fake_load(restaurant_id, *a, **k):
        calls.append(restaurant_id)
        return MENU

    async def fake_extract(prompt):
        return nodes.IntentExtraction(intent="ask_menu")

    monkeypatch.setattr(graph, "load_menu", fake_load)
    monkeypatch.setattr(nodes, "extract_intent", fake_extract)

    st = initial_order_state("c", "s", "r", "+91")
    st["latest_speech"] = "menu?"
    st.update(asyncio.run(graph.order_agent.ainvoke(st)))
    assert st["menu_items"] == MENU and calls == ["r"]
    st["latest_speech"] = "menu again"
    st.update(asyncio.run(graph.order_agent.ainvoke(st)))
    assert calls == ["r"]                         # cached: fetched once per call

    empty = initial_order_state("c", "s", "r2", "+91")
    empty["menu_items"] = []                      # valid empty menu
    empty["latest_speech"] = "menu?"
    asyncio.run(graph.order_agent.ainvoke(empty))
    assert calls == ["r"]                         # [] is not "uninitialized"
