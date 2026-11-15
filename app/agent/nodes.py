r"""Conversation logic for the Spice Villa voice agent.

State machine:
    ordering --checkout--> awaiting_confirmation --confirm--> collecting_details
        ^                                                               |
        |                                                               v (both slots valid)
        \------ modify ---------------------------------------- closing (ready for DB commit)
"""
import json
import logging
import re
from decimal import Decimal
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from app.agent.matching import MatchResult, normalize, resolve_item
from app.agent.state import OrderState
from app.voice.timing import span

logger = logging.getLogger("order-nodes")

MAX_HISTORY = 12
PROMPT_HISTORY = 6
MAX_LINE_QTY = 20
MAX_CART_ITEMS = 15
MAX_CONSECUTIVE_UNCLEAR = 3
MAX_DETAILS_RETRIES = 3

Intent = Literal[
    "add", "remove", "change_qty", "replace", "checkout", "cancel",
    "ask_menu", "repeat_order", "forward_to_owner", "unclear",
]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ExtractedItem(BaseModel):
    item_name: str = Field(description="Menu item name as English text")
    target_name: Optional[str] = Field(default=None, description="For item replacement: the existing item to replace.")
    quantity: int = Field(default=1, ge=0, le=MAX_LINE_QTY)
    operation: Literal["add", "set", "decrease", "remove_all"] = Field(
        default="add",
        description="add = increment, set = absolute qty, decrease = decrement, remove_all = delete."
    )
    remove_all: bool = Field(default=False)

    @field_validator("item_name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("item_name must not be empty")
        return v


class IntentExtraction(BaseModel):
    intent: Intent
    items: List[ExtractedItem] = Field(default_factory=list, max_length=20)


class CustomerDetailsExtraction(BaseModel):
    name: Optional[str] = Field(default=None, description="Customer personal name only. Null if not provided.")
    address: Optional[str] = Field(default=None, description="Delivery location, flat, building, society, or area. Null if not provided.")


# --------------------------------------------------------------------------- #
# LLM Providers & Invocations
# --------------------------------------------------------------------------- #
_llm = None
_structured = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq
        from app.config import settings
        kwargs = dict(temperature=0, model_name="openai/gpt-oss-20b", groq_api_key=settings.GROQ_API_KEY)
        try:
            _llm = ChatGroq(reasoning_effort="low", **kwargs)
        except Exception:
            _llm = ChatGroq(**kwargs)
    return _llm


def _get_structured_llm():
    global _structured
    if _structured is None:
        for method in ("json_schema", "function_calling"):
            try:
                _structured = _get_llm().with_structured_output(IntentExtraction, method=method)
                break
            except Exception:
                pass
        if _structured is None:
            raise RuntimeError("No structured-output method available")
    return _structured


async def extract_intent(prompt: str) -> IntentExtraction:
    return await _get_structured_llm().ainvoke(prompt)


async def classify_confirmation(prompt: str) -> str:
    result = await _get_llm().ainvoke(prompt)
    return str(result.content)


# --------------------------------------------------------------------------- #
# Fast-Path & Disambiguated Confirmation Matching
# --------------------------------------------------------------------------- #
_EXPLICIT_CANCEL_REGEX = re.compile(
    r"\b(order\s+cancel|poora\s+cancel|pura\s+cancel|cancel\s+kar\s+do|cancel\s+the\s+order)\b|"
    r"(ऑर्डर\s*कैंसिल|पूरा\s*कैंसिल|कैंसिल\s*कर\s*दो)",
    re.IGNORECASE
)

_DECLINE_REGEX = re.compile(
    r"\b(nahi|nahin|na|no|nope|ruk|ruko|wait|nahi\s+chahiye|mat\s+karo)\b|"
    r"(नहीं|ना|रुको|रुकिए|अभी\s*नहीं|नहीं\s*चाहिए|मत\s*करो)",
    re.IGNORECASE
)

_EXPLICIT_CONFIRM_PHRASES_REGEX = re.compile(
    r"\b(haan\s+confirm|han\s+confirm|ji\s+haan|haan\s+ji|confirm\s+kar\s+do|order\s+confirm|"
    r"confirm\s+hai|yes\s+confirm|haan\s+kar\s+do|pakka\s+confirm|bilkul\s+kar\s+do|okay\s+done|ok\s+done)\b|"
    r"(हाँ\s*कन्फर्म|हां\s*कन्फर्म|कन्फर्म\s*कर\s*दो|ऑर्डर\s*कन्फर्म|जी\s*हाँ|हाँ\s*जी|पक्का\s*कन्फर्म|कर\s*दो\s*कन्फर्म)",
    re.IGNORECASE
)

_STANDALONE_AFFIRMATIVE_REGEX = re.compile(
    r"^(haan|han|ha|ji|yes|yep|yeah|okay|ok|bilkul|pakka|sahi|done|theek\s+hai|thik\s+hai|"
    r"हाँ|हां|जी|बिल्कुल|पक्का|सही|ठीक\s*है|कन्फर्म)$",
    re.IGNORECASE
)


def _fast_classify_confirmation(spoken: str) -> Optional[str]:
    """Zero-latency confirmation evaluator with strict negation and decline precedence."""
    clean = re.sub(r"[^\w\s\u0900-\u097F]", "", spoken).strip()
    text = clean.lower()

    if not text:
        return None

    if _EXPLICIT_CANCEL_REGEX.search(text):
        return "cancel"

    has_decline = bool(_DECLINE_REGEX.search(text))
    if has_decline and not (text.startswith("haan") or text.startswith("yes") or text.startswith("जी")):
        return "decline"

    if _EXPLICIT_CONFIRM_PHRASES_REGEX.search(text) and not has_decline:
        return "confirm"

    if _STANDALONE_AFFIRMATIVE_REGEX.match(text) and not has_decline:
        return "confirm"

    return None


# --------------------------------------------------------------------------- #
# Strict Context-Aware Name Extraction & Filtering
# --------------------------------------------------------------------------- #
_NAME_STOP_WORDS = {
    "hai", "hu", "hoon", "shukriya", "ji", "mera", "naam", "kya", "bataiye",
    "batao", "bata", "boli", "boliye", "bolo", "hoga", "sir", "madam", "please",
    "theek", "ok", "yes", "no", "haan", "bol", "raha", "rahi"
}

_NAME_PATTERNS = [
    re.compile(r"(?:mera\s+naam|naam\s+hai|my\s+name\s+is|this\s+is)\s+([a-zA-Z\u0900-\u097F]+(?:\s+[a-zA-Z\u0900-\u097F]+){0,2})", re.IGNORECASE),
    re.compile(r"(?:मेरा\s*नाम|नाम\s*है)\s+([a-zA-Z\u0900-\u097F]+(?:\s+[a-zA-Z\u0900-\u097F]+){0,2})", re.IGNORECASE),
    re.compile(r"(?:main|mai|मैं)\s+([a-zA-Z\u0900-\u097F]+(?:\s+[a-zA-Z\u0900-\u097F]+){0,2})\s+(?:bol\s+raha|bol\s+rahi|बोल\s*रहा|बोल\s*रही)", re.IGNORECASE),
]


def _clean_and_validate_name(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    clean = re.sub(r"[^\w\s\u0900-\u097F]", "", candidate).strip()
    tokens = [t for t in clean.split() if t.lower() not in _NAME_STOP_WORDS]
    if not tokens or len(tokens) > 3:
        return None
    for token in tokens:
        if not re.match(r"^[a-zA-Z\u0900-\u097F]+$", token):
            return None
    return " ".join(tokens).title()


def _regex_extract_name(text: str) -> Optional[str]:
    clean = text.strip()
    for pattern in _NAME_PATTERNS:
        match = pattern.search(clean)
        if match:
            validated = _clean_and_validate_name(match.group(1))
            if validated:
                return validated
    return None


async def extract_delivery_details(spoken: str, history: List[dict]) -> CustomerDetailsExtraction:
    """Extracts customer name and address with schema validation and bare-name fallback."""
    quick_name = _regex_extract_name(spoken)

    clean_spoken = spoken.strip()
    tokens = clean_spoken.split()
    # Check if input is a plausible standalone name without keywords
    if not quick_name and 1 <= len(tokens) <= 3:
        candidate = _clean_and_validate_name(clean_spoken)
        if candidate and candidate.lower() not in {"delhi", "noida", "greater noida", "sector"}:
            quick_name = candidate

    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-PROMPT_HISTORY:])
    prompt = f"""You extract delivery info from restaurant phone callers (Hindi / Hinglish / English).

RECENT CONVERSATION:
{convo}

CALLER JUST SAID: "{spoken}"

Instructions:
1. 'name': Personal first and last name only (e.g. 'Ayush', 'Ayush Shukla'). If the caller says just their name, extract it. If not provided, return null. Never use words like 'bataiye', 'sir', 'madam', or restaurant names.
2. 'address': Delivery location, flat, building, society, or area. If not stated, return null.
3. Output valid JSON with keys "name" and "address".
"""
    try:
        extractor = _get_llm().with_structured_output(CustomerDetailsExtraction, method="json_schema")
        res = await extractor.ainvoke(prompt)
        if isinstance(res, CustomerDetailsExtraction):
            final_name = _clean_and_validate_name(res.name) or quick_name
            res.name = final_name
            return res
    except Exception as exc:
        logger.warning(f"Structured schema extraction failed ({exc}); trying fallback parsing.")

    try:
        fallback_prompt = f'{prompt}\n\nReturn ONLY a raw JSON object like: {{"name": "...", "address": "..."}}'
        raw_res = await _get_llm().ainvoke(fallback_prompt)
        raw_text = str(raw_res.content).strip()
        match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            name = _clean_and_validate_name(data.get("name")) or quick_name
            address = data.get("address")
            return CustomerDetailsExtraction(name=name, address=address)
    except Exception as raw_exc:
        logger.error(f"Fallback extraction failed: {raw_exc}")

    return CustomerDetailsExtraction(name=quick_name, address=None)


# --------------------------------------------------------------------------- #
# Helpers & Cart Computations
# --------------------------------------------------------------------------- #
_DEVANAGARI = re.compile("[ऀ-ॿ]")


def _spoken(item: dict) -> str:
    for alias in item.get("aliases") or []:
        if _DEVANAGARI.search(alias):
            return alias
    return item["name"]


def _money(value: Decimal) -> str:
    return str(int(value)) if value == value.to_integral_value() else f"{value:.2f}"


def _cart_total(cart: List[dict], menu_by_id: dict) -> Decimal:
    total = Decimal("0")
    for line in cart:
        price = menu_by_id.get(line["item_id"], {}).get("base_price", line["unit_price"])
        total += Decimal(str(price)) * int(line["quantity"])
    return total


def _cart_summary(cart: List[dict], menu_by_id: dict) -> str:
    return ", ".join(f"{line['quantity']} {_spoken(menu_by_id.get(line['item_id'], line))}" for line in cart)


def _options(items: List[dict]) -> str:
    return ", ".join(_spoken(i) for i in items[:4])


def _scope_for_cart(cart: List[dict], menu_by_id: dict) -> List[dict]:
    return [menu_by_id.get(c["item_id"]) or {"id": c["item_id"], "name": c["name"], "aliases": []} for c in cart]


def _resolve_all(extracted: List[ExtractedItem], pool: List[dict]):
    resolved, problems = [], []
    for ex in extracted:
        res: MatchResult = resolve_item(ex.item_name, pool)
        if res.is_match:
            resolved.append((res.item, ex))
        elif res.status == "ambiguous":
            problems.append(f"{ex.item_name} में कौन सा चाहिए: {_options(res.candidates)}?")
        else:
            problems.append(f"{ex.item_name} नहीं मिला।")
    return resolved, problems


def _clarify(problems: List[str], nothing_changed: str) -> str:
    return " ".join(problems) + f" {nothing_changed}"


def _trim(history: List[dict]) -> List[dict]:
    return history[-MAX_HISTORY:]


def _is_relative_increment_phrase(text: str) -> bool:
    text_norm = normalize(text)
    patterns = [
        r"\b(ek|do|teen|char|paanch|\d+)\s+aur\b",
        r"\baur\s+(ek|do|teen|char|paanch|\d+)\b",
        r"\b(badha|badhao|extra|more)\b",
        r"\bone\s+more\b",
    ]
    return any(re.search(p, text_norm) for p in patterns)


def _extract_relative_amount(text: str) -> int:
    text_norm = normalize(text)
    words = {"ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5}
    for word, num in words.items():
        if re.search(rf"\b({word}\s+aur|aur\s+{word})\b", text_norm):
            return num
    match = re.search(r"\b(\d+)\s+aur\b", text_norm) or re.search(r"\baur\s+(\d+)\b", text_norm)
    return max(1, int(match.group(1))) if match else 1


def _out(history, reply, cart, step, status, retries, details_retries=0) -> dict:
    history = _trim(history + [{"role": "agent", "content": reply}])
    return {
        "cart": cart,
        "reply": reply,
        "step": step,
        "confirmation_status": status,
        "retries": retries,
        "details_retries": details_retries,
        "conversation_history": history,
    }


def _build_prompt(spoken: str, menu: List[dict], cart: List[dict], history: List[dict]) -> str:
    menu_lines = "\n".join(
        f"- {m['name']} (₹{m['base_price']})" + (f" | aliases: {', '.join(m['aliases'])}" if m.get("aliases") else "")
        for m in menu
    ) or "(menu empty)"
    cart_lines = "\n".join(f"- {c['quantity']} x {c['name']}" for c in cart) or "(empty)"
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-PROMPT_HISTORY:]) or "(start of call)"
    
    return f"""You extract the intent of a restaurant phone caller (Hindi / Hinglish / English). Output structured data only.

MENU:
{menu_lines}

CURRENT CART:
{cart_lines}

RECENT CONVERSATION (oldest first):
{convo}

CALLER JUST SAID: "{spoken}"

Rules:
1. intent: add, remove, change_qty, replace, checkout, cancel, ask_menu, repeat_order, forward_to_owner, unclear.
   - checkout: caller signals they are done ordering and ready to finalize.
     * Direct requests: "order confirm kar do", "confirm kar do", "bas itna hi".
     * Negative completion: when responding to "aur kuch?" with "nahi", "nahin", "nahi aur kuch nahi", "nahi bas itna hi", "kuch nahi chahiye", or "nahi aur kuch nahi lunga".
       (NOTE: Do NOT treat an isolated "nahi" as checkout without completion context).
   - repeat_order: asking for bill, total, or cart status:
     * Examples: "bill kitna hua?", "total kitna hua?", "kitne paise hue?", "kya kya add kiya?".
     (NOTE: Asking for the total is repeat_order, NOT checkout).
   - cancel: explicitly canceling the entire order ("order cancel kar do", "nahi chahiye poora cancel").
2. compound checkout: If caller adds items AND asks to checkout in the same utterance:
   - Example: "Ek cold coffee add kar do aur bas order confirm kar do"
   - Intent MUST be 'checkout' and items MUST include the new item: [{{"item_name": "Cold Coffee", "quantity": 1, "operation": "add"}}].
3. items: list EVERY food item mentioned with explicit quantity and operation (add, set, decrease, remove_all).
4. item_name: Transliterate to English alphabet.
"""


# --------------------------------------------------------------------------- #
# Cart Operations
# --------------------------------------------------------------------------- #
def _do_add(extracted: List[ExtractedItem], menu: List[dict], menu_by_id: dict, cart: List[dict]) -> Tuple[Optional[List[dict]], str, bool]:
    if not extracted:
        return None, "कौन सा आइटम? नाम बताइए।", True

    resolved, problems = _resolve_all(extracted, menu)
    qty_by_id, order = {}, []

    for item, ex in resolved:
        if not item.get("is_available", True):
            problems.append(f"माफ़ कीजिए, {_spoken(item)} अभी उपलब्ध नहीं है।")
            continue
        if ex.quantity <= 0:
            problems.append(f"{ex.item_name} की मात्रा समझ नहीं आई।")
            continue
        if item["id"] not in qty_by_id:
            order.append(item)
        qty_by_id[item["id"]] = qty_by_id.get(item["id"], 0) + ex.quantity

    existing = {c["item_id"]: c["quantity"] for c in cart}
    valid_order = []
    for item in order:
        if item["id"] not in existing and len(cart) >= MAX_CART_ITEMS:
            problems.append(f"अधिकतम {MAX_CART_ITEMS} प्रकार के आइटम ही जोड़े जा सकते हैं।")
            continue
        if existing.get(item["id"], 0) + qty_by_id[item["id"]] > MAX_LINE_QTY:
            problems.append(f"{_spoken(item)} की अधिकतम सीमा {MAX_LINE_QTY} है।")
        else:
            valid_order.append(item)

    if not valid_order:
        return None, _clarify(problems, "अभी कुछ जोड़ा नहीं। फिर से बताइए?"), True

    parts = []
    for item in valid_order:
        qty = qty_by_id[item["id"]]
        line = next((c for c in cart if c["item_id"] == item["id"]), None)
        if line:
            line["quantity"] += qty
        else:
            cart.append({"item_id": item["id"], "name": item["name"], "quantity": qty,
                         "unit_price": item["base_price"]})
        parts.append(f"{qty} {_spoken(item)}")

    reply = f"{', '.join(parts)} जोड़ दिया।"
    reply += (" " + " ".join(problems)) if problems else " और कुछ?"
    has_problems = bool(problems)
    return cart, reply, has_problems


def _do_change_qty(extracted: List[ExtractedItem], menu_by_id: dict, cart: List[dict]) -> Tuple[Optional[List[dict]], str, bool]:
    if not cart:
        return None, "कार्ट खाली है।", True
    if not extracted:
        return None, "किस आइटम की संख्या बदलनी है?", True

    resolved, problems = _resolve_all(extracted, _scope_for_cart(cart, menu_by_id))
    if problems and not resolved:
        return None, _clarify(problems, "कार्ट में बदलाव नहीं किया।"), True

    parts = []
    for item, ex in resolved:
        line = next((c for c in cart if c["item_id"] == item["id"]), None)
        if not line:
            problems.append(f"{_spoken(item)} कार्ट में नहीं है।")
            continue

        spoken = _spoken(menu_by_id.get(item["id"], line))
        current_qty = line["quantity"]

        if ex.operation == "set":
            new_qty = ex.quantity
        elif ex.operation == "decrease":
            new_qty = max(0, current_qty - ex.quantity)
        elif ex.operation == "remove_all":
            new_qty = 0
        elif ex.operation == "add":
            new_qty = current_qty + ex.quantity
        else:
            new_qty = ex.quantity

        if new_qty <= 0:
            cart = [c for c in cart if c["item_id"] != item["id"]]
            parts.append(f"{spoken} हटा दिया")
        elif new_qty > MAX_LINE_QTY:
            problems.append(f"{spoken} की अधिकतम सीमा {MAX_LINE_QTY} है।")
        else:
            line["quantity"] = new_qty
            parts.append(f"{spoken} अब {new_qty}")

    if not parts and problems:
        return None, _clarify(problems, "कार्ट में बदलाव नहीं किया।"), True

    reply = f"{', '.join(parts)}।"
    reply += (" " + " ".join(problems)) if problems else " और कुछ?"
    has_problems = bool(problems)
    return cart, reply, has_problems


def _do_remove(extracted: List[ExtractedItem], menu_by_id: dict, cart: List[dict]) -> Tuple[Optional[List[dict]], str, bool]:
    if not cart:
        return None, "कार्ट खाली है।", True
    if not extracted:
        return None, "कौन सा आइटम हटाना है?", True

    resolved, problems = _resolve_all(extracted, _scope_for_cart(cart, menu_by_id))
    valid_changes = []
    for item, ex in resolved:
        line = next((c for c in cart if c["item_id"] == item["id"]), None)
        if line is None:
            problems.append(f"{_spoken(item)} कार्ट में नहीं है।")
            continue
        valid_changes.append((item, ex, line))

    if not valid_changes:
        return None, _clarify(problems, "कार्ट में बदलाव नहीं किया।"), True

    parts = []
    for item, ex, line in valid_changes:
        spoken = _spoken(menu_by_id.get(item["id"], line))
        if ex.remove_all or ex.quantity == 0 or ex.quantity >= line["quantity"]:
            cart = [c for c in cart if c["item_id"] != item["id"]]
            parts.append(f"{spoken} हटा दिया")
        else:
            line["quantity"] -= ex.quantity
            parts.append(f"{spoken} {ex.quantity} कम किया")

    reply = f"{', '.join(parts)}।"
    reply += (" " + " ".join(problems)) if problems else " और कुछ?"
    has_problems = bool(problems)
    return cart, reply, has_problems


def _do_replace(extracted: List[ExtractedItem], menu: List[dict], menu_by_id: dict, cart: List[dict]) -> Tuple[Optional[List[dict]], str, bool]:
    if not cart:
        return None, "कार्ट खाली है।", True
    if not extracted:
        return None, "क्या बदलना है, स्पष्ट बताइए।", True

    ex = extracted[0]
    if not ex.target_name:
        return None, f"{ex.item_name} की जगह कौन सा आइटम चाहिए, यह स्पष्ट नहीं है।", True

    resolved_target, _ = _resolve_all([ExtractedItem(item_name=ex.target_name)], _scope_for_cart(cart, menu_by_id))
    if not resolved_target:
        return None, f"{ex.target_name} आपके कार्ट में नहीं है।", True

    target_item_id = resolved_target[0][0]["id"]
    target_line = next((c for c in cart if c["item_id"] == target_item_id), None)
    preserved_qty = target_line["quantity"] if target_line else ex.quantity

    resolved_rep, problems = _resolve_all([ExtractedItem(item_name=ex.item_name, quantity=preserved_qty)], menu)
    if not resolved_rep or problems:
        return None, f"माफ़ कीजिए, {ex.item_name} उपलब्ध नहीं है। पुराना आइटम सुरक्षित है।", True

    rep_item, _ = resolved_rep[0]
    if not rep_item.get("is_available", True):
        return None, f"{rep_item['name']} अभी उपलब्ध नहीं है।", True

    cart = [c for c in cart if c["item_id"] != target_item_id]
    cart.append({
        "item_id": rep_item["id"],
        "name": rep_item["name"],
        "quantity": preserved_qty,
        "unit_price": rep_item["base_price"]
    })

    target_spoken = _spoken(menu_by_id.get(target_item_id, {"name": ex.target_name}))
    rep_spoken = _spoken(rep_item)
    return cart, f"{target_spoken} की जगह {preserved_qty} {rep_spoken} कर दिया है। और कुछ?", False


async def _handle_ordering(spoken, cart, menu, history, retries, *, from_confirmation=False):
    menu_by_id = {m["id"]: m for m in menu}
    unclear_reply = "ठीक है, क्या बदलना है?" if from_confirmation else "माफ़ कीजिए, फिर से बताइए?"

    try:
        with span("llm"):
            ext = await extract_intent(_build_prompt(spoken, menu, cart, history[:-1]))
    except Exception as exc:
        logger.exception("LLM intent extraction failure: %s", exc)
        return _out(history, "माफ़ कीजिए, आवाज़ कट रही थी। क्या आप एक बार फिर बोल सकते हैं?",
                    cart, "ordering", "pending", retries)

    intent = ext.intent if ext else "unclear"

    # Contextual Negative Completion: If the caller says "nahi" / "no" after "aur kuch?",
    # treat it as checkout intent and read back the cart total for confirmation.
    norm_spoken = normalize(spoken)
    if (intent in ("unclear", "checkout") or ext is None) and norm_spoken in (
        "nahi", "nahin", "na", "no", "nope", "nahi ji", "ji nahi", "नहीं", "ना", "नहीं जी"
    ):
        if cart:
            total = _money(_cart_total(cart, menu_by_id))
            reply = f"{_cart_summary(cart, menu_by_id)}, {total} रुपये। कन्फर्म कर दूँ?"
            return _out(history, reply, cart, "awaiting_confirmation", "pending", 0)
        else:
            return _out(history, "ठीक है। क्या ऑर्डर करेंगे?", cart, "ordering", "pending", 0)

    # Relative quantity fallback
    if (ext is None or not ext.items) and intent in ("unclear", "add") and _is_relative_increment_phrase(spoken):
        rel_qty = _extract_relative_amount(spoken)
        if cart:
            target_line = cart[-1]
            ex = ExtractedItem(item_name=target_line["name"], quantity=rel_qty, operation="add")
            new_cart, reply, has_problems = _do_change_qty([ex], menu_by_id, cart)
            if new_cart is not None:
                turn_retries = (retries + 1) if has_problems else 0
                return _out(history, reply, new_cart, "ordering", "pending", turn_retries)

    if ext is None or intent == "unclear":
        return _out(history, unclear_reply, cart, "ordering", "pending", retries + 1)

    if intent == "checkout":
        if ext.items:
            additions = [item for item in ext.items if not item.remove_all and item.quantity > 0]
            if additions:
                new_cart, add_reply, has_problems = _do_add(additions, menu, menu_by_id, cart)
                if new_cart is None:
                    return _out(history, add_reply, cart, "ordering", "pending", retries + 1)
                cart = new_cart

        if not cart:
            return _out(history, "कार्ट खाली है। क्या ऑर्डर करेंगे?", cart, "ordering", "pending", 0)

        total = _money(_cart_total(cart, menu_by_id))
        reply = f"{_cart_summary(cart, menu_by_id)}, {total} रुपये। कन्फर्म कर दूँ?"
        return _out(history, reply, cart, "awaiting_confirmation", "pending", 0)

    if intent in ("add", "change_qty", "remove", "replace"):
        if intent == "add":
            new_cart, reply, has_problems = _do_add(ext.items, menu, menu_by_id, cart)
        elif intent == "change_qty":
            new_cart, reply, has_problems = _do_change_qty(ext.items, menu_by_id, cart)
        elif intent == "remove":
            new_cart, reply, has_problems = _do_remove(ext.items, menu_by_id, cart)
        else:
            new_cart, reply, has_problems = _do_replace(ext.items, menu, menu_by_id, cart)

        if new_cart is None:
            return _out(history, reply, cart, "ordering", "pending", retries + 1)

        turn_retries = (retries + 1) if has_problems else 0
        return _out(history, reply, new_cart, "ordering", "pending", turn_retries)

    if intent == "ask_menu":
        if not menu:
            reply = "अभी मेनू उपलब्ध नहीं है।"
        else:
            shown = ", ".join(_spoken(m) for m in menu[:8])
            more = " और भी कई आइटम हैं।" if len(menu) > 8 else ""
            reply = f"हमारे पास है: {shown}।{more} आप क्या लेना चाहेंगे?"
        return _out(history, reply, cart, "ordering", "pending", 0)

    if intent == "repeat_order":
        if not cart:
            reply = "कार्ट खाली है। क्या लेंगे?"
        else:
            total = _money(_cart_total(cart, menu_by_id))
            reply = f"कार्ट में है: {_cart_summary(cart, menu_by_id)}, कुल {total} रुपये। और कुछ?"
        return _out(history, reply, cart, "ordering", "pending", 0)

    if intent == "cancel":
        return _out(history, "ठीक है, ऑर्डर कैंसिल कर दिया।", cart, "closing", "cancelled", 0)

    if intent == "forward_to_owner":
        return _out(history, "एक पल, ओनर से बात कराने की कोशिश करता हूँ।", cart, "forward", "pending", 0)

    return _out(history, unclear_reply, cart, "ordering", "pending", retries + 1)


# --------------------------------------------------------------------------- #
# Public Entrypoint
# --------------------------------------------------------------------------- #
async def process_order_turn(state: OrderState, spoken_text: str, menu_items: Optional[list]) -> dict:
    spoken = (spoken_text or "").strip()
    menu = list(menu_items or [])
    cart = [dict(c) for c in (state.get("cart") or [])]
    history = list(state.get("conversation_history") or [])
    retries = state.get("retries", 0)
    details_retries = state.get("details_retries", 0)
    step = state.get("step", "ordering")

    if not spoken:
        return _out(history, "आवाज़ साफ़ नहीं आई। फिर से बोलिए?",
                    cart, step, state.get("confirmation_status", "pending"), retries, details_retries)

    history.append({"role": "user", "content": spoken})

    # 1. Cart Confirmation Stage
    if step == "awaiting_confirmation":
        fast_label = _fast_classify_confirmation(spoken)
        if fast_label:
            label = fast_label
        else:
            prompt = (
                f"A restaurant bot read this order back to the caller: {cart}.\n"
                f'The caller replied: "{spoken}"\n'
                "Classify reply into EXACTLY one of these tokens: confirm, cancel, modify, decline, or unclear. "
                "Output ONLY the single token without explanation or punctuation."
            )
            try:
                raw = await classify_confirmation(prompt)
                raw_clean = re.sub(r"[^\w]", "", str(raw)).strip().lower()
                if raw_clean == "confirm":
                    label = "confirm"
                elif raw_clean == "cancel":
                    label = "cancel"
                elif raw_clean in ("modify", "change"):
                    label = "modify"
                elif raw_clean in ("decline", "no"):
                    label = "decline"
                else:
                    label = "unclear"
            except Exception:
                return _out(history, "माफ़ कीजिए, आवाज़ कट रही थी। क्या आप एक बार फिर बोल सकते हैं?",
                            cart, "awaiting_confirmation", "pending", retries, details_retries)

        if label == "confirm":
            reply = "ऑर्डर नोट हो गया! डिलीवरी के लिए अपना नाम और पता बता दीजिए।"
            return _out(history, reply, cart, "collecting_details", "pending", 0, 0)
        if label == "cancel":
            return _out(history, "ठीक है, ऑर्डर कैंसिल कर दिया।", cart, "closing", "cancelled", 0, 0)
        if label == "decline":
            return _out(history, "ठीक है। कुछ बदलना है, या पूरा ऑर्डर कैंसिल करूँ?",
                        cart, "awaiting_confirmation", "pending", retries, details_retries)
        if label == "modify":
            return await _handle_ordering(spoken, cart, menu, history, retries, from_confirmation=True)

        return _out(history, "समझा नहीं। कन्फर्म कर दूँ? हाँ या ना?",
                    cart, "awaiting_confirmation", "pending", retries + 1, details_retries)

    # 2. Independent Name & Address Slot Filling
    if step == "collecting_details":
        try:
            extracted = await extract_delivery_details(spoken, history)
        except Exception as exc:
            logger.exception("Unexpected error in extract_delivery_details: %s", exc)
            extracted = CustomerDetailsExtraction(name=None, address=None)

        existing_name = state.get("customer_name")
        existing_address = state.get("delivery_address")

        updated_name = extracted.name or existing_name
        updated_address = extracted.address or existing_address

        if not updated_name and not updated_address:
            if details_retries + 1 >= MAX_DETAILS_RETRIES:
                return _out(history, "माफ़ कीजिए, डिलीवरी की जानकारी नहीं मिल पाई। कॉल समाप्त की जा रही है।",
                            cart, "closing", "cancelled", 0, details_retries + 1)
            return _out(history, "माफ़ कीजिए, डिलीवरी के लिए आपका नाम और पूरा पता चाहिए। कृपया दोनों बताइए।",
                        cart, "collecting_details", "pending", 0, details_retries + 1)

        if updated_name and not updated_address:
            if details_retries + 1 >= MAX_DETAILS_RETRIES:
                return _out(history, "माफ़ कीजिए, बिना डिलीवरी पते के ऑर्डर नहीं भेजा जा सकता। कॉल समाप्त की जा रही है।",
                            cart, "closing", "cancelled", 0, details_retries + 1)
            res = _out(history, f"धन्यवाद {updated_name}! अब कृपया अपना डिलीवरी पता बता दीजिए।",
                       cart, "collecting_details", "pending", 0, details_retries + 1)
            res["customer_name"] = updated_name
            return res

        if updated_address and not updated_name:
            if details_retries + 1 < MAX_DETAILS_RETRIES:
                res = _out(history, "पता नोट कर लिया। कृपया अपना नाम भी बता दीजिए।",
                           cart, "collecting_details", "pending", 0, details_retries + 1)
                res["delivery_address"] = updated_address
                return res

        if not updated_address:
            return _out(history, "माफ़ कीजिए, डिलीवरी के लिए पता अनिवार्य है। कृपया अपना पता बताइए।",
                        cart, "collecting_details", "pending", 0, details_retries + 1)

        res = _out(
            history,
            "धन्यवाद! आपका ऑर्डर दर्ज किया जा रहा है।",
            cart,
            "closing",
            "pending",
            0,
            0
        )
        res["customer_name"] = updated_name
        res["delivery_address"] = updated_address
        return res

    return await _handle_ordering(spoken, cart, menu, history, retries)