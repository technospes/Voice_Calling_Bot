"""Semantic order-relevance gate (semantic filtering only; no speaker/voice analysis).

    Deepgram FINAL_TRANSCRIPT -> evaluate() -> accepted cleaned_text -> order_agent (LangGraph)

Stage A (deterministic): accepts ONLY utterances made entirely of quantity words, polite
    order filler, and menu vocabulary where every item phrase resolves via
    ``matching.resolve_item``; explicit negative closure answering 'aur kuch?';
    customer names during details collection; plus whole-utterance confirmations, but
    only while the state is ``awaiting_confirmation``. Bare affirmatives in ordering
    are rejected deterministically as AMBIGUOUS. Anything else goes to Stage B.
Stage B (Groq): reuses ``nodes._get_llm`` for a structured classification, then
    strictly lexically grounds ``cleaned_text`` in the original transcript using
    monotonic one-to-one token alignment.

Every failure path returns AMBIGUOUS / cleaned_text=None / confidence=0.0, i.e. nothing
reaches LangGraph.
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, List, Literal, Optional

from pydantic import BaseModel, Field

from app.agent.matching import normalize, resolve_item

logger = logging.getLogger("relevance-gate")

Category = Literal[
    "ORDER_ACTION", "ORDER_DETAIL", "ORDER_QUESTION", "CONFIRMATION", "UNRELATED", "AMBIGUOUS",
]

RELEVANT_CATEGORIES = frozenset({"ORDER_ACTION", "ORDER_DETAIL", "ORDER_QUESTION", "CONFIRMATION"})
MIN_CONFIDENCE = 0.70
GATE_LLM_TIMEOUT_SEC = 4.0
FAST_PATH_MAX_TOKENS = 14
CONFIRMATION_MAX_TOKENS = 6
PROMPT_HISTORY = 6

# Categories the existing pipeline can meaningfully consume at each step.
# Strictly preserves disallowing CONFIRMATION in 'ordering'.
_ALLOWED_BY_STEP = {
    "ordering": frozenset({"ORDER_ACTION", "ORDER_DETAIL", "ORDER_QUESTION"}),
    "awaiting_confirmation": RELEVANT_CATEGORIES,
    "collecting_details": frozenset({"ORDER_ACTION", "ORDER_DETAIL", "ORDER_QUESTION"}),
}


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
class LLMGateOutput(BaseModel):
    category: Category
    cleaned_text: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


@dataclass
class GateResult:
    category: Category
    cleaned_text: Optional[str]
    confidence: float
    reason: str
    stage: str = "none"  # fast | llm | none
    failed: bool = False

    @property
    def is_relevant(self) -> bool:
        return (
            self.category in RELEVANT_CATEGORIES
            and bool(self.cleaned_text and self.cleaned_text.strip())
            and self.confidence >= MIN_CONFIDENCE
        )


def _reject(category: Category, reason: str, stage: str = "none") -> GateResult:
    return GateResult(category, None, 0.0, reason, stage)


def _ambiguous(reason: str, stage: str = "none") -> GateResult:
    return _reject("AMBIGUOUS", reason, stage)


def _failure(reason: str, stage: str = "llm") -> GateResult:
    return GateResult("AMBIGUOUS", None, 0.0, reason, stage, failed=True)


# --------------------------------------------------------------------------- #
# Tokenisation / Canonicalisation / Cross-Script Alignment
# --------------------------------------------------------------------------- #
_NUMBER_WORDS = {
    "ek": "1", "एक": "1", "one": "1", "do": "2", "दो": "2", "two": "2",
    "teen": "3", "तीन": "3", "three": "3", "char": "4", "chaar": "4", "चार": "4", "four": "4",
    "paanch": "5", "panch": "5", "पांच": "5", "पाँच": "5", "five": "5",
    "chhe": "6", "chha": "6", "che": "6", "छह": "6", "छः": "6", "six": "6",
    "saat": "7", "सात": "7", "seven": "7", "aath": "8", "आठ": "8", "eight": "8",
    "nau": "9", "नौ": "9", "nine": "9", "das": "10", "दस": "10", "ten": "10",
}
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_DEVANAGARI_TRANSLIT = {
    # Functional verbs & affirmative/negation words
    "कर": "kar", "दो": "do", "दीजिए": "dijiye", "देना": "dena", "दीजिये": "dijiye",
    "चाहिए": "chahiye", "हाँ": "haan", "हां": "haan", "नहीं": "nahi", "हटाओ": "hatao",
    "हटा": "hata", "कम": "kam", "बढ़ाओ": "badhao", "जोड़ो": "jodo", "ऑर्डर": "order",
    "कन्फर्म": "confirm", "बस": "bas", "इतना": "itna", "ही": "hi", "और": "aur",
    "कुछ": "kuch", "लूंगा": "lunga", "लूँगा": "lunga", "लेंगे": "lenge",
    # Common restaurant vocabulary
    "बटर": "butter", "नान": "naan", "पनीर": "paneer", "मसाला": "masala",
    "रोटी": "roti", "तंदूरी": "tandoori", "दाल": "dal", "मखनी": "makhani",
    "चावल": "chawal", "चावलों": "chawal", "कॉफी": "coffee", "चाय": "chai",
    "टिक्का": "tikka", "ग्रेवी": "gravy", "कोल्ड": "cold",
}

_FILLER = {
    "dena", "de", "dijiye", "dijie", "dijiyega", "chahiye", "chahie", "aur", "and", "plus",
    "bhi", "please", "pls", "mujhe", "hume", "humein", "kar", "karo", "lena", "lo", "la",
    "bhej", "bhejna", "order", "add", "ji", "mere", "liye", "kardo", "karo", "do",
    "देना", "दीजिए", "दीजिये", "चाहिए", "और", "भी", "कृपया", "मुझे", "कर", "करो", "जी", "लो", "ऐड", "डाल",
}
_REFERENTIAL = re.compile(
    r"\b(aur|woh|wo|usko|isko|hata|hatao|remove|more|same)\b|(और|वो|वह|उसको|इसको|हटा|हटाओ)",
    re.IGNORECASE,
)
_SPLIT_ITEMS = re.compile(r"[,;]|\baur\b|\band\b|\bplus\b|और", re.IGNORECASE)

_NAME_FAST_PATTERN = re.compile(
    r"^(mera\s+naam|naam\s+hai|my\s+name\s+is|this\s+is|मेरा\s*नाम|नाम\s*है)\s+[a-zA-Z\u0900-\u097F]+",
    re.IGNORECASE,
)

_NON_NAME_WORDS = frozenset({
    "theek", "thik", "ok", "okay", "haan", "ha", "haa", "yes", "yep", "yeah", "nahi", "nahin",
    "na", "no", "kya", "kyu", "kyun", "kaise", "kab", "kaha", "kahan", "kitna", "kitne",
    "hello", "namaste", "suno", "arre", "bhai", "ji", "sir", "madam",
    "cancel", "order", "confirm", "stop", "wait", "ruko",
    "ठीक", "हाँ", "हां", "नहीं", "ना", "क्या", "नमस्ते", "सुनो", "भाई", "जी",
})

_NEGATIVE_CLOSURE_WORDS = frozenset({
    "nahi", "nahin", "na", "no", "nope", "nahi ji", "ji nahi",
    "नहीं", "ना", "नहीं जी", "जी नहीं", "बस", "bas", "bas itna",
})

_STANDALONE_AFFIRMATIVE_WORDS = frozenset({
    "haan", "han", "ha", "haa", "yes", "yep", "yeah", "ji haan", "haan ji", "ji",
    "हाँ", "हां", "जी", "जी हाँ", "हाँ जी", "यस",
})


def _tokens(text: Optional[str]) -> List[str]:
    return normalize(text).split()


def _canon(tok: str) -> str:
    t = tok.translate(_DEVANAGARI_DIGITS)
    return _NUMBER_WORDS.get(t, t)


def _canonical_token(tok: str) -> str:
    """Normalizes digits, number words, and Devanagari transliteration."""
    t = _canon(tok)
    return _DEVANAGARI_TRANSLIT.get(t, t)


def _is_number(tok: str) -> bool:
    return _canon(tok).isdigit()


def _menu_vocab(menu: List[dict]) -> set:
    vocab = set()
    for item in menu or []:
        for form in [item.get("name")] + list(item.get("aliases") or []):
            vocab.update(_tokens(form))
    return vocab


def _is_plausible_bare_name(text: str) -> bool:
    """Checks if a short token string can be a standalone person's name."""
    clean = text.strip()
    toks = clean.split()
    if not (1 <= len(toks) <= 3):
        return False
    for t in toks:
        stripped = re.sub(r"[^\w\u0900-\u097F]", "", t).lower()
        if not stripped or stripped in _NON_NAME_WORDS or _is_number(stripped):
            return False
        if not re.match(r"^[a-zA-Z\u0900-\u097F]+$", stripped):
            return False
    return True


def _is_assistant_asking_more(history: List[dict]) -> bool:
    """Checks if the assistant's last spoken turn asked 'aur kuch?' / 'anything else?'."""
    if not history:
        return False
    last_assistant_msg = ""
    for msg in reversed(history):
        if msg.get("role") in ("assistant", "agent"):
            last_assistant_msg = msg.get("content", "")
            break
    norm = normalize(last_assistant_msg)
    return any(p in norm for p in ("aur kuch", "kuch aur", "or kuch", "और कुछ", "कुछ और"))


# --------------------------------------------------------------------------- #
# Stage A: deterministic fast path
# --------------------------------------------------------------------------- #
def _fast_path(text: str, state: dict, menu: List[dict]) -> Optional[GateResult]:
    step = state.get("step", "ordering")
    toks = _tokens(text)
    if not toks:
        return None

    clean_text = text.strip()
    history = state.get("conversation_history") or []

    # Fast-Path 1: Bare affirmative during ordering without confirmation context is AMBIGUOUS
    if step == "ordering" and normalize(clean_text) in _STANDALONE_AFFIRMATIVE_WORDS:
        return GateResult("AMBIGUOUS", None, 0.0, "fast:bare_affirmation_in_ordering", "fast")

    # Fast-Path 2: Answering "aur kuch?" with "nahi" during ordering -> ORDER_ACTION (checkout)
    if step == "ordering" and state.get("cart"):
        if _is_assistant_asking_more(history) and normalize(clean_text) in _NEGATIVE_CLOSURE_WORDS:
            return GateResult("ORDER_ACTION", clean_text, 0.98, "fast:negative_checkout", "fast")

    if step == "collecting_details":
        if _NAME_FAST_PATTERN.search(clean_text) and len(toks) <= 6:
            return GateResult("ORDER_DETAIL", clean_text, 0.98, "fast:name_declaration", "fast")
        if not state.get("customer_name") and _is_plausible_bare_name(clean_text):
            return GateResult("ORDER_DETAIL", clean_text, 0.95, "fast:bare_name", "fast")
        return None

    if step == "awaiting_confirmation":
        from app.agent.nodes import _fast_classify_confirmation
        if len(toks) <= CONFIRMATION_MAX_TOKENS:
            label = _fast_classify_confirmation(text)
            if label:
                return GateResult("CONFIRMATION", clean_text, 0.95, f"fast:{label}", "fast")
        return None

    if step != "ordering" or len(toks) > FAST_PATH_MAX_TOKENS or not menu:
        return None

    vocab = _menu_vocab(menu)
    saw_item = False
    for chunk in _SPLIT_ITEMS.split(text):
        ctoks = _tokens(chunk)
        phrase = [t for t in ctoks if not _is_number(t) and t not in _FILLER]
        if not ctoks:
            continue
        if not phrase:
            continue
        if any(t not in vocab for t in phrase):
            return None
        if resolve_item(" ".join(phrase), menu).status == "none":
            return None
        saw_item = True
    if not saw_item:
        return None
    return GateResult("ORDER_ACTION", clean_text, 0.95, "fast:menu_only", "fast")


# --------------------------------------------------------------------------- #
# Lexical Grounding: Monotonic One-to-One Alignment
# --------------------------------------------------------------------------- #
def _token_match(clean_tok: str, orig_tok: str) -> bool:
    """Evaluates strict numeric equality or conservative non-numeric drift."""
    c_clean = _canonical_token(clean_tok)
    c_orig = _canonical_token(orig_tok)

    if c_clean == c_orig:
        return True

    # Numbers and quantities must be exact 1:1 matches; no fuzzy tolerance
    if _is_number(clean_tok) or _is_number(orig_tok):
        return False

    # Short tokens (< 4 chars) must match exactly
    if len(clean_tok) < 4 or len(orig_tok) < 4:
        return False

    # Non-numeric spelling drift capped at >= 0.85 similarity
    return SequenceMatcher(None, clean_tok, orig_tok).ratio() >= 0.85


def is_grounded(cleaned: str, original: str) -> bool:
    """Validates that 'cleaned' is a strict, ordered, one-to-one supported subsequence of 'original'.

    Rejects:
    - Duplicated token hallucinations (e.g., 'paneer paneer' from 'paneer naan')
    - Token order inversions (e.g., 'naan paneer' from 'paneer naan')
    - Altered quantities (e.g., '3 naan' from '2 naan')
    """
    clean_toks = _tokens(cleaned)
    orig_toks = _tokens(original)

    if not clean_toks or len(clean_toks) > len(orig_toks):
        return False

    orig_idx = 0
    clean_idx = 0

    while clean_idx < len(clean_toks) and orig_idx < len(orig_toks):
        if _token_match(clean_toks[clean_idx], orig_toks[orig_idx]):
            clean_idx += 1
            orig_idx += 1  # Consume original token; guarantees one-to-one consumption
        else:
            orig_idx += 1  # Skip unextracted / side-talk token

    return clean_idx == len(clean_toks)


# --------------------------------------------------------------------------- #
# Stage B: Groq semantic extraction
# --------------------------------------------------------------------------- #
_structured_gate = None


def _get_gate_llm():
    global _structured_gate
    if _structured_gate is None:
        from app.agent.nodes import _get_llm
        for method in ("json_schema", "function_calling"):
            try:
                _structured_gate = _get_llm().with_structured_output(LLMGateOutput, method=method)
                break
            except Exception as exc:
                logger.debug("Gate structured output via %s unavailable: %s", method, type(exc).__name__)
        if _structured_gate is None:
            raise RuntimeError("No structured-output method available for relevance gate")
    return _structured_gate


async def _default_llm_call(prompt: str) -> Any:
    return await _get_gate_llm().ainvoke(prompt)


def build_prompt(transcript: str, state: dict, menu: List[dict]) -> str:
    menu_lines = "\n".join(
        f"- {m.get('name')}" + (f" | aliases: {', '.join(m.get('aliases') or [])}" if m.get("aliases") else "")
        for m in menu or []
    ) or "(menu empty)"
    cart = state.get("cart") or []
    cart_lines = "\n".join(f"- {c.get('quantity')} x {c.get('name')}" for c in cart) or "(empty)"
    history = (state.get("conversation_history") or [])[-PROMPT_HISTORY:]
    convo = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in history) or "(start of call)"
    current_step = state.get("step", "ordering")

    return f"""You filter speech heard on a restaurant ordering phone line (Hindi / Hinglish / English).
The line may also pick up background TV, office, or domestic chatter. Anyone's order-related speech is valid;
completely unrelated background chatter must be dropped. Do not try to identify speakers.

CURRENT DIALOGUE STEP: {current_step}
  - ordering: Taking food items, modifications, or concluding cart.
  - awaiting_confirmation: Bot just read back items/total and asked for confirmation.
  - collecting_details: Bot specifically asked for the customer's name and/or delivery address.

CURRENT CART:
{cart_lines}

RECENT CONVERSATION (oldest first):
{convo}

TRANSCRIPT TO EVALUATE: "{transcript}"

CRITICAL CATEGORY RULES:
1. ORDER_ACTION:
   - Adding, modifying, replacing, or changing item quantities.
   - CHECKOUT INTENT: Direct requests to finish ("order confirm kar do", "bas itna hi") OR negative answers to "aur kuch?" ("nahi", "aur kuch nahi").
   - Cancellation ("cancel kar do").
   - A bare "haan" or "yes" alone during 'ordering' does not specify an action and is AMBIGUOUS.

2. ORDER_DETAIL:
   - Any response providing customer identity or location details (names, addresses, areas, buildings, landmarks, delivery instructions).
   - CONTEXTUAL PRIORITY IN 'collecting_details':
     * When CURRENT DIALOGUE STEP is 'collecting_details', any substantive utterance spoken in response to the bot's request for name/address MUST be classified as ORDER_DETAIL unless it is indisputable third-party background speech (e.g. talking about unrelated office work, TV audio, or weather).
     * Speech-to-text engines frequently mistranscribe spoken locations, accents, or street names (e.g., mishearing "Knowledge Park" as "Knowledge back", "bag", "bark", or foreign words). Do NOT classify misheard or phonetically garbled location/name fragments as UNRELATED if they are the customer's direct reply to the bot.

3. ORDER_QUESTION:
   - Inquiries about food items, recommendations, prices, or asking for the total/bill ("bill kitna hua?").

4. CONFIRMATION:
   - Answering the bot's direct order confirmation question ("...kya confirm kar doon?").
   - Valid ONLY when CURRENT DIALOGUE STEP is 'awaiting_confirmation'.

5. UNRELATED:
   - Third-party background chatter that has no conversational intent with the bot (e.g., "Papa phone kahan hai", "Turn off the TV", "Meeting kab hai").

6. AMBIGUOUS:
   - Isolated tokens or fragments that cannot be interpreted even with recent dialogue context.

CLEANED_TEXT RULES (CRITICAL):
1. For ANY relevant category (ORDER_ACTION, ORDER_DETAIL, ORDER_QUESTION, CONFIRMATION):
   - You MUST provide `cleaned_text`. It must NEVER be null or empty.
   - Copy the exact relevant words from the transcript in their original sequence.
   - For ORDER_ACTION: Copy the order instructions and items verbatim (e.g. "एक पनीर butter मसाला और दो butter नान add कर दो", "order confirm kar do", "nahi").
   - For ORDER_DETAIL: Copy the name or address phrase verbatim.
   - For ORDER_QUESTION: Copy the inquiry words verbatim.
   - For CONFIRMATION: Copy the confirmation response verbatim ("haan", "confirm kar do").
2. Drop ONLY completely unrelated background side-talk or noise.
3. For UNRELATED or AMBIGUOUS: Return null or empty string for `cleaned_text`.
"""

def _coerce(raw: Any) -> LLMGateOutput:
    if isinstance(raw, LLMGateOutput):
        return raw
    if isinstance(raw, dict):
        return LLMGateOutput.model_validate(raw)
    content = getattr(raw, "content", raw)
    if isinstance(content, str):
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return LLMGateOutput.model_validate(json.loads(match.group(0)))
    raise ValueError("unparseable gate output")


async def _llm_stage(
    transcript: str, state: dict, menu: List[dict], llm_call: Callable[[str], Awaitable[Any]]
) -> GateResult:
    try:
        raw = await asyncio.wait_for(llm_call(build_prompt(transcript, state, menu)), GATE_LLM_TIMEOUT_SEC)
        out = _coerce(raw)
    except asyncio.TimeoutError:
        logger.warning("Relevance gate LLM timed out after %.1fs; treating as AMBIGUOUS.", GATE_LLM_TIMEOUT_SEC)
        return _failure("llm_timeout")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Relevance gate LLM failed (%s); treating as AMBIGUOUS.", type(exc).__name__)
        return _failure(f"llm_error:{type(exc).__name__}")

    cleaned = (out.cleaned_text or "").strip()
    if out.category not in RELEVANT_CATEGORIES:
        return GateResult(out.category, None, out.confidence, out.reason[:120], "llm")
    if not cleaned:
        return _ambiguous("empty_cleaned_text", "llm")
    if out.confidence < MIN_CONFIDENCE:
        return _ambiguous(f"low_confidence:{out.confidence:.2f}", "llm")

    step = state.get("step", "ordering")
    if out.category not in _ALLOWED_BY_STEP.get(step, frozenset()):
        return _ambiguous(f"category_{out.category}_not_valid_in_{step}", "llm")
    if not is_grounded(cleaned, transcript):
        logger.warning("Relevance gate rejected ungrounded extraction.")
        return _failure("ungrounded_extraction")

    # Referential phrases ("ek aur", "woh hata do") need cart context to mean anything.
    if (
        step == "ordering"
        and not (state.get("cart") or [])
        and _REFERENTIAL.search(normalize(cleaned))
        and not any(t in _menu_vocab(menu) for t in _tokens(cleaned))
    ):
        return _ambiguous("referential_phrase_with_empty_cart", "llm")

    return GateResult(out.category, cleaned, out.confidence, out.reason[:120], "llm")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
async def evaluate(
    transcript: Optional[str],
    state: dict,
    menu: Optional[List[dict]],
    llm_call: Optional[Callable[[str], Awaitable[Any]]] = None,
) -> GateResult:
    text = (transcript or "").strip()
    if not text or not _tokens(text):
        return _reject("UNRELATED", "empty_transcript")
    if state.get("step") == "closing":
        return _ambiguous("closing_state")
    if state.get("step") not in _ALLOWED_BY_STEP:
        return _ambiguous(f"step_not_gated:{state.get('step')}")

    menu = list(menu or [])
    fast = _fast_path(text, state, menu)
    if fast is not None:
        return fast
    return await _llm_stage(text, state, menu, llm_call or _default_llm_call)


@dataclass
class TurnOutcome:
    gate: GateResult
    new_state: Optional[dict] = field(default=None)


async def run_gated_turn(
    user_text: str,
    state: dict,
    menu: Optional[List[dict]],
    invoke_agent: Callable[[dict], Awaitable[dict]],
    on_relevant: Optional[Callable[[], Awaitable[None]]] = None,
    llm_call: Optional[Callable[[str], Awaitable[Any]]] = None,
) -> TurnOutcome:
    """Gate, then (only if relevant) run the existing pipeline.

    Caller must hold its turn lock. ``on_relevant`` runs only for accepted speech and is where
    the worker refreshes the inactivity clock and cancels the obsolete idle prompt. Rejected
    speech never touches ``state``, never calls ``invoke_agent``, never calls ``on_relevant``.
    """
    gate = await evaluate(user_text, state, menu, llm_call)
    if state.get("step") == "closing" or not gate.is_relevant:
        return TurnOutcome(gate, None)
    if on_relevant is not None:
        await on_relevant()
    state["latest_speech"] = gate.cleaned_text
    new_state = await invoke_agent(state)
    return TurnOutcome(gate, new_state)