"""Menu matching: map a spoken/extracted item name to a menu entry.

Order of resolution (first stage that yields a result wins):
  1. exact canonical-name match
  2. exact normalized alias match
  3. token-subset match ("naan" -> every menu entry containing the token "naan")
  4. unique, clearly-best fuzzy match
  5. otherwise: ambiguous (several plausible) or none

Pure Python (difflib only) -- no external dependency, no I/O.
"""
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional

FUZZY_MIN_SCORE = 0.70   # best candidate must score at least this
FUZZY_MARGIN = 0.15      # ...and beat the runner-up by at least this much


def normalize(text: Optional[str]) -> str:
    """Lowercase, drop punctuation/symbols (keeps Devanagari marks), collapse spaces."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    cleaned = "".join(
        " " if (unicodedata.category(c)[0] in "PS" or c == "_") else c for c in text
    )
    return " ".join(cleaned.split())


@dataclass
class MatchResult:
    status: str  # "match" | "ambiguous" | "none"
    item: Optional[dict] = None
    candidates: List[dict] = field(default_factory=list)

    @property
    def is_match(self) -> bool:
        return self.status == "match"


def _forms(item: dict) -> List[str]:
    """Normalized canonical name followed by normalized aliases."""
    forms = [normalize(item.get("name"))]
    forms += [normalize(a) for a in (item.get("aliases") or [])]
    return [f for f in forms if f]


def _unique(items: List[dict]) -> List[dict]:
    seen, out = set(), []
    for it in items:
        if it["id"] not in seen:
            seen.add(it["id"])
            out.append(it)
    return out


def _result(cands: List[dict]) -> Optional[MatchResult]:
    cands = _unique(cands)
    if len(cands) == 1:
        return MatchResult("match", cands[0], cands)
    if len(cands) > 1:
        return MatchResult("ambiguous", None, cands)
    return None


def resolve_item(query: str, menu: List[dict]) -> MatchResult:
    q = normalize(query)
    if not q or not menu:
        return MatchResult("none")

    # 1. exact canonical name
    hit = _result([m for m in menu if normalize(m.get("name")) == q])
    if hit:
        return hit

    # 2. exact alias
    hit = _result([m for m in menu if q in [normalize(a) for a in (m.get("aliases") or [])]])
    if hit:
        return hit

    # 3. every query token appears in one of the item's forms
    q_tokens = set(q.split())
    hit = _result([m for m in menu if any(q_tokens <= set(f.split()) for f in _forms(m))])
    if hit:
        return hit

    # 4. fuzzy
    scored = []
    for m in menu:
        best = max((SequenceMatcher(None, q, f).ratio() for f in _forms(m)), default=0.0)
        scored.append((best, m))
    scored.sort(key=lambda s: s[0], reverse=True)
    best_score, best_item = scored[0]
    if best_score < FUZZY_MIN_SCORE:
        return MatchResult("none")
    close = [m for s, m in scored if s >= FUZZY_MIN_SCORE and best_score - s < FUZZY_MARGIN]
    if len(close) == 1:
        return MatchResult("match", best_item, [best_item])
    return MatchResult("ambiguous", None, close)
