"""What new POS rows say about articles (KATALOG-1 ticket 12, decision D1). Pure functions, no database.

  known code + name            -> nothing
  known code, similar new name -> "rename": tracked under the old class, marked "na proveri", one click to confirm
  known code, different name   -> "code_reused": alarm, the new name is not tracked until someone decides
  new code, known name         -> "second_code": same article, marked "na proveri"
  new code, new name           -> "new_article": goes to the labeler
  same code, unit price jumps  -> "price_jump": alarm (an exact double or half is a 2-piece row or a half portion, not a new article)

Rename rule, calibrated on the 9 real renames in the POS history (all have every word of the shorter name in the longer
one; plain similarity is as low as 0.63 for ASPALL -> ASPALL JABUKA): words contained >= 0.8 or similarity >= 0.85.
"""
from __future__ import annotations

import difflib
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

WORDS_CONTAINED = 0.8
SIMILAR = 0.85
PRICE_JUMP = 0.5


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().upper())


def _words(name: str) -> List[str]:
    return re.findall(r"[A-ZČĆŠĐŽ0-9.,]+", norm(name))


def similarity(a: str, b: str) -> Tuple[float, float]:
    """(plain similarity, share of the shorter name's words found in the longer one)."""
    plain = difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
    wa, wb = _words(a), _words(b)
    small, big = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    found = sum(1 for w in small if any(difflib.SequenceMatcher(None, w, u).ratio() >= 0.8 for u in big))
    return plain, found / max(1, len(small))


def is_rename(old: str, new: str) -> bool:
    plain, contained = similarity(old, new)
    return contained >= WORDS_CONTAINED or plain >= SIMILAR


@dataclass
class Known:
    names_by_code: Dict[str, Set[str]] = field(default_factory=dict)   # code -> names as written
    codes_by_name: Dict[str, Set[str]] = field(default_factory=dict)   # normalized name -> codes
    unit_price: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (code, normalized name) -> last unit price

    def add(self, code: str, name: str, price: Optional[float] = None) -> None:
        self.names_by_code.setdefault(code, set()).add(name.strip())
        self.codes_by_name.setdefault(norm(name), set()).add(code)
        if price:
            self.unit_price[(code, norm(name))] = price


def _price_jump(old: float, new: float) -> bool:
    if old <= 0 or new <= 0 or abs(new - old) / old <= PRICE_JUMP:
        return False
    ratio = new / old
    return not (abs(ratio - 2) < 0.02 or abs(ratio - 0.5) < 0.005)


def detect(known: Known, sales: Dict[Tuple[str, str], Sequence[float]]) -> List[Dict[str, object]]:
    """sales: (code, name as written) -> unit prices in the new report. Returns findings, best sellers first."""
    out: List[Dict[str, object]] = []
    for (code, name), prices in sorted(sales.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        name = name.strip()
        price = statistics.median(prices) if prices else 0.0
        names = known.names_by_code.get(code, set())
        if any(norm(n) == norm(name) for n in names):
            old_price = known.unit_price.get((code, norm(name)))
            if old_price and _price_jump(old_price, price):
                out.append({"kind": "price_jump", "code": code, "name": name, "old_price": old_price, "new_price": price,
                            "sold": len(prices)})
            continue
        if names:
            similar = sorted((n for n in names if is_rename(n, name)), key=lambda n: -similarity(n, name)[0])
            kind = "rename" if similar else "code_reused"
            out.append({"kind": kind, "code": code, "name": name, "old_name": (similar or sorted(names))[0],
                        "old_names": sorted(names), "sold": len(prices), "new_price": price})
        elif known.codes_by_name.get(norm(name)):
            out.append({"kind": "second_code", "code": code, "name": name,
                        "other_codes": sorted(known.codes_by_name[norm(name)]), "sold": len(prices), "new_price": price})
        else:
            out.append({"kind": "new_article", "code": code, "name": name, "sold": len(prices), "new_price": price})
    return out
