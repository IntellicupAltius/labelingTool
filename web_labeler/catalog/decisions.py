"""Manual decisions on top of the imported sources.

Decisions are only ever appended. The current picture of every POS variation is worked out on read:
first what the sources say, then every decision that is in force on the given day, oldest first.
That is why a decision survives every later import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

EFFECT_KINDS = ("classes", "provisional", "ignore", "merge", "class_target")
KINDS = EFFECT_KINDS + ("revoke", "propose", "reject")
# Owner 24.11.2026 (ticket 20): whoever is at the tool decides, one pass; the role only opens admin screens.
LABELER_KINDS = set(KINDS)
MODES = ("fallback", "any")
IGNORE_WHY = {"assortment": "izbačeno iz asortimana"}  # X with a kind of reason; no kind = plain X

KIND_LABELS = {"classes": "Promena klase", "provisional": "Zvezdica", "ignore": "Ne pratimo (X)", "merge": "Spajanje artikala",
               "revoke": "Poništavanje", "propose": "Predlog", "reject": "Odbijen predlog",
               "class_target": "Klasa za novi artikal"}


@dataclass
class Variation:
    identity_id: int
    article_id: int
    code: Optional[str]
    name: str
    state: str
    sold: float
    last_sale: Optional[str]
    not_verified: bool
    provisional: Optional[str]
    mode: Optional[str]
    links: List[Tuple[str, str]] = field(default_factory=list)
    decided: bool = False
    source_article_id: int = 0
    last_decision: int = 0
    ignore_why: Optional[str] = None
    awaiting_release: bool = False
    under_review: bool = False  # "na proveri": tracked because of a till finding nobody confirmed yet
    # class_target (D9): the class this variation goes to once the live model knows it; mapped only from then on
    target: Optional[Tuple[str, str]] = None
    target_opened: bool = False


@dataclass
class Decision:
    decision_id: int
    kind: str
    article_id: Optional[int]
    identity_id: Optional[int]
    payload: Dict[str, Any]
    valid_from: str
    target: Optional[int]


def revoked_ids(decisions: Iterable[Decision]) -> set:
    return {d.target for d in decisions if d.kind == "revoke" and d.target is not None}


def in_force(decisions: List[Decision], as_of: str) -> List[Decision]:
    """Effect decisions not revoked and already valid on as_of, in the order they take effect."""
    gone = revoked_ids(decisions)
    active = [d for d in decisions if d.kind in EFFECT_KINDS and d.decision_id not in gone and d.valid_from <= as_of]
    return sorted(active, key=lambda d: (d.valid_from, d.decision_id))


def merge_map(active: Iterable[Decision]) -> Dict[int, int]:
    return {d.article_id: int(d.payload["into"]) for d in active if d.kind == "merge"}


def resolve(article_id: int, merges: Mapping[int, int]) -> int:
    seen = set()
    while article_id in merges and article_id not in seen:
        seen.add(article_id)
        article_id = merges[article_id]
    return article_id


def apply(variations: Dict[int, Variation], decisions: List[Decision], as_of: str) -> Dict[int, int]:
    """Change variations in place. Returns the merges in force (article -> article it now belongs to)."""
    active = in_force(decisions, as_of)
    merges = merge_map(active)
    for v in variations.values():
        v.source_article_id = v.article_id
        v.article_id = resolve(v.article_id, merges)
    for d in active:
        if d.kind == "classes":
            v = variations.get(d.identity_id)
            if v is None:
                continue
            v.links = [(link["model"], link["class"]) for link in d.payload.get("links", [])]
            v.mode = d.payload.get("mode") or "fallback"
            v.provisional = d.payload.get("provisional") or None
            v.state = "tracked" if v.links else "pending"
            v.not_verified = bool(v.links)
            v.target, v.target_opened = None, False
            v.decided, v.last_decision = True, max(v.last_decision, d.decision_id)
        elif d.kind == "class_target":
            v = variations.get(d.identity_id)
            if v is None:
                continue
            # no mapping until the live model knows the class (resolved by the caller, which knows the live model)
            v.target, v.target_opened = (d.payload["model"], d.payload["class"]), bool(d.payload.get("opened"))
            v.links, v.mode, v.state, v.not_verified = [], None, "pending", False
            v.decided, v.last_decision = True, max(v.last_decision, d.decision_id)
        elif d.kind == "provisional":
            v = variations.get(d.identity_id)
            if v is not None:
                v.provisional = d.payload.get("provisional") or None
                v.decided, v.last_decision = True, max(v.last_decision, d.decision_id)
        elif d.kind == "ignore":
            target = resolve(d.article_id, merges)
            for v in variations.values():
                if v.article_id == target:
                    v.state, v.links, v.decided = "ignored", [], True
                    v.target, v.target_opened = None, False
                    v.ignore_why = d.payload.get("why") or None
                    v.last_decision = max(v.last_decision, d.decision_id)
        elif d.kind == "merge":
            for v in variations.values():
                if v.source_article_id == d.article_id:
                    v.decided, v.last_decision = True, max(v.last_decision, d.decision_id)
    return merges
