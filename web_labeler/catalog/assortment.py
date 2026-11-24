"""Read today's POS reports, article map and blacklist into one import plan. Never writes."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml

TRACKED, IGNORED, PENDING, DORMANT = "tracked", "ignored", "pending", "dormant"
Pair = Tuple[str, str]


class AssortmentError(Exception):
    pass


def normalize_name(name: str) -> str:
    """Same comparison the matching pipeline uses: outer spaces and letter case do not count."""
    return str(name).strip().upper()


@dataclass(frozen=True)
class PosRow:
    code: str
    name: str
    ring_time: datetime
    quantity: float
    price: float
    row_key: Tuple[str, ...]


@dataclass
class PosRead:
    rows: List[PosRow] = field(default_factory=list)
    rows_read: int = 0
    unpaid: int = 0
    duplicates_removed: int = 0
    bad_rows: List[str] = field(default_factory=list)


def _number(text: str) -> float:
    return float(str(text).replace(",", ""))


def read_blaznavac_csv(path: Path) -> Tuple[List[PosRow], int, int, List[str]]:
    """Columns: 16 ring time, 23 code, 24 name, 25 quantity, 26 price. Codes stay text."""
    rows: List[PosRow] = []
    total = unpaid = 0
    bad: List[str] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for line_no, row in enumerate(csv.reader(handle), 1):
            total += 1
            if "NENAPLACENO" in row:
                unpaid += 1
                continue
            try:
                rows.append(PosRow(row[23], row[24], datetime.strptime(row[16], "%Y.%m.%d %H:%M"),
                                   _number(row[25]), _number(row[26]), tuple(row[16:28])))
            except (IndexError, ValueError):
                bad.append(f"{path.name}:{line_no}")
    return rows, total, unpaid, bad


POS_ADAPTERS = {"blaznavac_csv": ("*.csv", read_blaznavac_csv)}


def read_pos_folder(folder: Path, adapter: str) -> PosRead:
    if adapter not in POS_ADAPTERS:
        raise AssortmentError(f"Unknown POS adapter: {adapter}")
    pattern, reader = POS_ADAPTERS[adapter]
    files = sorted(Path(folder).glob(pattern)) if Path(folder).is_dir() else []
    if not files:
        raise AssortmentError(f"No POS reports found in {folder}")
    result = PosRead()
    kept: Dict[Tuple[str, ...], int] = Counter()
    sample: Dict[Tuple[str, ...], PosRow] = {}
    seen_rows = 0
    for path in files:
        try:
            rows, total, unpaid, bad = reader(path)
        except (OSError, UnicodeDecodeError, csv.Error) as e:
            raise AssortmentError(f"{path.name}: cannot read POS report ({e})")
        result.rows_read += total
        result.unpaid += unpaid
        result.bad_rows.extend(bad)
        seen_rows += len(rows)
        # A row repeated inside one report is a real sale; the same row in another report is not.
        in_file = Counter(r.row_key for r in rows)
        for row in rows:
            sample.setdefault(row.row_key, row)
        for key, count in in_file.items():
            kept[key] = max(kept[key], count)
    for key, count in kept.items():
        result.rows.extend([sample[key]] * count)
    result.duplicates_removed = seen_rows - len(result.rows)
    result.rows.sort(key=lambda r: (r.ring_time, r.row_key))
    return result


@dataclass
class MapEntry:
    pos_name: str
    status: str
    mapping_mode: Optional[str]
    provisional: Optional[str]
    price: Optional[float]
    notes: Optional[str]
    links: List[Dict[str, Any]]
    keys: List[str]
    names: List[str] = field(default_factory=list)  # pos_name + aliases as written, for the generated map


def read_article_map(path: Path) -> Tuple[List[MapEntry], Dict[str, int], List[str]]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw_entries = data["articles"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
        raise AssortmentError(f"Cannot read article map: {e}")
    entries: List[MapEntry] = []
    owner: Dict[str, int] = {}
    warnings: List[str] = []
    for raw in raw_entries:
        links = [{"model": str(m["model"]).lower(), "class": str(m["class"]), "priority": m.get("priority"),
                  "note": m.get("note")} for m in (raw.get("mappings") or []) if m.get("model") and m.get("class")]
        names = [str(raw.get("pos_name", ""))] + [str(a) for a in (raw.get("aliases") or [])]
        entry = MapEntry(names[0], str(raw.get("status", "incl")), raw.get("mapping_mode"), raw.get("provisional"),
                         raw.get("price_rsd"), raw.get("notes"), links, [], [n.strip() for n in names if n.strip()])
        index = len(entries)
        entries.append(entry)
        for name in names:
            key = normalize_name(name)
            if not key or key in entry.keys:
                continue
            if key in owner:
                if [(l["model"], l["class"]) for l in entries[owner[key]].links] != [(l["model"], l["class"]) for l in links]:
                    warnings.append(f"{name}: appears twice in the map with different classes, the first is kept")
                continue
            owner[key] = index
            entry.keys.append(key)
    return entries, owner, warnings


def read_blacklist(path: Path) -> Set[str]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise AssortmentError(f"Cannot read blacklist: {e}")
    return {normalize_name(l) for l in lines if l.strip() and not l.strip().startswith("#")}


@dataclass
class PairStats:
    first_seen: datetime
    last_seen: datetime
    sold: float = 0.0
    sale_rows: int = 0
    last_price: float = 0.0
    last_sale: Optional[datetime] = None  # cancellations (negative rows) are not sales


def summarize(rows: Iterable[PosRow], until: Optional[datetime] = None) -> Dict[Pair, PairStats]:
    stats: Dict[Pair, PairStats] = {}
    for row in rows:  # sorted by ring time
        if until is not None and row.ring_time > until:
            continue
        item = stats.setdefault((row.code, row.name), PairStats(row.ring_time, row.ring_time))
        item.last_seen = row.ring_time
        item.sold += row.quantity
        item.sale_rows += 1
        if row.quantity > 0:
            item.last_sale = row.ring_time
            item.last_price = row.price
    return stats


def _is_mapped(entry: Optional[MapEntry]) -> bool:
    return entry is not None and entry.status != "skip" and bool(entry.links)


def classify(stats: Dict[Pair, PairStats], entries: List[MapEntry], owner: Dict[str, int], blacklist: Set[str],
             active_days: int) -> Dict[Pair, str]:
    if not stats:
        return {}
    newest = max(s.last_seen for s in stats.values())
    states: Dict[Pair, str] = {}
    for pair, item in stats.items():
        key = normalize_name(pair[1])
        entry = entries[owner[key]] if key in owner else None
        if key in blacklist or (entry is not None and entry.status == "skip"):
            states[pair] = IGNORED  # the matcher checks the blacklist first
        elif _is_mapped(entry):
            states[pair] = TRACKED
        elif (newest - item.last_seen).days <= active_days:
            states[pair] = PENDING
        else:
            states[pair] = DORMANT
    return states


def group_pairs(pairs: Iterable[Pair]) -> List[List[Pair]]:
    """One article per connected group: the same POS code, or the same name under another code."""
    parent: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def find(x: Tuple[str, str]) -> Tuple[str, str]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = sorted(pairs)
    for code, name in pairs:
        parent[find(("code", code))] = find(("name", normalize_name(name)))
    groups: Dict[Tuple[str, str], List[Pair]] = defaultdict(list)
    for pair in pairs:
        groups[find(("code", pair[0]))].append(pair)
    return sorted(groups.values())


@dataclass
class AssortmentPlan:
    stats: Dict[Pair, PairStats]
    states: Dict[Pair, str]
    groups: List[List[Pair]]
    entries: List[MapEntry]
    owner: Dict[str, int]
    report: Dict[str, Any]
    blacklist_lines: List[str] = field(default_factory=list)  # the file as written, comments included


def _baseline_report(baseline: Dict[str, Any], read: PosRead, states: Dict[Pair, str], entries: List[MapEntry],
                     owner: Dict[str, int], blacklist: Set[str], active_days: int) -> Dict[str, Any]:
    until = datetime.strptime(str(baseline["latest_sale"]), "%Y-%m-%d").replace(hour=23, minute=59)
    old_states = classify(summarize(read.rows, until), entries, owner, blacklist, active_days)
    expected = {k: int(baseline["counts"][k]) for k in (TRACKED, IGNORED, PENDING, DORMANT)}
    old_counts = Counter(old_states.values())
    changes = [{"code": c, "name": n, "from": old_states.get((c, n)), "to": state}
               for (c, n), state in sorted(states.items()) if old_states.get((c, n)) != state]
    return {"expected": expected, "now": dict(Counter(states.values())), "explained_by_newer_reports": changes,
            "unexplained": {k: old_counts.get(k, 0) - v for k, v in expected.items() if old_counts.get(k, 0) != v}}


def build_plan(pos_dir: Path, adapter: str, article_map: Path, blacklist_path: Path, active_days: int = 30,
               baseline: Optional[Dict[str, Any]] = None) -> AssortmentPlan:
    read = read_pos_folder(pos_dir, adapter)
    entries, owner, warnings = read_article_map(article_map)
    blacklist = read_blacklist(blacklist_path)
    stats = summarize(read.rows)
    states = classify(stats, entries, owner, blacklist, active_days)
    groups = group_pairs(stats)
    counts = Counter(states.values())
    unclassified = [{"code": c, "name": n, "state": states[(c, n)], "last_sale": stats[(c, n)].last_seen.isoformat(sep=" ")}
                    for c, n in sorted(states) if states[(c, n)] in (PENDING, DORMANT)]
    by_code: Dict[str, Dict[str, Tuple[str, ...]]] = defaultdict(dict)
    for code, name in stats:
        key = normalize_name(name)
        links = entries[owner[key]].links if key in owner else []
        by_code[code][name] = tuple(f"{l['model']}.{l['class']}" for l in links)
    split = [{"code": code, "names": {n: list(c) for n, c in sorted(names.items())}}
             for code, names in sorted(by_code.items()) if len(set(names.values())) > 1]
    report: Dict[str, Any] = {
        "pairs": len(stats), "articles": len(groups),
        "counts": {k: counts.get(k, 0) for k in (TRACKED, IGNORED, PENDING, DORMANT)},
        "latest_sale": max(s.last_seen for s in stats.values()).isoformat(sep=" ") if stats else None,
        "rows_read": read.rows_read, "unpaid_rows_skipped": read.unpaid,
        "duplicate_rows_removed": read.duplicates_removed, "bad_rows": read.bad_rows,
        "unclassified": unclassified, "same_code_different_classes": split, "map_warnings": warnings,
    }
    if baseline:
        report["baseline"] = _baseline_report(baseline, read, states, entries, owner, blacklist, active_days)
    lines = Path(blacklist_path).read_text(encoding="utf-8").splitlines()
    return AssortmentPlan(stats, states, groups, entries, owner, report, lines)
