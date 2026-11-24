"""POS report intake (KATALOG-1 ticket 11): read the drop folder, never change it.

Pure part: find files that are complete and new, read them, and say what is wrong with them. The database part
(storing versions, rows and findings) is in store.receive_pos_reports. Nothing here touches the map, the
blacklist, RAW or inference.
"""
from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from web_labeler.catalog import assortment

SETTLE_SECONDS = 120  # a file is read only after it has not changed for 2 minutes (D3)
_NAME_DAY = re.compile(r"(\d{1,2})\.(\d{1,2})")


@dataclass
class IncomingReport:
    name: str
    path: Path
    sha256: str
    size: int
    report_day: Optional[str]          # from the content
    name_day: Optional[str]            # from the file name, only for a warning
    rows: Dict[Tuple[str, ...], Tuple[int, assortment.PosRow]] = field(default_factory=dict)  # row key -> (count, row)
    rows_total: int = 0
    rows_unpaid: int = 0
    bad_rows: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def rows_paid(self) -> int:
        return sum(c for c, _ in self.rows.values())

    @property
    def ring_span(self) -> Tuple[Optional[str], Optional[str]]:
        times = [r.ring_time for _, r in self.rows.values()]
        return (min(times).isoformat(sep=" "), max(times).isoformat(sep=" ")) if times else (None, None)


def _blaznavac_report_day(path: Path) -> Optional[str]:
    """Column 3 holds the report date (e.g. 31/10/2026)."""
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) > 3 and row[3].strip():
                return datetime.strptime(row[3].strip(), "%d/%m/%Y").strftime("%Y-%m-%d")
    return None


REPORT_DAY = {"blaznavac_csv": _blaznavac_report_day}


def _name_day(name: str, year: Optional[str]) -> Optional[str]:
    m = _NAME_DAY.search(name)
    if not m or not year:
        return None
    try:
        return datetime(int(year), int(m.group(2)), int(m.group(1))).strftime("%Y-%m-%d")
    except ValueError:
        return None


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_new(folder: Path, adapter: str, known: Set[str], now: datetime,
             settle_seconds: int = SETTLE_SECONDS) -> Tuple[List[Path], List[str]]:
    """(complete files not seen before, names of files still being written). Same bytes are never read twice."""
    pattern, _ = assortment.POS_ADAPTERS[adapter]
    ready, waiting = [], []
    files = list(Path(folder).glob(pattern)) if Path(folder).is_dir() else []
    for path in sorted(files, key=lambda p: (p.stat().st_mtime, p.name)):  # arrival order: the last one is the newest version
        if now.timestamp() - path.stat().st_mtime < settle_seconds:
            waiting.append(path.name)
            continue
        if sha256_of(path) not in known:
            ready.append(path)
    return ready, waiting


def read_report(path: Path, adapter: str) -> IncomingReport:
    _, reader = assortment.POS_ADAPTERS[adapter]
    item = IncomingReport(path.name, path, sha256_of(path), path.stat().st_size, None, None)
    try:
        rows, total, unpaid, bad = reader(path)
        item.report_day = REPORT_DAY[adapter](path)
    except (OSError, UnicodeDecodeError, csv.Error, ValueError) as e:
        item.error = f"Fajl ne može da se pročita ({e})."
        return item
    item.rows_total, item.rows_unpaid, item.bad_rows = total, unpaid, bad
    counts = Counter(r.row_key for r in rows)  # a row repeated inside one report is a real sale
    for r in rows:
        item.rows.setdefault(r.row_key, (counts[r.row_key], r))
    year = item.report_day[:4] if item.report_day else (str(rows[0].ring_time.year) if rows else None)
    item.name_day = _name_day(path.name, year)
    if item.report_day is None:
        item.error = "U fajlu nema datuma izveštaja."
    elif not item.rows:
        item.error = "U fajlu nema nijednog naplaćenog reda."
    return item


def compare_versions(old_keys: Iterable[Tuple[str, ...]], new: IncomingReport) -> Dict[str, int]:
    old, fresh = set(old_keys), set(new.rows)
    return {"only_new": len(fresh - old), "only_old": len(old - fresh), "both": len(fresh & old)}


def fmt_day(day: Optional[str]) -> str:
    return f"{day[8:10]}.{day[5:7]}.{day[0:4]}." if day else "nepoznat dan"
