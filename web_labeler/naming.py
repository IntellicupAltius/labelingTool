from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


_GUID_RE = re.compile(r"\{[0-9A-Fa-f-]+\}")
_PAREN_RE = re.compile(r"\([0-9]+\)$")
_TS_RE = re.compile(r"(?:^|_)(\d{14})(?:$|_)")
_BAR_COUNTER_RE = re.compile(r"(SANK_[A-Z0-9]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedVideoName:
    timestamp_14: Optional[str]
    prefix: str
    bar_counter: Optional[str]


def _normalize_tokens(prefix: str) -> str:
    # Keep underscores, collapse other separators to underscore, uppercase.
    s = re.sub(r"[^\w]+", "_", prefix)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


def parse_video_name(filename: str, bar_counter_options: List[str]) -> ParsedVideoName:
    """
    Extracts:
    - timestamp (14 digits) from the video name (usually at the end)
    - prefix without GUID / timestamp / trailing (001)
    - bar_counter (must be one of bar_counter_options, case-insensitive)
    """
    p = Path(filename)
    stem = p.stem

    # remove trailing (001)
    stem = _PAREN_RE.sub("", stem)

    # remove GUID block(s)
    stem_wo_guid = _GUID_RE.sub("", stem)
    stem_wo_guid = re.sub(r"_+", "_", stem_wo_guid).strip("_")

    # timestamp: prefer the last 14-digit group separated by underscores
    timestamp = None
    for m in _TS_RE.finditer(stem_wo_guid):
        timestamp = m.group(1)

    # remove timestamp part (if present) from the end-ish
    stem_wo = stem_wo_guid
    if timestamp:
        # remove last occurrence of _<timestamp> (or leading <timestamp>)
        stem_wo = re.sub(rf"(?:^|_){re.escape(timestamp)}$", "", stem_wo_guid)
        stem_wo = re.sub(r"_+", "_", stem_wo).strip("_")

    prefix = _normalize_tokens(stem_wo)

    # find bar counter token
    normalized_options = [o.upper() for o in (bar_counter_options or [])]
    bar_counter = None

    # First: exact token match
    tokens = [t for t in prefix.split("_") if t]
    joined = "_".join(tokens)
    # Match patterns like SANK_LEVO, SANK_DESNO, etc.
    m2 = _BAR_COUNTER_RE.search(joined)
    if m2:
        candidate = m2.group(1).upper()
        # If options exist, enforce them; otherwise accept anything matching pattern.
        if normalized_options:
            if candidate in normalized_options:
                bar_counter = candidate
        else:
            bar_counter = candidate

    return ParsedVideoName(timestamp_14=timestamp, prefix=prefix, bar_counter=bar_counter)


def export_base_name(parsed: ParsedVideoName, frame_idx: int) -> str:
    """
    Base export name (no extension).
    Example:
      20251205090046_BLAZNAVAC_NVR_01_G_SANK_LEVO_f001036
    Uses 1-based frame numbering with 6-digit zero padding.
    """
    ts = parsed.timestamp_14 or "UNKNOWN_TS"
    return f"{ts}_{parsed.prefix}_f{(int(frame_idx) + 1):06d}"


def output_video_dirname(filename: str, max_len: int = 140) -> str:
    """
    Folder name for exports based on the *video name*, but without GUID / trailing (001).
    Keeps timestamp (so different captures don't collide) and avoids Windows path limits by truncating.

    Examples:
      blaznavac_sank_desno_{GUID}_20250509155935(001).avi -> BLAZNAVAC_SANK_DESNO_20250509155935
      BLAZNAVAC_NVR_01_G_SANK_LEVO_{GUID}_20251205090046.mp4 -> BLAZNAVAC_NVR_01_G_SANK_LEVO_20251205090046
    """
    p = Path(filename)
    stem = p.stem
    stem = _PAREN_RE.sub("", stem)
    stem = _GUID_RE.sub("", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    stem = _normalize_tokens(stem)
    if len(stem) > max_len:
        stem = stem[:max_len].rstrip("_")
    return stem or "VIDEO"


