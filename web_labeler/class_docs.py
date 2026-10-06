"""Class-picker helpers (LR-3 / LR-4): which classes the labeler may pick, and a short description.

Source of truth is the owner's documentation, NOT the training config (the config excludes B53,
the owner decided B53 is labeled):
  * hidden classes  <- IntelliCup/training_pipeline/config/documented_classes/<model>.yaml `excluded:`
  * descriptions    <- docs/labeling/kartice_<model>_v1.md ("Prepoznaje se" bullet of each "### CLASS")

Read-only. Fail-soft: missing/unreadable file => nothing hidden / no description (the picker then
behaves exactly as before). The model's own class list (and so class ids) is never altered here.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Set

import yaml

PROJECTS = Path(os.environ.get("INTELLICUP_PROJECTS_DIR", str(Path.home() / "Projects")))
DOCUMENTED_CLASSES_DIR = Path(os.environ.get(
    "LABELER_DOCUMENTED_CLASSES_DIR",
    str(PROJECTS / "IntelliCup/training_pipeline/config/documented_classes")))
KARTICE_DIR = Path(os.environ.get("LABELER_KARTICE_DIR", str(PROJECTS / "docs/labeling")))

# Cards are owner-confirmed only for shots; the other models' cards are drafts (LR-11).
DESCRIPTION_MODELS = {"shots"}
MAX_DESC_LEN = 110
# Cards put this warning on its own line, not in "Prepoznaje se"; the owner wants it visible in the picker.
DESC_SUFFIX = {"SHOT_BLURRY": " (NE mutan snimak)"}


def hidden_classes(model: str, directory: Path = None) -> Set[str]:
    p = (directory or DOCUMENTED_CLASSES_DIR) / f"{model.lower()}.yaml"
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return set()
    out: Set[str] = set()
    for e in data.get("excluded") or []:
        name = e.get("class") if isinstance(e, dict) else e
        if isinstance(name, str) and name:
            out.add(name)
    return out


def _clean(text: str) -> str:
    text = re.sub(r"[*_`]+", "", text).replace("⚠️", "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > MAX_DESC_LEN:
        text = text[:MAX_DESC_LEN - 1].rstrip(" ,;:.") + "…"
    return text


def descriptions(model: str, directory: Path = None) -> Dict[str, str]:
    if model.lower() not in DESCRIPTION_MODELS:
        return {}
    p = (directory or KARTICE_DIR) / f"kartice_{model.lower()}_v1.md"
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}
    out: Dict[str, str] = {}
    cur = None
    for ln in lines:
        m = re.match(r"^###\s+(\S+)\s*$", ln)
        if m:
            cur = m.group(1)
            continue
        if cur and cur not in out:
            m = re.match(r"^-\s+\*\*Prepoznaje se:\*\*\s*(.+)$", ln)
            if m:
                out[cur] = _clean(m.group(1)).rstrip(".") + DESC_SUFFIX.get(cur, "")
    return out


def picker_classes(model: str, names: List[str]) -> List[dict]:
    """Classes to offer in the picker, in model order: [{name, description}]."""
    hide = hidden_classes(model)
    desc = descriptions(model)
    return [{"name": n, "description": desc.get(n, "")} for n in names if n not in hide]
