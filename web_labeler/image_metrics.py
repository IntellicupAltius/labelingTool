"""
image_metrics.py — MDQ-7: per-image review-queue comparison metrics.

For ONE flagged image at a time (never a pool-wide pass — that's MDQ-1's checker), computes
the same four indicators MDQ-1's `check_data_quality.py::class_baseline()` computes at the
class level, then compares this image's own value against that class's already-written
baseline (`<pool>_class_baseline_stats.json`, produced by `IntelliCup/tests/check_data_quality.py`,
schema_version 1):

  1. bbox-area ratio (normalised box w*h)
  2. bbox aspect ratio (pixel w/h)
  3. hue / saturation over the box crop (saturation-weighted circular hue, same as MDQ-1)
  4. embedding distance from the class's centroid (cosine, same encoder MDQ-1 used)

The first three are pure cv2/numpy — computed in-process (this server's own interpreter has
neither `torch` nor `torchvision`). The embedding needs the real
`intellicup_deep_sort/tracking/appearance.py::AppearanceEncoder` (torch + torchvision), which
this interpreter cannot import — that one call is farmed out to `embedding_worker.py` via
subprocess under a torch-capable interpreter, the same "small standalone worker script, JSON
in/out, exit 0/1" shape as the existing `analyzer_worker.py`.

Never writes anything, never changes an approve/keep decision — purely informational, always
shown as a raw number next to its badge, per MDQ-7's own acceptance criteria.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# -- locating the newest baseline-stats artifact -------------------------------------

_TRAINING_ROOT_DEFAULT = Path.home() / "Projects" / "IntelliCup"


def _training_root() -> Path:
    v = os.environ.get("INTELLICUP_TRAINING_ROOT")
    return Path(v).expanduser().resolve() if v else _TRAINING_ROOT_DEFAULT


def find_baseline_stats_path(pool: str) -> Optional[Path]:
    """Newest `<pool>_class_baseline_stats.json` under IntelliCup's DATA-4 holdout output dir.

    Directory names are `<YYYYMMDD_HHMMSS>_<suffix>` (`check_data_quality.py`'s own convention),
    so lexicographic sort of the name is chronological. A directory with "INVALID" in its name
    (an archived, superseded run — see `data4_eval_confusion_transposed_and_low_conf.md`) is
    skipped, same discipline `check_data_quality.py` itself uses for the eval-JSON side.
    """
    holdout_dir = _training_root() / "tests" / "test_results_data4_holdout"
    if not holdout_dir.is_dir():
        return None
    best_name: Optional[str] = None
    best_path: Optional[Path] = None
    for d in holdout_dir.iterdir():
        if not d.is_dir() or "INVALID" in d.name:
            continue
        f = d / f"{pool}_class_baseline_stats.json"
        if f.is_file() and (best_name is None or d.name > best_name):
            best_name, best_path = d.name, f
    return best_path


_stats_cache: Dict[str, Tuple[int, int, dict]] = {}  # pool -> (mtime_ns, size, parsed doc)


def load_baseline_stats(pool: str) -> Optional[dict]:
    """Parsed `<pool>_class_baseline_stats.json`, cached by (path, mtime, size)."""
    path = find_baseline_stats_path(pool)
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    cached = _stats_cache.get(pool)
    if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    _stats_cache[pool] = (st.st_mtime_ns, st.st_size, data)
    return data


def find_class_entry(stats: dict, raw_class: str) -> Optional[Tuple[str, dict]]:
    """(canonical_class, entry) whose `members` (RAW folder names) contains `raw_class`.

    A RAW folder can be a `unite`-aliased member of a differently-named canonical class, or
    can be entirely absent from the baseline (excluded from training) — both handled by the
    caller reading `None`.
    """
    for canonical, entry in (stats.get("classes") or {}).items():
        if raw_class in (entry.get("members") or []):
            return canonical, entry
    return None


# -- per-image visual metrics (cv2/numpy only, mirrors check_data_quality.py::class_baseline) --

def compute_box_visuals(img_bgr: np.ndarray, boxes: List[dict]) -> dict:
    """`boxes`: normalised `{xc, yc, w, h}` dicts, already filtered to the class in question.

    Aggregates across every box of that class present in this one image (an image can have
    more than one) the same way MDQ-1 aggregates across a whole class's boxes: area/aspect are
    a plain mean of the per-box values, hue is a saturation-weighted circular mean over every
    contributing box's pixels (not box-then-average), saturation is a mean of per-box means.
    """
    H, W = img_bgr.shape[:2]
    areas: List[float] = []
    aspects: List[float] = []
    sats: List[float] = []
    hue_sin = hue_cos = 0.0
    boxes_tlwh_px: List[List[int]] = []

    for b in boxes:
        cx, cy, w, h = b["xc"], b["yc"], b["w"], b["h"]
        if w <= 0 or h <= 0:
            continue
        x1 = max(0, int(round((cx - w / 2) * W)))
        y1 = max(0, int(round((cy - h / 2) * H)))
        x2 = min(W, int(round((cx + w / 2) * W)))
        y2 = min(H, int(round((cy + h / 2) * H)))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img_bgr[y1:y2, x1:x2]
        areas.append(w * h)
        aspects.append((w * W) / (h * H))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sats.append(float(hsv[..., 1].mean()))
        weights = hsv[..., 1].astype(np.float64) / 255.0
        if weights.sum() > 1e-6:
            ang = hsv[..., 0].astype(np.float64) * (2 * math.pi / 180.0)
            hue_sin += float((weights * np.sin(ang)).sum())
            hue_cos += float((weights * np.cos(ang)).sum())
        boxes_tlwh_px.append([x1, y1, x2 - x1, y2 - y1])

    hsv_out = None
    if sats:
        mean_hue = None
        if hue_sin != 0.0 or hue_cos != 0.0:
            mean_hue = (math.atan2(hue_sin, hue_cos) % (2 * math.pi)) * 180.0 / (2 * math.pi)
        hsv_out = {"mean_hue": mean_hue, "mean_sat": sum(sats) / len(sats)}

    return {
        "n_boxes": len(boxes_tlwh_px),
        "bbox_area_ratio": (sum(areas) / len(areas)) if areas else None,
        "bbox_aspect_ratio": (sum(aspects) / len(aspects)) if aspects else None,
        "hsv": hsv_out,
        "boxes_tlwh_px": boxes_tlwh_px,
    }


# -- badges ---------------------------------------------------------------------------

# Deviation thresholds, documented rather than tuned: |z| < 1.5 covers ~87% of a normal
# distribution ("normal"), 1.5-3 is a real but modest outlier ("moderate"), >= 3 is the
# conventional statistical-outlier line ("large"). For the embedding distance (no mean/std,
# only percentiles), p75/p90 are used instead, deliberately the SAME p90 line MDQ-10's own
# design already plans to flag outliers at — consistent language across MDQ-7 and MDQ-10.
Z_MODERATE = 1.5
Z_LARGE = 3.0


def zscore_badge(value: Optional[float], mean: Optional[float], std: Optional[float],
                  *, circular_period: Optional[float] = None) -> Optional[dict]:
    """`circular_period` (e.g. 180 for OpenCV hue) uses the shortest signed wrap-around
    distance instead of a plain subtraction, so a value near 0 isn't scored as "far" from a
    mean near 180 when they're actually adjacent on the hue wheel."""
    if value is None or mean is None:
        return None
    d = value - mean
    if circular_period:
        d = ((d + circular_period / 2) % circular_period) - circular_period / 2
    if std is None or std <= 1e-9:
        badge = "normal" if abs(d) <= 1e-9 else "large"
        z = None
    else:
        z = d / std
        az = abs(z)
        badge = "large" if az >= Z_LARGE else "moderate" if az >= Z_MODERATE else "normal"
        z = round(z, 3)
    return {"value": round(value, 6), "mean": round(mean, 6), "std": std, "z": z, "badge": badge}


def percentile_badge(value: Optional[float], p50: Optional[float], p75: Optional[float],
                      p90: Optional[float]) -> Optional[dict]:
    if value is None or p75 is None or p90 is None:
        return None
    badge = "large" if value > p90 else "moderate" if value > p75 else "normal"
    return {"value": round(value, 6), "p50": p50, "p75": p75, "p90": p90, "badge": badge}


# -- embedding distance (subprocess, torch-capable interpreter) -----------------------

_WORKER_SCRIPT = Path(__file__).parent / "embedding_worker.py"


def compute_embedding_distance(
    python_bin: str,
    deep_sort_root: Path,
    img_path: Path,
    boxes_tlwh_px: List[List[int]],
    centroid: List[float],
    timeout: float = 30.0,
) -> Tuple[Optional[float], Optional[str]]:
    """`(distance, error)` — exactly one is not-None. `distance` is the mean cosine distance
    (1 - cos) of this image's boxes to the already-unit-normalised class centroid, computed by
    a short-lived subprocess (device forced to CPU — this is an on-demand, single-image call
    triggered by a person clicking a button, not a batch job, and must never contend with a
    real GPU inference run for the card)."""
    if not boxes_tlwh_px or not centroid:
        return None, "no boxes or no stored centroid for this class"
    if not Path(python_bin).exists():
        return None, f"metrics interpreter not found: {python_bin}"
    payload = json.dumps({
        "image": str(img_path),
        "boxes_tlwh": boxes_tlwh_px,
        "centroid": centroid,
        "deep_sort_root": str(deep_sort_root),
        "device": "cpu",
    })
    try:
        proc = subprocess.run(
            [python_bin, str(_WORKER_SCRIPT)],
            input=payload, capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"embedding worker failed to run: {e}"
    out: dict = {}
    if proc.stdout and proc.stdout.strip():
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except ValueError:
            out = {}
    if proc.returncode != 0 or "error" in out:
        reason = out.get("error") or (proc.stderr.strip()[-500:] if proc.stderr else "embedding worker failed")
        return None, reason
    dist = out.get("distance")
    return (float(dist) if dist is not None else None), (None if dist is not None else "worker returned no distance")
