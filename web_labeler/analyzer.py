"""
analyzer.py — orchestration for the Visual Analyzer tab.

Runs under INTELLICUP_LABELING_TOOL interpreter.
Calls analyzer_worker.py via subprocess using INTELLICUP_MODELS python.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from web_labeler import flag_store as _flags

logger = logging.getLogger("labeler.analyzer")

# ── Paths ─────────────────────────────────────────────────────────────────────

WORKER_SCRIPT = Path(__file__).parent / "analyzer_worker.py"

# Resolved at startup from config / env vars
_models_root: Optional[Path] = None   # /opt/intellicup/models
_raw_root: Optional[Path] = None      # /opt/intellicup/datasets/raw/blaznavac
_models_python: Optional[str] = None  # /opt/interpreters/INTELLICUP_MODELS/bin/python

# MDQ-9: the same per-pool flag sidecar MDQ-3/MDQ-6 write, so overlap_report's own findings
# land in the owner's one review queue instead of a report only the labeler-tool UI can show.
# This is a write OUTSIDE the RAW/models tree (raw_review/), not a violation of this module's
# read-only guarantee toward /opt/intellicup/models and /opt/intellicup/datasets/raw — see
# .claude/rules/analyzer.md.
_flag_store: Optional[_flags.FlagStore] = None

# Caps from the MDQ-9 ticket text: at most 5 new image-flag entries per class per run (the
# class's own strongest confusion pair first), at most 30 total per pool per run across every
# class (pool-wide priority = confusion rate, descending). The existing overlap_report
# threshold (>=15%) and per-class sample size (20) are untouched — this only decides which of
# the already-computed misclassifications get written to the shared sidecar.
MAX_FLAGS_PER_CLASS = 5
MAX_FLAGS_PER_POOL = 30


def configure(models_root: str, raw_root: str, models_python: str, flag_store: Optional[_flags.FlagStore] = None):
    global _models_root, _raw_root, _models_python, _flag_store
    _models_root = Path(models_root).expanduser().resolve() if models_root else None
    _raw_root = Path(raw_root).expanduser().resolve() if raw_root else None
    _models_python = models_python or None
    _flag_store = flag_store


def is_available() -> bool:
    if not _models_python or not Path(_models_python).exists():
        return False
    if not _models_root or not _models_root.exists():
        return False
    return True


def get_available_models() -> List[str]:
    if not _models_root:
        return []
    out = []
    for d in sorted(_models_root.iterdir()):
        if not d.is_dir():
            continue
        pt = d / "current.pt"
        if pt.exists():
            out.append(d.name)
    return out


def _model_pt(model_name: str) -> Optional[Path]:
    if not _models_root:
        return None
    pt = _models_root / model_name / "current.pt"
    return pt if pt.exists() else None


def _raw_dir(model_name: str) -> Optional[Path]:
    if not _raw_root:
        return None
    d = _raw_root / model_name
    return d if d.exists() else None


def _data_yaml(model_name: str) -> Optional[Path]:
    raw = _raw_dir(model_name)
    if not raw:
        return None
    y = raw / "data.yaml"
    return y if y.exists() else None


# ── Classify ──────────────────────────────────────────────────────────────────

def classify_image(model_name: str, image_bytes: bytes, bbox: Optional[Dict] = None) -> List[Dict]:
    """
    Run model inference on image_bytes (JPEG/PNG).
    bbox: {x1, y1, x2, y2} in pixel coords of the uploaded image.
    Returns [{class, confidence}, ...] sorted descending.
    """
    pt = _model_pt(model_name)
    if pt is None:
        raise ValueError(f"No current.pt for model '{model_name}'")

    suffix = ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(image_bytes)
        tmp_path = tf.name

    try:
        cmd = [
            _models_python,
            str(WORKER_SCRIPT),
            "classify",
            "--model-pt", str(pt),
            "--image", tmp_path,
        ]
        if bbox:
            cmd += ["--bbox", f"{bbox['x1']},{bbox['y1']},{bbox['x2']},{bbox['y2']}"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.error("Worker stderr: %s", result.stderr)
            raise RuntimeError(f"Inference failed: {result.stderr[-300:]}")

        return json.loads(result.stdout.strip())
    finally:
        os.unlink(tmp_path)


# ── Class examples ────────────────────────────────────────────────────────────

def get_class_examples(model_name: str, class_name: str, n: int = 4) -> List[str]:
    """
    Return up to n base64-encoded JPEG crops from the raw dataset for the given class.
    Returns list of data URLs ("data:image/jpeg;base64,...").
    """
    import cv2

    raw = _raw_dir(model_name)
    if not raw:
        return []

    img_dir = raw / "images" / class_name
    if not img_dir.exists():
        return []

    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not imgs:
        return []

    random.shuffle(imgs)
    imgs = imgs[:n]

    label_dir = raw / "labels" / class_name
    results = []

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Try to find a bbox from the label file
        lbl_path = label_dir / (img_path.stem + ".txt")
        crop = None
        if lbl_path.exists():
            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                # Pick first bbox
                parts = lines[0].split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = max(0, int((cx - bw / 2) * w))
                    y1 = max(0, int((cy - bh / 2) * h))
                    x2 = min(w, int((cx + bw / 2) * w))
                    y2 = min(h, int((cy + bh / 2) * h))
                    if x2 > x1 and y2 > y1:
                        pad = 10
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w, x2 + pad)
                        y2 = min(h, y2 + pad)
                        crop = img[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            # Fall back to full image thumbnail
            crop = img

        # Resize to max 200px on longest side
        ch, cw = crop.shape[:2]
        if max(ch, cw) > 200:
            scale = 200 / max(ch, cw)
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)))

        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        results.append(f"data:image/jpeg;base64,{b64}")

    return results


# ── Overlap report (background job) ──────────────────────────────────────────

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def start_overlap_report(model_name: str, samples: int = 20) -> str:
    """Start background overlap report job. Returns job_id."""
    pt = _model_pt(model_name)
    raw = _raw_dir(model_name)
    yaml = _data_yaml(model_name)

    if pt is None:
        raise ValueError(f"No current.pt for model '{model_name}'")
    if raw is None:
        raise ValueError(f"No raw dataset for model '{model_name}'")
    if yaml is None:
        raise ValueError(f"No data.yaml for model '{model_name}'")

    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "model": model_name,
            "progress": {"done": 0, "total": 0, "current_class": ""},
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_overlap_job, args=(job_id, model_name, pt, raw, yaml, samples), daemon=True
    )
    thread.start()
    return job_id


def _select_and_flag(pool: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Write up to MAX_FLAGS_PER_CLASS/MAX_FLAGS_PER_POOL new sidecar entries from one
    overlap_report result. Never raises — a flagging problem must not lose the report itself.
    """
    summary = {"available": False, "flagged": 0, "skipped_existing": 0, "classes_flagged": 0}
    if _flag_store is None or not _flag_store.available:
        return summary
    summary["available"] = True

    misclassified: Dict[str, List[Dict]] = result.get("misclassified") or {}
    flagged_pairs: List[Dict] = result.get("flagged_pairs") or []  # already sorted by rate desc
    per_class_flagged: Dict[str, int] = {}
    total_flagged = 0
    skipped_existing = 0

    for pair in flagged_pairs:
        if total_flagged >= MAX_FLAGS_PER_POOL:
            break
        cls_a = pair.get("class_a")
        cls_b = pair.get("class_b")
        rate = pair.get("rate")
        if not cls_a or not cls_b:
            continue
        # Images of cls_a that were specifically confused as THIS pair's cls_b, strongest
        # (most confident) misclassification first.
        candidates = [
            m for m in misclassified.get(cls_a, []) if m.get("confused_as") == cls_b
        ]
        candidates.sort(key=lambda m: m.get("confidence", 0.0), reverse=True)

        for m in candidates:
            if per_class_flagged.get(cls_a, 0) >= MAX_FLAGS_PER_CLASS:
                break
            if total_flagged >= MAX_FLAGS_PER_POOL:
                break
            image_key = f"{cls_a}/{m.get('image')}"
            signal = {
                "metric": "confused_as",
                "value": cls_b,
                "rate": rate,
                "confidence": m.get("confidence"),
            }
            try:
                entry = _flag_store.add_auto_flag(
                    pool, image_key, _flags.SOURCE_ANALYZER_AUTO, signal,
                    flagged_by="analyzer_overlap_report",
                )
            except _flags.FlagError as e:
                logger.warning("MDQ-9: could not auto-flag %s/%s: %s", pool, image_key, e)
                continue
            if entry is None:
                skipped_existing += 1
                continue
            total_flagged += 1
            per_class_flagged[cls_a] = per_class_flagged.get(cls_a, 0) + 1

    summary["flagged"] = total_flagged
    summary["skipped_existing"] = skipped_existing
    summary["classes_flagged"] = len(per_class_flagged)
    return summary


def _run_overlap_job(job_id: str, model_name: str, pt: Path, raw: Path, data_yaml: Path, samples: int):
    try:
        cmd = [
            _models_python,
            str(WORKER_SCRIPT),
            "overlap",
            "--model-pt", str(pt),
            "--raw-dir", str(raw),
            "--data-yaml", str(data_yaml),
            "--samples", str(samples),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        result = None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")
            if msg_type == "total":
                with _jobs_lock:
                    _jobs[job_id]["progress"]["total"] = msg.get("total", 0)
            elif msg_type == "progress":
                with _jobs_lock:
                    _jobs[job_id]["progress"]["done"] = msg.get("done", 0)
                    _jobs[job_id]["progress"]["current_class"] = msg.get("class", "")
            elif msg_type == "result":
                result = msg

        proc.wait()
        stderr = proc.stderr.read()

        if proc.returncode != 0:
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = stderr[-500:] if stderr else "Worker exited with error"
        elif result is None:
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = "No result received from worker"
        else:
            try:
                auto_flag_summary = _select_and_flag(model_name, result)
            except Exception:
                # MDQ-9 flagging is a best-effort side effect of the report — never let it
                # cost the labeler the report itself.
                logger.exception("MDQ-9: auto-flagging failed for overlap job %s", job_id)
                auto_flag_summary = {"available": False, "flagged": 0, "skipped_existing": 0,
                                      "classes_flagged": 0, "error": "auto-flagging failed, see server log"}
            with _jobs_lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = result
                _jobs[job_id]["auto_flag_summary"] = auto_flag_summary

    except Exception as e:
        logger.exception("Overlap job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)


def get_job_status(job_id: str) -> Optional[Dict]:
    with _jobs_lock:
        return dict(_jobs.get(job_id) or {})
