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

logger = logging.getLogger("labeler.analyzer")

# ── Paths ─────────────────────────────────────────────────────────────────────

WORKER_SCRIPT = Path(__file__).parent / "analyzer_worker.py"

# Resolved at startup from config / env vars
_models_root: Optional[Path] = None   # /opt/intellicup/models
_raw_root: Optional[Path] = None      # /opt/intellicup/datasets/raw/blaznavac
_models_python: Optional[str] = None  # /opt/interpreters/INTELLICUP_MODELS/bin/python


def configure(models_root: str, raw_root: str, models_python: str):
    global _models_root, _raw_root, _models_python
    _models_root = Path(models_root).expanduser().resolve() if models_root else None
    _raw_root = Path(raw_root).expanduser().resolve() if raw_root else None
    _models_python = models_python or None


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

    thread = threading.Thread(target=_run_overlap_job, args=(job_id, pt, raw, yaml, samples), daemon=True)
    thread.start()
    return job_id


def _run_overlap_job(job_id: str, pt: Path, raw: Path, data_yaml: Path, samples: int):
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
            with _jobs_lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = result

    except Exception as e:
        logger.exception("Overlap job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)


def get_job_status(job_id: str) -> Optional[Dict]:
    with _jobs_lock:
        return dict(_jobs.get(job_id) or {})
