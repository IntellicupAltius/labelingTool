"""Per-camera bar ROI / crop rectangle lookup for the review viewer overlay (read-only).

Training crops every image to the bounding rectangle of the camera's ``bar_roi`` + ``pickup_roi``
polygons and drops labels whose centre falls outside it; production inference crops each frame
to the same rectangle plus 20 px. Anything outside that rectangle never reaches training or
production, which is what the overlay shows the reviewer.

The logic is REPLICATED here, not imported: IntelliCup's ``dataset_builder.py`` is a training
module (pulls in its augmentation engine at import time and hardcodes the config path), and this
repo has no code dependency on IntelliCup — only on-disk data files (see ``article_map_path``).
Sources, as of IntelliCup commit 641a571:

- Camera from filename: ``IntelliCup/training_pipeline/tmp_training_builder/dataset_builder.py``
  ``_detect_camera`` (lines 46-55): first camera in ``cameras.yaml`` order whose name, lowercased,
  is a substring of the lowercased file name. Same rule in production:
  ``intellicup_deep_sort/tracking/deep_sort_tracker_simplified_v2.py`` ``get_roi_points`` (135-139).
- Crop rectangle: same file, ``_camera_crop_rect`` (lines 58-67): min/max over all points of
  ``bar_roi`` and ``pickup_roi`` (normalized, no padding).
- Label drop rule: same file, ``_apply_roi_crop`` (lines 564-596): a label is dropped when its
  centre is outside the rectangle.
- Production padding: ``deep_sort_tracker_simplified_v2.py`` lines 382-388 (``_PAD = 20`` px on
  each side of the same rectangle, computed on the pixel polygon points).

If any of those change in IntelliCup, this file must follow.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CAMERAS_YAML = "/opt/intellicup/config/cameras.yaml"
INFERENCE_PAD_PX = 20   # deep_sort_tracker_simplified_v2.py:384

_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {"path": None, "mtime": None, "cameras": None}


def _load_cameras(path: Path) -> Dict[str, dict]:
    """``cameras:`` mapping from cameras.yaml, re-read only when the file's mtime changes."""
    mtime = path.stat().st_mtime
    with _cache_lock:
        if _cache["path"] == str(path) and _cache["mtime"] == mtime:
            return _cache["cameras"]
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cams = data.get("cameras") or {}
    if not isinstance(cams, dict):
        raise ValueError("'cameras' is not a mapping")
    with _cache_lock:
        _cache.update(path=str(path), mtime=mtime, cameras=cams)
    return cams


def detect_camera(filename: str, cameras: Dict[str, dict]) -> Optional[str]:
    """dataset_builder.py:46-55, minus the raise: None when no camera name is in the file name."""
    name = Path(filename).name.lower()
    for cam in cameras:
        if cam.lower() in name:
            return cam
    return None


def _points(poly: Any) -> Optional[List[List[float]]]:
    if not isinstance(poly, list) or not poly:
        return None
    out = []
    for p in poly:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        out.append([float(p[0]), float(p[1])])
    return out


def _unknown(image_key: Optional[str], reason: str, camera: Optional[str] = None) -> dict:
    return {"image_key": image_key, "camera": camera or "unknown", "status": "unknown", "reason": reason,
            "crop_rect": None, "bar_roi": None, "pickup_roi": None, "inference_pad_px": INFERENCE_PAD_PX}


def camera_roi(image_key: str, cameras_yaml: Optional[str] = None) -> dict:
    """ROI data for the camera derived from ``image_key``'s file name. Never raises.

    ``status`` is "ok" or "unknown"; for "unknown", ``reason`` says why and the geometry is null.
    Coordinates are normalized [0, 1] image coordinates, crop_rect = {x0, y0, x1, y1}.
    """
    path = Path(cameras_yaml or DEFAULT_CAMERAS_YAML)
    try:
        cameras = _load_cameras(path)
    except FileNotFoundError:
        return _unknown(image_key, f"camera config not found: {path}")
    except Exception as e:  # unreadable / malformed yaml: report, don't 500
        return _unknown(image_key, f"camera config unreadable: {path} ({e})")

    cam = detect_camera(image_key.split("/")[-1], cameras)
    if cam is None:
        return _unknown(image_key, f"no camera name ({', '.join(cameras)}) in the file name")

    cfg = cameras.get(cam) or {}
    bar, pickup = _points(cfg.get("bar_roi")), _points(cfg.get("pickup_roi"))
    all_pts = (bar or []) + (pickup or [])     # dataset_builder.py:62-64
    if not all_pts:
        return _unknown(image_key, f"camera '{cam}' has no bar_roi/pickup_roi in {path}", camera=cam)
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    return {
        "image_key": image_key,
        "camera": cam,
        "status": "ok",
        "reason": None,
        "crop_rect": {"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)},   # dataset_builder.py:65-67
        "bar_roi": bar,
        "pickup_roi": pickup,
        "inference_pad_px": INFERENCE_PAD_PX,
    }
