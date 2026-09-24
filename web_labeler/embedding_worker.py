#!/usr/bin/env python3
"""
embedding_worker.py — runs under a torch-capable interpreter (default
/opt/interpreters/INTELLICUP_MODELS/bin/python — see LABELER_METRICS_PYTHON in server.py).

Called as a subprocess by image_metrics.py (MDQ-7). Same shape as the existing
analyzer_worker.py: small standalone script, JSON in on stdin / JSON out on stdout, exit 0 on
success and 1 on failure — this server's own interpreter has neither torch nor torchvision, so
the one step that needs the real AppearanceEncoder (intellicup_deep_sort/tracking/appearance.py)
is farmed out here rather than imported directly.

stdin (one JSON object):
    {"image": "<path>", "boxes_tlwh": [[x, y, w, h], ...] (pixel space),
     "centroid": [<already L2-normalised floats, as stored in the baseline stats JSON>],
     "deep_sort_root": "<path to intellicup_deep_sort repo>", "device": "cpu" | "cuda"}

stdout: {"distance": <float>} — mean cosine distance (1 - cos) of the given boxes to the
centroid — or {"error": "<message>"}.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def main() -> int:
    try:
        req = json.load(sys.stdin)
    except Exception as e:  # noqa: BLE001 — any stdin/JSON problem is reported the same way
        print(json.dumps({"error": f"bad stdin: {e}"}))
        return 1

    try:
        import cv2
        import numpy as np

        deep_sort_root = Path(req["deep_sort_root"])
        appearance_path = deep_sort_root / "tracking" / "appearance.py"
        if not appearance_path.is_file():
            print(json.dumps({"error": f"appearance.py not found under {deep_sort_root}"}))
            return 1
        spec = importlib.util.spec_from_file_location("mdq7_appearance", appearance_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        img = cv2.imread(req["image"])
        if img is None:
            print(json.dumps({"error": f"cannot read image: {req['image']}"}))
            return 1

        boxes = [tuple(float(v) for v in b) for b in (req.get("boxes_tlwh") or [])]
        if not boxes:
            print(json.dumps({"error": "no boxes given"}))
            return 1

        centroid = req.get("centroid") or []
        if not centroid:
            print(json.dumps({"error": "no centroid given"}))
            return 1

        encoder = mod.AppearanceEncoder(device=req.get("device", "cpu"))
        feats = encoder.encode(img, boxes)
        if not feats:
            print(json.dumps({"error": "encoder returned no features for these boxes"}))
            return 1

        c = np.asarray(centroid, dtype=np.float64)
        unit = c / max(float(np.linalg.norm(c)), 1e-12)
        dists = []
        for f in feats:
            v = np.asarray(f, dtype=np.float64)
            norm = max(float(np.linalg.norm(v)), 1e-12)
            dists.append(1.0 - float(v @ unit) / norm)

        print(json.dumps({"distance": sum(dists) / len(dists)}))
        return 0
    except Exception as e:  # noqa: BLE001 — report, don't crash the parent's subprocess.run
        print(json.dumps({"error": str(e)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
