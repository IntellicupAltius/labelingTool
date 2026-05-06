#!/usr/bin/env python3
"""
analyzer_worker.py — runs under INTELLICUP_MODELS interpreter.

Called as subprocess by analyzer.py (labeling tool). Outputs JSON lines to stdout.
Errors go to stderr. Exit 0 on success, 1 on failure.

Modes:
  classify  --model-pt PATH --image PATH [--bbox x1,y1,x2,y2]
  overlap   --model-pt PATH --raw-dir PATH --data-yaml PATH [--samples N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_model(model_pt: str):
    from ultralytics import YOLO
    return YOLO(model_pt)


def _run_classify(args):
    import cv2
    import tempfile
    import os

    model = _load_model(args.model_pt)

    img = cv2.imread(args.image)
    if img is None:
        print(json.dumps({"error": f"Cannot read image: {args.image}"}), flush=True)
        sys.exit(1)

    h, w = img.shape[:2]

    if args.bbox:
        x1, y1, x2, y2 = [int(v) for v in args.bbox.split(",")]
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        # Run on full image — model was trained on full frames
        results = model.predict(args.image, conf=0.01, verbose=False, save=False, imgsz=640)
        # Collect detections overlapping drawn bbox
        scores: dict[str, float] = {}
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
                # IoU with drawn bbox
                ix1, iy1 = max(bx1, x1), max(by1, y1)
                ix2, iy2 = min(bx2, x2), min(by2, y2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                drawn_area = max(1, (x2 - x1) * (y2 - y1))
                overlap_frac = inter / drawn_area
                if overlap_frac < 0.2:
                    continue
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if cls_name not in scores or conf > scores[cls_name]:
                    scores[cls_name] = conf

        # Fallback: if nothing overlaps, run on crop directly
        if not scores:
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                    tmp_path = tf.name
                try:
                    cv2.imwrite(tmp_path, crop)
                    results2 = model.predict(tmp_path, conf=0.01, verbose=False, save=False, imgsz=640)
                    for r in results2:
                        if r.boxes is None:
                            continue
                        for box in r.boxes:
                            cls_name = model.names[int(box.cls[0])]
                            conf = float(box.conf[0])
                            if cls_name not in scores or conf > scores[cls_name]:
                                scores[cls_name] = conf
                finally:
                    os.unlink(tmp_path)
    else:
        # No bbox — run on full image, return all detections
        results = model.predict(args.image, conf=0.01, verbose=False, save=False, imgsz=640)
        scores: dict[str, float] = {}
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if cls_name not in scores or conf > scores[cls_name]:
                    scores[cls_name] = conf

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(json.dumps([{"class": c, "confidence": round(v, 4)} for c, v in sorted_scores]), flush=True)


def _run_overlap(args):
    import cv2
    import yaml
    import random

    model = _load_model(args.model_pt)

    # Read class names from data.yaml
    with open(args.data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names_raw = data.get("names", [])
    if isinstance(names_raw, dict):
        names = {int(k): v for k, v in names_raw.items()}
    else:
        names = {i: v for i, v in enumerate(names_raw)}

    raw_images = Path(args.raw_dir) / "images"
    if not raw_images.exists():
        print(json.dumps({"error": f"images dir not found: {raw_images}"}), flush=True)
        sys.exit(1)

    class_dirs = sorted([d for d in raw_images.iterdir() if d.is_dir()])
    if not class_dirs:
        print(json.dumps({"error": "no class directories found"}), flush=True)
        sys.exit(1)

    samples_per_class = args.samples
    total = sum(
        min(samples_per_class, len(list(d.glob("*.jpg")) + list(d.glob("*.png"))))
        for d in class_dirs
    )

    # Report total so client can show progress bar
    print(json.dumps({"type": "total", "total": total}), flush=True)

    per_class: dict[str, dict] = {}
    done = 0

    for class_dir in class_dirs:
        cls_name = class_dir.name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not images:
            continue
        random.shuffle(images)
        images = images[:samples_per_class]

        correct = 0
        confused_as: dict[str, int] = {}

        for img_path in images:
            results = model.predict(str(img_path), conf=0.1, verbose=False, save=False, imgsz=640)

            # Find highest confidence detection
            best_cls = None
            best_conf = 0.0
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_cls = model.names[int(box.cls[0])]

            if best_cls is None:
                # No detection
                confused_as["__none__"] = confused_as.get("__none__", 0) + 1
            elif best_cls == cls_name:
                correct += 1
            else:
                confused_as[best_cls] = confused_as.get(best_cls, 0) + 1

            done += 1
            print(json.dumps({"type": "progress", "done": done, "total": total, "class": cls_name}), flush=True)

        per_class[cls_name] = {
            "total": len(images),
            "correct": correct,
            "accuracy": round(correct / len(images), 3) if images else 0.0,
            "confused_as": dict(sorted(confused_as.items(), key=lambda x: x[1], reverse=True)),
        }

    # Build flagged pairs: class A was confused as class B with rate > threshold
    threshold = 0.15
    flagged = []
    for cls_a, info in per_class.items():
        for cls_b, count in info["confused_as"].items():
            if cls_b.startswith("__"):
                continue
            rate = count / info["total"] if info["total"] > 0 else 0
            if rate >= threshold:
                flagged.append({
                    "class_a": cls_a,
                    "class_b": cls_b,
                    "count": count,
                    "rate": round(rate, 3),
                })
    flagged.sort(key=lambda x: x["rate"], reverse=True)

    print(json.dumps({
        "type": "result",
        "model": str(Path(args.model_pt).name),
        "total_images": done,
        "per_class": per_class,
        "flagged_pairs": flagged,
    }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_cls = sub.add_parser("classify")
    p_cls.add_argument("--model-pt", required=True)
    p_cls.add_argument("--image", required=True)
    p_cls.add_argument("--bbox", default=None, help="x1,y1,x2,y2")

    p_ov = sub.add_parser("overlap")
    p_ov.add_argument("--model-pt", required=True)
    p_ov.add_argument("--raw-dir", required=True)
    p_ov.add_argument("--data-yaml", required=True)
    p_ov.add_argument("--samples", type=int, default=20)

    args = parser.parse_args()

    if args.mode == "classify":
        _run_classify(args)
    elif args.mode == "overlap":
        _run_overlap(args)


if __name__ == "__main__":
    main()
