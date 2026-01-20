from __future__ import annotations

from pathlib import Path
from typing import List

import yaml


def load_yolo_names_from_yaml(yaml_path: Path) -> List[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names")
    if isinstance(names, dict):
        items = sorted(names.items(), key=lambda kv: int(kv[0]))
        return [v for _, v in items]
    if isinstance(names, list):
        return names
    raise ValueError(f"'names' not found or invalid in {yaml_path}")


def yolo_line(class_id: int, x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> str:
    box_w = x2 - x1
    box_h = y2 - y1
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0
    return f"{class_id} {cx / img_w:.6f} {cy / img_h:.6f} {box_w / img_w:.6f} {box_h / img_h:.6f}"


