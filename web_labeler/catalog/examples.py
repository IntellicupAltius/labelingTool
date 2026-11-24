"""Random dataset examples for a class: a crop around one labeled box, made in memory.

Only reads the dataset. Nothing is created, changed or cached there, and an example shows the class,
never which POS article it is.
"""
from __future__ import annotations

import random
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_MAX_SIDE = 420
_PAD = 0.35


class ExampleNotFound(Exception):
    pass


def _class_dirs(root: Path, model: str, class_name: str) -> Tuple[Path, Path]:
    if not class_name or "/" in class_name or "\\" in class_name or class_name in (".", ".."):
        raise ExampleNotFound(class_name)
    base = root / model.lower()
    return base / "images" / class_name, base / "labels" / class_name


def _boxes(label_file: Path, class_id: int) -> List[Tuple[float, float, float, float]]:
    try:
        lines = label_file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if cid == class_id and w > 0 and h > 0:
            boxes.append((cx, cy, w, h))
    return boxes


def pick_examples(root: Optional[Path], model: str, class_name: str, class_id: int,
                  count: int = 6, seed: Optional[int] = None) -> Dict[str, object]:
    """Pick up to count random images of the class that have a box of that class."""
    if root is None or not root.is_dir():
        return {"available": False, "total_images": 0, "examples": []}
    images_dir, labels_dir = _class_dirs(root, model, class_name)
    if not images_dir.is_dir():
        return {"available": True, "total_images": 0, "examples": []}
    files = sorted(p.name for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
    rng = random.Random(seed)
    rng.shuffle(files)
    examples = []
    for name in files[:max(count * 6, 30)]:
        boxes = _boxes(labels_dir / f"{Path(name).stem}.txt", class_id)
        if boxes:
            examples.append({"file": name, "box": rng.randrange(len(boxes))})
            if len(examples) >= count:
                break
    return {"available": True, "total_images": len(files), "examples": examples}


def crop_example(root: Optional[Path], model: str, class_name: str, class_id: int, file_name: str, box: int) -> bytes:
    """JPEG bytes of one labeled box with some surroundings."""
    from PIL import Image

    if root is None or not root.is_dir():
        raise ExampleNotFound(file_name)
    images_dir, labels_dir = _class_dirs(root, model, class_name)
    if not file_name or Path(file_name).name != file_name or Path(file_name).suffix.lower() not in _IMAGE_SUFFIXES:
        raise ExampleNotFound(file_name)
    image_path = images_dir / file_name
    if not image_path.is_file():
        raise ExampleNotFound(file_name)
    boxes = _boxes(labels_dir / f"{Path(file_name).stem}.txt", class_id)
    if not 0 <= box < len(boxes):
        raise ExampleNotFound(file_name)
    cx, cy, w, h = boxes[box]
    with Image.open(image_path) as im:
        full_w, full_h = im.size
        x1 = max(0.0, (cx - w / 2 - w * _PAD)) * full_w
        x2 = min(1.0, (cx + w / 2 + w * _PAD)) * full_w
        y1 = max(0.0, (cy - h / 2 - h * _PAD)) * full_h
        y2 = min(1.0, (cy + h / 2 + h * _PAD)) * full_h
        crop = im.convert("RGB").crop((int(x1), int(y1), int(x2), int(y2)))
    crop.thumbnail((_MAX_SIDE, _MAX_SIDE))
    out = BytesIO()
    crop.save(out, format="JPEG", quality=85)
    return out.getvalue()


def class_id_of(names: Sequence[str], class_name: str) -> int:
    try:
        return list(names).index(class_name)
    except ValueError:
        raise ExampleNotFound(class_name)
