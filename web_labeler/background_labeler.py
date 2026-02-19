from __future__ import annotations

import os
import random
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sanitize_part(s: str) -> str:
    import re

    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (s or "").strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def normalize_text(s: str) -> str:
    return (s or "").lower().replace(" ", "").replace("-", "").replace("_", "")


@dataclass
class BgSession:
    id: str
    dataset_name: str
    dataset_path: Path
    images_dir: Path
    target_model: str
    out_root: Path
    camera_filter: str = "ALL"
    shuffled: bool = True
    seed: int = 1337

    img_files: List[Path] = field(default_factory=list)
    idx: int = 0
    selected: Set[int] = field(default_factory=set)  # indices marked background
    skipped: Set[int] = field(default_factory=set)

    def current_path(self) -> Optional[Path]:
        if self.idx < 0 or self.idx >= len(self.img_files):
            return None
        return self.img_files[self.idx]

    def done(self) -> bool:
        return self.idx >= len(self.img_files)


def build_candidate_list(dataset_images_dir: Path, camera_filter: str) -> List[Path]:
    files = [p for p in dataset_images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: p.name.lower())
    if camera_filter and camera_filter.upper() != "ALL":
        key = normalize_text(camera_filter)
        files = [p for p in files if key in normalize_text(p.name)]
    return [p.resolve() for p in files]


def start_session(
    datasets_dir: Path,
    dataset_name: str,
    target_model: str,
    camera_filter: str,
    shuffled: bool,
    seed: int,
) -> BgSession:
    ds_path = (datasets_dir / dataset_name).resolve()
    images_dir = ds_path / "images"
    if ds_path.parent != datasets_dir or not images_dir.is_dir():
        raise ValueError("Dataset not found or missing images/")

    img_files = build_candidate_list(images_dir, camera_filter=camera_filter)
    if shuffled:
        rnd = random.Random(int(seed))
        rnd.shuffle(img_files)

    out_root = (ds_path.parent / f"{dataset_name}_background" / sanitize_part(target_model)).resolve()
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    return BgSession(
        id=uuid.uuid4().hex,
        dataset_name=dataset_name,
        dataset_path=ds_path,
        images_dir=images_dir,
        target_model=target_model,
        out_root=out_root,
        camera_filter=camera_filter or "ALL",
        shuffled=bool(shuffled),
        seed=int(seed),
        img_files=img_files,
    )


def copy_background_image(sess: BgSession, img_path: Path) -> None:
    images_out = sess.out_root / "images"
    labels_out = sess.out_root / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    dst_img = images_out / img_path.name
    dst_txt = labels_out / f"{img_path.stem}.txt"

    if not dst_img.exists():
        try:
            os.link(img_path, dst_img)
        except Exception:
            shutil.copy2(img_path, dst_img)

    # Empty label file (overwrite to ensure empty)
    dst_txt.write_text("", encoding="utf-8")


def remove_background_image(sess: BgSession, img_path: Path) -> None:
    images_out = sess.out_root / "images"
    labels_out = sess.out_root / "labels"
    dst_img = images_out / img_path.name
    dst_txt = labels_out / f"{img_path.stem}.txt"
    try:
        if dst_img.exists():
            dst_img.unlink()
    except Exception:
        pass
    try:
        if dst_txt.exists():
            dst_txt.unlink()
    except Exception:
        pass


