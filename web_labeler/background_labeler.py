from __future__ import annotations

import os
import random
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Dict


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
    # Required fields (no defaults) must come first
    id: str
    mode: str  # "folder" or "existing"
    dataset_name: str  # for folder mode: folder name; for existing: base dataset name
    dataset_path: Path
    target_model: str
    out_root: Path
    # Optional fields (with defaults) come after
    images_dir: Optional[Path] = None  # only for folder mode
    camera_filter: str = "ALL"
    shuffled: bool = True
    seed: int = 1337

    img_files: List[Path] = field(default_factory=list)
    idx: int = 0
    selected: Set[int] = field(default_factory=set)  # indices marked background
    skipped: Set[int] = field(default_factory=set)
    exported_paths: Dict[str, Set[str]] = field(default_factory=dict)  # target_model -> set of exported image paths (for duplicate protection)

    def current_path(self) -> Optional[Path]:
        if self.idx < 0 or self.idx >= len(self.img_files):
            return None
        return self.img_files[self.idx]

    def done(self) -> bool:
        return self.idx >= len(self.img_files)


def scan_images_recursive(folder: Path, camera_filter: str = "ALL") -> List[Path]:
    """Scan folder recursively for image files, optionally filter by camera token."""
    files = []
    for ext in IMG_EXTS:
        files.extend(folder.rglob(f"*{ext}"))
        files.extend(folder.rglob(f"*{ext.upper()}"))
    files = [p for p in files if p.is_file()]
    files.sort(key=lambda p: p.name.lower())
    if camera_filter and camera_filter.upper() != "ALL":
        key = normalize_text(camera_filter)
        files = [p for p in files if key in normalize_text(p.name)]
    return [p.resolve() for p in files]


def build_candidate_list(dataset_images_dir: Path, camera_filter: str) -> List[Path]:
    """Legacy: scan single images/ folder (non-recursive)."""
    files = [p for p in dataset_images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: p.name.lower())
    if camera_filter and camera_filter.upper() != "ALL":
        key = normalize_text(camera_filter)
        files = [p for p in files if key in normalize_text(p.name)]
    return [p.resolve() for p in files]


def start_session_folder_mode(
    folder_path: Path,
    target_model: str,
    camera_filter: str,
    shuffled: bool,
    seed: int,
    datasets_dir: Path,
) -> BgSession:
    """Mode 1: Open folder / Manual set - scan any folder recursively."""
    folder_path = folder_path.resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Folder not found: {folder_path}")

    img_files = scan_images_recursive(folder_path, camera_filter=camera_filter)
    if not img_files:
        raise ValueError(f"No image files found in: {folder_path}")

    if shuffled:
        rnd = random.Random(int(seed))
        rnd.shuffle(img_files)

    # Output: datasets/<foldername>_background/<model>/
    folder_name = sanitize_part(folder_path.name)
    out_root = (datasets_dir / f"{folder_name}_background" / sanitize_part(target_model)).resolve()
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    return BgSession(
        id=uuid.uuid4().hex,
        mode="folder",
        dataset_name=folder_name,
        dataset_path=folder_path,
        images_dir=None,
        target_model=target_model,
        out_root=out_root,
        camera_filter=camera_filter or "ALL",
        shuffled=bool(shuffled),
        seed=int(seed),
        img_files=img_files,
    )


def start_session_existing_mode(
    existing_datasets_dir: Path,
    target_model: str,
    camera_filter: str,
    shuffled: bool,
    seed: int,
) -> BgSession:
    """Mode 2: From existing datasets - scan all model subdirs except target."""
    existing_datasets_dir = existing_datasets_dir.resolve()
    if not existing_datasets_dir.exists() or not existing_datasets_dir.is_dir():
        raise ValueError(f"Existing datasets directory not found: {existing_datasets_dir}")

    # Find all model subdirs (must have images/ folder)
    model_dirs = []
    for item in existing_datasets_dir.iterdir():
        if not item.is_dir():
            continue
        images_subdir = item / "images"
        if images_subdir.exists() and images_subdir.is_dir():
            model_dirs.append(item)

    if len(model_dirs) < 2:
        raise ValueError(f"Need at least 2 model subdirectories with images/ in: {existing_datasets_dir}")

    # Filter out target model
    target_normalized = normalize_text(target_model)
    source_dirs = [d for d in model_dirs if normalize_text(d.name) != target_normalized]

    if not source_dirs:
        raise ValueError(f"No source model directories found (excluding target: {target_model})")

    # Collect images from all source model dirs
    img_files = []
    for model_dir in source_dirs:
        images_dir = model_dir / "images"
        files = scan_images_recursive(images_dir, camera_filter=camera_filter)
        img_files.extend(files)

    if not img_files:
        raise ValueError("No image files found in source model directories")

    if shuffled:
        rnd = random.Random(int(seed))
        rnd.shuffle(img_files)

    # Output: datasets/existing/<target>_background_<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_name = f"{sanitize_part(target_model)}_background_{timestamp}"
    out_root = (existing_datasets_dir / out_name).resolve()
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    return BgSession(
        id=uuid.uuid4().hex,
        mode="existing",
        dataset_name="existing",
        dataset_path=existing_datasets_dir,
        images_dir=None,
        target_model=target_model,
        out_root=out_root,
        camera_filter=camera_filter or "ALL",
        shuffled=bool(shuffled),
        seed=int(seed),
        img_files=img_files,
    )


def start_session(
    datasets_dir: Path,
    dataset_name: str,
    target_model: str,
    camera_filter: str,
    shuffled: bool,
    seed: int,
) -> BgSession:
    """Legacy: single-dataset mode (kept for backward compatibility)."""
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
        mode="legacy",
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


def copy_background_image(sess: BgSession, img_path: Path) -> bool:
    """
    Copy background image and create empty label file.
    Returns True if copied (new), False if already exists (duplicate protection).
    """
    images_out = sess.out_root / "images"
    labels_out = sess.out_root / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Duplicate protection: check if this image was already exported for this target_model
    img_key = str(img_path.resolve())
    if sess.target_model not in sess.exported_paths:
        sess.exported_paths[sess.target_model] = set()
    if img_key in sess.exported_paths[sess.target_model]:
        # Already exported, skip
        return False

    # Handle filename collisions (if same name exists, add suffix)
    dst_img = images_out / img_path.name
    base_name = img_path.stem
    suffix = img_path.suffix
    counter = 1
    while dst_img.exists():
        dst_img = images_out / f"{base_name}_{counter}{suffix}"
        counter += 1

    dst_txt = labels_out / f"{dst_img.stem}.txt"

    # Copy image
    try:
        os.link(img_path, dst_img)
    except Exception:
        shutil.copy2(img_path, dst_img)

    # Empty label file (overwrite to ensure empty)
    dst_txt.write_text("", encoding="utf-8")

    # Mark as exported
    sess.exported_paths[sess.target_model].add(img_key)
    return True


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


