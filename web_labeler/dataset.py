from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from web_labeler.video_io import clamp


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetAnnotation:
    id: str
    image_idx: int
    class_id: int
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    # MDQ-17c: a model proposal in a relabel batch (proposal=True until confirmed/re-classed; never saved
    # while it is a proposal). proposal_id/proposal_class stay after confirming, to record the decision.
    proposal: bool = False
    proposal_id: Optional[str] = None
    proposal_class: Optional[str] = None
    proposal_conf: Optional[float] = None
    proposal_agreement: Optional[str] = None
    decoy: bool = False          # QA decoy: never written to labels/, never sent to the browser


@dataclass
class DatasetSession:
    dataset_path: Path
    images_dir: Path
    labels_dir: Path
    model: str
    img_files: List[Path] = field(default_factory=list)  # absolute paths
    img_sizes: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # idx -> (w,h)
    ann_by_image: Dict[int, List[DatasetAnnotation]] = field(default_factory=dict)
    background_images: set[int] = field(default_factory=set)
    deleted_images: set[int] = field(default_factory=set)  # image indices marked for deletion
    image_tags: Dict[int, Dict] = field(default_factory=dict)  # image idx → {frame_tags: [], bbox_tags: {ann_id: []}}
    relabel_items: Dict[str, Dict] = field(default_factory=dict)  # MDQ-15c-2 relabel batch: image file name → manifest item
    proposals_by_image: Dict[int, List[Dict]] = field(default_factory=dict)  # MDQ-17c: proposals loaded (still undecided) per image

    def new_id(self) -> str:
        return uuid.uuid4().hex

    def image_path(self, idx: int) -> Path:
        return self.img_files[int(idx)]

    def image_name(self, idx: int) -> str:
        return self.image_path(idx).name

    def image_stem(self, idx: int) -> str:
        return self.image_path(idx).stem

    def label_path(self, idx: int) -> Path:
        return self.labels_dir / f"{self.image_stem(idx)}.txt"


def list_dataset_folders(datasets_dir: Path) -> List[str]:
    if not datasets_dir.exists():
        return []
    out = []
    for p in sorted(datasets_dir.iterdir()):
        if not p.is_dir():
            continue
        if (p / "images").is_dir() and (p / "labels").is_dir():
            out.append(p.name)
    return out


def _load_image_list(images_dir: Path) -> List[Path]:
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: p.name.lower())
    return [p.resolve() for p in files]


def _read_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (w,h)


def _yolo_to_xyxy(line: str, w: int, h: int) -> Optional[Tuple[int, int, int, int, int]]:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
    except Exception:
        return None
    x1 = int((cx - bw / 2.0) * w)
    y1 = int((cy - bh / 2.0) * h)
    x2 = int((cx + bw / 2.0) * w)
    y2 = int((cy + bh / 2.0) * h)
    x1 = clamp(x1, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    x2 = clamp(x2, 0, w - 1)
    y2 = clamp(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return class_id, x1, y1, x2, y2


def _xyxy_to_yolo(class_id: int, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> str:
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"{class_id} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def load_dataset_session(dataset_path: Path, model: str, class_names: List[str]) -> DatasetSession:
    dataset_path = Path(dataset_path).resolve()
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError("Dataset must contain images/ and labels/ folders")

    img_files = _load_image_list(images_dir)
    sess = DatasetSession(dataset_path=dataset_path, images_dir=images_dir, labels_dir=labels_dir, model=model, img_files=img_files)

    # Preload labels (fast enough for typical datasets; if big we can lazy-load later)
    for i, img_path in enumerate(sess.img_files):
        w, h = _read_image_size(img_path)
        sess.img_sizes[i] = (w, h)
        txt = sess.label_path(i)
        anns: List[DatasetAnnotation] = []
        if txt.exists():
            try:
                lines = txt.read_text(encoding="utf-8").splitlines()
            except Exception:
                lines = []
            for line in lines:
                parsed = _yolo_to_xyxy(line, w, h)
                if not parsed:
                    continue
                class_id, x1, y1, x2, y2 = parsed
                cname = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
                anns.append(
                    DatasetAnnotation(
                        id=sess.new_id(),
                        image_idx=i,
                        class_id=class_id,
                        class_name=cname,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
        sess.ann_by_image[i] = anns
    return sess


def save_dataset_session(sess: DatasetSession, strategy: str) -> Path:
    """
    strategy:
      - overwrite: write labels into the same dataset folder, delete images/labels marked for deletion
      - create_new: write into <dataset>_fixed, copying/linking images and writing labels (skip deleted)
    Returns the output dataset path.
    """
    strategy = (strategy or "").lower().strip()
    if strategy not in ("overwrite", "create_new"):
        raise ValueError("strategy must be overwrite or create_new")
    # Relabel batch (MDQ-15c-2): only overwrite, and Delete never removes files here. The Delete /
    # Background marks are written by the server to the batch's fixer_decisions.json and acted on
    # by apply_relabel.py (Delete = archive every RAW copy).
    relabel = bool(sess.relabel_items)
    if relabel and strategy != "overwrite":
        raise ValueError("a relabel batch can only be saved with overwrite")

    if strategy == "overwrite":
        out_path = sess.dataset_path
    else:
        out_path = sess.dataset_path.parent / f"{sess.dataset_path.name}_fixed"

    images_dir = out_path / "images"
    labels_dir = out_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Copy/link images if create_new (skip deleted images)
    if strategy == "create_new":
        for i, src in enumerate(sess.img_files):
            if i in sess.deleted_images:
                continue  # skip deleted images
            dst = images_dir / src.name
            if dst.exists():
                continue
            try:
                os.link(src, dst)  # hardlink if possible
            except Exception:
                shutil.copy2(src, dst)

    # Write labels (skip deleted images; a relabel batch keeps them, with their (empty) label)
    for i, img_path in enumerate(sess.img_files):
        if i in sess.deleted_images and not relabel:
            continue  # skip deleted images
        w, h = sess.img_sizes[i]
        anns = sess.ann_by_image.get(i, [])
        out_txt = labels_dir / f"{img_path.stem}.txt"
        with open(out_txt, "w", encoding="utf-8") as fp:
            for a in anns:
                if a.proposal or a.decoy:   # MDQ-17c: an undecided proposal / a QA decoy never reaches labels/
                    continue
                fp.write(_xyxy_to_yolo(a.class_id, a.x1, a.y1, a.x2, a.y2, w, h) + "\n")

    # Delete images and labels marked for deletion (only when overwriting)
    if strategy == "overwrite" and sess.deleted_images and not relabel:
        for i in sess.deleted_images:
            if i >= len(sess.img_files):
                continue
            img_path = sess.img_files[i]
            # Delete image file
            img_file = sess.images_dir / img_path.name
            if img_file.exists():
                try:
                    img_file.unlink()
                except Exception:
                    pass
            # Delete label file
            label_file = sess.labels_dir / f"{img_path.stem}.txt"
            if label_file.exists():
                try:
                    label_file.unlink()
                except Exception:
                    pass

    return out_path


def ann_to_dict(a: DatasetAnnotation) -> Dict:
    d = asdict(a)
    d.pop("decoy", None)   # the browser must not be able to tell a QA decoy from a real proposal
    return d


def add_proposals(sess: DatasetSession, proposals: Dict[str, List[Dict]], decisions: Dict[str, Dict[str, str]],
                  class_names: List[str]) -> int:
    """MDQ-17c: put every undecided proposal of the batch on its image as a proposal box. Returns how many.
    A proposal whose class is not in the model is skipped (it could never be saved)."""
    n = 0
    for i, img_path in enumerate(sess.img_files):
        props = proposals.get(img_path.name) or []
        done = decisions.get(img_path.name, {})
        w, h = sess.img_sizes[i]
        for pr in props:
            if pr["pid"] in done or pr["class_name"] not in class_names:
                continue
            x1, y1, x2, y2 = pr["box"]
            a = DatasetAnnotation(id=sess.new_id(), image_idx=i, class_id=class_names.index(pr["class_name"]),
                                  class_name=pr["class_name"],
                                  x1=clamp(int(x1 * w), 0, w - 1), y1=clamp(int(y1 * h), 0, h - 1),
                                  x2=clamp(int(x2 * w), 0, w - 1), y2=clamp(int(y2 * h), 0, h - 1),
                                  proposal=True, proposal_id=pr["pid"], proposal_class=pr["class_name"],
                                  proposal_conf=pr.get("conf"), proposal_agreement=pr.get("agreement"),
                                  decoy=bool(pr.get("decoy")))
            if a.x2 <= a.x1 or a.y2 <= a.y1:
                continue
            sess.ann_by_image.setdefault(i, []).append(a)
            sess.proposals_by_image.setdefault(i, []).append(pr)
            n += 1
    return n


def proposal_decisions(sess: DatasetSession) -> Dict[str, Dict[str, str]]:
    """MDQ-17c: what happened to each proposal loaded in this session (still-undecided ones are left out)."""
    out: Dict[str, Dict[str, str]] = {}
    for i, props in sess.proposals_by_image.items():
        anns = {a.proposal_id: a for a in sess.ann_by_image.get(i, []) if a.proposal_id}
        dec = {}
        for pr in props:
            a = anns.get(pr["pid"])
            if a is None:
                dec[pr["pid"]] = "rejected"
            elif not a.proposal:
                dec[pr["pid"]] = "confirmed" if a.class_name == a.proposal_class else "reclassed"
        if dec:
            out[sess.image_name(i)] = dec
    return out


def pending_count(sess: DatasetSession, idx: Optional[int] = None) -> int:
    images = [idx] if idx is not None else list(sess.ann_by_image.keys())
    return sum(1 for i in images for a in sess.ann_by_image.get(i, []) if a.proposal)


