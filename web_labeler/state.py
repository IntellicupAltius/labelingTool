from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid


@dataclass
class Annotation:
    id: str
    frame_idx: int
    model: str
    class_name: str
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class AppState:
    # Models
    models_dir: Path
    model_to_names: Dict[str, List[str]] = field(default_factory=dict)

    # Videos
    videos_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "videos")
    video_path: Optional[Path] = None

    # OpenCV state (kept in server module)
    total_frames: int = 0
    fps: float = 0.0
    img_w: int = 0
    img_h: int = 0

    # Annotations
    ann_by_frame: Dict[int, List[Annotation]] = field(default_factory=dict)
    last_annotation: Optional[Tuple[str, str]] = None  # (model, class_name)
    # Background markers: frame_idx -> set(models)
    background_by_frame: Dict[int, set[str]] = field(default_factory=dict)

    # Where we save work/export
    output_base_dir: Path = field(default_factory=lambda: Path.cwd() / "output")

    def reset_video_state(self):
        self.video_path = None
        self.total_frames = 0
        self.fps = 0.0
        self.img_w = 0
        self.img_h = 0
        self.ann_by_frame.clear()
        self.last_annotation = None
        self.background_by_frame.clear()

    def new_annotation_id(self) -> str:
        return uuid.uuid4().hex


