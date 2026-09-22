from __future__ import annotations

import base64
import json
import os
import re
import shutil
import threading
import logging
import platform
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Stability knobs ---
# We've seen FFmpeg/OpenCV crash with pthread_frame assertions on some Linux builds
# when decoding in multiple threads or when VideoCapture is accessed concurrently.
# Force conservative thread usage by default (can be overridden by env vars).
os.environ.setdefault("OPENCV_FFMPEG_THREAD_COUNT", "1")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2  # noqa: E402
from fastapi import FastAPI, HTTPException, Query, Request, Body
from fastapi import UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web_labeler import __version__ as app_version
from web_labeler.state import Annotation, AppState
from web_labeler.video_io import VideoReader, clamp
from web_labeler.yolo import load_yolo_names_from_yaml, yolo_line
from web_labeler.naming import ParsedVideoName, parse_video_name, export_base_name, output_video_dirname, batch_dirname
from web_labeler.dataset import (
    DatasetSession,
    DatasetAnnotation,
    list_dataset_folders,
    load_dataset_session,
    save_dataset_session,
    ann_to_dict,
    _read_image_size,
    _yolo_to_xyxy,
)
from web_labeler.background_labeler import (
    BgSession,
    start_session as bg_start_session,
    start_session_folder_mode,
    start_session_existing_mode,
    copy_background_image,
    remove_background_image,
    sanitize_part,
)
from web_labeler import analyzer as _analyzer
from web_labeler import flag_store as _flags
from web_labeler.flag_store import FlagConflict, FlagError, FlagStore, FlagUnavailable, normalize_pool, validate_image_key


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


logger = logging.getLogger("labeler")


def _pos_to_variation_label(pos_name: str) -> str:
    """Derive a short variation label from a POS article name."""
    label = pos_name.strip()
    label = re.sub(r'\s+0[,\.]\d+\s*$', '', label).strip()   # strip size suffix
    label = re.sub(r'^Aleksic\s+', '', label, flags=re.IGNORECASE).strip()  # strip "Aleksic " prefix
    label = label.upper().replace(' ', '_')
    return label


def _build_class_variations(article_map_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Parse article_map.yaml → {model: {class: [variation_labels]}}.
    Only classes with ≥2 active (status: incl) POS articles get an entry.
    """
    if not article_map_path or not article_map_path.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(article_map_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not parse article_map for class variations: %s", e)
        return {}

    class_articles: Dict[tuple, List[str]] = {}
    for article in data.get("articles", []):
        if article.get("status") != "incl":
            continue
        pos_name = str(article.get("pos_name", "")).strip()
        if not pos_name:
            continue
        for mapping in article.get("mappings", []):
            model = mapping.get("model")
            cls = mapping.get("class")
            if not model or not cls:
                continue
            key = (model, cls)
            if key not in class_articles:
                class_articles[key] = []
            if pos_name not in class_articles[key]:
                class_articles[key].append(pos_name)

    result: Dict[str, Dict[str, List[str]]] = {}
    for (model, cls), pos_names in class_articles.items():
        labels = list(dict.fromkeys(_pos_to_variation_label(p) for p in pos_names))  # ordered dedup
        if len(labels) < 2:
            continue
        if model not in result:
            result[model] = {}
        result[model][cls] = labels

    return result


def sanitize_filename_part(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def video_stem_without_uuid(stem: str) -> str:
    # matches what you did in v6_1: split on "{...}" and trim
    return stem.split("{")[0].rstrip("_-")


class LoadVideoRequest(BaseModel):
    video_name: str = Field(..., description="Filename under videos directory (or under the batch subfolder, if batch is set)")
    load_existing_exports: bool = Field(False, description="If true, load annotations from existing exported labels on disk")
    batch: Optional[str] = Field(None, description="MDQ-4b: handoff_<session_id> subfolder to load video_name from, instead of the flat videos directory")


class AddAnnotationRequest(BaseModel):
    frame_idx: int
    model: str
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int


class MarkBackgroundRequest(BaseModel):
    frame_idx: int
    model: str


class ExportRequest(BaseModel):
    bar_counter: Optional[str] = Field(None, description="Override BAR_COUNTER_INFO (e.g. SANK_LEVO)")

#
# Dataset Fixer API models (module-level to avoid FastAPI/Pydantic edge cases)
#
class DatasetLoadRequest(BaseModel):
    dataset_name: str
    model: str


class DatasetSaveRequest(BaseModel):
    strategy: str = Field(..., description="overwrite or create_new")


class DatasetAddAnnRequest(BaseModel):
    image_idx: int
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int


class DatasetUpdateAnnRequest(BaseModel):
    class_name: Optional[str] = None
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None


class DatasetMarkBackgroundRequest(BaseModel):
    image_idx: int


class DatasetMarkDeleteRequest(BaseModel):
    image_idx: int


class BgStartRequest(BaseModel):
    mode: str = "folder"  # "folder" or "existing"
    dataset_name: Optional[str] = None  # for folder mode: folder name/path; for existing: not used
    folder_path: Optional[str] = None  # for folder mode: explicit folder path (relative to datasets_dir or absolute)
    existing_datasets_dir: Optional[str] = None  # for existing mode: path to datasets/existing/ (relative to datasets_dir or absolute)
    target_model: str
    camera_filter: str = "ALL"  # e.g. SANK_DESNO
    shuffled: bool = True
    seed: int = 1337


class BgDecisionRequest(BaseModel):
    action: str  # "background" | "skip"


class TestModeRequest(BaseModel):
    test_mode: bool

class ImageTagsRequest(BaseModel):
    image_idx: int
    tags: List[str]


class FrameTagsRequest(BaseModel):
    frame_idx: int
    frame_tags: List[str] = []
    bbox_tags: Dict[str, List[str]] = {}       # ann_id → [tags]
    bbox_variations: Dict[str, str] = {}       # ann_id → variation label (single string)


class DatasetTagsRequest(BaseModel):
    image_idx: int
    frame_tags: List[str] = []
    bbox_tags: Dict[str, List[str]] = {}       # ann_id → [tags]
    bbox_variations: Dict[str, str] = {}       # ann_id → variation label (single string)


class AnalyzerClassifyRequest(BaseModel):
    model: str
    image_b64: str  # base64-encoded image (JPEG/PNG)
    bbox: Optional[Dict] = None  # {x1,y1,x2,y2} in pixels


class AnalyzerOverlapRequest(BaseModel):
    model: str
    samples: int = 20


class RawFlagRequest(BaseModel):
    image_key: str  # "<class>/<filename>", relative to <pool>/images/
    category: str
    comment: str = ""
    pool: Optional[str] = None  # defaults to the loaded raw session's model


class ReviewDecisionRequest(BaseModel):
    pool: str
    image_key: str
    decision: Optional[str] = None  # "approve_delete" | "keep" | null (back to pending)


class FrameFlagRequest(BaseModel):
    model: str  # the pool the frame is flagged for
    frame_idx: int
    category: str
    comment: str = ""


def create_app() -> FastAPI:
    # Directories (configurable via env vars)
    base_dir = Path(__file__).resolve().parents[1]
    config_path = base_dir / "labeler_config.json"

    def _default_data_root() -> Path:
        # Cross-platform (Windows/Linux): puts data outside the project by default.
        return (Path.home() / "LabelingToolData").resolve()

    def _load_or_create_config() -> Dict[str, str]:
        default_cfg = {
            # Use a portable "~" form so config can be copied between Windows/Linux.
            "data_root": "~/LabelingToolData",
            # Optional overrides (if empty/missing, derived from data_root)
            "models_dir": "",
            "videos_dir": "",
            "output_dir": "",
            "datasets_dir": "",
            "existing_datasets_dir": "",  # defaults to {datasets_dir}/existing
            "bar_counter_options": ["SANK_LEVO", "SANK_DESNO", "SANK_TOCILICA"],
        }
        if not config_path.exists():
            config_path.write_text(json.dumps(default_cfg, indent=2), encoding="utf-8")
            return default_cfg
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(cfg, dict):
                return default_cfg

            # If someone copied a Linux config onto Windows (or vice versa), paths can break.
            # Detect and ignore clearly non-portable absolute paths on Windows.
            if platform.system().lower().startswith("win"):
                for k in ("models_dir", "videos_dir", "output_dir", "datasets_dir"):
                    v = cfg.get(k)
                    if isinstance(v, str) and v.strip().startswith("/"):
                        cfg[k] = ""

            # allow overriding only some keys
            merged = dict(default_cfg)
            for k in ("data_root", "models_dir", "videos_dir", "output_dir", "datasets_dir", "existing_datasets_dir", "raw_base_path", "raw_review_dir"):
                if isinstance(cfg.get(k), str) and cfg.get(k).strip():
                    merged[k] = cfg[k].strip()
            if isinstance(cfg.get("bar_counter_options"), list) and cfg.get("bar_counter_options"):
                merged["bar_counter_options"] = cfg["bar_counter_options"]
            return merged
        except Exception:
            return default_cfg

    cfg = _load_or_create_config()

    def _resolve_cfg_path(key: str, fallback: Path) -> Path:
        # Env vars win (useful for advanced setups).
        env_map = {
            "models_dir": "LABELER_MODELS_DIR",
            "videos_dir": "LABELER_VIDEOS_DIR",
            "output_dir": "LABELER_OUTPUT_DIR",
        }
        env_key = env_map.get(key)
        if env_key and os.getenv(env_key):
            return Path(os.getenv(env_key)).expanduser().resolve()
        # If user set data_root, derive other folders from it unless explicitly overridden.
        if key in ("models_dir", "videos_dir", "output_dir"):
            dr = cfg.get("data_root")
            if isinstance(dr, str) and dr.strip():
                root = Path(dr).expanduser()
                if key == "models_dir":
                    fallback = root / "Models"
                elif key == "videos_dir":
                    fallback = root / "videos"
                else:
                    fallback = root / "output"
        if key == "datasets_dir":
            dr = cfg.get("data_root")
            if isinstance(dr, str) and dr.strip():
                fallback = Path(dr).expanduser() / "datasets"
        v = cfg.get(key)
        if isinstance(v, str) and v.strip():
            return Path(v).expanduser().resolve()
        return fallback.resolve()

    # Defaults if config/env missing (kept for backwards compatibility)
    models_dir = _resolve_cfg_path("models_dir", base_dir / "Models")
    videos_dir = _resolve_cfg_path("videos_dir", base_dir / "data" / "videos")
    output_dir = _resolve_cfg_path("output_dir", base_dir / "output")
    datasets_dir = _resolve_cfg_path("datasets_dir", Path(_default_data_root() / "datasets"))
    
    # Resolve existing_datasets_dir (defaults to {datasets_dir}/existing)
    existing_datasets_dir_cfg = cfg.get("existing_datasets_dir", "").strip()
    if existing_datasets_dir_cfg:
        existing_datasets_dir = Path(existing_datasets_dir_cfg).expanduser().resolve()
    else:
        existing_datasets_dir = datasets_dir / "existing"

    # Resolve raw_base_path (optional — raw datasets organised by model/class)
    raw_base_path_cfg = cfg.get("raw_base_path", "").strip()
    raw_base_path: Optional[Path] = Path(raw_base_path_cfg).expanduser().resolve() if raw_base_path_cfg else None

    # MDQ-3: flag sidecars live OUTSIDE the RAW tree. Enabled when raw_review_dir is configured
    # (config key or LABELER_RAW_REVIEW_DIR), or on the server where /opt/intellicup/datasets
    # exists; unavailable (503 on the flag routes) on labeler PCs that have neither.
    _review_cfg = (os.getenv("LABELER_RAW_REVIEW_DIR") or cfg.get("raw_review_dir") or "").strip()
    if _review_cfg:
        _review_root: Optional[Path] = Path(_review_cfg).expanduser()
    elif Path("/opt/intellicup/datasets").is_dir():
        _review_root = Path("/opt/intellicup/datasets/raw_review")
    else:
        _review_root = None
    try:
        flag_store = FlagStore(_review_root, forbidden_roots=[raw_base_path] if raw_base_path else None)
    except ValueError as e:
        logger.error("Flagging disabled: %s", e)
        flag_store = FlagStore(None)
    logger.info("Flagging available: %s (%s)", flag_store.available, flag_store.root)

    debug = os.getenv("LABELER_DEBUG", "").strip() not in ("", "0", "false", "False")
    logging.basicConfig(level=(logging.DEBUG if debug else logging.INFO))
    if debug:
        logger.debug("Debug logging enabled (LABELER_DEBUG=1)")

    ensure_dir(videos_dir)
    ensure_dir(output_dir)
    ensure_dir(models_dir)
    ensure_dir(datasets_dir)

    state = AppState(models_dir=models_dir, videos_dir=videos_dir, output_base_dir=output_dir)
    reader = VideoReader()
    video_lock = threading.Lock()
    dataset_session: Optional[DatasetSession] = None
    test_mode: bool = False
    bg_session: Optional[BgSession] = None
    raw_session: Optional[DatasetSession] = None

    # Build class variations from article_map (optional — graceful if missing)
    _article_map_path_cfg = cfg.get("article_map_path", "").strip()
    _article_map_path = (
        Path(_article_map_path_cfg).expanduser().resolve()
        if _article_map_path_cfg
        else Path.home() / "Projects" / "IntelliCup" / "utils" / "blaznavac_article_map.yaml"
    )
    class_variations: Dict[str, Dict[str, List[str]]] = _build_class_variations(_article_map_path)
    logger.info("Class variations loaded for models: %s", list(class_variations.keys()))

    # Configure analyzer (server-side YOLO inference)
    _analyzer.configure(
        models_root=cfg.get("analyzer_models_root", "").strip()
                   or os.getenv("ANALYZER_MODELS_ROOT", "/opt/intellicup/models"),
        raw_root=cfg.get("analyzer_raw_root", "").strip()
                or os.getenv("ANALYZER_RAW_ROOT", "/opt/intellicup/datasets/raw/blaznavac"),
        models_python=cfg.get("analyzer_models_python", "").strip()
                     or os.getenv("ANALYZER_MODELS_PYTHON", "/opt/interpreters/INTELLICUP_MODELS/bin/python"),
    )
    logger.info("Analyzer available: %s", _analyzer.is_available())

    try:
        # Avoid OpenCV internal thread pools competing with FFmpeg (stability/perf).
        cv2.setNumThreads(0)
    except Exception:
        pass

    static_dir = Path(__file__).resolve().parent / "static"
    app = FastAPI(title="Labeling Tool (Browser)", version="0.1.0")

    @app.exception_handler(Exception)
    async def _json_error_handler(request: Request, exc: Exception):
        # Without this, an unhandled exception falls through to Starlette's
        # plain-text 500 page, which breaks the frontend's `await r.json()`
        # calls (it sees "Internal Server Error" instead of JSON).
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": str(exc) or "Internal server error"})

    @app.exception_handler(FlagError)
    async def _flag_error_handler(request: Request, exc: FlagError):
        body = {"detail": str(exc)}
        if isinstance(exc, FlagConflict) and exc.entry is not None:
            body["entry"] = exc.entry
        if exc.status >= 500 and not isinstance(exc, FlagUnavailable):
            logger.error("Flag store error on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=exc.status, content=body)

    @app.middleware("http")
    async def no_cache_for_static_and_root(request: Request, call_next):
        resp = await call_next(request)
        p = request.url.path or ""
        if p == "/" or p.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-store, max-age=0"
            resp.headers["Pragma"] = "no-cache"
        return resp

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def root():
        return FileResponse(str(static_dir / "index.html"))

    def refresh_models() -> Dict[str, List[str]]:
        model_to_names: Dict[str, List[str]] = {}
        model_to_yaml: Dict[str, Path] = {}
        if not state.models_dir.exists():
            return model_to_names
        for yaml_path in list(state.models_dir.glob("*.yaml")) + list(state.models_dir.glob("*.yml")):
            try:
                model_name = yaml_path.stem
                model_to_names[model_name] = load_yolo_names_from_yaml(yaml_path)
                model_to_yaml[model_name] = yaml_path
            except Exception:
                continue
        state.model_to_names = model_to_names
        state.model_to_yaml_path = model_to_yaml
        return model_to_names

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
    HANDOFF_BATCH_PREFIX = "handoff_"   # must match intellicup_deep_sort/api/serve.py's own prefix

    def _is_safe_batch_name(batch: Optional[str]) -> bool:
        """A batch is a single path segment we join onto videos_dir — reject
        anything that isn't exactly a `handoff_<...>` folder name we ourselves
        would have listed, so a crafted `batch` can never escape videos_dir
        (no slashes, no `..`, no absolute path)."""
        if not batch:
            return True   # no batch = flat mode, always fine
        if not batch.startswith(HANDOFF_BATCH_PREFIX):
            return False
        if "/" in batch or "\\" in batch or ".." in batch:
            return False
        return True

    def _videos_root_for(batch: Optional[str]) -> Optional[Path]:
        """The directory list_videos()/load_video() actually read from — either
        videos_dir itself (flat, batch=None) or one of its handoff_* subfolders.
        Returns None for an invalid/unsafe batch name (caller decides how to fail)."""
        if not batch:
            return state.videos_dir
        if not _is_safe_batch_name(batch):
            return None
        return state.videos_dir / batch

    def list_videos(batch: Optional[str] = None) -> List[str]:
        root = _videos_root_for(batch)
        if root is None or not root.exists():
            return []
        vids = [p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        vids.sort()
        return vids

    def list_video_batches() -> List[Dict[str, Any]]:
        """MDQ-4b: the handoff_* session subfolders written by intellicup_deep_sort's
        send_to_labeling — each is one batch a labeler can load instead of the flat
        video list. Reads that same repo's own `_session.json` shape directly (no
        import across repos — this is just a JSON file on shared disk, same
        cross-repo pattern as blaznavac_article_map.yaml elsewhere in this file),
        tolerant of it being missing/unreadable (an old or hand-made folder still
        shows up, just without session metadata)."""
        if not state.videos_dir.exists():
            return []
        out: List[Dict[str, Any]] = []
        for p in sorted(state.videos_dir.iterdir()):
            if not p.is_dir() or not p.name.startswith(HANDOFF_BATCH_PREFIX):
                continue
            clip_count = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
            meta: Dict[str, Any] = {}
            meta_path = p / "_session.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            out.append({
                "name":         p.name,
                "session_id":   meta.get("session_id", p.name[len(HANDOFF_BATCH_PREFIX):]),
                "opened_at":    meta.get("opened_at"),
                "closed_at":    meta.get("closed_at"),
                "clip_count":   clip_count,
            })
        return out

    def is_supported_video_name(name: str) -> bool:
        return Path(name).suffix.lower() in VIDEO_EXTS

    # NOTE: We intentionally do NOT persist/restore annotations to disk.
    # Requirement: switching videos and restarting the server should start clean.

    @app.get("/api/config")
    def get_config():
        models = refresh_models()
        videos = list_videos()
        bar_counter_options = cfg.get("bar_counter_options") or ["SANK_LEVO", "SANK_DESNO"]
        detected = None
        if state.video_path is not None:
            detected = parse_video_name(state.video_path.name, bar_counter_options=bar_counter_options).bar_counter
        return {
            "app_version": app_version,
            "config_file": str(config_path),
            "models_dir": str(state.models_dir),
            "videos_dir": str(state.videos_dir),
            "output_dir": str(state.output_base_dir),
            "datasets_dir": str(datasets_dir),
            "models": [{"name": k, "class_count": len(v)} for k, v in sorted(models.items())],
            "videos": videos,
            "video_batches": list_video_batches(),   # MDQ-4b: handoff_* subfolders, flat "videos" list is unaffected
            "bar_counter_options": bar_counter_options,
            "bar_counter_detected": detected,
            "raw_configured": raw_base_path is not None and raw_base_path.is_dir(),
            "flagging_available": flag_store.available,
        }

    @app.get("/api/video/list")
    def api_list_videos(batch: Optional[str] = Query(None)):
        """MDQ-4b: video names inside one handoff_* batch (or the flat videos_dir
        list, same as /api/config's own `videos`, when batch is omitted) — used
        when the labeler switches the batch selector without a full page reload."""
        if batch and not _is_safe_batch_name(batch):
            raise HTTPException(status_code=400, detail="Invalid batch name")
        return {"videos": list_videos(batch)}

    @app.get("/api/class_variations")
    def get_class_variations():
        """Return {model: {class: [variation_labels]}} for classes with ≥2 active POS articles."""
        return class_variations

    # ---------------- Dataset Fixer API ----------------
    @app.get("/api/datasets")
    def datasets_list():
        return {"datasets_dir": str(datasets_dir), "datasets": list_dataset_folders(datasets_dir)}

    # ---------------- Raw Dataset API ----------------

    @app.get("/api/raw/models")
    def raw_list_models():
        if not raw_base_path or not raw_base_path.is_dir():
            raise HTTPException(status_code=404, detail="raw_base_path not configured or missing")
        models = sorted(p.name for p in raw_base_path.iterdir() if p.is_dir() and (p / "images").is_dir())
        return {"raw_base_path": str(raw_base_path), "models": models}

    @app.get("/api/raw/classes")
    def raw_list_classes(model: str = Query(...)):
        if not raw_base_path or not raw_base_path.is_dir():
            raise HTTPException(status_code=404, detail="raw_base_path not configured or missing")
        images_dir = raw_base_path / model / "images"
        if not images_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"images dir not found for model '{model}'")
        classes = sorted(p.name for p in images_dir.iterdir() if p.is_dir())
        return {"model": model, "classes": classes}

    @app.post("/api/raw/load")
    def raw_load(req: dict = Body(...)):
        nonlocal raw_session
        model = req.get("model", "").strip()
        class_name = req.get("class_name", "").strip()
        if not model or not class_name:
            raise HTTPException(status_code=400, detail="model and class_name are required")
        if not raw_base_path or not raw_base_path.is_dir():
            raise HTTPException(status_code=404, detail="raw_base_path not configured or missing")
        images_dir = raw_base_path / model / "images" / class_name
        labels_dir = raw_base_path / model / "labels" / class_name
        if not images_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"images dir not found: {images_dir}")
        if not labels_dir.is_dir():
            labels_dir.mkdir(parents=True, exist_ok=True)
        refresh_models()
        # case-insensitive lookup — YAML files may be capitalized (e.g. Bottles.yaml vs bottles)
        _key = next((k for k in state.model_to_names if k.lower() == model.lower()), None)
        class_names = state.model_to_names.get(_key) if _key else None
        class_names = class_names or [class_name]
        camera = req.get("camera", "").strip().upper()
        img_files = [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        if camera and camera != "ALL":
            img_files = [p for p in img_files if camera in p.name.upper()]
        if not img_files:
            raise HTTPException(status_code=404, detail=f"No images found in {images_dir}" + (f" for camera {camera}" if camera and camera != "ALL" else ""))
        sess = DatasetSession(
            dataset_path=raw_base_path / model,
            images_dir=images_dir,
            labels_dir=labels_dir,
            model=model,
            img_files=[p.resolve() for p in img_files],
        )
        for i, img_path in enumerate(sess.img_files):
            w, h = _read_image_size(img_path)
            sess.img_sizes[i] = (w, h)
            txt = labels_dir / f"{img_path.stem}.txt"
            anns = []
            if txt.exists():
                try:
                    lines = txt.read_text(encoding="utf-8").splitlines()
                except Exception:
                    lines = []
                for line in lines:
                    parsed = _yolo_to_xyxy(line, w, h)
                    if not parsed:
                        continue
                    cid, x1, y1, x2, y2 = parsed
                    cname = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
                    anns.append(DatasetAnnotation(
                        id=sess.new_id(), image_idx=i, class_id=cid, class_name=cname,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                    ))
            sess.ann_by_image[i] = anns
        raw_session = sess
        return {
            "model": model,
            "class_name": class_name,
            "images_dir": str(images_dir),
            "image_count": len(sess.img_files),
        }

    @app.get("/api/raw/image")
    def raw_image(index: int = Query(..., ge=0)):
        if raw_session is None:
            raise HTTPException(status_code=400, detail="No raw session loaded")
        if index >= len(raw_session.img_files):
            raise HTTPException(status_code=400, detail="Index out of range")
        img_path = raw_session.img_files[index]
        media = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return FileResponse(str(img_path), media_type=media, headers={
            "X-Image-Index": str(index),
            "X-Image-Name": img_path.name,
            "Cache-Control": "no-store",
        })

    @app.get("/api/raw/annotations")
    def raw_annotations(index: int = Query(..., ge=0)):
        if raw_session is None:
            raise HTTPException(status_code=400, detail="No raw session loaded")
        idx = int(index)
        anns = raw_session.ann_by_image.get(idx, [])
        image_key = f"{raw_session.images_dir.name}/{raw_session.img_files[idx].name}" if idx < len(raw_session.img_files) else None
        flag = None
        flag_error = None
        if image_key and flag_store.available:
            try:
                flag = flag_store.get(raw_session.model, image_key)
            except FlagError as e:  # a bad sidecar must not break browsing
                flag_error = str(e)
        return {
            "image_idx": idx,
            "annotations": [ann_to_dict(a) for a in anns],
            "is_background": False,
            "is_deleted": False,
            "image_key": image_key,
            "flag": flag,
            "flag_error": flag_error,
        }

    # ---------------- Flag API (MDQ-3) ----------------
    # Flags only ever write the per-pool sidecar (flag_store.py) — never anything under RAW.

    def _need_flags() -> None:
        if not flag_store.available:
            raise FlagUnavailable("flagging is not available on this machine (no raw_review_dir)")

    def _raw_pool(pool: Optional[str]) -> str:
        return normalize_pool(pool or (raw_session.model if raw_session is not None else None))

    def _require_raw_image(pool: str, image_key: str) -> None:
        if raw_base_path is None or not raw_base_path.is_dir():
            raise HTTPException(status_code=404, detail="raw_base_path not configured or missing")
        cls, fname = image_key.split("/")
        if not (raw_base_path / pool / "images" / cls / fname).is_file():
            raise HTTPException(status_code=404, detail=f"No such RAW image: {pool}/images/{image_key}")

    @app.get("/api/raw/flag")
    def raw_flag_get(image_key: str = Query(...), pool: Optional[str] = Query(None)):
        _need_flags()
        p = _raw_pool(pool)
        validate_image_key(image_key)
        return {"pool": p, "image_key": image_key, "entry": flag_store.get(p, image_key)}

    @app.post("/api/raw/flag")
    def raw_flag_add(req: RawFlagRequest):
        _need_flags()
        p = _raw_pool(req.pool)
        validate_image_key(req.image_key)
        _require_raw_image(p, req.image_key)
        entry = flag_store.add_manual_flag(p, req.image_key, req.category, req.comment, _flags.CONTEXT_RETROACTIVE)
        return {"pool": p, "image_key": req.image_key, "entry": entry}

    @app.delete("/api/raw/flag")
    def raw_flag_remove(image_key: str = Query(...), pool: Optional[str] = Query(None)):
        _need_flags()
        p = _raw_pool(pool)
        validate_image_key(image_key)
        return {"pool": p, "image_key": image_key, "removed": flag_store.remove_manual_flag(p, image_key)}

    def _frame_flag_key(frame_idx: int) -> str:
        if state.video_path is None:
            raise HTTPException(status_code=400, detail="No video loaded")
        if frame_idx < 0 or frame_idx >= max(state.total_frames, 1):
            raise HTTPException(status_code=400, detail="frame_idx out of range")
        parsed = parse_video_name(state.video_path.name, bar_counter_options=cfg.get("bar_counter_options") or ["SANK_LEVO", "SANK_DESNO"])
        return f"{_flags.FORWARD_DIR}/{export_base_name(parsed, frame_idx)}.jpg"

    @app.get("/api/frame/flag")
    def frame_flag_get(model: str = Query(...), frame_idx: int = Query(..., ge=0)):
        _need_flags()
        p = normalize_pool(model)
        key = _frame_flag_key(int(frame_idx))
        return {"pool": p, "image_key": key, "entry": flag_store.get(p, key)}

    @app.post("/api/frame/flag")
    def frame_flag_add(req: FrameFlagRequest):
        _need_flags()
        p = normalize_pool(req.model)
        idx = int(req.frame_idx)
        key = _frame_flag_key(idx)
        classes = sorted({a.class_name for a in state.ann_by_frame.get(idx, []) if (a.model or "").lower() == p})
        entry = flag_store.add_manual_flag(
            p, key, req.category, req.comment, _flags.CONTEXT_FORWARD,
            extra={"video": state.video_path.name, "frame_idx": idx, "annotation_classes": classes},
        )
        return {"pool": p, "image_key": key, "entry": entry}

    @app.delete("/api/frame/flag")
    def frame_flag_remove(model: str = Query(...), frame_idx: int = Query(..., ge=0)):
        _need_flags()
        p = normalize_pool(model)
        key = _frame_flag_key(int(frame_idx))
        return {"pool": p, "image_key": key, "removed": flag_store.remove_manual_flag(p, key)}

    # ---------------- Owner review queue (MDQ-6) ----------------
    # One list over every pool's sidecar, whatever the source (manual_goca / analyzer_auto /
    # data4_audit). The owner's approve-delete / keep verdict goes back into the same sidecar
    # entry (owner_decision). RAW is only ever READ here (thumbnail + label boxes); moving an
    # approved image out of RAW is MDQ-8's archive script.

    def _raw_image_path(pool: str, image_key: str) -> Optional[Path]:
        """RAW image for a non-forward key, or None if there is none (forward frame, missing file)."""
        if raw_base_path is None or image_key.startswith(_flags.FORWARD_DIR + "/"):
            return None
        try:
            validate_image_key(image_key)
        except FlagError:
            return None
        cls, fname = image_key.split("/")
        p = raw_base_path / pool / "images" / cls / fname
        return p if p.is_file() else None

    def _raw_label_boxes(pool: str, image_key: str, img_path: Path) -> List[dict]:
        """Normalized YOLO boxes from the image's label file (read-only), with class names."""
        cls = image_key.split("/")[0]
        txt = raw_base_path / pool / "labels" / cls / f"{img_path.stem}.txt"
        if not txt.is_file():
            return []
        _key = next((k for k in state.model_to_names if k.lower() == pool), None)
        names = state.model_to_names.get(_key) if _key else None
        boxes: List[dict] = []
        try:
            lines = txt.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                xc, yc, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            cname = names[cid] if names and 0 <= cid < len(names) else f"class_{cid}"
            boxes.append({"class_id": cid, "class_name": cname, "xc": xc, "yc": yc, "w": w, "h": h})
        return boxes

    @app.get("/api/review/queue")
    def review_queue():
        _need_flags()
        if not state.model_to_names:
            refresh_models()
        pools = {}
        items = []
        for pool, res in flag_store.list_all().items():
            if "error" in res:
                pools[pool] = {"count": 0, "error": res["error"]}
                continue
            pools[pool] = {"count": len(res["entries"]), "error": None}
            for key, entry in res["entries"].items():
                forward = key.startswith(_flags.FORWARD_DIR + "/")
                img_path = _raw_image_path(pool, key)
                reason = _flags.describe_reason(entry)
                items.append({
                    "pool": pool,
                    "image_key": key,
                    "class_name": None if forward else key.split("/")[0],
                    "filename": key.split("/")[-1],
                    "is_forward": forward,
                    "raw_exists": img_path is not None,
                    "reason": reason,
                    "reason_missing": reason is None,
                    "boxes": _raw_label_boxes(pool, key, img_path) if img_path else [],
                    "entry": entry,
                })
        items.sort(key=lambda it: it["entry"].get("flagged_at") or "", reverse=True)
        return {"pools": pools, "items": items, "sources": sorted({it["entry"].get("source") or "" for it in items})}

    @app.get("/api/review/image")
    def review_image(pool: str = Query(...), image_key: str = Query(...), thumb: int = Query(1)):
        _need_flags()
        p = normalize_pool(pool)
        validate_image_key(image_key)
        img_path = _raw_image_path(p, image_key)
        if img_path is None:
            raise HTTPException(status_code=404, detail=f"No such RAW image: {p}/images/{image_key}")
        if thumb:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                scale = min(1.0, 480.0 / max(w, 1))
                if scale < 1.0:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    return Response(content=buf.tobytes(), media_type="image/jpeg",
                                    headers={"Cache-Control": "private, max-age=300"})
        media = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return FileResponse(str(img_path), media_type=media, headers={"Cache-Control": "private, max-age=300"})

    @app.post("/api/review/decision")
    def review_decision(req: ReviewDecisionRequest):
        _need_flags()
        p = normalize_pool(req.pool)
        entry = flag_store.set_owner_decision(p, req.image_key, req.decision)
        return {"pool": p, "image_key": req.image_key, "entry": entry}

    # ---------------- Background Labeler API ----------------
    @app.get("/api/background_labeler/config")
    def bg_config():
        refresh_models()
        models = sorted(list(state.model_to_names.keys()))
        # camera filters from bar counter options + ALL
        bar_counter_options = cfg.get("bar_counter_options") or ["SANK_LEVO", "SANK_DESNO", "SANK_TOCILICA"]
        camera_filters = ["ALL"] + [str(x) for x in bar_counter_options]
        # Use configured existing_datasets_dir
        has_existing = existing_datasets_dir.exists() and existing_datasets_dir.is_dir()
        return {
            "datasets_dir": str(datasets_dir),
            "datasets": list_dataset_folders(datasets_dir),
            "models": models,
            "camera_filters": camera_filters,
            "has_existing": has_existing,
            "existing_path": str(existing_datasets_dir),
        }

    @app.post("/api/background_labeler/start")
    def bg_start(req: BgStartRequest = Body(...)):
        nonlocal bg_session
        refresh_models()
        if req.target_model not in state.model_to_names:
            raise HTTPException(status_code=400, detail="Invalid model")

        mode = (req.mode or "folder").lower().strip()
        try:
            if mode == "folder":
                # Mode 1: Open folder / Manual set
                if req.folder_path:
                    # Explicit path (can be absolute or relative to datasets_dir)
                    folder_path = Path(req.folder_path)
                    if not folder_path.is_absolute():
                        folder_path = datasets_dir / folder_path
                elif req.dataset_name:
                    # Legacy: treat dataset_name as folder name
                    folder_path = datasets_dir / req.dataset_name
                else:
                    raise ValueError("folder_path or dataset_name required for folder mode")
                bg_session = start_session_folder_mode(
                    folder_path=folder_path,
                    target_model=req.target_model,
                    camera_filter=req.camera_filter,
                    shuffled=bool(req.shuffled),
                    seed=int(req.seed),
                    datasets_dir=datasets_dir,
                )
            elif mode == "existing":
                # Mode 2: From existing datasets - always use configured path
                bg_session = start_session_existing_mode(
                    existing_datasets_dir=existing_datasets_dir,
                    target_model=req.target_model,
                    camera_filter=req.camera_filter,
                    shuffled=bool(req.shuffled),
                    seed=int(req.seed),
                )
            else:
                raise ValueError(f"Invalid mode: {mode} (must be 'folder' or 'existing')")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "session_id": bg_session.id,
            "mode": bg_session.mode,
            "dataset_name": bg_session.dataset_name,
            "target_model": bg_session.target_model,
            "total": len(bg_session.img_files),
            "out_root": str(bg_session.out_root),
        }

    @app.get("/api/background_labeler/status")
    def bg_status():
        if bg_session is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "session_id": bg_session.id,
            "mode": bg_session.mode,
            "dataset_name": bg_session.dataset_name,
            "target_model": bg_session.target_model,
            "idx": bg_session.idx,
            "total": len(bg_session.img_files),
            "selected": len(bg_session.selected),
            "skipped": len(bg_session.skipped),
            "out_root": str(bg_session.out_root),
            "done": bg_session.done(),
        }

    @app.get("/api/background_labeler/image")
    def bg_image():
        if bg_session is None:
            raise HTTPException(status_code=400, detail="No background session loaded")
        p = bg_session.current_path()
        if p is None:
            raise HTTPException(status_code=400, detail="No more images")
        media = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        decision = "background" if bg_session.idx in bg_session.selected else "skip"
        return FileResponse(
            str(p),
            media_type=media,
            headers={
                "X-Index": str(bg_session.idx),
                "X-Name": p.name,
                "X-Decision": decision,
                "Cache-Control": "no-store, max-age=0",
                "Pragma": "no-cache",
            },
        )

    @app.post("/api/background_labeler/set_index")
    def bg_set_index(index: int = Query(..., ge=0)):
        if bg_session is None:
            raise HTTPException(status_code=400, detail="No background session loaded")
        bg_session.idx = max(0, min(int(index), len(bg_session.img_files)))
        return {"ok": True, "idx": bg_session.idx, "total": len(bg_session.img_files)}

    @app.post("/api/background_labeler/decide")
    def bg_decide(req: BgDecisionRequest = Body(...)):
        if bg_session is None:
            raise HTTPException(status_code=400, detail="No background session loaded")
        if bg_session.done():
            return {"ok": True, "done": True}
        action = (req.action or "").lower().strip()
        cur_idx = bg_session.idx
        p = bg_session.current_path()
        if p is None:
            return {"ok": True, "done": True}

        if action == "background":
            # copy immediately for speed (duplicate protection built-in)
            copied = copy_background_image(bg_session, p)
            if copied:
                bg_session.selected.add(cur_idx)
                bg_session.skipped.discard(cur_idx)
            # If duplicate, still mark as selected (already exported)
            else:
                bg_session.selected.add(cur_idx)
                bg_session.skipped.discard(cur_idx)
        elif action == "skip":
            # if previously selected, remove file outputs
            if cur_idx in bg_session.selected:
                remove_background_image(bg_session, p)
                bg_session.selected.discard(cur_idx)
            bg_session.skipped.add(cur_idx)
        else:
            raise HTTPException(status_code=400, detail="Invalid action")

        bg_session.idx += 1
        return {"ok": True, "idx": bg_session.idx, "done": bg_session.done(), "selected": len(bg_session.selected), "skipped": len(bg_session.skipped)}

    @app.post("/api/background_labeler/finish")
    def bg_finish():
        nonlocal bg_session
        if bg_session is None:
            return {"ok": True, "cleared": True}
        if len(bg_session.selected) == 0:
            # Do not clear; allow user to continue selecting backgrounds.
            raise HTTPException(
                status_code=400,
                detail="No images marked as BACKGROUND. Select at least 1 background (B) before finishing.",
            )
        out_root = bg_session.out_root
        model = bg_session.target_model

        # Copy data.yaml (needed by ingestion pipeline)
        refresh_models()
        yaml_src = state.model_to_yaml_path.get(model)
        if yaml_src and yaml_src.exists():
            yaml_dst = out_root / "data.yaml"
            if not yaml_dst.exists():
                try:
                    shutil.copy2(yaml_src, yaml_dst)
                except Exception:
                    pass

        # Zip and remove folder.
        # Naming convention: zip name == folder inside zip (so unzipped content is identifiable).
        # folder/legacy: {DATASET_NAME}_{TIMESTAMP}_BACKGROUND
        # existing:      {MODEL}_BACKGROUND_{TIMESTAMP}  (out_root.name already encodes this)
        model_slug = re.sub(r"[^A-Z0-9]+", "_", (model or "").upper()).strip("_")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if bg_session.mode in ("folder", "legacy"):
            # out_root = .../datasets/{folder_name}_background/{model}/
            # zip lands next to {folder_name}_background/, i.e. in datasets_dir
            zip_dir = out_root.parent.parent
            zip_dir.mkdir(parents=True, exist_ok=True)
            ds_slug = re.sub(r"[^A-Z0-9]+", "_", (bg_session.dataset_name or "DATASET").upper()).strip("_")
            batch_name = f"{ds_slug}_{timestamp}_BACKGROUND"
            zip_target = zip_dir / f"{batch_name}.zip"
        else:
            # existing mode: out_root.name is already {model}_background_{timestamp}
            batch_name = out_root.name
            zip_target = out_root.parent / f"{batch_name}.zip"
        zip_path_str = str(out_root)
        if out_root.exists():
            try:
                zp = _zip_batch(out_root, zip_path=zip_target, arcname=batch_name)
                zip_path_str = str(zp)
                shutil.rmtree(out_root)
            except Exception as e:
                logger.warning("Could not zip background batch %s: %s", out_root, e)

        bg_session = None
        return {"ok": True, "cleared": True, "zip_path": zip_path_str}

    # ── Analyzer API ──────────────────────────────────────────────────────────

    @app.get("/api/analyzer/config")
    def analyzer_config():
        available = _analyzer.is_available()
        models = _analyzer.get_available_models() if available else []
        return {"available": available, "models": models}

    @app.post("/api/analyzer/classify")
    def analyzer_classify(req: AnalyzerClassifyRequest = Body(...)):
        if not _analyzer.is_available():
            raise HTTPException(status_code=503, detail="Analyzer not available on this machine")
        try:
            image_bytes = base64.b64decode(req.image_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")
        try:
            scores = _analyzer.classify_image(req.model, image_bytes, req.bbox)
            return {"scores": scores}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.exception("classify error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/analyzer/examples")
    def analyzer_examples(model: str = Query(...), class_name: str = Query(...), n: int = Query(4)):
        if not _analyzer.is_available():
            raise HTTPException(status_code=503, detail="Analyzer not available on this machine")
        try:
            examples = _analyzer.get_class_examples(model, class_name, n=n)
            return {"examples": examples}
        except Exception as e:
            logger.exception("examples error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/analyzer/overlap_report")
    def analyzer_overlap_report(req: AnalyzerOverlapRequest = Body(...)):
        if not _analyzer.is_available():
            raise HTTPException(status_code=503, detail="Analyzer not available on this machine")
        try:
            job_id = _analyzer.start_overlap_report(req.model, samples=req.samples)
            return {"job_id": job_id}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.exception("overlap report start error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/analyzer/overlap_status")
    def analyzer_overlap_status(job_id: str = Query(...)):
        status = _analyzer.get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        return status

    # ── Datasets API ──────────────────────────────────────────────────────────

    @app.post("/api/datasets/load")
    def datasets_load(req: DatasetLoadRequest = Body(...)):
        nonlocal dataset_session, test_mode
        refresh_models()
        names = state.model_to_names.get(req.model)
        if not names:
            raise HTTPException(status_code=400, detail="Invalid model")
        ds_path = (datasets_dir / req.dataset_name).resolve()
        if ds_path.parent != datasets_dir or not ds_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        dataset_session = load_dataset_session(ds_path, model=req.model, class_names=names)
        test_mode = False
        mismatch = (req.model.lower() not in req.dataset_name.lower())
        return {
            "dataset_name": req.dataset_name,
            "dataset_path": str(ds_path),
            "model": req.model,
            "image_count": len(dataset_session.img_files),
            "dataset_name_contains_model": (not mismatch),
        }

    @app.get("/api/datasets/image")
    def datasets_image(index: int = Query(..., ge=0)):
        if dataset_session is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        if index >= len(dataset_session.img_files):
            raise HTTPException(status_code=400, detail="Index out of range")
        img_path = dataset_session.image_path(index)
        # Let the browser decode; serve bytes directly
        media = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return FileResponse(str(img_path), media_type=media, headers={
            "X-Image-Index": str(index),
            "X-Image-Name": img_path.name,
            "Cache-Control": "no-store",
        })

    @app.get("/api/datasets/annotations")
    def datasets_annotations(index: int = Query(..., ge=0)):
        if dataset_session is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        idx = int(index)
        anns = dataset_session.ann_by_image.get(idx, [])
        is_background = idx in dataset_session.background_images
        is_deleted = idx in dataset_session.deleted_images
        return {
            "image_idx": idx,
            "annotations": [ann_to_dict(a) for a in anns],
            "is_background": is_background,
            "is_deleted": is_deleted,
        }

    @app.post("/api/datasets/annotations")
    def datasets_add_annotation(req: DatasetAddAnnRequest = Body(...)):
        if dataset_session is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        idx = int(req.image_idx)
        # Background and boxes are mutually exclusive.
        if idx in dataset_session.background_images:
            raise HTTPException(
                status_code=400,
                detail="This image is marked as BACKGROUND. Remove background mark before adding any annotations.",
            )
        names = state.model_to_names.get(dataset_session.model, [])
        if req.class_name not in names:
            raise HTTPException(status_code=400, detail="Invalid class")
        class_id = names.index(req.class_name)
        a = DatasetAnnotation(
            id=dataset_session.new_id(),
            image_idx=idx,
            class_id=class_id,
            class_name=req.class_name,
            x1=int(req.x1),
            y1=int(req.y1),
            x2=int(req.x2),
            y2=int(req.y2),
        )
        dataset_session.ann_by_image.setdefault(a.image_idx, []).append(a)
        return {"ok": True, "annotation": ann_to_dict(a)}

    @app.put("/api/datasets/annotations/{ann_id}")
    def datasets_update_annotation(ann_id: str, req: DatasetUpdateAnnRequest = Body(...)):
        if dataset_session is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        names = state.model_to_names.get(dataset_session.model, [])
        for i, anns in dataset_session.ann_by_image.items():
            for a in anns:
                if a.id != ann_id:
                    continue
                if req.class_name is not None:
                    if req.class_name not in names:
                        raise HTTPException(status_code=400, detail="Invalid class")
                    a.class_name = req.class_name
                    a.class_id = names.index(req.class_name)
                for k in ("x1", "y1", "x2", "y2"):
                    v = getattr(req, k)
                    if v is not None:
                        setattr(a, k, int(v))
                return {"ok": True, "annotation": ann_to_dict(a)}
        raise HTTPException(status_code=404, detail="Annotation not found")

    @app.delete("/api/datasets/annotations/{ann_id}")
    def datasets_delete_annotation(ann_id: str):
        if dataset_session is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        removed = False
        for i in list(dataset_session.ann_by_image.keys()):
            anns = dataset_session.ann_by_image.get(i, [])
            new_anns = [a for a in anns if a.id != ann_id]
            if len(new_anns) != len(anns):
                dataset_session.ann_by_image[i] = new_anns
                removed = True
        return {"ok": True, "removed": removed}

    @app.get("/api/datasets/status")
    def datasets_status():
        if dataset_session is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "total_images": len(dataset_session.img_files),
            "deleted_count": len(dataset_session.deleted_images),
            "background_count": len(dataset_session.background_images),
        }

    @app.post("/api/datasets/save")
    def datasets_save(req: DatasetSaveRequest = Body(...)):
        nonlocal dataset_session, test_mode
        if dataset_session is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        deleted_count = len(dataset_session.deleted_images)
        out_path = save_dataset_session(dataset_session, req.strategy)

        # Copy data.yaml file to output dataset directory
        model = dataset_session.model
        yaml_src = state.model_to_yaml_path.get(model)
        if yaml_src and yaml_src.exists():
            yaml_dst = out_path / "data.yaml"
            if not yaml_dst.exists():
                try:
                    shutil.copy2(yaml_src, yaml_dst)
                except Exception as e:
                    logger.warning("Could not copy data.yaml for model %s to %s: %s", model, yaml_dst, e)

        # Write image_tags.json if any tags are set
        if dataset_session.image_tags:
            tags_by_stem = {}
            for idx, entry in dataset_session.image_tags.items():
                if not (0 <= idx < len(dataset_session.img_files)):
                    continue
                # Handle old flat format
                if isinstance(entry, list):
                    entry = {"frame_tags": entry, "bbox_tags": {}}
                frame_tags = entry.get("frame_tags", [])
                stored_bbox_tags = entry.get("bbox_tags", {})
                stored_bbox_variations = entry.get("bbox_variations", {})
                if not frame_tags and not any(stored_bbox_tags.values()) and not any(stored_bbox_variations.values()):
                    continue
                stem = dataset_session.img_files[idx].stem
                anns = dataset_session.ann_by_image.get(idx, [])
                bbox_tags_list = []
                for a in anns:
                    ann_tags = stored_bbox_tags.get(a.id, [])
                    ann_variation = stored_bbox_variations.get(a.id, "")
                    entry_dict = {
                        "class": a.class_name,
                        "bbox": [a.x1, a.y1, a.x2, a.y2],
                        "tags": ann_tags,
                    }
                    if ann_variation:
                        entry_dict["variation"] = ann_variation
                    bbox_tags_list.append(entry_dict)
                tags_by_stem[stem] = {"frame_tags": frame_tags, "bbox_tags": bbox_tags_list}
            if tags_by_stem:
                (out_path / "image_tags.json").write_text(
                    json.dumps(tags_by_stem, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        # Zip the output so the labeler can drop it directly.
        # Naming convention: zip name == folder inside zip (so unzipped content is identifiable).
        # Format: {DATASET_NAME}_{MODEL}_{TIMESTAMP}
        # For create_new: delete the _fixed folder after zipping.
        # For overwrite: keep the folder (it's the labeler's source dataset).
        model_slug = re.sub(r"[^A-Z0-9]+", "_", (model or "").upper()).strip("_")
        ds_slug = re.sub(r"[^A-Z0-9]+", "_", dataset_session.dataset_path.name.upper()).strip("_")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        batch_name = f"{ds_slug}_{model_slug}_{timestamp}"
        zip_target = out_path.parent / f"{batch_name}.zip"
        zip_path_str = str(out_path)
        try:
            zp = _zip_batch(out_path, zip_path=zip_target, arcname=batch_name)
            zip_path_str = str(zp)
            if req.strategy == "create_new":
                shutil.rmtree(out_path)
        except Exception as e:
            logger.warning("Could not zip dataset %s: %s", out_path, e)

        result = {"ok": True, "zip_path": zip_path_str, "cleared": True}
        if deleted_count > 0:
            result["deleted_count"] = deleted_count
        dataset_session = None
        test_mode = False
        return result

    @app.post("/api/datasets/background")
    def datasets_mark_background(req: DatasetMarkBackgroundRequest = Body(...)):
        if dataset_session is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        idx = int(req.image_idx)
        if dataset_session.ann_by_image.get(idx):
            raise HTTPException(status_code=400, detail="Please remove all annotations from current image to select it as background.")
        dataset_session.background_images.add(idx)
        return {"ok": True}

    @app.delete("/api/datasets/background")
    def datasets_unmark_background(image_idx: int = Query(...)):
        if dataset_session is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        idx = int(image_idx)
        removed = idx in dataset_session.background_images
        dataset_session.background_images.discard(idx)
        return {"ok": True, "removed": removed}

    @app.post("/api/datasets/delete")
    def datasets_mark_delete(req: DatasetMarkDeleteRequest = Body(...)):
        if dataset_session is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        idx = int(req.image_idx)
        if idx < 0 or idx >= len(dataset_session.img_files):
            raise HTTPException(status_code=400, detail="Invalid image index")
        # Can't delete if it has annotations (must remove annotations first)
        if dataset_session.ann_by_image.get(idx):
            raise HTTPException(
                status_code=400,
                detail="Please remove all annotations from current image before marking it for deletion.",
            )
        dataset_session.deleted_images.add(idx)
        return {"ok": True, "deleted": True}

    @app.delete("/api/datasets/delete")
    def datasets_unmark_delete(image_idx: int = Query(...)):
        if dataset_session is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        idx = int(image_idx)
        removed = idx in dataset_session.deleted_images
        dataset_session.deleted_images.discard(idx)
        return {"ok": True, "removed": removed}

    @app.get("/api/datasets/test_mode")
    def datasets_get_test_mode():
        return {"test_mode": test_mode}

    @app.post("/api/datasets/test_mode")
    def datasets_set_test_mode(req: TestModeRequest = Body(...)):
        nonlocal test_mode
        test_mode = req.test_mode
        return {"ok": True, "test_mode": test_mode}

    @app.get("/api/datasets/tags")
    def datasets_get_tags(image_idx: int = Query(..., ge=0)):
        if dataset_session is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        idx = int(image_idx)
        entry = dataset_session.image_tags.get(idx, {})
        if isinstance(entry, list):  # old format
            entry = {"frame_tags": entry, "bbox_tags": {}}
        frame_tags = entry.get("frame_tags", [])
        bbox_tags = entry.get("bbox_tags", {})
        bbox_variations = entry.get("bbox_variations", {})
        return {"image_idx": idx, "frame_tags": frame_tags, "bbox_tags": bbox_tags, "bbox_variations": bbox_variations}

    @app.put("/api/datasets/tags")
    def datasets_set_tags(req: DatasetTagsRequest = Body(...)):
        if dataset_session is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        idx = int(req.image_idx)
        # Validate frame_tags
        valid_frame_tags = {"low_light", "busy", "force_day", "force_night"}
        frame_tags = [t for t in req.frame_tags if t in valid_frame_tags]
        # force_day and force_night are mutually exclusive
        if "force_day" in frame_tags and "force_night" in frame_tags:
            frame_tags = [t for t in frame_tags if t != "force_night"]
        # Validate bbox_tags
        valid_bbox_tags = {"occlusion", "partial", "blurry"}
        bbox_tags = {ann_id: [t for t in tags if t in valid_bbox_tags] for ann_id, tags in req.bbox_tags.items()}
        # Store bbox_variations (any non-empty string value accepted)
        bbox_variations = {ann_id: v for ann_id, v in req.bbox_variations.items() if v and isinstance(v, str)}
        if frame_tags or any(bbox_tags.values()) or any(bbox_variations.values()):
            dataset_session.image_tags[idx] = {"frame_tags": frame_tags, "bbox_tags": bbox_tags, "bbox_variations": bbox_variations}
        else:
            dataset_session.image_tags.pop(idx, None)
        return {"ok": True, "image_idx": idx, "frame_tags": frame_tags, "bbox_tags": bbox_tags, "bbox_variations": bbox_variations}

    @app.get("/api/frame/tags")
    def get_frame_tags(frame_idx: int = Query(..., ge=0)):
        entry = state.frame_tags_by_frame.get(int(frame_idx), {})
        return {
            "frame_idx": int(frame_idx),
            "frame_tags": entry.get("frame_tags", []),
            "bbox_tags": entry.get("bbox_tags", {}),
            "bbox_variations": entry.get("bbox_variations", {}),
        }

    @app.put("/api/frame/tags")
    def set_frame_tags(req: FrameTagsRequest = Body(...)):
        frame_idx = int(req.frame_idx)
        # Validate frame_tags
        valid_frame_tags = {"low_light", "busy", "force_day", "force_night"}
        frame_tags = [t for t in req.frame_tags if t in valid_frame_tags]
        # Enforce mutual exclusion
        if "force_day" in frame_tags and "force_night" in frame_tags:
            frame_tags = [t for t in frame_tags if t != "force_night"]
        # Validate bbox_tags
        valid_bbox_tags = {"occlusion", "partial", "blurry"}
        bbox_tags = {ann_id: [t for t in tags if t in valid_bbox_tags] for ann_id, tags in req.bbox_tags.items()}
        # Store bbox_variations (any non-empty string value accepted)
        bbox_variations = {ann_id: v for ann_id, v in req.bbox_variations.items() if v and isinstance(v, str)}
        if frame_tags or any(bbox_tags.values()) or any(bbox_variations.values()):
            state.frame_tags_by_frame[frame_idx] = {"frame_tags": frame_tags, "bbox_tags": bbox_tags, "bbox_variations": bbox_variations}
        else:
            state.frame_tags_by_frame.pop(frame_idx, None)
        return {"ok": True, "frame_idx": frame_idx, "frame_tags": frame_tags, "bbox_tags": bbox_tags, "bbox_variations": bbox_variations}

    def _zip_batch(folder: Path, zip_path: Optional[Path] = None, arcname: Optional[str] = None) -> Path:
        """
        Zip a batch folder. Default zip name is <folder>.zip beside it.
        Pass zip_path to use a custom output path/name.
        Pass arcname to override the root folder name inside the zip (defaults to folder.name).
        The zip contains the folder itself at the root (pipeline expects a single
        top-level folder inside the zip with images/ labels/ data.yaml).
        Returns the zip path.
        """
        if zip_path is None:
            zip_path = folder.parent / f"{folder.name}.zip"
        root_name = arcname or folder.name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(folder.rglob("*")):
                if file.is_file():
                    zf.write(file, Path(root_name) / file.relative_to(folder))
        return zip_path

    def _output_root_for_video_name(video_name: str) -> Path:
        """
        Compute per-video output folder name, keeping it short (Windows path limits).
        """
        return state.output_base_dir / sanitize_filename_part(output_video_dirname(video_name))

    def _load_existing_exports_into_state(video_name: str) -> tuple:
        """
        Loads existing exported YOLO labels back into the in-memory workspace.
        Searches output_base_dir for batch folders (VIDEO_PREFIX_MODEL/) and zip files
        (VIDEO_PREFIX_MODEL.zip) that match this video — handles both in-progress exports
        (folders still on disk) and completed exports (zipped).
        Returns (loaded_count, last_annotated_frame_idx).
        """
        if state.video_path is None:
            return 0, 0
        if not state.output_base_dir.exists():
            return 0, 0

        video_prefix = output_video_dirname(video_name)  # e.g., BLAZNAVAC_SANK_LEVO_20251205090046
        sep = video_prefix + "_"

        loaded = 0
        last_frame = -1
        state.ann_by_frame.clear()
        state.last_annotation = None

        def _resolve_model_key(raw: str) -> str:
            """Find the actual model key in state.model_to_names (case-insensitive)."""
            for k in state.model_to_names:
                if k.lower() == raw.lower():
                    return k
            return raw.lower()

        def _parse_label_lines(lines, frame_idx: int, model_key: str):
            nonlocal loaded, last_frame
            names = state.model_to_names.get(model_key, [])
            img_w = max(1, int(state.img_w))
            img_h = max(1, int(state.img_h))
            any_loaded = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    class_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                except Exception:
                    continue
                x1 = int((cx - bw / 2.0) * img_w)
                y1 = int((cy - bh / 2.0) * img_h)
                x2 = int((cx + bw / 2.0) * img_w)
                y2 = int((cy + bh / 2.0) * img_h)
                x1 = clamp(x1, 0, img_w - 1)
                y1 = clamp(y1, 0, img_h - 1)
                x2 = clamp(x2, 0, img_w - 1)
                y2 = clamp(y2, 0, img_h - 1)
                if x2 <= x1 or y2 <= y1:
                    continue
                class_name = names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"
                ann = Annotation(
                    id=state.new_annotation_id(),
                    frame_idx=frame_idx,
                    model=model_key,
                    class_name=class_name,
                    class_id=class_id,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
                state.ann_by_frame.setdefault(frame_idx, []).append(ann)
                loaded += 1
                any_loaded = True
            if any_loaded and frame_idx > last_frame:
                last_frame = frame_idx

        def _load_label_text(text: str, filename_stem: str, model_key: str):
            m = re.search(r"_f(\d{6})$", filename_stem)
            if not m:
                return
            frame_idx = int(m.group(1)) - 1
            if frame_idx < 0:
                return
            _parse_label_lines(text.splitlines(), frame_idx, model_key)

        # 1. Batch folders on disk: output_base_dir / VIDEO_PREFIX_MODEL /
        #    (present when export was interrupted before zip+delete)
        models_from_folders: set = set()
        for batch_dir in sorted(state.output_base_dir.iterdir()):
            if not batch_dir.is_dir() or not batch_dir.name.startswith(sep):
                continue
            raw_model = batch_dir.name[len(sep):]
            mk = _resolve_model_key(raw_model)
            labels_dir = batch_dir / "labels"
            if labels_dir.exists():
                for txt_path in sorted(labels_dir.glob("*.txt")):
                    try:
                        _load_label_text(txt_path.read_text(encoding="utf-8"), txt_path.stem, mk)
                    except Exception:
                        pass
            models_from_folders.add(mk)

        # 2. Zip files: output_base_dir / VIDEO_PREFIX_MODEL.zip
        #    (normal case after a completed export)
        for zip_path in sorted(state.output_base_dir.glob("*.zip")):
            if not zip_path.stem.startswith(sep):
                continue
            raw_model = zip_path.stem[len(sep):]
            mk = _resolve_model_key(raw_model)
            if mk in models_from_folders:
                continue  # already loaded from folder; skip to avoid duplicates
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for entry in zf.namelist():
                        norm = entry.replace("\\", "/")
                        parts_e = norm.split("/")
                        if len(parts_e) >= 2 and parts_e[-2] == "labels" and norm.endswith(".txt"):
                            try:
                                text = zf.read(entry).decode("utf-8")
                                _load_label_text(text, Path(parts_e[-1]).stem, mk)
                            except Exception:
                                pass
            except Exception as e:
                logger.warning("Could not read zip %s: %s", zip_path, e)

        if debug:
            logger.debug("Loaded %s existing annotations, last_frame=%s from %s", loaded, last_frame, state.output_base_dir)
        return loaded, max(0, last_frame)

    @app.get("/api/video/info")
    def video_info(video_name: str = Query(...)):
        """
        Lightweight info used by the UI to warn if a video already has exported files.
        Checks for batch folders (VIDEO_PREFIX_MODEL/) and zip files (VIDEO_PREFIX_MODEL.zip)
        in the output_base_dir — both layouts produced by the export pipeline.
        """
        video_prefix = output_video_dirname(video_name)
        sep = video_prefix + "_"
        img_count = 0
        lbl_count = 0

        if state.output_base_dir.exists():
            for p in state.output_base_dir.iterdir():
                if not p.name.startswith(sep):
                    continue
                if p.is_dir():
                    for f in p.rglob("*"):
                        if not f.is_file():
                            continue
                        suf = f.suffix.lower()
                        if suf in (".jpg", ".jpeg", ".png") and f.parent.name == "images":
                            img_count += 1
                        elif suf == ".txt" and f.parent.name == "labels":
                            lbl_count += 1
                elif p.suffix.lower() == ".zip":
                    try:
                        with zipfile.ZipFile(p, "r") as zf:
                            for entry in zf.namelist():
                                norm = entry.replace("\\", "/")
                                if "/labels/" in norm and norm.endswith(".txt"):
                                    lbl_count += 1
                                elif "/images/" in norm and norm.lower().endswith((".jpg", ".jpeg", ".png")):
                                    img_count += 1
                    except Exception:
                        lbl_count += 1  # fallback: at least count the zip

        return {
            "video_name": video_name,
            "output_root": str(state.output_base_dir),
            "image_files": img_count,
            "label_files": lbl_count,
            "has_exports": (img_count > 0 or lbl_count > 0),
        }

    @app.post("/api/video/upload")
    async def upload_video(file: UploadFile = File(...)):
        """
        Upload a video from the browser into the server's videos folder.
        This is intentionally simple for non-technical labelers.
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        if not is_supported_video_name(file.filename):
            raise HTTPException(status_code=400, detail="Unsupported video type")

        original = Path(file.filename).name
        suffix = Path(original).suffix.lower()
        stem = sanitize_filename_part(Path(original).stem)
        safe_name = f"{stem}{suffix}"

        ensure_dir(state.videos_dir)
        dest = (state.videos_dir / safe_name).resolve()
        if dest.parent != state.videos_dir:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # avoid overwrite by adding suffix
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            for i in range(1, 1000):
                cand = state.videos_dir / f"{stem}__{i}{suffix}"
                if not cand.exists():
                    dest = cand
                    break

        try:
            with open(dest, "wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        finally:
            try:
                await file.close()
            except Exception:
                pass

        return {"ok": True, "video_name": dest.name}

    @app.get("/api/model/{model_name}/classes")
    def get_model_classes(model_name: str):
        if not state.model_to_names:
            refresh_models()
        names = state.model_to_names.get(model_name)
        if names is None:
            raise HTTPException(status_code=404, detail="Model not found")
        return {"model": model_name, "classes": names}

    @app.post("/api/video/load")
    def load_video(req: LoadVideoRequest):
        refresh_models()

        if req.batch and not _is_safe_batch_name(req.batch):
            raise HTTPException(status_code=400, detail="Invalid batch name")
        root = _videos_root_for(req.batch)   # videos_dir itself, or videos_dir/handoff_<id> (MDQ-4b)
        p = (root / req.video_name).resolve()
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail="Video not found in videos directory")
        if p.parent != root.resolve():
            raise HTTPException(status_code=400, detail="Invalid video name")

        with video_lock:
            meta = reader.open(p)
            # Always start clean when a video is loaded (no carryover between videos).
            state.ann_by_frame.clear()
            state.last_annotation = None
        state.video_path = p
        state.total_frames = meta.total_frames
        state.fps = meta.fps
        state.img_w = meta.width
        state.img_h = meta.height
        if debug:
            logger.debug("Loaded video=%s frames=%s fps=%s size=%sx%s", p.name, state.total_frames, state.fps, state.img_w, state.img_h)

        loaded_existing = 0
        last_annotated_frame = 0
        if req.load_existing_exports:
            with video_lock:
                loaded_existing, last_annotated_frame = _load_existing_exports_into_state(p.name)

        return {
            "video_name": p.name,
            "total_frames": state.total_frames,
            "fps": state.fps,
            "width": state.img_w,
            "height": state.img_h,
            "models_loaded": len(state.model_to_names),
            "loaded_existing": loaded_existing,
            "last_annotated_frame": last_annotated_frame,
        }

    @app.get("/api/video/status")
    def video_status():
        if state.video_path is None or reader.meta is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "video_name": state.video_path.name,
            "total_frames": state.total_frames,
            "fps": state.fps,
            "width": state.img_w,
            "height": state.img_h,
        }

    @app.get("/api/video/hints")
    def get_video_hints(video_name: str = Query(...), batch: Optional[str] = Query(None)):
        if batch and not _is_safe_batch_name(batch):
            return {"hints": None}
        root = _videos_root_for(batch) or videos_dir
        stem = Path(video_name).stem
        hints_path = root / f"{stem}_hints.json"
        if not hints_path.exists():
            return {"hints": None}
        try:
            return json.loads(hints_path.read_text(encoding="utf-8"))
        except Exception:
            return {"hints": None}

    @app.get("/api/video/case_info")
    def get_video_case_info(video_name: str = Query(...), batch: Optional[str] = Query(None)):
        if batch and not _is_safe_batch_name(batch):
            return {"instruction": None}
        root = _videos_root_for(batch) or videos_dir
        stem = Path(video_name).stem
        info_path = root / f"{stem}_info.json"
        if not info_path.exists():
            return {"instruction": None}
        try:
            return json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            return {"instruction": None}

    @app.get("/api/frame")
    def get_frame(index: int = Query(..., ge=0)):
        if state.video_path is None or reader.meta is None:
            raise HTTPException(status_code=400, detail="No video loaded")
        # Frame count can be unreliable. If it exists (>0) and the client asks past the end,
        # clamp to the last frame instead of erroring (makes End/playback robust).
        requested_index = int(index)
        if state.total_frames > 0 and requested_index >= state.total_frames:
            requested_index = max(0, state.total_frames - 1)

        try:
            with video_lock:
                frame_idx, frame_bgr = reader.safe_seek_read(requested_index)
                end_reached = bool(getattr(reader, "end_reached", False))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        # For UX: we want deterministic navigation. If seeking is imperfect for a codec,
        # still report what we actually returned.
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode frame")
        # If we couldn't reach the requested frame (returned an earlier one), treat it as end reached.
        end_reached = end_reached or (frame_idx < requested_index)
        if debug and (end_reached or requested_index % 250 == 0):
            logger.debug(
                "frame req=%s got=%s end=%s total=%s fps=%s",
                requested_index,
                frame_idx,
                int(end_reached),
                state.total_frames,
                state.fps,
            )
        headers = {"X-Frame-Index": str(frame_idx), "X-Requested-Index": str(requested_index)}
        if end_reached:
            headers["X-End-Reached"] = "1"
        return Response(content=buf.tobytes(), media_type="image/jpeg", headers=headers)

    @app.get("/api/frame/next")
    def get_next_frame(step: int = Query(1, ge=1, le=64)):
        """
        Sequential playback endpoint: reads forward from the current decoder position.
        This avoids seek jitter/rewinds near end-of-video for some codecs.
        """
        if state.video_path is None or reader.meta is None:
            raise HTTPException(status_code=400, detail="No video loaded")

        with video_lock:
            if reader.cap is None:
                raise HTTPException(status_code=400, detail="Video not opened")

            # advance step-1 frames, then read
            for _ in range(max(0, int(step) - 1)):
                ok = reader.cap.grab()
                if not ok:
                    break
            ok, frame_bgr = reader.cap.read()
            if not ok or frame_bgr is None:
                # End reached or decode failure: return last good frame if we have it.
                last = getattr(reader, "_last_bgr", None)
                if last is None:
                    raise HTTPException(status_code=500, detail="Could not read next frame")
                frame_bgr = last
                end_reached = True
            else:
                # update reader state
                pos = int(reader.cap.get(cv2.CAP_PROP_POS_FRAMES))
                reader.frame_idx = max(0, pos - 1)
                reader._last_bgr = frame_bgr
                end_reached = False

            ok2, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok2:
                raise HTTPException(status_code=500, detail="Could not encode frame")

            headers = {"X-Frame-Index": str(reader.frame_idx)}
            if end_reached:
                headers["X-End-Reached"] = "1"
            if debug and (end_reached or reader.frame_idx % 250 == 0):
                logger.debug("next step=%s got=%s end=%s", step, reader.frame_idx, int(end_reached))
            return Response(content=buf.tobytes(), media_type="image/jpeg", headers=headers)

    @app.get("/api/annotations")
    def get_annotations(frame: Optional[int] = None):
        if frame is None:
            # global list for sidebar
            items = []
            frames = set(state.ann_by_frame.keys())
            frames.update(state.background_by_frame.keys())
            for f in sorted(frames):
                for a in state.ann_by_frame.get(f, []):
                    d = asdict(a)
                    d["kind"] = "box"
                    items.append(d)
                for m in sorted(list(state.background_by_frame.get(f, set()))):
                    items.append(
                        {
                            "id": f"bg:{f}:{m}",
                            "kind": "background",
                            "frame_idx": int(f),
                            "model": str(m),
                            "class_name": "BACKGROUND",
                            "class_id": -1,
                            "x1": 0,
                            "y1": 0,
                            "x2": 0,
                            "y2": 0,
                        }
                    )
            return {"annotations": items}

        anns = state.ann_by_frame.get(int(frame), [])
        bgs = sorted(list(state.background_by_frame.get(int(frame), set())))
        return {"frame_idx": int(frame), "annotations": [asdict(a) for a in anns], "background_models": bgs}

    @app.post("/api/annotations")
    def add_annotation(req: AddAnnotationRequest):
        if state.video_path is None:
            raise HTTPException(status_code=400, detail="No video loaded")
        if req.model not in state.model_to_names:
            refresh_models()
        names = state.model_to_names.get(req.model)
        if not names:
            raise HTTPException(status_code=400, detail="Invalid model")
        if req.class_name not in names:
            raise HTTPException(status_code=400, detail="Invalid class")

        frame_idx = int(req.frame_idx)
        # Background and boxes are mutually exclusive.
        if state.background_by_frame.get(frame_idx):
            raise HTTPException(
                status_code=400,
                detail="This frame is marked as BACKGROUND. Remove background mark before adding any annotations.",
            )

        class_id = names.index(req.class_name)
        ann = Annotation(
            id=state.new_annotation_id(),
            frame_idx=frame_idx,
            model=req.model,
            class_name=req.class_name,
            class_id=int(class_id),
            x1=int(req.x1),
            y1=int(req.y1),
            x2=int(req.x2),
            y2=int(req.y2),
        )
        state.ann_by_frame.setdefault(ann.frame_idx, []).append(ann)
        state.last_annotation = (req.model, req.class_name)
        return {"ok": True, "annotation": asdict(ann)}

    @app.post("/api/background")
    def mark_background(req: MarkBackgroundRequest = Body(...)):
        """
        Mark a frame as background for a model.
        Export will write the frame image and an EMPTY label file for that model.
        """
        if state.video_path is None:
            raise HTTPException(status_code=400, detail="No video loaded")
        refresh_models()
        if req.model not in state.model_to_names:
            raise HTTPException(status_code=400, detail="Invalid model")
        frame_idx = int(req.frame_idx)
        if state.ann_by_frame.get(frame_idx):
            raise HTTPException(status_code=400, detail="Please remove all annotations from current frame to select it as background.")
        state.background_by_frame.setdefault(frame_idx, set()).add(req.model)
        return {"ok": True}

    @app.delete("/api/background")
    def unmark_background(frame_idx: int = Query(...), model: str = Query(...)):
        s = state.background_by_frame.get(int(frame_idx))
        if not s:
            return {"ok": True, "removed": False}
        if model in s:
            s.remove(model)
            if not s:
                state.background_by_frame.pop(int(frame_idx), None)
            return {"ok": True, "removed": True}
        return {"ok": True, "removed": False}

    @app.delete("/api/annotations/{ann_id}")
    def delete_annotation(ann_id: str):
        removed = False
        for f in list(state.ann_by_frame.keys()):
            anns = state.ann_by_frame.get(f) or []
            new_anns = [a for a in anns if a.id != ann_id]
            if len(new_anns) != len(anns):
                state.ann_by_frame[f] = new_anns
                removed = True
                if not new_anns:
                    # keep empty frames out
                    state.ann_by_frame.pop(f, None)
        if removed:
            pass
        return {"ok": True, "removed": removed}

    @app.post("/api/export")
    def export_all(req: ExportRequest = ExportRequest()):
        if state.video_path is None or reader.meta is None:
            raise HTTPException(status_code=400, detail="No video loaded")

        frames = set(f for f, anns in state.ann_by_frame.items() if anns)
        frames.update(state.background_by_frame.keys())
        frames = sorted(frames)
        if not frames:
            raise HTTPException(status_code=400, detail="Nothing to export")

        bar_counter_options = cfg.get("bar_counter_options") or ["SANK_LEVO", "SANK_DESNO"]
        parsed = parse_video_name(state.video_path.name, bar_counter_options=bar_counter_options)

        override_bc = (req.bar_counter or "").strip().upper() if req else ""
        if override_bc:
            if bar_counter_options and override_bc not in [x.upper() for x in bar_counter_options]:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "BAR_COUNTER_INVALID", "options": bar_counter_options, "provided": override_bc},
                )
            parsed = ParsedVideoName(timestamp_14=parsed.timestamp_14, prefix=parsed.prefix, bar_counter=override_bc)  # type: ignore[name-defined]

        if not parsed.bar_counter:
            raise HTTPException(
                status_code=400,
                detail={"error": "BAR_COUNTER_MISSING", "options": bar_counter_options, "video_name": state.video_path.name},
            )

        # Per-model batch folders (ingestion-pipeline compatible).
        # Layout:
        #   <output>/<VIDEO_PREFIX>_<MODEL>/images/<BASE>.jpg
        #   <output>/<VIDEO_PREFIX>_<MODEL>/labels/<BASE>.txt
        #   <output>/<VIDEO_PREFIX>_<MODEL>/data.yaml
        model_roots: Dict[str, Path] = {}

        written_images = 0
        written_label_files = 0
        copied_yaml_models: set = set()

        for f in frames:
            with video_lock:
                frame_idx, frame_bgr = reader.safe_seek_read(f)
            base = export_base_name(parsed, frame_idx)

            # Group annotations per model
            by_model: Dict[str, List[Annotation]] = {}
            for a in state.ann_by_frame.get(f, []):
                by_model.setdefault(a.model, []).append(a)

            bg_models = state.background_by_frame.get(f, set())

            for model, anns in by_model.items():
                if model not in model_roots:
                    model_roots[model] = state.output_base_dir / batch_dirname(state.video_path.name, model)
                model_root = model_roots[model]
                images_dir = model_root / "images"
                labels_dir = model_root / "labels"
                ensure_dir(images_dir)
                ensure_dir(labels_dir)

                # Image and label MUST have the same base filename
                img_out = images_dir / f"{base}.jpg"
                txt_out = labels_dir / f"{base}.txt"

                ok = cv2.imwrite(str(img_out), frame_bgr)
                if ok:
                    written_images += 1

                with open(txt_out, "w", encoding="utf-8") as fp:
                    for a in anns:
                        fp.write(yolo_line(a.class_id, a.x1, a.y1, a.x2, a.y2, state.img_w, state.img_h) + "\n")
                written_label_files += 1

                # Copy data.yaml file to model output directory (once per model)
                if model not in copied_yaml_models:
                    yaml_src = state.model_to_yaml_path.get(model)
                    if yaml_src and yaml_src.exists():
                        yaml_dst = model_root / "data.yaml"
                        if not yaml_dst.exists():
                            try:
                                shutil.copy2(yaml_src, yaml_dst)
                                copied_yaml_models.add(model)
                            except Exception as e:
                                logger.warning("Could not copy data.yaml for model %s to %s: %s", model, yaml_dst, e)

            # Background exports (empty label file). Image/label names must match.
            for model in sorted(bg_models):
                if model not in model_roots:
                    model_roots[model] = state.output_base_dir / batch_dirname(state.video_path.name, model)
                model_root = model_roots[model]
                images_dir = model_root / "images"
                labels_dir = model_root / "labels"
                ensure_dir(images_dir)
                ensure_dir(labels_dir)

                img_out = images_dir / f"{base}.jpg"
                txt_out = labels_dir / f"{base}.txt"

                ok = cv2.imwrite(str(img_out), frame_bgr)
                if ok:
                    written_images += 1
                with open(txt_out, "w", encoding="utf-8") as fp:
                    fp.write("")
                written_label_files += 1

                # Copy data.yaml file to model output directory (once per model)
                if model not in copied_yaml_models:
                    yaml_src = state.model_to_yaml_path.get(model)
                    if yaml_src and yaml_src.exists():
                        yaml_dst = model_root / "data.yaml"
                        if not yaml_dst.exists():
                            try:
                                shutil.copy2(yaml_src, yaml_dst)
                                copied_yaml_models.add(model)
                            except Exception as e:
                                logger.warning("Could not copy data.yaml for model %s to %s: %s", model, yaml_dst, e)

        # Write image_tags.json per model batch (before zipping)
        if state.frame_tags_by_frame:
            for model, mr in model_roots.items():
                if not mr.exists():
                    continue
                # Collect all frames that belong to this model batch
                model_frames = set()
                for f, anns in state.ann_by_frame.items():
                    if any(a.model == model for a in anns):
                        model_frames.add(f)
                for f, bg_models in state.background_by_frame.items():
                    if model in bg_models:
                        model_frames.add(f)
                tags_by_stem = {}
                for f in sorted(model_frames):
                    entry = state.frame_tags_by_frame.get(f, {})
                    frame_tags = entry.get("frame_tags", [])
                    stored_bbox_tags = entry.get("bbox_tags", {})
                    if not frame_tags and not any(stored_bbox_tags.values()):
                        continue
                    base = export_base_name(parsed, f)
                    # Build bbox_tags_list: ALL annotations for this model on this frame
                    frame_anns = [a for a in state.ann_by_frame.get(f, []) if a.model == model]
                    bbox_tags_list = []
                    for a in frame_anns:
                        ann_tags = stored_bbox_tags.get(a.id, [])
                        bbox_tags_list.append({
                            "class": a.class_name,
                            "bbox": [a.x1, a.y1, a.x2, a.y2],
                            "tags": ann_tags,
                        })
                    tags_by_stem[base] = {"frame_tags": frame_tags, "bbox_tags": bbox_tags_list}
                if tags_by_stem:
                    (mr / "image_tags.json").write_text(
                        json.dumps(tags_by_stem, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

        # Zip each model batch folder and remove the source folder.
        zip_paths = []
        for model, mr in sorted(model_roots.items()):
            if mr.exists():
                try:
                    zp = _zip_batch(mr)
                    zip_paths.append(str(zp))
                    shutil.rmtree(mr)
                except Exception as e:
                    logger.warning("Could not zip %s: %s", mr, e)
                    zip_paths.append(str(mr))

        # After export: clear workspace (close video + remove annotations), as requested.
        with video_lock:
            try:
                reader.close()
            except Exception:
                pass
            state.reset_video_state()

        return JSONResponse(
            {
                "ok": True,
                "zip_paths": zip_paths,
                "written_images": written_images,
                "written_label_files": written_label_files,
                "frames_labeled": len(frames),
                "cleared": True,
            }
        )

    return app


app = create_app()


