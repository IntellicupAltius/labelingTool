from __future__ import annotations

import json
import os
import re
import shutil
import threading
import logging
import platform
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

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


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


logger = logging.getLogger("labeler")


def sanitize_filename_part(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def video_stem_without_uuid(stem: str) -> str:
    # matches what you did in v6_1: split on "{...}" and trim
    return stem.split("{")[0].rstrip("_-")


class LoadVideoRequest(BaseModel):
    video_name: str = Field(..., description="Filename under videos directory")
    load_existing_exports: bool = Field(False, description="If true, load annotations from existing exported labels on disk")


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
            for k in ("data_root", "models_dir", "videos_dir", "output_dir", "datasets_dir", "existing_datasets_dir"):
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
    bg_session: Optional[BgSession] = None

    try:
        # Avoid OpenCV internal thread pools competing with FFmpeg (stability/perf).
        cv2.setNumThreads(0)
    except Exception:
        pass

    static_dir = Path(__file__).resolve().parent / "static"
    app = FastAPI(title="Labeling Tool (Browser)", version="0.1.0")

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

    def list_videos() -> List[str]:
        exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
        if not state.videos_dir.exists():
            return []
        vids = [p.name for p in state.videos_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        vids.sort()
        return vids

    def is_supported_video_name(name: str) -> bool:
        exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
        return Path(name).suffix.lower() in exts

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
            "bar_counter_options": bar_counter_options,
            "bar_counter_detected": detected,
        }

    # ---------------- Dataset Fixer API ----------------
    @app.get("/api/datasets")
    def datasets_list():
        return {"datasets_dir": str(datasets_dir), "datasets": list_dataset_folders(datasets_dir)}

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

        # Zip and remove folder
        zip_path_str = str(out_root)
        if out_root.exists():
            try:
                zp = _zip_batch(out_root)
                zip_path_str = str(zp)
                shutil.rmtree(out_root)
            except Exception as e:
                logger.warning("Could not zip background batch %s: %s", out_root, e)

        bg_session = None
        return {"ok": True, "cleared": True, "zip_path": zip_path_str}

    @app.post("/api/datasets/load")
    def datasets_load(req: DatasetLoadRequest = Body(...)):
        nonlocal dataset_session
        refresh_models()
        names = state.model_to_names.get(req.model)
        if not names:
            raise HTTPException(status_code=400, detail="Invalid model")
        ds_path = (datasets_dir / req.dataset_name).resolve()
        if ds_path.parent != datasets_dir or not ds_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        dataset_session = load_dataset_session(ds_path, model=req.model, class_names=names)
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
        return FileResponse(str(img_path), media_type=media, headers={"X-Image-Index": str(index), "X-Image-Name": img_path.name})

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
        nonlocal dataset_session
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
                except Exception:
                    pass  # ignore copy errors
        
        # Zip the output so the labeler can drop it directly.
        # For create_new: delete the folder after zipping (it's a derivative copy).
        # For overwrite: keep the folder (it's the labeler's source dataset).
        zip_path_str = str(out_path)
        try:
            zp = _zip_batch(out_path)
            zip_path_str = str(zp)
            if req.strategy == "create_new":
                shutil.rmtree(out_path)
        except Exception as e:
            logger.warning("Could not zip dataset %s: %s", out_path, e)

        result = {"ok": True, "zip_path": zip_path_str, "cleared": True}
        if deleted_count > 0:
            result["deleted_count"] = deleted_count
        dataset_session = None
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

    def _zip_batch(folder: Path) -> Path:
        """
        Zip a batch folder into <folder>.zip beside it.
        The zip contains the folder itself at the root (pipeline expects a single
        top-level folder inside the zip with images/ labels/ data.yaml).
        Returns the zip path.
        """
        zip_path = folder.parent / f"{folder.name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(folder.rglob("*")):
                if file.is_file():
                    zf.write(file, Path(folder.name) / file.relative_to(folder))
        return zip_path

    def _output_root_for_video_name(video_name: str) -> Path:
        """
        Compute per-video output folder name, keeping it short (Windows path limits).
        """
        return state.output_base_dir / sanitize_filename_part(output_video_dirname(video_name))

    def _load_existing_exports_into_state(video_name: str) -> int:
        """
        Loads existing exported YOLO labels (per-model folders) back into the in-memory workspace.
        Expects files:
          <output>/<VIDEO_NAME>/<MODEL>/labels/<BASE>.txt
        where <BASE> ends with `_f%06d`.
        """
        if state.video_path is None:
            return 0
        out_root = _output_root_for_video_name(video_name)
        if not out_root.exists():
            return 0

        loaded = 0
        # clear again just in case
        state.ann_by_frame.clear()
        state.last_annotation = None

        for model_root in sorted([p for p in out_root.iterdir() if p.is_dir()]):
            model = model_root.name
            labels_dir = model_root / "labels"
            if not labels_dir.exists():
                continue

            # class names for this model (if present)
            names = state.model_to_names.get(model, [])

            for txt_path in sorted(labels_dir.glob("*.txt")):
                m = re.search(r"_f(\d{6})$", txt_path.stem)
                if not m:
                    continue
                frame_idx = int(m.group(1)) - 1
                if frame_idx < 0:
                    continue
                try:
                    lines = txt_path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    continue
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                    except Exception:
                        continue

                    # YOLO normalized -> pixel box
                    img_w = max(1, int(state.img_w))
                    img_h = max(1, int(state.img_h))
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
                        model=model,
                        class_name=class_name,
                        class_id=class_id,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                    state.ann_by_frame.setdefault(frame_idx, []).append(ann)
                    loaded += 1

        if debug:
            logger.debug("Loaded %s existing annotations from %s", loaded, out_root)
        return loaded

    @app.get("/api/video/info")
    def video_info(video_name: str = Query(...)):
        """
        Lightweight info used by the UI to warn if a video already has exported files.
        """
        out_root = _output_root_for_video_name(video_name)
        # New export layout is per-model:
        #   <out_root>/<model>/images/*.jpg
        #   <out_root>/<model>/labels/*.txt
        img_count = 0
        lbl_count = 0
        if out_root.exists():
            for p in out_root.rglob("*"):
                if not p.is_file():
                    continue
                suf = p.suffix.lower()
                if suf in (".jpg", ".jpeg", ".png") and p.parent.name == "images":
                    img_count += 1
                elif suf == ".txt" and p.parent.name == "labels":
                    lbl_count += 1

        return {
            "video_name": video_name,
            "output_root": str(out_root),
            "images_dir": str(out_root),
            "labels_dir": str(out_root),
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

        p = (state.videos_dir / req.video_name).resolve()
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail="Video not found in videos directory")
        if p.parent != state.videos_dir:
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
        if req.load_existing_exports:
            with video_lock:
                loaded_existing = _load_existing_exports_into_state(p.name)

        return {
            "video_name": p.name,
            "total_frames": state.total_frames,
            "fps": state.fps,
            "width": state.img_w,
            "height": state.img_h,
            "models_loaded": len(state.model_to_names),
            "loaded_existing": loaded_existing,
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
                            except Exception:
                                pass  # ignore copy errors

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
                            except Exception:
                                pass  # ignore copy errors

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


