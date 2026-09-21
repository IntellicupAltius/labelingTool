"""
Per-pool sidecar store for flagged images (MDQ-3).

A flag marks an image for the owner's later review — it never deletes, edits or moves
anything. The sidecar lives OUTSIDE the RAW tree (default
``/opt/intellicup/datasets/raw_review/<pool>_flagged.json``, one file per pool) and is the
only thing this module ever writes.

Schema (``schema_version`` 1)::

    {"schema_version": 1,
     "entries": {"<class>/<filename.jpg>": {
         "source": "manual_goca" | "analyzer_auto" | "data4_audit",
         "category": one of CATEGORIES or null,
         "comment": "<free text>",
         "signal": {"metric": ..., "value": ...} | null,
         "flagged_at": "<ISO timestamp>",
         "flagged_by": "goca" | "analyzer_overlap_report" | "data4_audit",
         "owner_decision": "approve_delete" | "keep" | null,
         "owner_decision_at": "<ISO timestamp>" | null,
         "flag_context": "retroactive_review" | "forward_labeling"}}}

Every entry must carry a ``category`` or a ``signal`` (never an empty reason).
Entries are keyed by the stable ``<class>/<filename>`` id, never by an index: RAW is
append-only and indices shift as it grows. A frame flagged while labeling a video (before it
is ingested into RAW) has no RAW file yet, so it is keyed ``_forward/<export base name>.jpg``
and additionally carries ``video`` / ``frame_idx`` / ``annotation_classes``.

Only ``source == "manual_goca"`` entries that the owner has not decided on can be changed or
removed through this module's manual API. The analyzer / audit sources (MDQ-9 / MDQ-10) and
the owner review queue (MDQ-6) will write the same files; concurrent writers are serialized
with a per-pool lock file and every write is temp-file + ``os.replace`` (no torn files).
A sidecar that fails to parse is never overwritten.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

try:  # POSIX only; the labeler also runs on Windows where flagging is unavailable
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore

SCHEMA_VERSION = 1
POOLS = ("glasses", "bottles", "cups", "pitchers", "shots")
CATEGORIES = ("gibberish", "wrong_frame_wrong_class", "near_duplicate", "mislabeled_background")
SOURCE_MANUAL = "manual_goca"
CONTEXT_RETROACTIVE = "retroactive_review"
CONTEXT_FORWARD = "forward_labeling"
FORWARD_DIR = "_forward"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_COMMENT_LEN = 2000
_EXTRA_KEYS = ("video", "frame_idx", "annotation_classes")


class FlagError(Exception):
    """Base class; ``status`` is the HTTP status the server maps it to."""
    status = 500


class FlagValidationError(FlagError):
    status = 400


class FlagUnavailable(FlagError):
    status = 503


class FlagStoreError(FlagError):
    """The sidecar on disk is unreadable / has an unknown schema. Never auto-repaired."""
    status = 500


class FlagConflict(FlagError):
    status = 409

    def __init__(self, message: str, entry: Optional[dict] = None):
        super().__init__(message)
        self.entry = entry


def normalize_pool(pool: Optional[str]) -> str:
    p = (pool or "").strip().lower()
    if p not in POOLS:
        raise FlagValidationError(f"pool must be one of {', '.join(POOLS)}")
    return p


def validate_image_key(image_key: str, *, forward: bool = False) -> str:
    """``<class>/<filename>``: exactly two clean path components, an image extension."""
    if not isinstance(image_key, str) or not image_key or "\x00" in image_key or "\\" in image_key:
        raise FlagValidationError("image_key must be '<class>/<filename>'")
    parts = image_key.split("/")
    if len(parts) != 2 or any(p in ("", ".", "..") for p in parts):
        raise FlagValidationError("image_key must be '<class>/<filename>'")
    if Path(parts[1]).suffix.lower() not in IMG_EXTS:
        raise FlagValidationError("image_key filename must be an image file")
    if (parts[0] == FORWARD_DIR) != forward:
        raise FlagValidationError(
            f"'{FORWARD_DIR}/' keys are for video-frame flags only" if not forward
            else f"forward keys must start with '{FORWARD_DIR}/'"
        )
    return image_key


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _is_mutable_by_manual(entry: dict) -> bool:
    return entry.get("source") == SOURCE_MANUAL and entry.get("owner_decision") is None


class FlagStore:
    """``root`` is the sidecar directory, or None when flagging is unavailable here."""

    def __init__(self, root: Optional[Path], forbidden_roots: Optional[List[Path]] = None):
        self.root: Optional[Path] = None
        self._thread_lock = threading.Lock()
        if root is None:
            return
        resolved = Path(root).expanduser().resolve()
        for f in forbidden_roots or []:
            f = Path(f).resolve()
            if resolved == f or f in resolved.parents:
                raise ValueError(f"flag store root {resolved} must be outside the RAW tree {f}")
        self.root = resolved

    @property
    def available(self) -> bool:
        return self.root is not None

    def _require(self) -> Path:
        if self.root is None:
            raise FlagUnavailable("flagging is not available on this machine (no raw_review_dir)")
        return self.root

    def path_for(self, pool: str) -> Path:
        return self._require() / f"{normalize_pool(pool)}_flagged.json"

    # -- persistence -------------------------------------------------------------

    def _read(self, pool: str) -> dict:
        path = self.path_for(pool)
        if not path.exists():
            return {"schema_version": SCHEMA_VERSION, "entries": {}}
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise FlagStoreError(f"{path} is unreadable ({e}); not overwriting it") from e
        if not isinstance(doc, dict) or doc.get("schema_version") != SCHEMA_VERSION \
                or not isinstance(doc.get("entries"), dict):
            raise FlagStoreError(f"{path} has an unsupported schema; not overwriting it")
        return doc

    def _write(self, pool: str, doc: dict) -> None:
        path = self.path_for(pool)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        try:
            with open(tmp, "w", encoding="utf-8") as fp:
                json.dump(doc, fp, indent=2, ensure_ascii=False)
                fp.write("\n")
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    @contextmanager
    def _locked(self, pool: str) -> Iterator[None]:
        root = self._require()
        root.mkdir(parents=True, exist_ok=True)
        lock_path = root / f"{normalize_pool(pool)}_flagged.json.lock"
        with self._thread_lock:
            with open(lock_path, "a+") as lock_fp:
                if fcntl is not None:
                    fcntl.flock(lock_fp, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_fp, fcntl.LOCK_UN)

    # -- reads -------------------------------------------------------------------

    def get(self, pool: str, image_key: str) -> Optional[dict]:
        if not self.available:
            return None
        entry = self._read(pool)["entries"].get(image_key)
        return dict(entry) if entry else None

    def list_entries(self, pool: str) -> Dict[str, dict]:
        return dict(self._read(pool)["entries"]) if self.available else {}

    # -- manual (Goca) writes ----------------------------------------------------

    def add_manual_flag(
        self,
        pool: str,
        image_key: str,
        category: str,
        comment: str = "",
        flag_context: str = CONTEXT_RETROACTIVE,
        extra: Optional[dict] = None,
    ) -> dict:
        pool = normalize_pool(pool)
        forward = flag_context == CONTEXT_FORWARD
        if flag_context not in (CONTEXT_RETROACTIVE, CONTEXT_FORWARD):
            raise FlagValidationError("unknown flag_context")
        validate_image_key(image_key, forward=forward)
        if category not in CATEGORIES:
            raise FlagValidationError(f"category must be one of {', '.join(CATEGORIES)}")
        comment = (comment or "").strip()
        if len(comment) > MAX_COMMENT_LEN:
            raise FlagValidationError(f"comment is longer than {MAX_COMMENT_LEN} characters")
        with self._locked(pool):
            doc = self._read(pool)
            existing = doc["entries"].get(image_key)
            if existing is not None and not _is_mutable_by_manual(existing):
                raise FlagConflict(_conflict_message(existing), existing)
            entry = {
                "source": SOURCE_MANUAL,
                "category": category,
                "comment": comment,
                "signal": None,
                "flagged_at": _now(),
                "flagged_by": "goca",
                "owner_decision": None,
                "owner_decision_at": None,
                "flag_context": flag_context,
            }
            for k in _EXTRA_KEYS:
                if extra and extra.get(k) is not None:
                    entry[k] = extra[k]
            doc["entries"][image_key] = entry
            self._write(pool, doc)
        return entry

    def remove_manual_flag(self, pool: str, image_key: str) -> bool:
        """True if an entry was removed, False if there was none."""
        pool = normalize_pool(pool)
        with self._locked(pool):
            doc = self._read(pool)
            existing = doc["entries"].get(image_key)
            if existing is None:
                return False
            if not _is_mutable_by_manual(existing):
                raise FlagConflict(_conflict_message(existing), existing)
            del doc["entries"][image_key]
            self._write(pool, doc)
        return True


def _conflict_message(existing: dict) -> str:
    if existing.get("owner_decision") is not None:
        return f"already reviewed by the owner ({existing['owner_decision']}); it can no longer be changed here"
    return f"already flagged by {existing.get('source')}; it is in the owner's review queue"
