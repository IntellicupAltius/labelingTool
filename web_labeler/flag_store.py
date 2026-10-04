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
         "owner_decision": "approve_delete" | "keep" | "relabel" | null,
         "owner_decision_at": "<ISO timestamp>" | null,
         "flag_context": "retroactive_review" | "forward_labeling"}}}

``relabel`` (MDQ-15c-1) means "keep the image but its RAW label must be corrected". It is the
only decision that adds a field: the optional ``relabel_snapshot`` object
(``image_sha256`` / ``label_sha256`` (null when the image has no label file) / ``recorded_at``) —
the sha256 of the RAW image and label at the moment of the decision, so the later apply step
(MDQ-15c-3) can refuse to act if the original changed in the meantime. Changing the decision to
anything else, or undoing it, drops the snapshot. MDQ-15c-3 adds a second optional field,
``relabel_result`` (``{"status": "applied"|"unchanged", "at", "batch_id", "apply_run"}``), set by
``apply_relabel.py`` when the correction was applied to RAW or found unchanged (nothing to apply). Old sidecars carry neither the decision nor the
field and are read unchanged (``schema_version`` stays 1).

Every entry must carry a ``category`` or a ``signal`` (never an empty reason).
Entries are keyed by the stable ``<class>/<filename>`` id, never by an index: RAW is
append-only and indices shift as it grows. A frame flagged while labeling a video (before it
is ingested into RAW) has no RAW file yet, so it is keyed ``_forward/<export base name>.jpg``
and additionally carries ``video`` / ``frame_idx`` / ``annotation_classes``.

Manual flags win over automated ones: ``add_manual_flag`` may REPLACE an undecided
``analyzer_auto`` / ``data4_audit`` entry (the entry is one per key). The replaced automated
flag is kept verbatim in the entry's optional top-level ``replaced_auto`` object
(``source/category/signal/comment/flagged_at/flagged_by/flag_context``); ``unflag_manual`` puts it
back, so removing the manual flag never loses the automated evidence. ``replaced_auto`` is an extra
key like ``video``/``frame_idx`` — ``schema_version`` stays 1 and old sidecars need no migration.
Nothing here ever replaces or removes an entry the owner has decided on.

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
CATEGORY_LABELS = {
    "gibberish": "Gibberish",
    "wrong_frame_wrong_class": "Wrong frame, wrong class",
    "near_duplicate": "Near-duplicate",
    "mislabeled_background": "Mislabeled background",
}
OWNER_DECISIONS = ("approve_delete", "keep", "relabel")
DECISION_RELABEL = "relabel"
SOURCE_MANUAL = "manual_goca"
SOURCE_ANALYZER_AUTO = "analyzer_auto"
SOURCE_DATA4_AUDIT = "data4_audit"
AUTO_SOURCES = (SOURCE_ANALYZER_AUTO, SOURCE_DATA4_AUDIT)
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


class FlagNotFound(FlagError):
    status = 404


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


def _is_replaceable_by_manual(entry: dict) -> bool:
    """A manual flag may be written over: own undecided manual flag, or an undecided automated one."""
    return entry.get("owner_decision") is None and entry.get("source") in (SOURCE_MANUAL, *AUTO_SOURCES)


_REPLACED_AUTO_KEYS = ("source", "category", "signal", "comment", "flagged_at", "flagged_by", "flag_context")


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
            if existing is not None and not _is_replaceable_by_manual(existing):
                raise FlagConflict(_conflict_message(existing), existing)
            replaced = None
            if existing is not None:
                if existing.get("source") in AUTO_SOURCES:
                    replaced = {k: existing.get(k) for k in _REPLACED_AUTO_KEYS}
                elif isinstance(existing.get("replaced_auto"), dict):
                    replaced = existing["replaced_auto"]      # re-saving own flag keeps the original automated one
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
            if replaced is not None:
                entry["replaced_auto"] = replaced
            doc["entries"][image_key] = entry
            self._write(pool, doc)
        return entry

    def unflag_manual(self, pool: str, image_key: str) -> dict:
        """Remove the manual flag. ``{"removed": bool, "restored": entry | None}``.

        If the manual flag had replaced an automated one (``replaced_auto``), that automated flag
        is written back (undecided, exactly as it was) and returned as ``restored``; otherwise the
        entry is simply deleted.
        """
        pool = normalize_pool(pool)
        with self._locked(pool):
            doc = self._read(pool)
            existing = doc["entries"].get(image_key)
            if existing is None:
                return {"removed": False, "restored": None}
            if not _is_mutable_by_manual(existing):
                raise FlagConflict(_conflict_message(existing), existing)
            ra = existing.get("replaced_auto")
            restored = None
            if isinstance(ra, dict) and ra.get("source") in AUTO_SOURCES:
                restored = {k: ra.get(k) for k in _REPLACED_AUTO_KEYS}
                restored["owner_decision"] = None
                restored["owner_decision_at"] = None
                doc["entries"][image_key] = restored
            else:
                del doc["entries"][image_key]
            self._write(pool, doc)
        return {"removed": True, "restored": dict(restored) if restored else None}

    def remove_manual_flag(self, pool: str, image_key: str) -> bool:
        """True if the manual flag was removed, False if there was none (restores a replaced automated flag)."""
        return self.unflag_manual(pool, image_key)["removed"]


    # -- owner review queue (MDQ-6) ---------------------------------------------

    def list_all(self) -> Dict[str, dict]:
        """``{pool: {"entries": {...}} | {"error": "..."}}`` for every pool.

        One unreadable sidecar must not hide the other four pools from the review queue, so a
        per-pool read failure is reported next to the others instead of raised.
        """
        self._require()
        out: Dict[str, dict] = {}
        for pool in POOLS:
            try:
                out[pool] = {"entries": dict(self._read(pool)["entries"])}
            except FlagStoreError as e:
                out[pool] = {"error": str(e)}
        return out

    # -- automated writes (MDQ-9 analyzer_auto / MDQ-10 data4_audit) -------------

    def add_auto_flag(
        self,
        pool: str,
        image_key: str,
        source: str,
        signal: dict,
        flagged_by: str,
        comment: str = "",
    ) -> Optional[dict]:
        """Add one automated flag, conservatively.

        Returns the new entry, or ``None`` if the key already carries ANY entry — a manual
        flag, a prior automated flag, or one the owner has already decided on. An automated
        source never overwrites an existing flag regardless of who wrote it or what state it
        is in; a re-run that would pick the same image again is a silent no-op, not a refresh.
        ``signal`` must be a non-empty dict describing the concrete evidence (never a bare
        "suspicious"), and this method never sets ``owner_decision`` — an automated source can
        flag for review, it can never approve its own deletion.
        """
        pool = normalize_pool(pool)
        if source not in AUTO_SOURCES:
            raise FlagValidationError(f"source must be one of {', '.join(AUTO_SOURCES)}")
        validate_image_key(image_key, forward=False)
        if not isinstance(signal, dict) or not signal:
            raise FlagValidationError("signal must be a non-empty dict")
        comment = (comment or "").strip()
        if len(comment) > MAX_COMMENT_LEN:
            raise FlagValidationError(f"comment is longer than {MAX_COMMENT_LEN} characters")
        with self._locked(pool):
            doc = self._read(pool)
            if image_key in doc["entries"]:
                return None
            entry = {
                "source": source,
                "category": None,
                "comment": comment,
                "signal": signal,
                "flagged_at": _now(),
                "flagged_by": flagged_by,
                "owner_decision": None,
                "owner_decision_at": None,
                "flag_context": CONTEXT_RETROACTIVE,
            }
            doc["entries"][image_key] = entry
            self._write(pool, doc)
        return entry

    def set_owner_decision(self, pool: str, image_key: str, decision: Optional[str],
                           relabel_snapshot: Optional[dict] = None) -> dict:
        """Record the owner's verdict on an existing flag, in place.

        ``decision`` is one of OWNER_DECISIONS, or None to put the item back to pending (a
        mis-click must be undoable). Only ``owner_decision`` / ``owner_decision_at`` change:
        the flag record itself (source, reason, signal, timestamps) is never removed or
        rewritten, and nothing outside the sidecar is touched — moving an approved image out
        of RAW is MDQ-8's archive script, not this.

        ``relabel`` requires ``relabel_snapshot`` (``image_sha256`` + ``label_sha256``, computed by
        the caller from RAW, read-only) and stores it in the entry; any other decision, or None,
        removes a previous snapshot.
        """
        pool = normalize_pool(pool)
        if decision is not None and decision not in OWNER_DECISIONS:
            raise FlagValidationError(f"decision must be one of {', '.join(OWNER_DECISIONS)} or null")
        if not isinstance(image_key, str) or not image_key:
            raise FlagValidationError("image_key is required")
        if decision == DECISION_RELABEL:
            snap = relabel_snapshot
            if not isinstance(snap, dict) or not isinstance(snap.get("image_sha256"), str) or not snap["image_sha256"] \
                    or (snap.get("label_sha256") is not None and not isinstance(snap["label_sha256"], str)):
                raise FlagValidationError("relabel needs a relabel_snapshot with image_sha256 (and label_sha256 or null)")
        with self._locked(pool):
            doc = self._read(pool)
            entry = doc["entries"].get(image_key)
            if entry is None:
                raise FlagNotFound(f"no flag for {pool}/{image_key}")
            entry["owner_decision"] = decision
            entry["owner_decision_at"] = _now() if decision is not None else None
            if decision == DECISION_RELABEL:
                entry["relabel_snapshot"] = {
                    "image_sha256": relabel_snapshot["image_sha256"],
                    "label_sha256": relabel_snapshot.get("label_sha256"),
                    "recorded_at": entry["owner_decision_at"],
                }
            else:
                entry.pop("relabel_snapshot", None)
            self._write(pool, doc)
        return dict(entry)

    def set_relabel_result(self, pool: str, image_key: str, result: dict) -> dict:
        """MDQ-15c-3: record that the relabel decision was resolved by ``apply_relabel.py``.

        Adds the optional ``relabel_result`` object (``status`` ``"applied"`` | ``"unchanged"``,
        ``at``, ``batch_id``, ``apply_run``). ``owner_decision`` / ``relabel_snapshot`` and the rest
        of the entry are not touched, the entry is never removed. Old sidecars carry no such field.
        """
        pool = normalize_pool(pool)
        if not isinstance(result, dict) or result.get("status") not in ("applied", "unchanged"):
            raise FlagValidationError("relabel_result.status must be 'applied' or 'unchanged'")
        with self._locked(pool):
            doc = self._read(pool)
            entry = doc["entries"].get(image_key)
            if entry is None:
                raise FlagNotFound(f"no flag for {pool}/{image_key}")
            entry["relabel_result"] = {**result, "at": result.get("at") or _now()}
            self._write(pool, doc)
        return dict(entry)

    def clear_relabel_result(self, pool: str, image_key: str) -> bool:
        """Remove ``relabel_result`` (rollback of an apply). True if there was one."""
        pool = normalize_pool(pool)
        with self._locked(pool):
            doc = self._read(pool)
            entry = doc["entries"].get(image_key)
            if entry is None or "relabel_result" not in entry:
                return False
            entry.pop("relabel_result")
            self._write(pool, doc)
        return True


def describe_reason(entry: dict) -> Optional[str]:
    """Human-readable reason for a flag, or None if the entry carries none (invalid entry).

    A category is shown by its label, a signal by its concrete metric/value; the comment is
    appended when present. Never returns a generic placeholder: an entry without a category
    or signal is reported as missing, not papered over.
    """
    parts: List[str] = []
    cat = entry.get("category")
    if cat:
        parts.append(CATEGORY_LABELS.get(cat, cat))
    sig = entry.get("signal")
    if isinstance(sig, dict) and sig:
        metric = sig.get("metric")
        value = sig.get("value")
        detail = ", ".join(f"{k}={v}" for k, v in sig.items() if k not in ("metric", "value"))
        text = f"{metric}: {value}" if metric is not None else ", ".join(f"{k}={v}" for k, v in sig.items())
        if metric is not None and detail:
            text += f" ({detail})"
        parts.append(text)
    if not parts:
        return None
    comment = (entry.get("comment") or "").strip()
    if comment:
        parts.append(f"“{comment}”")
    return " — ".join(parts)


def _conflict_message(existing: dict) -> str:
    if existing.get("owner_decision") is not None:
        return f"already reviewed by the owner ({existing['owner_decision']}); it can no longer be changed here"
    return f"already flagged by {existing.get('source')}; it is in the owner's review queue"  # only reached for decided / unknown-source entries
