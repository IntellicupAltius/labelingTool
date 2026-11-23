"""Catalog core: the only code that reads or changes the Catalog database."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from web_labeler.yolo import load_yolo_names_from_yaml

SCHEMA_VERSION = 1
_TABLES = {"venues", "models", "classes", "import_runs"}

_SCHEMA = """
CREATE TABLE venues (
    venue_id   TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE models (
    venue_id  TEXT NOT NULL REFERENCES venues(venue_id),
    model_key TEXT NOT NULL,
    PRIMARY KEY (venue_id, model_key)
);
CREATE TABLE classes (
    venue_id    TEXT NOT NULL,
    model_key   TEXT NOT NULL,
    class_id    INTEGER NOT NULL CHECK (class_id >= 0),
    name        TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    imported_at TEXT NOT NULL,
    PRIMARY KEY (venue_id, model_key, class_id),
    UNIQUE (venue_id, model_key, name),
    FOREIGN KEY (venue_id, model_key) REFERENCES models(venue_id, model_key)
);
CREATE TABLE import_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id     TEXT NOT NULL REFERENCES venues(venue_id),
    at           TEXT NOT NULL,
    summary_json TEXT NOT NULL
);
CREATE TRIGGER classes_no_delete BEFORE DELETE ON classes
BEGIN SELECT RAISE(ABORT, 'catalog classes are never deleted'); END;
CREATE TRIGGER classes_keep_identity BEFORE UPDATE OF venue_id, model_key, class_id ON classes
BEGIN SELECT RAISE(ABORT, 'catalog class numbers never change'); END;
"""

MSG_NOT_CONFIGURED = "Katalog još nije podešen na ovom računaru. Javi administratoru."
MSG_MISSING = "Katalog još nije napravljen. Javi administratoru. Ostatak alata radi normalno."
MSG_INVALID = "Katalog trenutno ne može da se pročita. Javi administratoru. Ostatak alata radi normalno."
MSG_EMPTY = "Katalog je prazan. Javi administratoru da pokrene uvoz klasa."


class CatalogError(Exception):
    """An import or write was refused; nothing was changed."""


class CatalogUnavailable(Exception):
    """The Catalog cannot be read. The message is safe to show to a labeler."""


class UnknownModel(Exception):
    pass


@dataclass(frozen=True)
class CatalogSettings:
    db_path: Optional[Path]
    venue_id: Optional[str]
    venue_name: Optional[str]
    source_models_dir: Path
    forbidden_roots: Tuple[Path, ...] = ()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], models_dir: Path,
                    raw_base_path: Optional[Path] = None) -> "CatalogSettings":
        venue_id = str(cfg.get("catalog_venue_id") or "").strip() or None
        venue_name = str(cfg.get("catalog_venue_name") or "").strip() or venue_id
        db_cfg = str(cfg.get("catalog_db_path") or "").strip()
        data_root = str(cfg.get("data_root") or "").strip()
        if db_cfg:
            db_path: Optional[Path] = Path(db_cfg).expanduser().resolve()
        elif data_root:
            db_path = (Path(data_root).expanduser() / "catalog" / "catalog.db").resolve()
        else:
            db_path = None
        roots = [Path(models_dir).resolve()]
        if raw_base_path is not None:
            roots.append(Path(raw_base_path).resolve())
        return cls(db_path, venue_id, venue_name, Path(models_dir).resolve(), tuple(roots))

    @classmethod
    def from_config_file(cls, config_path: Path) -> "CatalogSettings":
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        env_models = os.getenv("LABELER_MODELS_DIR")
        if env_models:
            models_dir = Path(env_models).expanduser()
        elif str(cfg.get("models_dir") or "").strip():
            models_dir = Path(cfg["models_dir"]).expanduser()
        else:
            models_dir = Path(cfg.get("data_root") or "~/LabelingToolData").expanduser() / "Models"
        raw = str(cfg.get("raw_base_path") or "").strip()
        return cls.from_config(cfg, models_dir, Path(raw).expanduser() if raw else None)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_sources(settings: CatalogSettings) -> Dict[str, List[str]]:
    """Read class lists the same way the tool does: every *.yaml/*.yml directly in the folder."""
    folder = settings.source_models_dir
    if not folder.is_dir():
        raise CatalogError(f"Class list folder does not exist: {folder}")
    sources: Dict[str, List[str]] = {}
    for path in list(folder.glob("*.yaml")) + list(folder.glob("*.yml")):
        try:
            names = load_yolo_names_from_yaml(path)
        except Exception as e:
            raise CatalogError(f"{path.name}: cannot read class list ({e})")
        if not names or not all(isinstance(n, str) and n.strip() for n in names):
            raise CatalogError(f"{path.name}: class names must be non-empty text")
        if len(set(names)) != len(names):
            raise CatalogError(f"{path.name}: duplicate class name")
        sources[path.stem] = list(names)
    if not sources:
        raise CatalogError(f"No class lists found in {folder}")
    return sources


def _open_for_write(settings: CatalogSettings) -> sqlite3.Connection:
    if settings.db_path is None or not settings.venue_id:
        raise CatalogError("Catalog is not configured: set catalog_venue_id and a data root or catalog_db_path")
    for root in settings.forbidden_roots:
        if _is_within(settings.db_path, root):
            raise CatalogError(f"Catalog database must not be inside {root}")
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.db_path), isolation_level=None)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        has_tables = conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] > 0
        if version == 0 and not has_tables:
            conn.executescript(f"BEGIN;{_SCHEMA}PRAGMA user_version = {SCHEMA_VERSION};COMMIT;")
        elif version != SCHEMA_VERSION:
            raise CatalogError(f"Unsupported Catalog database version: {version}")
    except sqlite3.Error as e:
        conn.close()
        raise CatalogError(f"Catalog database is not usable: {e}")
    except CatalogError:
        conn.close()
        raise
    return conn


def import_class_lists(settings: CatalogSettings) -> Dict[str, Any]:
    """Bring the tool's class lists into the Catalog, all or nothing.

    Safe to repeat: existing numbers never change and new classes are only appended. If a source
    disagrees with a number already in the Catalog, the whole import is refused.
    """
    sources = _read_sources(settings)
    conn = _open_for_write(settings)
    venue = settings.venue_id
    report: Dict[str, Any] = {"venue_id": venue, "models": {}, "added": 0}
    try:
        conn.execute("BEGIN IMMEDIATE")
        at = _now()
        conn.execute("INSERT OR IGNORE INTO venues (venue_id, name, created_at) VALUES (?, ?, ?)",
                     (venue, settings.venue_name or venue, at))
        conflicts: List[str] = []
        for model, names in sorted(sources.items()):
            stored = dict(conn.execute("SELECT class_id, name FROM classes WHERE venue_id = ? AND model_key = ?",
                                       (venue, model)).fetchall())
            for class_id, old in sorted(stored.items()):
                new = names[class_id] if class_id < len(names) else None
                if new != old:
                    conflicts.append(f"{model} #{class_id}: Catalog has {old!r}, source has {new!r}")
            added = len(names) - len(stored) if len(names) > len(stored) else 0
            report["models"][model] = {"classes": len(names), "added": added}
            report["added"] += added
            if conflicts:
                continue
            conn.execute("INSERT OR IGNORE INTO models (venue_id, model_key) VALUES (?, ?)", (venue, model))
            conn.executemany(
                "INSERT INTO classes (venue_id, model_key, class_id, name, imported_at) VALUES (?, ?, ?, ?, ?)",
                [(venue, model, i, names[i], at) for i in range(len(stored), len(names))])
        if conflicts:
            raise CatalogError("Import refused, nothing changed. " + "; ".join(conflicts))
        if report["added"]:
            conn.execute("INSERT INTO import_runs (venue_id, at, summary_json) VALUES (?, ?, ?)",
                         (venue, at, json.dumps(report, ensure_ascii=False, sort_keys=True)))
        conn.execute("COMMIT")
        return report
    except sqlite3.Error as e:
        conn.rollback()
        raise CatalogError(f"Import failed, nothing changed: {e}")
    except CatalogError:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def _open_for_read(settings: CatalogSettings) -> Iterator[sqlite3.Connection]:
    if settings.db_path is None or not settings.venue_id:
        raise CatalogUnavailable(MSG_NOT_CONFIGURED)
    if not settings.db_path.is_file():
        raise CatalogUnavailable(MSG_MISSING)
    conn = None
    try:
        conn = sqlite3.connect(settings.db_path.as_uri() + "?mode=ro", uri=True)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if version != SCHEMA_VERSION or not _TABLES <= tables:
            raise CatalogUnavailable(MSG_INVALID)
        yield conn
    except sqlite3.Error:
        raise CatalogUnavailable(MSG_INVALID)
    finally:
        if conn is not None:
            conn.close()


def list_models(settings: CatalogSettings) -> List[Dict[str, Any]]:
    with _open_for_read(settings) as conn:
        rows = conn.execute(
            "SELECT m.model_key, COUNT(c.class_id) FROM models m "
            "LEFT JOIN classes c ON c.venue_id = m.venue_id AND c.model_key = m.model_key "
            "WHERE m.venue_id = ? GROUP BY m.model_key ORDER BY m.model_key",
            (settings.venue_id,)).fetchall()
    return [{"name": name, "class_count": count} for name, count in rows]


def list_classes(settings: CatalogSettings, model: str) -> List[Dict[str, Any]]:
    with _open_for_read(settings) as conn:
        known = conn.execute("SELECT 1 FROM models WHERE venue_id = ? AND model_key = ?",
                             (settings.venue_id, model)).fetchone()
        if known is None:
            raise UnknownModel(model)
        rows = conn.execute(
            "SELECT class_id, name, active FROM classes WHERE venue_id = ? AND model_key = ? ORDER BY class_id",
            (settings.venue_id, model)).fetchall()
    return [{"id": i, "name": name, "active": bool(active)} for i, name, active in rows]


def catalog_status(settings: CatalogSettings) -> Dict[str, Any]:
    try:
        models = list_models(settings)
    except CatalogUnavailable as e:
        return {"available": False, "message": str(e), "venue_name": settings.venue_name, "model_count": 0}
    if not models:
        return {"available": False, "message": MSG_EMPTY, "venue_name": settings.venue_name, "model_count": 0}
    return {"available": True, "message": "", "venue_name": settings.venue_name, "model_count": len(models)}
