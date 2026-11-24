"""Catalog core: the only code that reads or changes the Catalog database."""
from __future__ import annotations

import html
import json
import re
import os
from collections import Counter
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from web_labeler.catalog import assortment as _assortment
from web_labeler.catalog import class_lists as _class_lists
from web_labeler.catalog import class_rules as _rules
from web_labeler.catalog import decisions as _decisions
from web_labeler.catalog import examples as _examples
from web_labeler.catalog import miro as _miro
from web_labeler.catalog import model_cards as _model_cards
from web_labeler.catalog import releases as _releases
from web_labeler.catalog import clips as _clips
from web_labeler.yolo import load_yolo_names_from_yaml

SCHEMA_VERSION = 13
_TABLES = {"venues", "models", "classes", "import_runs", "articles", "pos_identities", "map_entries",
           "map_keys", "class_links", "class_training", "class_rule_texts", "source_findings",
           "miro_pages", "miro_suggested", "miro_reviews", "miro_links", "evidence", "evidence_events", "decisions",
           "map_names", "source_lines", "training_classes", "releases", "locks",
           "pos_files", "pos_file_rows", "findings", "finding_events", "pos_day_marks", "inbox_snoozes", "clip_events"}

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

_SCHEMA_V2 = """
CREATE TABLE articles (
    article_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id     TEXT NOT NULL REFERENCES venues(venue_id),
    display_name TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE pos_identities (
    identity_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id     TEXT NOT NULL REFERENCES venues(venue_id),
    article_id   INTEGER NOT NULL REFERENCES articles(article_id),
    code         TEXT,
    name         TEXT NOT NULL,
    norm_name    TEXT NOT NULL,
    state        TEXT NOT NULL CHECK (state IN ('tracked', 'ignored', 'pending', 'dormant')),
    not_verified INTEGER NOT NULL DEFAULT 0 CHECK (not_verified IN (0, 1)),
    first_seen   TEXT,
    last_seen    TEXT,
    last_sale    TEXT,
    sold_qty     REAL NOT NULL DEFAULT 0,
    sale_rows    INTEGER NOT NULL DEFAULT 0,
    last_price   REAL
);
CREATE UNIQUE INDEX pos_identity_pair ON pos_identities (venue_id, IFNULL(code, ''), name);
CREATE INDEX pos_identity_norm ON pos_identities (venue_id, norm_name);
CREATE TABLE map_entries (
    entry_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id     TEXT NOT NULL REFERENCES venues(venue_id),
    pos_name     TEXT NOT NULL,
    status       TEXT NOT NULL,
    mapping_mode TEXT,
    provisional  TEXT,
    price        REAL,
    notes        TEXT
);
CREATE TABLE map_keys (
    venue_id  TEXT NOT NULL,
    norm_name TEXT NOT NULL,
    entry_id  INTEGER NOT NULL REFERENCES map_entries(entry_id),
    PRIMARY KEY (venue_id, norm_name)
);
CREATE TABLE class_links (
    entry_id   INTEGER NOT NULL REFERENCES map_entries(entry_id),
    position   INTEGER NOT NULL,
    model_key  TEXT NOT NULL,
    class_name TEXT NOT NULL,
    priority   INTEGER,
    note       TEXT,
    PRIMARY KEY (entry_id, position)
);
CREATE INDEX class_links_class ON class_links (model_key, class_name);
"""

_SCHEMA_V3 = """
CREATE TABLE class_training (
    venue_id     TEXT NOT NULL,
    model_key    TEXT NOT NULL,
    class_id     INTEGER NOT NULL,
    trained_name TEXT,
    state        TEXT NOT NULL CHECK (state IN ('trained', 'excluded', 'not_in_model', 'unknown')),
    PRIMARY KEY (venue_id, model_key, class_id),
    FOREIGN KEY (venue_id, model_key, class_id) REFERENCES classes(venue_id, model_key, class_id)
);
CREATE TABLE class_rule_texts (
    venue_id   TEXT NOT NULL,
    model_key  TEXT NOT NULL,
    class_name TEXT NOT NULL,
    position   INTEGER NOT NULL,
    source     TEXT NOT NULL,
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL,
    PRIMARY KEY (venue_id, model_key, class_name, source, position)
);
CREATE TABLE source_findings (
    finding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id   TEXT NOT NULL,
    model_key  TEXT,
    class_name TEXT,
    severity   TEXT NOT NULL CHECK (severity IN ('high', 'medium', 'info')),
    title      TEXT NOT NULL,
    text       TEXT NOT NULL,
    sources    TEXT NOT NULL
);
"""

_SCHEMA_V4 = """
CREATE TABLE miro_pages (
    venue_id    TEXT NOT NULL REFERENCES venues(venue_id),
    page_key    TEXT NOT NULL,
    model_key   TEXT NOT NULL,
    source_file TEXT NOT NULL,
    page_no     INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    heading     TEXT NOT NULL DEFAULT '',
    confidence  TEXT NOT NULL DEFAULT 'normal',
    note        TEXT NOT NULL DEFAULT '',
    imported_at TEXT NOT NULL,
    PRIMARY KEY (venue_id, page_key)
);
CREATE TABLE miro_suggested (
    venue_id   TEXT NOT NULL,
    page_key   TEXT NOT NULL,
    class_name TEXT NOT NULL,
    PRIMARY KEY (venue_id, page_key, class_name)
);
CREATE TABLE miro_reviews (
    venue_id   TEXT NOT NULL,
    page_key   TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    PRIMARY KEY (venue_id, page_key)
);
CREATE TABLE miro_links (
    venue_id   TEXT NOT NULL,
    page_key   TEXT NOT NULL,
    model_key  TEXT NOT NULL,
    class_name TEXT NOT NULL,
    PRIMARY KEY (venue_id, page_key, class_name)
);
CREATE INDEX miro_links_class ON miro_links (venue_id, model_key, class_name);
"""

_SCHEMA_V5 = """
CREATE TABLE evidence (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id    TEXT NOT NULL REFERENCES venues(venue_id),
    identity_id INTEGER NOT NULL REFERENCES pos_identities(identity_id),
    article_id  INTEGER NOT NULL REFERENCES articles(article_id),
    model_key   TEXT,
    class_name  TEXT,
    sha256      TEXT NOT NULL,
    media_type  TEXT NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    size_bytes  INTEGER NOT NULL,
    source      TEXT NOT NULL,
    added_by    TEXT NOT NULL,
    added_role  TEXT NOT NULL,
    added_at    TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('pending', 'confirmed', 'rejected')),
    decided_by  TEXT,
    decided_at  TEXT,
    reason      TEXT,
    UNIQUE (venue_id, sha256)
);
CREATE INDEX evidence_identity ON evidence (venue_id, identity_id, status);
CREATE TABLE evidence_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id    TEXT NOT NULL,
    evidence_id INTEGER NOT NULL REFERENCES evidence(evidence_id),
    at          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    role        TEXT NOT NULL,
    action      TEXT NOT NULL CHECK (action IN ('added', 'confirmed', 'rejected')),
    note        TEXT
);
CREATE TRIGGER evidence_no_delete BEFORE DELETE ON evidence
BEGIN SELECT RAISE(ABORT, 'catalog images are never deleted'); END;
CREATE TRIGGER evidence_events_append_only BEFORE UPDATE ON evidence_events
BEGIN SELECT RAISE(ABORT, 'image history is append-only'); END;
CREATE TRIGGER evidence_events_no_delete BEFORE DELETE ON evidence_events
BEGIN SELECT RAISE(ABORT, 'image history is append-only'); END;
"""

_SCHEMA_V6 = """
CREATE TABLE decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id    TEXT NOT NULL REFERENCES venues(venue_id),
    kind        TEXT NOT NULL CHECK (kind IN ('classes', 'provisional', 'ignore', 'merge', 'revoke', 'propose', 'reject')),
    article_id  INTEGER REFERENCES articles(article_id),
    identity_id INTEGER REFERENCES pos_identities(identity_id),
    target_id   INTEGER REFERENCES decisions(decision_id),
    payload     TEXT NOT NULL,
    reason      TEXT,
    evidence    TEXT,
    actor       TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('admin', 'labeler')),
    created_at  TEXT NOT NULL,
    valid_from  TEXT NOT NULL
);
CREATE INDEX decisions_article ON decisions (venue_id, article_id);
CREATE INDEX decisions_identity ON decisions (venue_id, identity_id);
CREATE TRIGGER decisions_no_update BEFORE UPDATE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions are append-only'); END;
CREATE TRIGGER decisions_no_delete BEFORE DELETE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions are append-only'); END;
"""

_SCHEMA_V7 = """
CREATE TABLE map_names (
    entry_id INTEGER NOT NULL REFERENCES map_entries(entry_id),
    position INTEGER NOT NULL,
    name     TEXT NOT NULL,
    PRIMARY KEY (entry_id, position)
);
CREATE TABLE source_lines (
    venue_id TEXT NOT NULL,
    source   TEXT NOT NULL,
    position INTEGER NOT NULL,
    line     TEXT NOT NULL,
    PRIMARY KEY (venue_id, source, position)
);
CREATE TABLE training_classes (
    venue_id  TEXT NOT NULL,
    model_key TEXT NOT NULL,
    config    TEXT NOT NULL,
    PRIMARY KEY (venue_id, model_key)
);
CREATE TABLE releases (
    release_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id       TEXT NOT NULL REFERENCES venues(venue_id),
    number         INTEGER NOT NULL CHECK (number >= 1),
    kind           TEXT NOT NULL CHECK (kind IN ('publish', 'rollback')),
    based_on       INTEGER,
    actor          TEXT NOT NULL,
    role           TEXT NOT NULL CHECK (role = 'admin'),
    reason         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    decisions_upto INTEGER NOT NULL DEFAULT 0,
    files          TEXT NOT NULL,
    checksums      TEXT NOT NULL,
    summary        TEXT NOT NULL,
    UNIQUE (venue_id, number)
);
CREATE TRIGGER releases_no_update BEFORE UPDATE ON releases
BEGIN SELECT RAISE(ABORT, 'a published release never changes'); END;
CREATE TRIGGER releases_no_delete BEFORE DELETE ON releases
BEGIN SELECT RAISE(ABORT, 'a published release never changes'); END;
"""

_SCHEMA_V8 = """
CREATE TABLE locks (
    lock_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id       TEXT NOT NULL REFERENCES venues(venue_id),
    action         TEXT NOT NULL CHECK (action IN ('lock', 'unlock')),
    release_number INTEGER,
    actor          TEXT NOT NULL,
    role           TEXT NOT NULL CHECK (role = 'admin'),
    reason         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    conditions     TEXT NOT NULL
);
CREATE TRIGGER locks_no_update BEFORE UPDATE ON locks
BEGIN SELECT RAISE(ABORT, 'lock history is append-only'); END;
CREATE TRIGGER locks_no_delete BEFORE DELETE ON locks
BEGIN SELECT RAISE(ABORT, 'lock history is append-only'); END;
"""

_SCHEMA_V9 = """
CREATE TABLE pos_files (
    file_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id    TEXT NOT NULL REFERENCES venues(venue_id),
    file_name   TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    size        INTEGER NOT NULL,
    received_at TEXT NOT NULL,
    report_day  TEXT,
    name_day    TEXT,
    ring_first  TEXT,
    ring_last   TEXT,
    rows_total  INTEGER NOT NULL,
    rows_paid   INTEGER NOT NULL,
    rows_unpaid INTEGER NOT NULL,
    bad_rows    INTEGER NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('ok', 'bad')),
    baseline    INTEGER NOT NULL DEFAULT 0 CHECK (baseline IN (0, 1)),
    UNIQUE (venue_id, sha256)
);
CREATE INDEX pos_files_day ON pos_files (venue_id, report_day);
CREATE TABLE pos_file_rows (
    file_id   INTEGER NOT NULL REFERENCES pos_files(file_id),
    row_key   TEXT NOT NULL,
    count     INTEGER NOT NULL,
    code      TEXT NOT NULL,
    name      TEXT NOT NULL,
    ring_time TEXT NOT NULL,
    quantity  REAL NOT NULL,
    price     REAL NOT NULL,
    PRIMARY KEY (file_id, row_key)
);
CREATE TABLE findings (
    finding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id   TEXT NOT NULL REFERENCES venues(venue_id),
    kind       TEXT NOT NULL,
    day        TEXT,
    file_id    INTEGER REFERENCES pos_files(file_id),
    title      TEXT NOT NULL,
    text       TEXT NOT NULL,
    data       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE finding_events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id   TEXT NOT NULL,
    finding_id INTEGER NOT NULL REFERENCES findings(finding_id),
    action     TEXT NOT NULL CHECK (action IN ('applied', 'denied', 'reopened')),
    actor      TEXT NOT NULL,
    role       TEXT NOT NULL,
    reason     TEXT,
    at         TEXT NOT NULL
);
CREATE TRIGGER pos_files_no_delete BEFORE DELETE ON pos_files
BEGIN SELECT RAISE(ABORT, 'received POS reports are never deleted'); END;
CREATE TRIGGER pos_files_no_update BEFORE UPDATE ON pos_files
BEGIN SELECT RAISE(ABORT, 'received POS reports never change'); END;
CREATE TRIGGER pos_file_rows_no_change BEFORE UPDATE ON pos_file_rows
BEGIN SELECT RAISE(ABORT, 'received POS rows never change'); END;
CREATE TRIGGER pos_file_rows_no_delete BEFORE DELETE ON pos_file_rows
BEGIN SELECT RAISE(ABORT, 'received POS rows never change'); END;
CREATE TRIGGER findings_no_update BEFORE UPDATE ON findings
BEGIN SELECT RAISE(ABORT, 'findings never change; decisions are events'); END;
CREATE TRIGGER findings_no_delete BEFORE DELETE ON findings
BEGIN SELECT RAISE(ABORT, 'findings are never deleted'); END;
CREATE TRIGGER finding_events_no_update BEFORE UPDATE ON finding_events
BEGIN SELECT RAISE(ABORT, 'finding history is append-only'); END;
CREATE TRIGGER finding_events_no_delete BEFORE DELETE ON finding_events
BEGIN SELECT RAISE(ABORT, 'finding history is append-only'); END;
"""

_SCHEMA_V10 = """
CREATE TABLE pos_day_marks (
    mark_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id TEXT NOT NULL REFERENCES venues(venue_id),
    day      TEXT NOT NULL,
    action   TEXT NOT NULL CHECK (action IN ('closed', 'reopened')),
    actor    TEXT NOT NULL,
    role     TEXT NOT NULL,
    reason   TEXT,
    at       TEXT NOT NULL
);
CREATE TRIGGER pos_day_marks_no_update BEFORE UPDATE ON pos_day_marks
BEGIN SELECT RAISE(ABORT, 'day marks are append-only'); END;
CREATE TRIGGER pos_day_marks_no_delete BEFORE DELETE ON pos_day_marks
BEGIN SELECT RAISE(ABORT, 'day marks are append-only'); END;
"""

_SCHEMA_V11 = """
CREATE TABLE inbox_snoozes (
    snooze_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id    TEXT NOT NULL REFERENCES venues(venue_id),
    identity_id INTEGER NOT NULL REFERENCES pos_identities(identity_id),
    until_day   TEXT NOT NULL,
    actor       TEXT NOT NULL,
    role        TEXT NOT NULL,
    reason      TEXT,
    at          TEXT NOT NULL
);
CREATE TRIGGER inbox_snoozes_no_update BEFORE UPDATE ON inbox_snoozes
BEGIN SELECT RAISE(ABORT, 'snoozes are append-only'); END;
CREATE TRIGGER inbox_snoozes_no_delete BEFORE DELETE ON inbox_snoozes
BEGIN SELECT RAISE(ABORT, 'snoozes are append-only'); END;
"""

# v12: decisions may also be class_target (D9). SQLite cannot change a CHECK, so the table is rebuilt with every row
# and id kept (run with foreign keys off, see _open_for_write); the append-only triggers come back unchanged.
_SCHEMA_V12 = """
CREATE TABLE decisions_v12 (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id    TEXT NOT NULL REFERENCES venues(venue_id),
    kind        TEXT NOT NULL CHECK (kind IN ('classes', 'provisional', 'ignore', 'merge', 'revoke', 'propose', 'reject',
                                              'class_target')),
    article_id  INTEGER REFERENCES articles(article_id),
    identity_id INTEGER REFERENCES pos_identities(identity_id),
    target_id   INTEGER REFERENCES decisions(decision_id),
    payload     TEXT NOT NULL,
    reason      TEXT,
    evidence    TEXT,
    actor       TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('admin', 'labeler')),
    created_at  TEXT NOT NULL,
    valid_from  TEXT NOT NULL
);
INSERT INTO decisions_v12 (decision_id, venue_id, kind, article_id, identity_id, target_id, payload, reason, evidence, actor,
                           role, created_at, valid_from)
    SELECT decision_id, venue_id, kind, article_id, identity_id, target_id, payload, reason, evidence, actor, role,
           created_at, valid_from FROM decisions;
DROP TABLE decisions;
ALTER TABLE decisions_v12 RENAME TO decisions;
CREATE INDEX decisions_article ON decisions (venue_id, article_id);
CREATE INDEX decisions_identity ON decisions (venue_id, identity_id);
CREATE TRIGGER decisions_no_update BEFORE UPDATE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions are append-only'); END;
CREATE TRIGGER decisions_no_delete BEFORE DELETE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions are append-only'); END;
"""

# v13: footage of new articles (ticket 16-18). Every step is an event; files on disk are only what events say.
_SCHEMA_V13 = """
CREATE TABLE clip_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id    TEXT NOT NULL REFERENCES venues(venue_id),
    identity_id INTEGER NOT NULL REFERENCES pos_identities(identity_id),
    action      TEXT NOT NULL CHECK (action IN ('planned', 'downloaded', 'failed', 'missing', 'deleted', 'warned',
                                                'preview', 'seen', 'not_visible', 'sent')),
    sale_time   TEXT,
    day         TEXT,
    bytes       INTEGER,
    folder      TEXT,
    detail      TEXT NOT NULL DEFAULT '{}',
    actor       TEXT,
    role        TEXT,
    at          TEXT NOT NULL
);
CREATE INDEX clip_events_identity ON clip_events (venue_id, identity_id);
CREATE TRIGGER clip_events_no_update BEFORE UPDATE ON clip_events
BEGIN SELECT RAISE(ABORT, 'clip history is append-only'); END;
CREATE TRIGGER clip_events_no_delete BEFORE DELETE ON clip_events
BEGIN SELECT RAISE(ABORT, 'clip history is append-only'); END;
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
    pos_dir: Optional[Path] = None
    pos_adapter: Optional[str] = None
    article_map_path: Optional[Path] = None
    blacklist_path: Optional[Path] = None
    active_days: int = 30
    import_baseline: Optional[Mapping[str, Any]] = None
    rule_sources: Optional[_rules.RuleSources] = None
    miro_dir: Optional[Path] = None
    miro_files: Optional[Mapping[str, str]] = None
    miro_suggestions_path: Optional[Path] = None
    media_dir: Optional[Path] = None
    dataset_dir: Optional[Path] = None
    backup_dir: Optional[Path] = None
    model_cards: Optional[_model_cards.ModelCardSources] = None
    release_dir: Optional[Path] = None
    pipeline_map_path: Optional[Path] = None
    trial_dir: Optional[Path] = None
    reminder_emails: Tuple[str, ...] = ()
    clips: Optional[_clips.ClipSettings] = None

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

        def optional_path(key: str) -> Optional[Path]:
            value = str(cfg.get(key) or "").strip()
            if not value:
                return None
            path = Path(value).expanduser()
            return (path if path.is_absolute() else Path(__file__).resolve().parents[2] / path).resolve()

        baseline = cfg.get("catalog_import_baseline")
        rule_sources = None
        training_dir = optional_path("catalog_training_config_dir")
        if training_dir:
            files = cfg.get("catalog_rule_files")
            rule_files = {str(m): (str(v["label"]), Path(str(v["path"])).expanduser().resolve())
                          for m, v in files.items()} if isinstance(files, dict) else {}
            rule_sources = _rules.RuleSources(
                training_dir, optional_path("catalog_documented_classes_dir"), optional_path("catalog_cards_dir"),
                optional_path("catalog_word_exports_dir"), optional_path("catalog_live_models_dir"), rule_files,
                optional_path("catalog_known_conflicts_path"))
        miro_files = cfg.get("catalog_miro_files")
        media_dir = optional_path("catalog_media_dir") or (db_path.parent / "media" if db_path else None)
        dataset_dir = optional_path("catalog_dataset_dir") or (Path(raw_base_path).resolve() if raw_base_path else None)
        if dataset_dir is not None and dataset_dir not in roots:
            roots.append(dataset_dir)
        live_models = optional_path("catalog_live_models_dir")
        model_cards = _model_cards.ModelCardSources(
            live_models, optional_path("catalog_promotions_path"), optional_path("catalog_training_root"),
            optional_path("catalog_training_archives_dir"),
            str(cfg.get("catalog_training_tenant") or "").strip() or venue_id) if live_models else None
        return cls(db_path, venue_id, venue_name, Path(models_dir).resolve(), tuple(roots),
                   optional_path("catalog_pos_dir"), str(cfg.get("catalog_pos_adapter") or "").strip() or None,
                   optional_path("article_map_path"), optional_path("catalog_blacklist_path"),
                   int(cfg.get("catalog_active_days") or 30), baseline if isinstance(baseline, dict) else None,
                   rule_sources, optional_path("catalog_miro_dir"),
                   {str(k): str(v) for k, v in miro_files.items()} if isinstance(miro_files, dict) else None,
                   optional_path("catalog_miro_suggestions_path"), media_dir, dataset_dir,
                   optional_path("catalog_backup_dir"), model_cards,
                   optional_path("catalog_release_dir") or (db_path.parent / "releases" if db_path else None),
                   optional_path("catalog_pipeline_map_path"),
                   optional_path("catalog_trial_dir") or (db_path.parent / "trial_days" if db_path else None),
                   tuple(str(e).strip() for e in (cfg.get("catalog_reminder_emails") or []) if str(e).strip()),
                   _clips.ClipSettings.from_config(cfg))

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


def model_cards_snapshot_path(settings: CatalogSettings) -> Optional[Path]:
    if settings.model_cards is None or settings.db_path is None:
        return None
    return settings.db_path.parent / _model_cards.SNAPSHOT_NAME


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
            conn.executescript(f"BEGIN;{_SCHEMA}{_SCHEMA_V2}{_SCHEMA_V3}{_SCHEMA_V4}{_SCHEMA_V5}{_SCHEMA_V6}{_SCHEMA_V7}{_SCHEMA_V8}{_SCHEMA_V9}{_SCHEMA_V10}{_SCHEMA_V11}{_SCHEMA_V12}{_SCHEMA_V13}"
                               f"PRAGMA user_version = {SCHEMA_VERSION};COMMIT;")
        elif version in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
            steps = [_SCHEMA_V2, _SCHEMA_V3, _SCHEMA_V4, _SCHEMA_V5, _SCHEMA_V6, _SCHEMA_V7, _SCHEMA_V8, _SCHEMA_V9,
                     _SCHEMA_V10, _SCHEMA_V11, _SCHEMA_V12, _SCHEMA_V13][version - 1:]
            conn.execute("PRAGMA foreign_keys = OFF")  # the decisions rebuild (v12) drops a table others point to
            before = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] if version >= 6 else 0
            conn.executescript(f"BEGIN;{''.join(steps)}PRAGMA user_version = {SCHEMA_VERSION};COMMIT;")
            problems = conn.execute("PRAGMA foreign_key_check").fetchall()
            after = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            conn.execute("PRAGMA foreign_keys = ON")
            if problems or after != before:
                raise CatalogError(f"Catalog upgrade check failed (decisions {before} -> {after}, references {len(problems)})")
        elif version != SCHEMA_VERSION:
            raise CatalogError(f"Unsupported Catalog database version: {version}")
    except sqlite3.Error as e:
        conn.close()
        raise CatalogError(f"Catalog database is not usable: {e}")
    except CatalogError:
        conn.close()
        raise
    return conn


def _build_assortment_plan(settings: CatalogSettings) -> _assortment.AssortmentPlan:
    missing = [label for label, value in (("catalog_pos_dir", settings.pos_dir),
                                          ("catalog_pos_adapter", settings.pos_adapter),
                                          ("article_map_path", settings.article_map_path),
                                          ("catalog_blacklist_path", settings.blacklist_path)) if not value]
    if missing:
        raise CatalogError("Catalog import is not configured: " + ", ".join(missing))
    try:
        return _assortment.build_plan(settings.pos_dir, settings.pos_adapter, settings.article_map_path,
                                      settings.blacklist_path, settings.active_days,
                                      dict(settings.import_baseline) if settings.import_baseline else None)
    except _assortment.AssortmentError as e:
        raise CatalogError(str(e))


def _apply_assortment(conn: sqlite3.Connection, venue: str, plan: _assortment.AssortmentPlan, at: str) -> None:
    """Refresh the map snapshot and POS identities. Article numbers stay the same across imports."""
    norm = _assortment.normalize_name
    old_entries = [r[0] for r in conn.execute("SELECT entry_id FROM map_entries WHERE venue_id = ?", (venue,))]
    conn.executemany("DELETE FROM class_links WHERE entry_id = ?", [(i,) for i in old_entries])
    conn.executemany("DELETE FROM map_names WHERE entry_id = ?", [(i,) for i in old_entries])
    conn.execute("DELETE FROM map_keys WHERE venue_id = ?", (venue,))
    conn.execute("DELETE FROM map_entries WHERE venue_id = ?", (venue,))
    entry_ids: List[int] = []
    for e in plan.entries:
        cur = conn.execute("INSERT INTO map_entries (venue_id, pos_name, status, mapping_mode, provisional, price, notes) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (venue, e.pos_name, e.status, e.mapping_mode, e.provisional, e.price, e.notes))
        entry_ids.append(cur.lastrowid)
        conn.executemany("INSERT INTO class_links (entry_id, position, model_key, class_name, priority, note) "
                         "VALUES (?, ?, ?, ?, ?, ?)",
                         [(cur.lastrowid, i, l["model"], l["class"], l["priority"], l["note"]) for i, l in enumerate(e.links)])
        conn.executemany("INSERT INTO map_keys (venue_id, norm_name, entry_id) VALUES (?, ?, ?)",
                         [(venue, k, cur.lastrowid) for k in e.keys])
        conn.executemany("INSERT INTO map_names (entry_id, position, name) VALUES (?, ?, ?)",
                         [(cur.lastrowid, i, n) for i, n in enumerate(e.names)])
    conn.execute("DELETE FROM source_lines WHERE venue_id = ? AND source = 'blacklist'", (venue,))
    conn.executemany("INSERT INTO source_lines (venue_id, source, position, line) VALUES (?, 'blacklist', ?, ?)",
                     [(venue, i, line) for i, line in enumerate(plan.blacklist_lines)])

    known_pairs = {(c, n): a for c, n, a in conn.execute(
        "SELECT code, name, article_id FROM pos_identities WHERE venue_id = ? AND code IS NOT NULL", (venue,))}
    by_code: Dict[str, int] = {c: a for (c, _), a in sorted(known_pairs.items())}
    by_name: Dict[str, int] = {norm(n): a for (_, n), a in sorted(known_pairs.items())}
    for n, a in conn.execute("SELECT norm_name, article_id FROM pos_identities WHERE venue_id = ? AND code IS NULL", (venue,)):
        by_name.setdefault(n, a)
    needs_review: List[Dict[str, Any]] = []
    pair_article: Dict[Tuple[str, str], int] = {}
    for group in plan.groups:
        existing = sorted({a for c, n in group for a in (by_code.get(c), by_name.get(norm(n))) if a is not None})
        if existing:
            article = existing[0]
        else:
            newest = max(group, key=lambda p: (plan.stats[p].last_seen, p))
            article = conn.execute("INSERT INTO articles (venue_id, display_name, created_at) VALUES (?, ?, ?)",
                                   (venue, newest[1].strip(), at)).lastrowid
        if len(existing) > 1:
            needs_review.append({"pairs": [{"code": c, "name": n} for c, n in group], "articles": existing})
        for pair in group:
            target = known_pairs.get(pair) or by_code.get(pair[0]) or by_name.get(norm(pair[1])) or article
            pair_article[pair] = target
            by_code.setdefault(pair[0], target)
            by_name.setdefault(norm(pair[1]), target)

    for pair, target in sorted(pair_article.items()):
        code, name = pair
        item = plan.stats[pair]
        state = plan.states[pair]
        last_sale = item.last_sale.isoformat(sep=" ") if item.last_sale else None
        values = (state, int(state == _assortment.TRACKED), item.first_seen.isoformat(sep=" "),
                  item.last_seen.isoformat(sep=" "), last_sale, item.sold, item.sale_rows, item.last_price)
        if pair in known_pairs:
            conn.execute("UPDATE pos_identities SET state = ?, not_verified = ?, first_seen = ?, last_seen = ?, last_sale = ?, "
                         "sold_qty = ?, sale_rows = ?, last_price = ? WHERE venue_id = ? AND code = ? AND name = ?",
                         values + (venue, code, name))
            continue
        placeholder = conn.execute("SELECT identity_id, article_id FROM pos_identities "
                                   "WHERE venue_id = ? AND code IS NULL AND norm_name = ?", (venue, norm(name))).fetchone()
        if placeholder:  # a name known only from the map has now been sold
            conn.execute("UPDATE pos_identities SET code = ?, name = ?, state = ?, not_verified = ?, first_seen = ?, "
                         "last_seen = ?, last_sale = ?, sold_qty = ?, sale_rows = ?, last_price = ? WHERE identity_id = ?",
                         (code, name) + values + (placeholder[0],))
        else:
            conn.execute("INSERT INTO pos_identities (venue_id, article_id, code, name, norm_name, state, not_verified, "
                         "first_seen, last_seen, last_sale, sold_qty, sale_rows, last_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (venue, target, code, name, norm(name)) + values)

    sold_names = {norm(n) for _, n in plan.stats}
    map_only = 0
    for index, entry in enumerate(plan.entries):
        for key in entry.keys:
            if key in sold_names:
                continue
            map_only += 1
            state = _assortment.IGNORED if entry.status == "skip" else (
                _assortment.TRACKED if entry.links else _assortment.DORMANT)
            row = conn.execute("SELECT identity_id FROM pos_identities WHERE venue_id = ? AND code IS NULL AND norm_name = ?",
                               (venue, key)).fetchone()
            if row:
                conn.execute("UPDATE pos_identities SET state = ?, not_verified = ? WHERE identity_id = ?",
                             (state, int(state == _assortment.TRACKED), row[0]))
                continue
            siblings = {pair_article[p] for p in pair_article if norm(p[1]) in entry.keys}
            if len(siblings) == 1:
                article = siblings.pop()
            else:
                article = conn.execute("INSERT INTO articles (venue_id, display_name, created_at) VALUES (?, ?, ?)",
                                       (venue, entry.pos_name.strip(), at)).lastrowid
            conn.execute("INSERT INTO pos_identities (venue_id, article_id, code, name, norm_name, state, not_verified) "
                         "VALUES (?, ?, NULL, ?, ?, ?, ?)", (venue, article, key, key, state, int(state == _assortment.TRACKED)))
    known_keys = {k for e in plan.entries for k in e.keys}
    for identity_id, key in conn.execute("SELECT identity_id, norm_name FROM pos_identities WHERE venue_id = ? AND code IS NULL",
                                         (venue,)).fetchall():
        if key not in known_keys:  # removed from the map: kept, but no longer tracked
            conn.execute("UPDATE pos_identities SET state = ?, not_verified = 0 WHERE identity_id = ?",
                         (_assortment.DORMANT, identity_id))
    for article_id, in conn.execute("SELECT article_id FROM articles WHERE venue_id = ?", (venue,)).fetchall():
        row = conn.execute("SELECT name FROM pos_identities WHERE article_id = ? ORDER BY last_seen IS NULL, last_seen DESC, name LIMIT 1",
                           (article_id,)).fetchone()
        if row:
            conn.execute("UPDATE articles SET display_name = ? WHERE article_id = ?", (row[0].strip(), article_id))
    plan.report["map_only_names"] = map_only
    plan.report["needs_review"] = needs_review


def _apply_rules(conn: sqlite3.Connection, venue: str, plan: _rules.RulesPlan) -> None:
    for table in ("class_training", "class_rule_texts", "source_findings", "training_classes"):
        conn.execute(f"DELETE FROM {table} WHERE venue_id = ?", (venue,))
    conn.executemany("INSERT INTO training_classes (venue_id, model_key, config) VALUES (?, ?, ?)",
                     [(venue, model, json.dumps(cfg, ensure_ascii=False, sort_keys=True))
                      for model, cfg in plan.training_classes.items()])
    conn.executemany("INSERT INTO class_training (venue_id, model_key, class_id, trained_name, state) VALUES (?, ?, ?, ?, ?)",
                     [(venue, model, row["class_id"], row["trained_name"], row["state"])
                      for model, rows in plan.classes.items() for row in rows.values()])
    conn.executemany("INSERT INTO class_rule_texts (venue_id, model_key, class_name, position, source, kind, text) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?)",
                     [(venue, t["model"], t["class"], t["position"], t["source"], t["kind"], t["text"]) for t in plan.texts])
    conn.executemany("INSERT INTO source_findings (venue_id, model_key, class_name, severity, title, text, sources) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?)",
                     [(venue, f["model"], f["class"], f["severity"], f["title"], f["text"],
                       json.dumps(f["sources"], ensure_ascii=False)) for f in plan.findings])


def _build_miro_plan(settings: CatalogSettings, sources: Mapping[str, List[str]]) -> Optional[_miro.MiroPlan]:
    if not settings.miro_dir and not settings.miro_files:
        return None
    if not settings.miro_dir or not settings.miro_files or settings.media_dir is None:
        raise CatalogError("Miro import needs both catalog_miro_dir and catalog_miro_files")
    for root in settings.forbidden_roots + ((settings.dataset_dir,) if settings.dataset_dir else ()):
        if _is_within(settings.media_dir.resolve(), root):
            raise CatalogError(f"Catalog images must not be stored inside {root}")
    try:
        return _miro.build_plan(settings.miro_dir, settings.miro_files, settings.miro_suggestions_path, sources)
    except _miro.MiroError as e:
        raise CatalogError(f"Import refused, nothing changed. {e}")


def _media_path(settings: CatalogSettings, sha256: str) -> Path:
    return settings.media_dir / "miro" / f"{sha256}.jpg"


def _store_miro_images(settings: CatalogSettings, plan: _miro.MiroPlan) -> None:
    """Images are stored by content, so writing them before the database commit is harmless."""
    for page in plan.pages:
        target = _media_path(settings, page.sha256)
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        tmp.write_bytes(page.data)
        os.replace(tmp, target)


def _apply_miro(conn: sqlite3.Connection, venue: str, plan: _miro.MiroPlan, at: str) -> None:
    """Pages and suggestions follow the sources. Admin decisions are never touched by an import."""
    for page in plan.pages:
        s = plan.suggestions.get(page.key) or _miro.Suggestion("", [])
        conn.execute(
            "INSERT INTO miro_pages (venue_id, page_key, model_key, source_file, page_no, sha256, width, height, heading, "
            "confidence, note, imported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (venue_id, page_key) DO UPDATE SET model_key = excluded.model_key, source_file = excluded.source_file, "
            "page_no = excluded.page_no, sha256 = excluded.sha256, width = excluded.width, height = excluded.height, "
            "heading = excluded.heading, confidence = excluded.confidence, note = excluded.note",
            (venue, page.key, page.model, page.source_file, page.page_no, page.sha256, page.width, page.height,
             s.heading, s.confidence, s.note, at))
    conn.execute("DELETE FROM miro_suggested WHERE venue_id = ?", (venue,))
    conn.executemany("INSERT INTO miro_suggested (venue_id, page_key, class_name) VALUES (?, ?, ?)",
                     [(venue, key, c) for key, s in plan.suggestions.items() for c in s.classes])
    keys = {p.key for p in plan.pages}
    stored = {r[0] for r in conn.execute("SELECT page_key FROM miro_pages WHERE venue_id = ?", (venue,))}
    plan.report["kept_but_missing_in_source"] = sorted(stored - keys)
    plan.report["confirmed"] = conn.execute("SELECT COUNT(*) FROM miro_reviews WHERE venue_id = ?", (venue,)).fetchone()[0]


def import_catalog(settings: CatalogSettings) -> Dict[str, Any]:
    """Full import: class lists, then articles and their POS history, in one all-or-nothing step."""
    plan = _build_assortment_plan(settings)
    rules = None
    if settings.rule_sources is not None:
        sold_names = {_assortment.normalize_name(n) for _, n in plan.stats}
        map_classes: Dict[str, Dict[str, bool]] = {}
        for entry in plan.entries:
            if entry.status != "skip":
                sold = any(k in sold_names for k in entry.keys)
                for link in entry.links:
                    model_classes = map_classes.setdefault(link["model"], {})
                    model_classes[link["class"]] = model_classes.get(link["class"], False) or sold
        rules = _rules.build_rules_plan(settings.rule_sources, _read_sources(settings), map_classes)
    miro = _build_miro_plan(settings, _read_sources(settings))
    if miro is not None:
        _store_miro_images(settings, miro)
    return _import(settings, plan, rules, miro)


def import_class_lists(settings: CatalogSettings) -> Dict[str, Any]:
    return _import(settings, None)


def _import(settings: CatalogSettings, plan: Optional[_assortment.AssortmentPlan],
            rules: Optional[_rules.RulesPlan] = None, miro: Optional[_miro.MiroPlan] = None) -> Dict[str, Any]:
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
        if plan is not None:
            _apply_assortment(conn, venue, plan, at)
            report["assortment"] = plan.report
        if rules is not None:
            _apply_rules(conn, venue, rules)
            report["class_rules"] = rules.report
        if miro is not None:
            _apply_miro(conn, venue, miro, at)
            report["miro"] = miro.report
        if report["added"] or plan is not None:
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


STATE_LABELS = {"tracked": "Praćeno", "ignored": "Ignorisano", "pending": "Čeka odluku", "dormant": "Uspavano"}


def _evidence_by_identity(conn: sqlite3.Connection, venue: str) -> Dict[int, Dict[str, Any]]:
    """Confirmed image (newest) and number of images waiting, per POS variation."""
    result: Dict[int, Dict[str, Any]] = {}
    for evidence_id, identity_id, status in conn.execute(
            "SELECT evidence_id, identity_id, status FROM evidence WHERE venue_id = ? AND status != 'rejected' "
            "ORDER BY evidence_id", (venue,)):
        item = result.setdefault(identity_id, {"confirmed": None, "pending": 0})
        if status == "confirmed":
            item["confirmed"] = evidence_id
        else:
            item["pending"] += 1
    return result


def mapping_text(v: _decisions.Variation) -> str:
    if v.state == "ignored":
        why = _decisions.IGNORE_WHY.get(v.ignore_why or "")
        return f"ne pratimo: {why}" if why else "ne pratimo"
    if not v.links and v.target:
        return f"čeka klasu {v.target[0]}.{v.target[1]} (živi model je još ne zna)"
    if not v.links:
        return "mapiranje nedostaje"
    return ", ".join(f"{m}.{c}" for m, c in v.links)


def _variation(v: _decisions.Variation, evidence: Optional[Mapping[int, Dict[str, Any]]] = None,
               flow: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    ev = (evidence or {}).get(v.identity_id, {"confirmed": None, "pending": 0})
    tags: List[str] = []
    if v.not_verified and ev["confirmed"] is None:
        tags.append("nije provereno na snimku")
    if v.provisional:
        tags.append("privremeno")
    if v.code is None:
        tags.append("nije prodavano")
    elif v.sold <= 0:
        tags.append("sve stornirano")
    if ev["pending"]:
        tags.append("slika čeka potvrdu")
    if v.under_review:
        tags.append("na proveri")
    if v.awaiting_release:
        tags.append("ručna odluka, čeka izdanje")
    if flow and flow["stage"] != "tracked":
        tags.append(f"nova klasa: {flow['label']}")
    image = None if ev["confirmed"] is None else {"evidence_id": ev["confirmed"], "url": _evidence_url(ev["confirmed"]),
                                                  "source_label": EVIDENCE_SOURCE_LABEL}
    return {"identity_id": v.identity_id, "article_id": v.article_id, "code": v.code, "name": v.name.strip(),
            "state": v.state, "state_label": STATE_LABELS[v.state], "sold": v.sold, "last_sale": v.last_sale, "tags": tags,
            "provisional_reason": v.provisional, "mapping_mode": v.mode, "classes": [f"{m}.{c}" for m, c in v.links],
            "mapping": mapping_text(v), "image": image, "pending_images": ev["pending"], "flow": dict(flow) if flow else None}


TRAINING_LABELS = {"trained": "U modelu", "excluded": "Izbačena iz treninga", "not_in_model": "Nije u živom modelu",
                   "unknown": "Nepoznato"}
RULE_KIND_LABELS = {"rule": "Važeće pravilo", "recognize": "Kako se prepoznaje", "examples": "Primeri",
                    "confused": "Česte greške", "note": "Napomene"}
_LINKED_IDENTITIES = (
    "FROM class_links l JOIN map_entries e ON e.entry_id = l.entry_id AND e.venue_id = ? AND e.status != 'skip' "
    "JOIN map_keys k ON k.entry_id = e.entry_id "
    "JOIN pos_identities i ON i.venue_id = k.venue_id AND i.norm_name = k.norm_name ")


def _training_rows(conn: sqlite3.Connection, venue: str, model: str) -> Dict[str, Dict[str, Any]]:
    rows = conn.execute("SELECT c.name, t.trained_name, t.state FROM classes c LEFT JOIN class_training t "
                        "ON t.venue_id = c.venue_id AND t.model_key = c.model_key AND t.class_id = c.class_id "
                        "WHERE c.venue_id = ? AND c.model_key = ? ORDER BY c.class_id", (venue, model)).fetchall()
    return {name: {"trained_name": trained, "state": state or "unknown"} for name, trained, state in rows}


def _class_scope(name: str, training: Mapping[str, Dict[str, Any]]) -> List[str]:
    trained = training.get(name, {}).get("trained_name")
    return sorted({name, trained} - {None})


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _load_decisions(conn: sqlite3.Connection, venue: str) -> List[_decisions.Decision]:
    return [_decisions.Decision(i, k, a, ident, json.loads(p), vf, t) for i, k, a, ident, p, vf, t in conn.execute(
        "SELECT decision_id, kind, article_id, identity_id, payload, valid_from, target_id FROM decisions "
        "WHERE venue_id = ? ORDER BY decision_id", (venue,))]


def _current(conn: sqlite3.Connection, venue: str, as_of: Optional[str] = None) -> Dict[int, _decisions.Variation]:
    """Every POS variation as the sources describe it, then with the decisions in force on as_of."""
    result: Dict[int, _decisions.Variation] = {}
    for row in conn.execute(
            "SELECT i.identity_id, i.article_id, i.code, i.name, i.state, i.sold_qty, i.last_sale, i.not_verified, "
            "e.provisional, e.mapping_mode, e.entry_id FROM pos_identities i "
            "LEFT JOIN map_keys k ON k.venue_id = i.venue_id AND k.norm_name = i.norm_name "
            "LEFT JOIN map_entries e ON e.entry_id = k.entry_id AND e.status != 'skip' "
            "WHERE i.venue_id = ? ORDER BY i.identity_id", (venue,)):
        identity_id, article_id, code, name, state, sold, last_sale, not_verified, provisional, mode, entry_id = row
        links = [(m, c) for m, c in conn.execute(
            "SELECT model_key, class_name FROM class_links WHERE entry_id = ? ORDER BY position", (entry_id,))] if entry_id else []
        result[identity_id] = _decisions.Variation(identity_id, article_id, code, name, state, sold, last_sale,
                                                   bool(not_verified), provisional, mode, links)
    _decisions.apply(result, _load_decisions(conn, venue), as_of or _today())
    _resolve_targets(conn, venue, result)
    released = conn.execute("SELECT MAX(decisions_upto) FROM releases WHERE venue_id = ?", (venue,)).fetchone()[0] or 0
    for v in result.values():
        v.awaiting_release = v.decided and v.last_decision > released
    _apply_till_findings(conn, venue, result)
    return result


def _trained_names(conn: sqlite3.Connection, venue: str) -> Dict[Tuple[str, str], str]:
    """(model lower, tool class name) -> the name the live model outputs, for classes the live model knows."""
    return {(m.lower(), n): t for m, n, t in conn.execute(
        "SELECT c.model_key, c.name, t.trained_name FROM classes c JOIN class_training t ON t.venue_id = c.venue_id "
        "AND t.model_key = c.model_key AND t.class_id = c.class_id WHERE c.venue_id = ? AND t.state = 'trained' "
        "AND t.trained_name IS NOT NULL", (venue,))}


def _resolve_targets(conn: sqlite3.Connection, venue: str, current: Dict[int, _decisions.Variation]) -> None:
    """A class chosen for a new article maps only once the live model knows it (D9). Until then: no mapping."""
    waiting = [v for v in current.values() if v.target and v.state != "ignored"]
    if not waiting:
        return
    trained = _trained_names(conn, venue)
    for v in waiting:
        name = trained.get(v.target)
        if name:
            v.links, v.state, v.not_verified, v.mode = [(v.target[0], name)], "tracked", True, "fallback"


def _till_findings(conn: sqlite3.Connection, venue: str) -> List[Tuple[int, Dict[str, Any], str]]:
    """Rename and second-code findings that still count (open = na proveri, applied = confirmed), newest first."""
    out = []
    for finding_id, kind, data, created_at in conn.execute(
            "SELECT finding_id, kind, data, created_at FROM findings WHERE venue_id = ? AND kind IN ('rename', 'second_code') "
            "ORDER BY finding_id DESC", (venue,)):
        state = _finding_status(conn, finding_id)
        if state != "denied":
            out.append((finding_id, dict(json.loads(data), kind=kind, created_at=created_at), state))
    return out


def _apply_till_findings(conn: sqlite3.Connection, venue: str, current: Dict[int, _decisions.Variation]) -> None:
    by_key = {((v.code or ""), _assortment.normalize_name(v.name)): v for v in current.values()}
    for _, data, state in _till_findings(conn, venue):
        v = by_key.get((data.get("code") or "", _assortment.normalize_name(data.get("name", ""))))
        if v is None or v.decided or v.state == "ignored":
            continue
        if data["kind"] == "rename" and not v.links and data.get("classes"):
            v.links, v.state = [tuple(link) for link in data["classes"]], "tracked"
        v.under_review = v.under_review or state == "open"


def _in_scope(v: _decisions.Variation, model: str, names: Sequence[str]) -> bool:
    return any(m == model.lower() and c in names for m, c in v.links)


def _sorted(items: Iterable[_decisions.Variation]) -> List[_decisions.Variation]:
    return sorted(items, key=lambda v: (-v.sold, v.name, v.code or ""))


def list_model_view(settings: CatalogSettings, model: str) -> Dict[str, Any]:
    """Classes as labelers draw them, how training treats them, and map classes with no home."""
    classes = list_classes(settings, model)
    venue = settings.venue_id
    with _open_for_read(settings) as conn:
        training = _training_rows(conn, venue, model)
        current = _current(conn, venue)
    for item in classes:
        info = training[item["name"]]
        scope = _class_scope(item["name"], training)
        item["variation_count"] = sum(_in_scope(v, model, scope) for v in current.values())
        item["trained_name"] = info["trained_name"]
        item["training_state"] = info["state"]
        item["training_label"] = TRAINING_LABELS[info["state"]]
    known = set(training) | {i["trained_name"] for i in training.values() if i["trained_name"]}
    linked = sorted({c for v in current.values() for m, c in v.links if m == model.lower()} - known)
    others = [{"name": n, "variation_count": sum(_in_scope(v, model, [n]) for v in current.values())} for n in linked]
    return {"model": model, "classes": classes, "map_only_classes": others}


def list_variations(settings: CatalogSettings, model: str, class_name: str) -> List[Dict[str, Any]]:
    view = list_model_view(settings, model)
    if class_name not in {c["name"] for c in view["classes"]} | {c["name"] for c in view["map_only_classes"]}:
        raise UnknownModel(f"{model}.{class_name}")
    venue = settings.venue_id
    with _open_for_read(settings) as conn:
        scope = _class_scope(class_name, _training_rows(conn, venue, model))
        current = _current(conn, venue)
        evidence = _evidence_by_identity(conn, venue)
    result = []
    for v in _sorted(x for x in current.values() if _in_scope(x, model, scope)):
        item = _variation(v, evidence)
        own = {c for m, c in v.links if m == model.lower()}
        item["via_class"] = None if class_name in own else next((n for n in scope if n in own), None)
        result.append(item)
    return result


def class_detail(settings: CatalogSettings, model: str, class_name: str) -> Dict[str, Any]:
    variations = list_variations(settings, model, class_name)
    venue = settings.venue_id
    with _open_for_read(settings) as conn:
        training = _training_rows(conn, venue, model)
        info = training.get(class_name, {"trained_name": None, "state": "unknown"})
        trained = info["trained_name"]
        drawn = [n for n, t in training.items() if trained and t["trained_name"] == trained]
        names = _class_scope(class_name, training)
        marks = ",".join("?" for _ in names)
        rows = conn.execute(f"SELECT class_name, kind, text, source FROM class_rule_texts WHERE venue_id = ? AND model_key = ? "
                            f"AND class_name IN ({marks}) ORDER BY class_name != ?, position",
                            (venue, model, *names, class_name)).fetchall()
        findings = conn.execute("SELECT severity, title, text, sources FROM source_findings WHERE venue_id = ? "
                                "AND model_key = ? AND class_name = ? ORDER BY finding_id",
                                (venue, model.lower(), class_name)).fetchall()
    sections = []
    for kind, label in RULE_KIND_LABELS.items():
        items = [{"text": text, "source": source, "from_class": None if owner == class_name else owner}
                 for owner, row_kind, text, source in rows if row_kind == kind]
        if items:
            sections.append({"kind": kind, "label": label, "items": items})
    return {"model": model, "class": class_name, "trained_name": trained, "training_state": info["state"],
            "training_label": TRAINING_LABELS[info["state"]], "drawn_as": drawn, "united": len(drawn) > 1,
            "rules": sections, "has_description": bool(sections), "variations": variations,
            "findings": [{"severity": s, "title": t, "text": x, "sources": json.loads(src)} for s, t, x, src in findings],
            "miro_pages": class_miro_pages(settings, model, class_name)}


def source_report(settings: CatalogSettings) -> Dict[str, Any]:
    with _open_for_read(settings) as conn:
        rows = conn.execute("SELECT model_key, class_name, severity, title, text, sources FROM source_findings "
                            "WHERE venue_id = ? ORDER BY finding_id", (settings.venue_id,)).fetchall()
    findings = [{"model": m, "class": c, "severity": s, "title": t, "text": x, "sources": json.loads(src)}
                for m, c, s, t, x, src in rows]
    return {"findings": findings, "counts": {k: sum(f["severity"] == k for f in findings) for k in ("high", "medium", "info")}}


def list_unclassified(settings: CatalogSettings) -> Dict[str, Any]:
    """Sold POS variations that are not tracked, grouped by state, best sellers first."""
    with _open_for_read(settings) as conn:
        current = _current(conn, settings.venue_id)
        evidence = _evidence_by_identity(conn, settings.venue_id)
    groups = {state: [] for state in ("pending", "dormant", "ignored")}
    for v in _sorted(x for x in current.values() if x.code is not None and x.state != "tracked"):
        groups[v.state].append(_variation(v, evidence))
    return {"groups": [{"state": s, "label": STATE_LABELS[s], "items": items} for s, items in groups.items()]}


def catalog_status(settings: CatalogSettings) -> Dict[str, Any]:
    try:
        models = list_models(settings)
    except CatalogUnavailable as e:
        return {"available": False, "message": str(e), "venue_name": settings.venue_name, "model_count": 0}
    if not models:
        return {"available": False, "message": MSG_EMPTY, "venue_name": settings.venue_name, "model_count": 0}
    return {"available": True, "message": "", "venue_name": settings.venue_name, "model_count": len(models)}


# ---------------- Old Miro pages and dataset examples ----------------

MIRO_SOURCE_LABEL = "Stara Miro strana"
EXAMPLE_SOURCE_LABEL = "Primer iz dataseta"


class UnknownPage(Exception):
    pass


def _page_rows(conn: sqlite3.Connection, venue: str, where: str = "", args: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT p.page_key, p.model_key, p.source_file, p.page_no, p.width, p.height, p.heading, p.confidence, p.note, "
        "r.decided_at FROM miro_pages p LEFT JOIN miro_reviews r ON r.venue_id = p.venue_id AND r.page_key = p.page_key "
        f"WHERE p.venue_id = ? {where} ORDER BY p.model_key, p.page_no", (venue, *args)).fetchall()
    suggested: Dict[str, List[str]] = {}
    for key, name in conn.execute("SELECT page_key, class_name FROM miro_suggested WHERE venue_id = ? ORDER BY class_name", (venue,)):
        suggested.setdefault(key, []).append(name)
    linked: Dict[str, List[str]] = {}
    for key, name in conn.execute("SELECT page_key, class_name FROM miro_links WHERE venue_id = ? ORDER BY class_name", (venue,)):
        linked.setdefault(key, []).append(name)
    pages = []
    for key, model, source, page_no, width, height, heading, confidence, note, decided_at in rows:
        confirmed = decided_at is not None
        classes = linked.get(key, []) if confirmed else suggested.get(key, [])
        pages.append({"key": key, "model": model, "source_file": source, "page_no": page_no, "width": width,
                      "height": height, "heading": heading, "note": note, "low_confidence": confidence == "low",
                      "confirmed": confirmed, "decided_at": decided_at, "classes": classes,
                      "suggested": suggested.get(key, []), "image_url": f"/api/catalog/miro-pages/{key}/image",
                      "source_label": MIRO_SOURCE_LABEL})
    return pages


def list_miro_pages(settings: CatalogSettings) -> Dict[str, Any]:
    """Every imported page, whether or not it has a class, with the admin's progress."""
    with _open_for_read(settings) as conn:
        pages = _page_rows(conn, settings.venue_id)
        classes: Dict[str, List[str]] = {}
        for model, name in conn.execute("SELECT model_key, name FROM classes WHERE venue_id = ? ORDER BY model_key, class_id",
                                        (settings.venue_id,)):
            classes.setdefault(model, []).append(name)
    counts = {"total": len(pages), "confirmed": sum(p["confirmed"] for p in pages),
              "without_class": sum(not p["classes"] for p in pages)}
    return {"pages": pages, "counts": counts, "classes": classes}


def save_miro_page(settings: CatalogSettings, page_key: str, class_names: Sequence[str]) -> Dict[str, Any]:
    """Admin confirms which classes a page belongs to. An empty list confirms a page without a class."""
    names = sorted({str(n).strip() for n in class_names if str(n).strip()})
    conn = _open_for_write(settings)
    venue = settings.venue_id
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT model_key FROM miro_pages WHERE venue_id = ? AND page_key = ?", (venue, page_key)).fetchone()
        if row is None:
            raise UnknownPage(page_key)
        model = row[0]
        known = {r[0] for r in conn.execute("SELECT name FROM classes WHERE venue_id = ? AND model_key = ?", (venue, model))}
        unknown = [n for n in names if n not in known]
        if unknown:
            raise CatalogError(f"Model {model} nema klasu: {', '.join(unknown)}")
        conn.execute("DELETE FROM miro_links WHERE venue_id = ? AND page_key = ?", (venue, page_key))
        conn.executemany("INSERT INTO miro_links (venue_id, page_key, model_key, class_name) VALUES (?, ?, ?, ?)",
                         [(venue, page_key, model, n) for n in names])
        conn.execute("INSERT INTO miro_reviews (venue_id, page_key, decided_at) VALUES (?, ?, ?) "
                     "ON CONFLICT (venue_id, page_key) DO UPDATE SET decided_at = excluded.decided_at",
                     (venue, page_key, _now()))
        conn.execute("COMMIT")
    except sqlite3.Error as e:
        conn.rollback()
        raise CatalogError(f"Nije sačuvano: {e}")
    except (CatalogError, UnknownPage):
        conn.rollback()
        raise
    finally:
        conn.close()
    return next(p for p in list_miro_pages(settings)["pages"] if p["key"] == page_key)


def reset_miro_page(settings: CatalogSettings, page_key: str) -> Dict[str, Any]:
    """Undo the admin's decision: the page goes back to the suggestion."""
    conn = _open_for_write(settings)
    venue = settings.venue_id
    try:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM miro_pages WHERE venue_id = ? AND page_key = ?", (venue, page_key)).fetchone() is None:
            raise UnknownPage(page_key)
        conn.execute("DELETE FROM miro_links WHERE venue_id = ? AND page_key = ?", (venue, page_key))
        conn.execute("DELETE FROM miro_reviews WHERE venue_id = ? AND page_key = ?", (venue, page_key))
        conn.execute("COMMIT")
    except sqlite3.Error as e:
        conn.rollback()
        raise CatalogError(f"Nije sačuvano: {e}")
    except UnknownPage:
        conn.rollback()
        raise
    finally:
        conn.close()
    return next(p for p in list_miro_pages(settings)["pages"] if p["key"] == page_key)


def miro_page_image(settings: CatalogSettings, page_key: str) -> Path:
    with _open_for_read(settings) as conn:
        row = conn.execute("SELECT sha256 FROM miro_pages WHERE venue_id = ? AND page_key = ?",
                           (settings.venue_id, page_key)).fetchone()
    if row is None or settings.media_dir is None:
        raise UnknownPage(page_key)
    path = _media_path(settings, row[0])
    if not path.is_file():
        raise UnknownPage(page_key)
    return path


def class_miro_pages(settings: CatalogSettings, model: str, class_name: str) -> List[Dict[str, Any]]:
    """Confirmed pages of the class, and pages only suggested for it while the admin has not decided."""
    with _open_for_read(settings) as conn:
        pages = _page_rows(conn, settings.venue_id, "AND p.model_key = ?", (model,))
    return [p for p in pages if class_name in p["classes"]]


def _class_names(settings: CatalogSettings, model: str) -> List[str]:
    return [c["name"] for c in list_classes(settings, model)]


def class_examples(settings: CatalogSettings, model: str, class_name: str, count: int = 6,
                   seed: Optional[int] = None) -> Dict[str, Any]:
    names = _class_names(settings, model)
    if class_name not in names:
        return {"available": settings.dataset_dir is not None, "total_images": 0, "examples": [],
                "source_label": EXAMPLE_SOURCE_LABEL}
    picked = _examples.pick_examples(settings.dataset_dir, model, class_name, names.index(class_name),
                                     max(1, min(int(count), 24)), seed)
    base = f"/api/catalog/models/{model}/classes/{class_name}/examples"
    picked["examples"] = [{"url": f"{base}/{e['file']}?box={e['box']}", "source_label": EXAMPLE_SOURCE_LABEL}
                          for e in picked["examples"]]
    picked["source_label"] = EXAMPLE_SOURCE_LABEL
    return picked


def example_image(settings: CatalogSettings, model: str, class_name: str, file_name: str, box: int) -> bytes:
    names = _class_names(settings, model)
    return _examples.crop_example(settings.dataset_dir, model, class_name,
                                  _examples.class_id_of(names, class_name), file_name, box)


# ---------------- Pasted images as evidence for a POS variation ----------------

EVIDENCE_SOURCE_LABEL = "Dokaz iz Kataloga"
EVIDENCE_MAX_BYTES = 10 * 1024 * 1024
EVIDENCE_MIN_SIDE = 40
EVIDENCE_MAX_SIDE = 8000
_EVIDENCE_FORMATS = {"PNG": ("image/png", "png"), "JPEG": ("image/jpeg", "jpg"), "WEBP": ("image/webp", "webp")}
EVIDENCE_STATUS_LABELS = {"pending": "Čeka potvrdu", "confirmed": "Potvrđeno", "rejected": "Odbijeno"}


class UnknownEvidence(Exception):
    pass


def _evidence_url(evidence_id: int) -> str:
    return f"/api/catalog/evidence/{evidence_id}/image"


def _evidence_path(settings: CatalogSettings, sha256: str, media_type: str) -> Path:
    ext = next(e for t, e in _EVIDENCE_FORMATS.values() if t == media_type)
    return settings.media_dir / "evidence" / f"{sha256}.{ext}"


def _check_image(data: bytes) -> Tuple[str, int, int]:
    """Refuse anything that is not a reasonable screenshot. Returns media type and size."""
    from io import BytesIO
    from PIL import Image
    if not data:
        raise CatalogError("Slika je prazna.")
    if len(data) > EVIDENCE_MAX_BYTES:
        raise CatalogError("Slika je prevelika (najviše 10 MB). Iseci manji deo ekrana.")
    try:
        with Image.open(BytesIO(data)) as im:
            fmt = im.format
            im.verify()
        with Image.open(BytesIO(data)) as im:
            width, height = im.size
    except Exception:
        raise CatalogError("Ovo nije slika koju alat može da pročita.")
    if fmt not in _EVIDENCE_FORMATS:
        raise CatalogError("Prihvataju se samo PNG, JPEG i WEBP slike.")
    if min(width, height) < EVIDENCE_MIN_SIDE:
        raise CatalogError("Slika je premala. Napravi veći isečak.")
    if max(width, height) > EVIDENCE_MAX_SIDE:
        raise CatalogError("Slika je prevelika po dimenzijama. Iseci manji deo ekrana.")
    return _EVIDENCE_FORMATS[fmt][0], width, height


def _clean_person(name: Any, role: Any) -> Tuple[str, str]:
    person = " ".join(str(name or "").split())[:60]
    if not person:
        raise CatalogError("Upiši svoje ime pre čuvanja.")
    return person, ("admin" if role == "admin" else "labeler")


def add_evidence(settings: CatalogSettings, identity_id: int, data: bytes, added_by: str, role: str,
                 model: Optional[str] = None, class_name: Optional[str] = None, source: str = "paste") -> Dict[str, Any]:
    """Save a pasted screenshot for one POS variation. It waits for the admin and never goes to RAW."""
    import hashlib
    person, role = _clean_person(added_by, role)
    media_type, width, height = _check_image(data)
    if settings.media_dir is None:
        raise CatalogError("Katalog nema folder za slike.")
    for root in settings.forbidden_roots:
        if _is_within(settings.media_dir.resolve(), root):
            raise CatalogError(f"Catalog images must not be stored inside {root}")
    sha = hashlib.sha256(data).hexdigest()
    conn = _open_for_write(settings)
    venue = settings.venue_id
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT article_id, name FROM pos_identities WHERE venue_id = ? AND identity_id = ?",
                           (venue, identity_id)).fetchone()
        if row is None:
            raise UnknownEvidence(identity_id)
        same = conn.execute("SELECT e.status, i.name FROM evidence e JOIN pos_identities i ON i.identity_id = e.identity_id "
                            "WHERE e.venue_id = ? AND e.sha256 = ?", (venue, sha)).fetchone()
        if same:
            raise CatalogError(f"Ova slika je već sačuvana za artikal {same[1].strip()} "
                               f"({EVIDENCE_STATUS_LABELS[same[0]].lower()}).")
        target = _evidence_path(settings, sha, media_type)
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".part")
            tmp.write_bytes(data)
            os.replace(tmp, target)
        at = _now()
        evidence_id = conn.execute(
            "INSERT INTO evidence (venue_id, identity_id, article_id, model_key, class_name, sha256, media_type, width, height, "
            "size_bytes, source, added_by, added_role, added_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
            (venue, identity_id, row[0], model, class_name, sha, media_type, width, height, len(data), source, person, role,
             at)).lastrowid
        conn.execute("INSERT INTO evidence_events (venue_id, evidence_id, at, actor, role, action) VALUES (?, ?, ?, ?, ?, 'added')",
                     (venue, evidence_id, at, person, role))
        conn.execute("COMMIT")
    except sqlite3.Error as e:
        conn.rollback()
        raise CatalogError(f"Nije sačuvano: {e}")
    except (CatalogError, UnknownEvidence):
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_evidence(settings, evidence_id)


def decide_evidence(settings: CatalogSettings, evidence_id: int, decision: str, decided_by: str,
                    reason: str = "") -> Dict[str, Any]:
    """Admin confirms or rejects a waiting image. A rejected image stays in the history."""
    if decision not in ("confirmed", "rejected"):
        raise CatalogError("Odluka mora biti potvrda ili odbijanje.")
    person, _ = _clean_person(decided_by, "admin")
    note = " ".join(str(reason or "").split())[:300] or None
    conn = _open_for_write(settings)
    venue = settings.venue_id
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM evidence WHERE venue_id = ? AND evidence_id = ?", (venue, evidence_id)).fetchone()
        if row is None:
            raise UnknownEvidence(evidence_id)
        if row[0] != "pending":
            raise CatalogError(f"O ovoj slici je već odlučeno ({EVIDENCE_STATUS_LABELS[row[0]].lower()}).")
        at = _now()
        conn.execute("UPDATE evidence SET status = ?, decided_by = ?, decided_at = ?, reason = ? WHERE evidence_id = ?",
                     (decision, person, at, note, evidence_id))
        conn.execute("INSERT INTO evidence_events (venue_id, evidence_id, at, actor, role, action, note) "
                     "VALUES (?, ?, ?, ?, 'admin', ?, ?)", (venue, evidence_id, at, person, decision, note))
        conn.execute("COMMIT")
    except sqlite3.Error as e:
        conn.rollback()
        raise CatalogError(f"Nije sačuvano: {e}")
    except (CatalogError, UnknownEvidence):
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_evidence(settings, evidence_id)


_EVIDENCE_SELECT = (
    "SELECT e.evidence_id, e.identity_id, e.article_id, i.code, i.name, e.model_key, e.class_name, e.width, e.height, "
    "e.added_by, e.added_role, e.added_at, e.status, e.decided_by, e.decided_at, e.reason "
    "FROM evidence e JOIN pos_identities i ON i.identity_id = e.identity_id WHERE e.venue_id = ? ")


def _evidence_item(conn: sqlite3.Connection, row: Sequence[Any]) -> Dict[str, Any]:
    (evidence_id, identity_id, article_id, code, name, model, class_name, width, height,
     added_by, added_role, added_at, status, decided_by, decided_at, reason) = row
    history = [{"at": at, "actor": actor, "role": role, "action": action, "note": note}
               for at, actor, role, action, note in conn.execute(
                   "SELECT at, actor, role, action, note FROM evidence_events WHERE evidence_id = ? ORDER BY event_id",
                   (evidence_id,))]
    return {"evidence_id": evidence_id, "identity_id": identity_id, "article_id": article_id, "code": code,
            "name": name.strip(), "model": model, "class": class_name, "width": width, "height": height,
            "added_by": added_by, "added_role": added_role, "added_at": added_at, "status": status,
            "status_label": EVIDENCE_STATUS_LABELS[status], "decided_by": decided_by, "decided_at": decided_at,
            "reason": reason, "url": _evidence_url(evidence_id), "source_label": EVIDENCE_SOURCE_LABEL, "history": history}


def get_evidence(settings: CatalogSettings, evidence_id: int) -> Dict[str, Any]:
    with _open_for_read(settings) as conn:
        row = conn.execute(_EVIDENCE_SELECT + "AND e.evidence_id = ?", (settings.venue_id, evidence_id)).fetchone()
        if row is None:
            raise UnknownEvidence(evidence_id)
        return _evidence_item(conn, row)


def list_evidence(settings: CatalogSettings, status: Optional[str] = "pending",
                  identity_id: Optional[int] = None) -> Dict[str, Any]:
    where, args = "", []
    if status:
        where += "AND e.status = ? "
        args.append(status)
    if identity_id is not None:
        where += "AND e.identity_id = ? "
        args.append(identity_id)
    with _open_for_read(settings) as conn:
        rows = conn.execute(_EVIDENCE_SELECT + where + "ORDER BY e.evidence_id DESC", (settings.venue_id, *args)).fetchall()
        items = [_evidence_item(conn, r) for r in rows]
        counts = dict(conn.execute("SELECT status, COUNT(*) FROM evidence WHERE venue_id = ? GROUP BY status",
                                   (settings.venue_id,)).fetchall())
    return {"items": items, "counts": {k: counts.get(k, 0) for k in EVIDENCE_STATUS_LABELS}}


def evidence_image(settings: CatalogSettings, evidence_id: int) -> Tuple[Path, str]:
    with _open_for_read(settings) as conn:
        row = conn.execute("SELECT sha256, media_type FROM evidence WHERE venue_id = ? AND evidence_id = ?",
                           (settings.venue_id, evidence_id)).fetchone()
    if row is None or settings.media_dir is None:
        raise UnknownEvidence(evidence_id)
    path = _evidence_path(settings, row[0], row[1])
    if not path.is_file():
        raise UnknownEvidence(evidence_id)
    return path, row[1]


def list_without_image(settings: CatalogSettings, model: Optional[str] = None) -> Dict[str, Any]:
    """Tracked, sold POS variations with a class but no confirmed image, best sellers first."""
    with _open_for_read(settings) as conn:
        current = _current(conn, settings.venue_id)
        evidence = _evidence_by_identity(conn, settings.venue_id)
    items = [v for v in current.values() if v.state == "tracked" and v.code is not None and v.links
             and (model is None or any(m == model.lower() for m, _ in v.links))
             and (evidence.get(v.identity_id) or {}).get("confirmed") is None]
    return {"model": model, "items": [_variation(v, evidence) for v in _sorted(items)]}


# ---------------- Manual decisions, proposals, the article list and the article card ----------------

class UnknownArticle(Exception):
    pass


class NotAllowed(Exception):
    """The chosen role may not make this decision. The message is safe to show."""


MODE_LABELS = {"fallback": "prvo glavna, pa rezerva", "any": "bilo koja od navedenih"}


def class_choices(settings: CatalogSettings) -> Dict[str, List[str]]:
    """Class names a decision may point to, per model: names the live model outputs, plus map classes in use."""
    venue = settings.venue_id
    with _open_for_read(settings) as conn:
        result: Dict[str, set] = {}
        for model in [r[0] for r in conn.execute("SELECT model_key FROM models WHERE venue_id = ?", (venue,))]:
            trained = {r[0] for r in conn.execute(
                "SELECT t.trained_name FROM class_training t WHERE t.venue_id = ? AND t.model_key = ? AND t.state = 'trained' "
                "AND t.trained_name IS NOT NULL", (venue, model))}
            drawn = {r[0] for r in conn.execute(
                "SELECT c.name FROM classes c LEFT JOIN class_training t ON t.venue_id = c.venue_id AND t.model_key = c.model_key "
                "AND t.class_id = c.class_id WHERE c.venue_id = ? AND c.model_key = ? AND t.state IS NULL", (venue, model))}
            result[model.lower()] = trained | drawn
        for v in _current(conn, venue).values():
            for m, c in v.links:
                result.setdefault(m, set()).add(c)
    return {m: sorted(names) for m, names in sorted(result.items())}


def _valid_date(value: Optional[str]) -> str:
    if not value:
        return _today()
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise CatalogError("Datum mora biti u obliku GGGG-MM-DD.")


def _clean_links(settings: CatalogSettings, links: Any) -> List[Dict[str, str]]:
    if not isinstance(links, list):
        raise CatalogError("Klase moraju biti spisak.")
    choices = class_choices(settings)
    result, seen = [], set()
    for link in links:
        model = str((link or {}).get("model") or "").strip().lower()
        cls = str((link or {}).get("class") or "").strip()
        if model not in choices or cls not in choices[model]:
            raise CatalogError(f"Nepoznata klasa: {model}.{cls}")
        if (model, cls) in seen:
            raise CatalogError(f"Klasa je navedena dvaput: {model}.{cls}")
        seen.add((model, cls))
        result.append({"model": model, "class": cls})
    return result


def _clean_change(settings: CatalogSettings, conn: sqlite3.Connection, venue: str, kind: str, payload: Mapping[str, Any],
                  article_id: Optional[int], identity_id: Optional[int]) -> Dict[str, Any]:
    if kind == "classes":
        mode = payload.get("mode") or "fallback"
        if mode not in _decisions.MODES:
            raise CatalogError("Način mora biti fallback ili any.")
        provisional = " ".join(str(payload.get("provisional") or "").split()) or None
        return {"links": _clean_links(settings, payload.get("links")), "mode": mode, "provisional": provisional}
    if kind == "provisional":
        return {"provisional": " ".join(str(payload.get("provisional") or "").split()) or None}
    if kind == "merge":
        into = payload.get("into")
        if not isinstance(into, int) or conn.execute("SELECT 1 FROM articles WHERE venue_id = ? AND article_id = ?",
                                                     (venue, into)).fetchone() is None:
            raise CatalogError("Izaberi postojeći artikal sa kojim se spaja.")
        merges = _decisions.merge_map(_decisions.in_force(_load_decisions(conn, venue), "9999-12-31"))
        if _decisions.resolve(into, merges) == _decisions.resolve(article_id, merges):
            raise CatalogError("Artikal je već deo tog artikla.")
        return {"into": into}
    if kind == "class_target":
        return _clean_target(conn, venue, payload)
    if kind == "ignore":
        why = payload.get("why")
        if why in (None, ""):
            return {}
        if why not in _decisions.IGNORE_WHY:
            raise CatalogError("Nepoznat razlog za X.")
        return {"why": why}
    raise CatalogError(f"Nepoznata vrsta odluke: {kind}")


def _clean_target(conn: sqlite3.Connection, venue: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Admin picks the class for a new article: a new class appended to a model, or an existing one (D9)."""
    models = {m.lower(): m for m, in conn.execute("SELECT model_key FROM models WHERE venue_id = ?", (venue,))}
    model_key = models.get(str(payload.get("model") or "").strip().lower())
    if model_key is None:
        raise CatalogError("Izaberi model.")
    rows = conn.execute("SELECT c.class_id, c.name, t.state, t.trained_name FROM classes c LEFT JOIN class_training t "
                        "ON t.venue_id = c.venue_id AND t.model_key = c.model_key AND t.class_id = c.class_id "
                        "WHERE c.venue_id = ? AND c.model_key = ? ORDER BY c.class_id", (venue, model_key)).fetchall()
    if payload.get("open_new"):
        try:
            name = _class_lists.clean_name(payload.get("class"))
        except _class_lists.ClassListError as e:
            raise CatalogError(str(e))
        taken = {str(n).upper() for _, n, _, _ in rows} | {str(t).upper() for _, _, _, t in rows if t}
        if name.upper() in taken:
            raise CatalogError(f"Klasa {name} već postoji u modelu {model_key}. Izaberi je kao postojeću klasu.")
        return {"model": model_key.lower(), "model_key": model_key, "class": name,
                "class_id": (rows[-1][0] + 1) if rows else 0, "opened": True}
    name = str(payload.get("class") or "").strip()
    hit = next((r for r in rows if r[1] == name), None)
    if hit is None:
        raise CatalogError("Izaberi postojeću klasu ovog modela.")
    if hit[2] == "excluded":
        raise CatalogError(f"Klasa {name} je izbačena iz treninga, pa je model nikad neće naučiti. Izaberi drugu ili otvori novu.")
    return {"model": model_key.lower(), "model_key": model_key, "class": name, "class_id": hit[0], "opened": False}


def _open_class(settings: CatalogSettings, conn: sqlite3.Connection, venue: str, change: Mapping[str, Any]) -> Tuple[Path, Path]:
    """Append the new class to the Catalog and to the end of the tool's class list. Returns (file, backup of the old file)."""
    model_key, name, class_id = change["model_key"], change["class"], change["class_id"]
    conn.execute("INSERT INTO classes (venue_id, model_key, class_id, name, imported_at) VALUES (?, ?, ?, ?, ?)",
                 (venue, model_key, class_id, name, _now()))
    expected = [n for n, in conn.execute("SELECT name FROM classes WHERE venue_id = ? AND model_key = ? AND class_id < ? "
                                         "ORDER BY class_id", (venue, model_key, class_id))]
    try:
        path = _class_lists.find_file(settings.source_models_dir, model_key)
        backup = _class_lists.append_class(path, name, expected, settings.db_path.parent / "class_list_backups")
    except _class_lists.ClassListError as e:
        raise CatalogError(str(e))
    return path, backup


def record_decision(settings: CatalogSettings, kind: str, actor: Any, role: Any, *, article_id: Optional[int] = None,
                    identity_id: Optional[int] = None, payload: Optional[Mapping[str, Any]] = None, reason: Any = "",
                    evidence: Any = "", valid_from: Optional[str] = None, target_id: Optional[int] = None) -> Dict[str, Any]:
    """Append one decision. Nothing already written ever changes; production files are never touched."""
    person, role = _clean_person(actor, role)
    if kind not in _decisions.KINDS:
        raise CatalogError(f"Nepoznata vrsta odluke: {kind}")
    if role != "admin" and kind not in _decisions.LABELER_KINDS:
        raise NotAllowed("Ovu odluku donosi admin. Kao labeler možeš da označiš X ili da pošalješ predlog.")
    why = " ".join(str(reason or "").split())[:500] or None
    if kind not in ("ignore", "provisional") and not why:
        raise CatalogError("Upiši zašto.")
    payload = dict(payload or {})
    day = _valid_date(valid_from) if role == "admin" else _today()
    venue = settings.venue_id
    conn = _open_for_write(settings)
    written: Optional[Tuple[Path, Path]] = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        if identity_id is not None:
            row = conn.execute("SELECT article_id FROM pos_identities WHERE venue_id = ? AND identity_id = ?",
                               (venue, identity_id)).fetchone()
            if row is None:
                raise UnknownArticle(identity_id)
            article_id = article_id or row[0]
        if article_id is not None and conn.execute("SELECT 1 FROM articles WHERE venue_id = ? AND article_id = ?",
                                                   (venue, article_id)).fetchone() is None:
            raise UnknownArticle(article_id)
        decisions = {d.decision_id: d for d in _load_decisions(conn, venue)}
        if kind in ("classes", "provisional", "class_target") and identity_id is None:
            raise CatalogError("Izaberi POS varijaciju (ime i šifru).")
        if kind in ("ignore", "merge", "propose") and article_id is None:
            raise CatalogError("Izaberi artikal.")
        if kind in ("revoke", "reject"):
            target = decisions.get(target_id)
            if target is None:
                raise CatalogError("Ta odluka ne postoji.")
            if kind == "revoke" and (target.kind not in _decisions.EFFECT_KINDS or target_id in _decisions.revoked_ids(decisions.values())):
                raise CatalogError("Ova odluka ne može da se poništi (već je poništena ili nije promena).")
            if kind == "reject" and (target.kind != "propose" or _proposal_status(target_id, decisions.values()) != "pending"):
                raise CatalogError("Ovaj predlog više ne čeka odluku.")
            article_id, identity_id = target.article_id, target.identity_id
            payload = {}
        elif kind == "propose":
            change_kind = str(payload.get("kind") or "")
            if change_kind not in ("classes", "provisional", "merge", "ignore", "new_class"):
                raise CatalogError("Predlog može biti promena klase, zvezdica, spajanje, X ili nova klasa.")
            if change_kind in ("classes", "provisional", "new_class") and identity_id is None:
                raise CatalogError("Izaberi POS varijaciju (ime i šifru).")
            if change_kind == "new_class":  # the admin opens the class (stage 3); until then it only waits
                change = {"note": " ".join(str((payload.get("change") or {}).get("note") or "").split())[:300] or None}
            else:
                change = _clean_change(settings, conn, venue, change_kind, payload.get("change") or {}, article_id, identity_id)
            payload = {"kind": change_kind, "change": change}
        else:
            from_proposal = payload.get("from_proposal")
            payload = _clean_change(settings, conn, venue, kind, payload, article_id, identity_id)
            if from_proposal is not None:
                proposal = decisions.get(from_proposal)
                if proposal is None or proposal.kind != "propose" or _proposal_status(from_proposal, decisions.values()) != "pending":
                    raise CatalogError("Ovaj predlog više ne čeka odluku.")
                payload["from_proposal"] = from_proposal
        decision_id = conn.execute(
            "INSERT INTO decisions (venue_id, kind, article_id, identity_id, target_id, payload, reason, evidence, actor, role, "
            "created_at, valid_from) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (venue, kind, article_id, identity_id, target_id if kind in ("revoke", "reject") else None,
             json.dumps(payload, ensure_ascii=False, sort_keys=True), why, " ".join(str(evidence or "").split())[:300] or None,
             person, role, _now(), day)).lastrowid
        if kind == "class_target" and payload.get("opened"):
            written = _open_class(settings, conn, venue, payload)  # the file is written last, right before COMMIT
        conn.execute("COMMIT")
    except sqlite3.Error as e:
        conn.rollback()
        if written:
            _class_lists.restore(*written)
        raise CatalogError(f"Nije sačuvano: {e}")
    except (CatalogError, UnknownArticle, NotAllowed):
        conn.rollback()
        if written:
            _class_lists.restore(*written)
        raise
    finally:
        conn.close()
    return get_decision(settings, decision_id)


def _proposal_status(proposal_id: int, decisions: Iterable[_decisions.Decision]) -> str:
    for d in decisions:
        if d.kind == "reject" and d.target == proposal_id:
            return "rejected"
        if d.kind in _decisions.EFFECT_KINDS and d.payload.get("from_proposal") == proposal_id:
            return "applied"
    return "pending"


_DECISION_SELECT = ("SELECT d.decision_id, d.kind, d.article_id, d.identity_id, d.target_id, d.payload, d.reason, d.evidence, "
                    "d.actor, d.role, d.created_at, d.valid_from, i.code, i.name, a.display_name FROM decisions d "
                    "LEFT JOIN pos_identities i ON i.identity_id = d.identity_id "
                    "LEFT JOIN articles a ON a.article_id = d.article_id WHERE d.venue_id = ? ")
PROPOSAL_STATUS_LABELS = {"pending": "Čeka odluku", "applied": "Primenjen", "rejected": "Odbijen"}


def _fmt_day(value: str) -> str:
    y, m, d = value[:10].split("-")
    return f"{d}.{m}.{y}."


def _describe(kind: str, payload: Mapping[str, Any], article_names: Mapping[int, str]) -> str:
    if kind == "class_target":
        where = f"{payload['model']}.{payload['class']}"
        head = (f"nova klasa {where} (broj {payload['class_id']} u modelu {payload['model_key']})" if payload.get("opened")
                else f"klasa {where}")
        return head + ", mapira se kad je živi model nauči"
    if kind == "new_class":
        return "treba nova klasa" + (f" ({payload['note']})" if payload.get("note") else "")
    if kind == "classes":
        links = payload.get("links") or []
        if not links:
            text = "bez klase (mapiranje nedostaje)"
        else:
            text = ", ".join(f"{l['model']}.{l['class']}" for l in links)
            if len(links) > 1:
                text += f" ({MODE_LABELS[payload.get('mode') or 'fallback']})"
        if payload.get("provisional"):
            text += f", privremeno (*): {payload['provisional']}"
        return text
    if kind == "provisional":
        return f"privremeno (*): {payload['provisional']}" if payload.get("provisional") else "skinuta zvezdica"
    if kind == "ignore":
        why = _decisions.IGNORE_WHY.get(payload.get("why") or "")
        return f"ne pratimo ({why})" if why else "ne pratimo"
    if kind == "merge":
        return f"spojen sa artiklom {article_names.get(payload.get('into'), payload.get('into'))}"
    if kind == "propose":
        return "predlog: " + _describe(payload.get("kind", ""), payload.get("change") or {}, article_names)
    return ""


def _miro_text(kind: str, subject: str, code: Optional[str], payload: Mapping[str, Any], valid_from: str, reason: Optional[str],
               article_names: Mapping[int, str]) -> Optional[str]:
    """Exact sentence for the owner to copy into Miro and Word until they are locked."""
    if kind not in ("classes", "ignore", "merge", "class_target"):
        return None
    who = f"POS artikal {subject}" + (f" (šifra {code})" if code else "")
    if kind == "class_target":
        text = f"{who} od {_fmt_day(valid_from)} ide u {payload['model']}.{payload['class']} čim je živi model nauči."
        if payload.get("opened"):
            text += f" Nova klasa {payload['class']} dodata je na kraj liste modela {payload['model_key']} (broj {payload['class_id']})."
        return text + (f" Razlog: {reason.rstrip('.')}." if reason else "")
    body = {"classes": f"od {_fmt_day(valid_from)} ide u {_describe(kind, payload, article_names)}",
            "ignore": (f"od {_fmt_day(valid_from)} je izbačen iz asortimana i ne prati se" if payload.get("why") == "assortment"
                       else f"od {_fmt_day(valid_from)} se ne prati"),
            "merge": f"od {_fmt_day(valid_from)} je isti artikal kao {article_names.get(payload.get('into'), payload.get('into'))}"}[kind]
    return f"{who} {body}." + (f" Razlog: {reason.rstrip('.')}." if reason else "")


def _decision_items(conn: sqlite3.Connection, venue: str, where: str = "", args: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    all_decisions = _load_decisions(conn, venue)
    revoked = _decisions.revoked_ids(all_decisions)
    locked = _is_locked(conn, venue)  # after the lock Miro and Word are an archive: no more sentences to copy
    revokers = {d.target: d.decision_id for d in all_decisions if d.kind == "revoke"}
    names = dict(conn.execute("SELECT article_id, display_name FROM articles WHERE venue_id = ?", (venue,)).fetchall())
    today = _today()
    items = []
    for row in conn.execute(_DECISION_SELECT + where + "ORDER BY d.decision_id DESC", (venue, *args)):
        (decision_id, kind, article_id, identity_id, target_id, payload, reason, evidence, actor, role, created_at, valid_from,
         code, name, article_name) = row
        payload = json.loads(payload)
        subject = (name or article_name or "").strip()
        item = {"decision_id": decision_id, "kind": kind, "kind_label": _decisions.KIND_LABELS[kind], "article_id": article_id,
                "identity_id": identity_id, "target_id": target_id, "subject": subject, "code": code, "payload": payload,
                "text": _describe(kind, payload, names), "reason": reason, "evidence": evidence, "actor": actor, "role": role,
                "created_at": created_at, "valid_from": valid_from, "revoked": decision_id in revoked,
                "revoked_by": revokers.get(decision_id), "future": valid_from > today,
                "miro_text": None if decision_id in revoked or locked else _miro_text(kind, subject, code, payload, valid_from, reason, names)}
        if kind == "propose":
            item["status"] = _proposal_status(decision_id, all_decisions)
            item["status_label"] = PROPOSAL_STATUS_LABELS[item["status"]]
        if kind in ("revoke", "reject"):
            item["text"] = f"odluka #{target_id}"
        items.append(item)
    return items


def get_decision(settings: CatalogSettings, decision_id: int) -> Dict[str, Any]:
    with _open_for_read(settings) as conn:
        items = _decision_items(conn, settings.venue_id, "AND d.decision_id = ? ", (decision_id,))
    if not items:
        raise UnknownArticle(decision_id)
    return items[0]


def list_decisions(settings: CatalogSettings, kind: Optional[str] = None) -> Dict[str, Any]:
    """All decisions, newest first; with kind 'propose' only proposals, with 'ignore' the X list."""
    where, args = ("AND d.kind = ? ", (kind,)) if kind else ("", ())
    with _open_for_read(settings) as conn:
        items = _decision_items(conn, settings.venue_id, where, args)
    return {"items": items}


FLOW_LABELS = {"waiting_class": "Čeka klasu", "labeling": "U označavanju", "training": "Čeka trening", "tracked": "Praćeno"}


def _raw_images(settings: CatalogSettings, model: str, class_name: str) -> Optional[int]:
    """Labeled images of a class in RAW (only counted, never changed). None when RAW is not reachable."""
    root = settings.dataset_dir
    if root is None or not root.is_dir():
        return None
    folder = root / model.lower() / "images" / class_name
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _examples._IMAGE_SUFFIXES)


def _flows(settings: CatalogSettings, current: Mapping[int, _decisions.Variation],
           decisions: Sequence[_decisions.Decision]) -> Dict[int, Dict[str, Any]]:
    """Where each article on the new class path is (D9): Čeka klasu -> U označavanju -> Čeka trening -> Praćeno."""
    asked: Dict[int, _decisions.Decision] = {}
    for d in decisions:
        if (d.kind == "propose" and d.payload.get("kind") == "new_class" and d.identity_id
                and _proposal_status(d.decision_id, decisions) == "pending"):
            asked[d.identity_id] = d
    images: Dict[Tuple[str, str], Optional[int]] = {}
    out: Dict[int, Dict[str, Any]] = {}
    for v in current.values():
        if v.state == "ignored":
            continue
        if v.target:
            model, cls = v.target
            if v.links:
                stage = "tracked"
            else:
                if v.target not in images:
                    images[v.target] = _raw_images(settings, model, cls)
                stage = "training" if images[v.target] else "labeling"
            out[v.identity_id] = {"stage": stage, "label": FLOW_LABELS[stage], "model": model, "class": cls,
                                  "opened": v.target_opened, "raw_images": images.get(v.target)}
        elif v.identity_id in asked:
            d = asked[v.identity_id]
            out[v.identity_id] = {"stage": "waiting_class", "label": FLOW_LABELS["waiting_class"], "model": None,
                                  "class": None, "opened": False, "raw_images": None, "proposal_id": d.decision_id,
                                  "note": (d.payload.get("change") or {}).get("note")}
    return out


def list_articles(settings: CatalogSettings, query: Optional[str] = None, state: Optional[str] = None) -> Dict[str, Any]:
    """Every POS variation of the venue with its mapping, for someone starting from the till."""
    q = _assortment.normalize_name(query or "")
    with _open_for_read(settings) as conn:
        current = _current(conn, settings.venue_id)
        evidence = _evidence_by_identity(conn, settings.venue_id)
        flows = _flows(settings, current, _load_decisions(conn, settings.venue_id))
    counts = {s: 0 for s in STATE_LABELS}
    counts["missing"] = 0
    counts["new_class"] = len(flows)
    items = []
    for v in _sorted(current.values()):
        counts[v.state] += 1
        missing = v.state != "ignored" and not v.links and not v.target
        counts["missing"] += missing
        if state == "missing" and not missing:
            continue
        if state == "new_class" and v.identity_id not in flows:
            continue
        if state and state not in ("missing", "new_class") and v.state != state:
            continue
        if q and q not in _assortment.normalize_name(v.name) and q not in (v.code or ""):
            continue
        items.append(_variation(v, evidence, flows.get(v.identity_id)))
    return {"items": items, "counts": counts, "state_labels": STATE_LABELS, "flow_labels": FLOW_LABELS}


def article_card(settings: CatalogSettings, article_id: int, as_of: Optional[str] = None) -> Dict[str, Any]:
    venue = settings.venue_id
    with _open_for_read(settings) as conn:
        current = _current(conn, venue, as_of)
        decisions = _load_decisions(conn, venue)
        merges = _decisions.merge_map(_decisions.in_force(decisions, as_of or _today()))
        row = conn.execute("SELECT display_name, created_at FROM articles WHERE venue_id = ? AND article_id = ?",
                           (venue, article_id)).fetchone()
        if row is None:
            raise UnknownArticle(article_id)
        home = _decisions.resolve(article_id, merges)
        members = sorted({v.source_article_id for v in current.values() if v.article_id == home} | {home})
        variations = _sorted(v for v in current.values() if v.article_id == home)
        evidence = _evidence_by_identity(conn, venue)
        ids = [v.identity_id for v in variations]
        details = {i: (fs, ls) for i, fs, ls in conn.execute(
            f"SELECT identity_id, first_seen, last_seen FROM pos_identities WHERE identity_id IN ({','.join('?' * len(ids)) or 'NULL'})", ids)}
        marks = ",".join("?" * len(members))
        history = _decision_items(conn, venue, f"AND (d.article_id IN ({marks}) OR d.identity_id IN "
                                  f"({','.join('?' * len(ids)) or 'NULL'})) ", (*members, *ids))
        images = []
        for i in ids:
            images.extend(list_evidence_conn(conn, venue, i))
        with_clips = {i for i, in conn.execute(
            f"SELECT DISTINCT identity_id FROM clip_events WHERE venue_id = ? AND action = 'downloaded' "
            f"AND identity_id IN ({','.join('?' * len(ids)) or 'NULL'})", (venue, *ids))}
        home_name = conn.execute("SELECT display_name FROM articles WHERE article_id = ?", (home,)).fetchone()[0]
    flows = _flows(settings, {v.identity_id: v for v in variations}, decisions)
    items = []
    for v in variations:
        item = _variation(v, evidence, flows.get(v.identity_id))
        item["first_seen"], item["last_seen"] = details.get(v.identity_id, (None, None))
        item["mode_label"] = MODE_LABELS.get(v.mode or "", "") if len(v.links) > 1 else ""
        item["has_clips"] = v.identity_id in with_clips and settings.clips is not None
        items.append(item)
    return {"article_id": article_id, "name": row[0].strip(), "home_article_id": home, "home_name": home_name.strip(),
            "merged_into": home if home != article_id else None, "member_articles": members, "variations": items,
            "history": history, "images": images, "as_of": as_of or _today()}


def list_evidence_conn(conn: sqlite3.Connection, venue: str, identity_id: int) -> List[Dict[str, Any]]:
    rows = conn.execute(_EVIDENCE_SELECT + "AND e.identity_id = ? ORDER BY e.evidence_id DESC", (venue, identity_id)).fetchall()
    return [_evidence_item(conn, r) for r in rows]


# ---------------- Releases ----------------

RELEASE_LIVE_LABEL = "važeće izdanje"
PRODUCTION_LABEL = "današnji fajlovi"


class UnknownRelease(Exception):
    pass


class ReleaseBlocked(Exception):
    """A check failed; nothing was published. `checks` says which."""

    def __init__(self, message: str, checks: List[Dict[str, Any]]):
        super().__init__(message)
        self.checks = checks


def _production_paths(settings: CatalogSettings) -> _releases.ProductionPaths:
    rules = settings.rule_sources
    return _releases.ProductionPaths(settings.pipeline_map_path, settings.blacklist_path, settings.source_models_dir,
                                     rules.training_config_dir if rules else None, rules.documented_dir if rules else None)


def _live_classes(settings: CatalogSettings, models: Iterable[str]) -> Dict[str, Optional[set]]:
    root = settings.model_cards.models_dir if settings.model_cards else (
        settings.rule_sources.live_models_dir if settings.rule_sources else None)
    out: Dict[str, Optional[set]] = {}
    for model in {m.lower() for m in models}:
        data = _rules._read_yaml(root / model / "active_classes.yaml") if root else None
        names = (data or {}).get("names")
        out[model] = {str(n) for n in names if str(n) != "background"} if isinstance(names, list) else None
    return out


def _release_sources(conn: sqlite3.Connection, venue: str) -> _releases.ReleaseSources:
    entries = []
    rows = conn.execute("SELECT entry_id, status FROM map_entries WHERE venue_id = ? ORDER BY entry_id", (venue,)).fetchall()
    for entry_id, status in rows:
        names = [n for n, in conn.execute("SELECT name FROM map_names WHERE entry_id = ? ORDER BY position", (entry_id,))]
        links = [(m, c) for m, c in conn.execute(
            "SELECT model_key, class_name FROM class_links WHERE entry_id = ? ORDER BY position", (entry_id,))]
        entries.append(_releases.MapSource(status, names, links))
    if rows and not any(e.names for e in entries):
        raise CatalogError("Pokreni uvoz još jednom: ova verzija Kataloga čuva imena iz mape tačno kako su napisana.")
    blacklist = [l for l, in conn.execute("SELECT line FROM source_lines WHERE venue_id = ? AND source = 'blacklist' "
                                          "ORDER BY position", (venue,))]
    current = _current(conn, venue)
    decisions = _load_decisions(conn, venue)
    active = _decisions.in_force(decisions, _today())
    merges = _decisions.merge_map(active)
    info = {i: {"id": i, "actor": a, "reason": r, "created_at": c} for i, a, r, c in conn.execute(
        "SELECT decision_id, actor, reason, created_at FROM decisions WHERE venue_id = ?", (venue,))}
    by_identity: Dict[int, List[Dict[str, Any]]] = {}
    by_article: Dict[int, List[Dict[str, Any]]] = {}
    for d in active:
        if d.kind in ("classes", "class_target") and d.identity_id is not None:
            by_identity.setdefault(d.identity_id, []).append(info[d.decision_id])
        elif d.kind == "ignore" and d.article_id is not None:
            by_article.setdefault(_decisions.resolve(d.article_id, merges), []).append(info[d.decision_id])
    variations = [_releases.VariationState(v.identity_id, v.article_id, v.code, v.name, v.state, list(v.links),
                                           by_identity.get(v.identity_id, []) + by_article.get(v.article_id, []),
                                           v.ignore_why)
                  for v in current.values()]
    classes: Dict[str, List[str]] = {}
    for model, name in conn.execute("SELECT model_key, name FROM classes WHERE venue_id = ? ORDER BY model_key, class_id", (venue,)):
        classes.setdefault(model, []).append(name)
    training = {m: json.loads(c) for m, c in conn.execute("SELECT model_key, config FROM training_classes WHERE venue_id = ?", (venue,))}
    extra = [(data["name"], [tuple(link) for link in data.get("classes") or []],
              {"id": f"nalaz {finding_id}", "actor": "kasa", "created_at": data["created_at"],
               "reason": f"preimenovano iz {data.get('old_name')}" + (", na proveri" if state == "open" else ", potvrđeno")})
             for finding_id, data, state in _till_findings(conn, venue) if data["kind"] == "rename"]
    return _releases.ReleaseSources(entries, blacklist, variations, classes, training, extra)


def _release_rows(conn: sqlite3.Connection, venue: str) -> List[Dict[str, Any]]:
    rows = conn.execute("SELECT number, kind, based_on, actor, reason, created_at, decisions_upto, files, checksums, summary "
                        "FROM releases WHERE venue_id = ? ORDER BY number", (venue,)).fetchall()
    return [{"number": n, "kind": k, "based_on": b, "actor": a, "reason": r, "created_at": c, "decisions_upto": u,
             "files": json.loads(f), "checksums": json.loads(cs), "summary": json.loads(sm)}
            for n, k, b, a, r, c, u, f, cs, sm in rows]


def _prepare(settings: CatalogSettings, conn: sqlite3.Connection) -> Dict[str, Any]:
    venue = settings.venue_id
    src = _release_sources(conn, venue)
    generated = _releases.generate(src)
    new = _releases.parse_files(generated.files)
    production = _releases.read_production(_production_paths(settings), src.classes)
    releases = _release_rows(conn, venue)
    live = releases[-1] if releases else None
    baseline = _releases.parse_files(live["files"]) if live else production
    checks = _releases.run_checks(new, baseline, generated, src, _live_classes(settings, src.classes))
    same = live is not None and live["files"] == generated.files
    return {"generated": generated, "new": new, "production": production, "live": live, "baseline": baseline,
            "checks": checks, "same_as_live": same,
            "decisions_upto": conn.execute("SELECT IFNULL(MAX(decision_id), 0) FROM decisions WHERE venue_id = ?",
                                           (venue,)).fetchone()[0]}


def _public(row: Dict[str, Any], current: Optional[int], intact: Optional[bool] = None) -> Dict[str, Any]:
    return {k: row[k] for k in ("number", "kind", "based_on", "actor", "reason", "created_at")} | {
        "live": row["number"] == current, "changes": row["summary"].get("changes", []),
        "files": [{"path": p, "sha256": h} for p, h in sorted(row["checksums"].items())], "intact": intact}


def next_release(settings: CatalogSettings) -> Dict[str, Any]:
    """The release that would be published now: changes, checks and parity with today's files. Changes nothing."""
    with _open_for_read(settings) as conn:
        prep = _prepare(settings, conn)
    live = prep["live"]
    changes = _releases.describe_changes(prep["baseline"], prep["new"])
    ok = all(c["ok"] for c in prep["checks"])
    return {"baseline_label": f"{RELEASE_LIVE_LABEL} {live['number']}" if live else PRODUCTION_LABEL,
            "live_number": live["number"] if live else None, "changes": changes, "checks": prep["checks"],
            "same_as_live": prep["same_as_live"], "can_publish": ok and not prep["same_as_live"],
            "parity": _releases.parity(prep["production"], prep["generated"], prep["new"]),
            "decision_changes": prep["generated"].changes,
            "files": [{"path": p, "sha256": _releases.checksum(t)} for p, t in sorted(prep["generated"].files.items())]}


def _verify_on_disk(settings: CatalogSettings, row: Dict[str, Any]) -> Optional[bool]:
    if settings.release_dir is None:
        return None
    folder = settings.release_dir / f"{row['number']:04d}"
    if not folder.is_dir():
        return False
    try:
        return all((folder / p).is_file() and _releases.checksum((folder / p).read_text(encoding="utf-8")) == h
                   for p, h in row["checksums"].items())
    except OSError:
        return False


def list_releases(settings: CatalogSettings) -> Dict[str, Any]:
    with _open_for_read(settings) as conn:
        rows = _release_rows(conn, settings.venue_id)
    current = rows[-1]["number"] if rows else None
    return {"live_number": current, "release_dir": str(settings.release_dir) if settings.release_dir else None,
            "releases": [_public(r, current, _verify_on_disk(settings, r)) for r in reversed(rows)]}


def release_file(settings: CatalogSettings, number: int, path: str) -> str:
    with _open_for_read(settings) as conn:
        rows = [r for r in _release_rows(conn, settings.venue_id) if r["number"] == number]
    if not rows or path not in rows[0]["files"]:
        raise UnknownRelease(f"{number}/{path}")
    return rows[0]["files"][path]


def _write_release_folder(settings: CatalogSettings, number: int, files: Mapping[str, str], meta: str) -> Path:
    root = settings.release_dir
    if root is None:
        raise CatalogError("Folder za izdanja nije podešen.")
    for forbidden in settings.forbidden_roots:
        if _is_within(root.resolve(), forbidden):
            raise CatalogError(f"Izdanja ne smeju u {forbidden}")
    production = [p for p in (settings.pipeline_map_path, settings.blacklist_path) if p]
    if any(p.resolve().parent == root.resolve() for p in production):
        raise CatalogError("Folder za izdanja ne sme biti folder produkcionih fajlova.")
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{number:04d}"
    if target.exists():
        raise CatalogError(f"Izdanje {number} već postoji na disku i ne menja se.")
    tmp = root / f".{number:04d}.tmp{os.getpid()}"
    for path, text in list(files.items()) + [("RELEASE.json", meta)]:
        out = tmp / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        out.chmod(0o444)
    for folder in sorted({p for p in tmp.rglob("*") if p.is_dir()} | {tmp}, key=lambda p: -len(p.parts)):
        folder.chmod(0o555)
    os.rename(tmp, target)
    pointer = root / ".CURRENT.tmp"
    pointer.write_text(f"{number}\n", encoding="utf-8")
    os.replace(pointer, root / "CURRENT")
    return target


def _remove_folder(path: Path) -> None:
    import shutil
    for p in [path] + list(path.rglob("*")):
        try:
            p.chmod(0o755 if p.is_dir() else 0o644)
        except OSError:
            pass
    shutil.rmtree(path, ignore_errors=True)


def _store_release(settings: CatalogSettings, conn: sqlite3.Connection, kind: str, based_on: Optional[int], person: str,
                   reason: str, files: Mapping[str, str], summary: Dict[str, Any], decisions_upto: int) -> Dict[str, Any]:
    venue = settings.venue_id
    number = conn.execute("SELECT IFNULL(MAX(number), 0) + 1 FROM releases WHERE venue_id = ?", (venue,)).fetchone()[0]
    at = _now()
    sums = {p: _releases.checksum(t) for p, t in files.items()}
    conn.execute("INSERT INTO releases (venue_id, number, kind, based_on, actor, role, reason, created_at, decisions_upto, "
                 "files, checksums, summary) VALUES (?, ?, ?, ?, ?, 'admin', ?, ?, ?, ?, ?, ?)",
                 (venue, number, kind, based_on, person, reason, at, decisions_upto,
                  json.dumps(dict(files), ensure_ascii=False, sort_keys=True), json.dumps(sums, sort_keys=True),
                  json.dumps(summary, ensure_ascii=False)))
    meta = _releases.release_json(number, kind, at, person, reason, based_on, sums, summary.get("decision_changes") or ())
    folder = _write_release_folder(settings, number, files, meta)
    try:
        conn.execute("COMMIT")
    except sqlite3.Error:
        _remove_folder(folder)
        raise
    return {"number": number, "kind": kind, "based_on": based_on, "created_at": at, "folder": str(folder)}


def _admin(actor: Any, role: Any, reason: Any) -> Tuple[str, str]:
    person, role = _clean_person(actor, role)
    if role != "admin":
        raise NotAllowed("Izdanje objavljuje samo admin.")
    why = " ".join(str(reason or "").split())[:500]
    if not why:
        raise CatalogError("Upiši kratku napomenu: šta ovo izdanje donosi.")
    return person, why


def publish_release(settings: CatalogSettings, actor: Any, role: Any, reason: Any) -> Dict[str, Any]:
    """Publish the next release into the release folder. The pipeline does not read it yet (shadow)."""
    person, why = _admin(actor, role, reason)
    conn = _open_for_write(settings)
    try:
        conn.execute("BEGIN IMMEDIATE")
        prep = _prepare(settings, conn)
        failed = [c for c in prep["checks"] if not c["ok"]]
        if failed:
            raise ReleaseBlocked("Objava nije moguća: " + "; ".join(c["label"].lower() for c in failed) + ".", prep["checks"])
        if prep["same_as_live"]:
            raise CatalogError("Nema promena u odnosu na važeće izdanje.")
        summary = {"changes": _releases.describe_changes(prep["baseline"], prep["new"]),
                   "decision_changes": prep["generated"].changes}
        return _store_release(settings, conn, "publish", None, person, why, prep["generated"].files, summary,
                              prep["decisions_upto"])
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def rollback_release(settings: CatalogSettings, number: int, actor: Any, role: Any, reason: Any) -> Dict[str, Any]:
    """A new release with exactly the files of an older one. Nothing is erased; the step stays in the list."""
    person, why = _admin(actor, role, reason)
    conn = _open_for_write(settings)
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = _release_rows(conn, settings.venue_id)
        target = next((r for r in rows if r["number"] == number), None)
        if target is None:
            raise UnknownRelease(number)
        live = rows[-1]
        if target["files"] == live["files"]:
            raise CatalogError("To izdanje je već važeće.")
        content = _releases.parse_files(target["files"])
        checks = _releases.run_checks(content, content, None, None, _live_classes(settings, content.class_lists))
        failed = [c for c in checks if not c["ok"]]
        if failed:
            raise ReleaseBlocked("Vraćanje nije moguće: " + "; ".join(c["label"].lower() for c in failed) + ".", checks)
        summary = {"changes": _releases.describe_changes(_releases.parse_files(live["files"]), content),
                   "decision_changes": target["summary"].get("decision_changes", [])}
        return _store_release(settings, conn, "rollback", number, person, why, target["files"], summary, live["decisions_upto"])
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------- Locking Miro and Word (D14) ----------------

LOCK_CONDITIONS = {
    "map": "Mapa iz Kataloga je ista kao današnja ili je svaka razlika objašnjena",
    "class_lists": "Liste klasa su iste za svih 5 modela",
    "trial_day": "Jedan ceo dan kroz uparivanje daje isti rezultat kao stara mapa",
}


def _is_locked(conn: sqlite3.Connection, venue: str) -> bool:
    row = conn.execute("SELECT action FROM locks WHERE venue_id = ? ORDER BY lock_id DESC LIMIT 1", (venue,)).fetchone()
    return bool(row and row[0] == "lock")


def _trial_reports(settings: CatalogSettings) -> List[Dict[str, Any]]:
    folder = settings.trial_dir
    if folder is None or not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("release"), int):
            data["file"] = path.name
            out.append(data)
    return out


def _lock_conditions(settings: CatalogSettings, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    venue = settings.venue_id
    rows = _release_rows(conn, venue)
    live = rows[-1] if rows else None
    if live is None:
        none = "Još nijedno izdanje nije objavljeno."
        return [{"key": k, "label": v, "ok": False, "detail": none} for k, v in LOCK_CONDITIONS.items()]
    content = _releases.parse_files(live["files"])
    models = list(content.class_lists)
    production = _releases.read_production(_production_paths(settings), models)
    generated = _releases.Generated(live["files"], live["summary"].get("decision_changes", []))
    parity = _releases.parity(production, generated, content)
    map_open = [i for i in parity["items"] if i["area"] in ("map", "blacklist", "missing") and not i["explained"]]
    class_diff = [i for i in parity["items"] if i["area"] == "class_lists"]
    missing_lists = [m for m in production.missing if m.startswith("lista klasa")]
    reports = [r for r in _trial_reports(settings) if r.get("release") == live["number"]]
    passed = [r for r in reports if r.get("complete") and r.get("unexplained") == 0]
    if passed:
        best = passed[-1]
        trial = f"Dan {best.get('day')} ({best.get('run')}): " + ("isto." if best.get("identical") else "razlike su objašnjene odlukama.")
    elif reports:
        last = reports[-1]
        trial = f"Dan {last.get('day')}: {last.get('unexplained')} neobjašnjenih razlika." if last.get("complete") else \
            f"Dan {last.get('day')}: poređenje nije završeno."
    else:
        trial = f"Za izdanje {live['number']} još nema probnog dana."
    return [
        {"key": "map", "label": LOCK_CONDITIONS["map"], "ok": not map_open,
         "detail": f"Izdanje {live['number']}: " + ("sve razlike su objašnjene." if not map_open else f"{len(map_open)} neobjašnjenih razlika.")},
        {"key": "class_lists", "label": LOCK_CONDITIONS["class_lists"], "ok": not class_diff and not missing_lists and len(models) >= 1,
         "detail": "Iste." if not class_diff and not missing_lists else f"Razlike: {', '.join(i['text'] for i in class_diff) or ', '.join(missing_lists)}"},
        {"key": "trial_day", "label": LOCK_CONDITIONS["trial_day"], "ok": bool(passed), "detail": trial},
    ]


def archive_text(day: str, release: Optional[int]) -> str:
    d = f"{day[8:10]}.{day[5:7]}.{day[0:4]}."
    return (f"Arhiva, stanje na dan {d} Važeći izvor je Katalog u labeling toolu (izdanje {release}). "
            "Ovaj dokument se više ne menja; nove odluke se vode samo u Katalogu.")


def lock_status(settings: CatalogSettings) -> Dict[str, Any]:
    venue = settings.venue_id
    with _open_for_read(settings) as conn:
        conditions = _lock_conditions(settings, conn)
        history = [{"action": a, "release": n, "actor": who, "reason": r, "created_at": c} for a, n, who, r, c in conn.execute(
            "SELECT action, release_number, actor, reason, created_at FROM locks WHERE venue_id = ? ORDER BY lock_id", (venue,))]
        locked = _is_locked(conn, venue)
    last = history[-1] if history else None
    return {"locked": locked, "since": last["created_at"] if locked else None, "release": last["release"] if locked else None,
            "conditions": conditions, "can_lock": not locked and all(c["ok"] for c in conditions),
            "archive_text": archive_text(last["created_at"], last["release"]) if locked else None, "history": list(reversed(history))}


def _lock_write(settings: CatalogSettings, action: str, actor: Any, role: Any, reason: Any) -> Dict[str, Any]:
    person, role = _clean_person(actor, role)
    if role != "admin":
        raise NotAllowed("Zaključava i otključava samo admin.")
    why = " ".join(str(reason or "").split())[:500]
    if not why:
        raise CatalogError("Upiši kratku napomenu.")
    venue = settings.venue_id
    conn = _open_for_write(settings)
    try:
        conn.execute("BEGIN IMMEDIATE")
        locked = _is_locked(conn, venue)
        conditions = _lock_conditions(settings, conn)
        if action == "lock":
            if locked:
                raise CatalogError("Već je zaključano.")
            failed = [c for c in conditions if not c["ok"]]
            if failed:
                raise ReleaseBlocked("Zaključavanje nije moguće: " + "; ".join(c["label"].lower() for c in failed) + ".", conditions)
        elif not locked:
            raise CatalogError("Nije zaključano.")
        live = conn.execute("SELECT MAX(number) FROM releases WHERE venue_id = ?", (venue,)).fetchone()[0]
        at = _now()
        conn.execute("INSERT INTO locks (venue_id, action, release_number, actor, role, reason, created_at, conditions) "
                     "VALUES (?, ?, ?, ?, 'admin', ?, ?, ?)",
                     (venue, action, live, person, why, at, json.dumps(conditions, ensure_ascii=False)))
        conn.execute("COMMIT")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return lock_status(settings)


def lock_catalog(settings: CatalogSettings, actor: Any, role: Any, reason: Any) -> Dict[str, Any]:
    """Owner's click: from now on the Catalog is the only source and Miro/Word are an archive."""
    return _lock_write(settings, "lock", actor, role, reason)


def unlock_catalog(settings: CatalogSettings, actor: Any, role: Any, reason: Any) -> Dict[str, Any]:
    """The way back: Miro and Word count again and the Miro/Word sentences return. Recorded, nothing erased."""
    return _lock_write(settings, "unlock", actor, role, reason)


# ---------------- POS report intake (ticket 11) ----------------

from web_labeler.catalog import pos_intake as _pos  # noqa: E402
from web_labeler.catalog import article_findings as _articles  # noqa: E402

FINDING_STATUS_LABELS = {"open": "Čeka odluku", "applied": "Primenjeno", "denied": "Odbijeno"}
FINDING_ACTIONS = {"changed_file": ("Koristi novu verziju", "Zadrži staru verziju"),
                   "name_mismatch": ("U redu, važi datum iz fajla", "Proveriti sa lokalom"),
                   "bad_file": ("Viđeno", "Proveriti sa lokalom"),
                   "rename": ("Isti artikal, potvrdi", "Nije isti, ne prati"),
                   "code_reused": ("Viđeno, odlučiću u Artiklima", "Greška u kasi"),
                   "second_code": ("Isti artikal, potvrdi", "Nije isti"),
                   "new_article": ("Viđeno", "Ne treba"),
                   "price_jump": ("U redu", "Proveriti šifru"),
                   "pos_missing": ("Stiglo je / rešeno", "Nije potrebno"),
                   "clip_late": ("Viđeno", "Nije potrebno"),
                   "clip_budget": ("Viđeno", "Nije potrebno")}


class UnknownFinding(Exception):
    pass


def _finding_status(conn: sqlite3.Connection, finding_id: int) -> str:
    row = conn.execute("SELECT action FROM finding_events WHERE finding_id = ? ORDER BY event_id DESC LIMIT 1",
                       (finding_id,)).fetchone()
    return "open" if row is None or row[0] == "reopened" else row[0]


def _add_finding(conn: sqlite3.Connection, venue: str, kind: str, day: Optional[str], file_id: Optional[int],
                 title: str, text: str, data: Mapping[str, Any], at: str) -> None:
    conn.execute("INSERT INTO findings (venue_id, kind, day, file_id, title, text, data, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (venue, kind, day, file_id, title, text, json.dumps(dict(data), ensure_ascii=False), at))


def _active_file(conn: sqlite3.Connection, venue: str, day: str) -> Optional[int]:
    """The version of a day that counts: the newest good one, unless the admin chose to keep an older one."""
    for file_id, in conn.execute("SELECT file_id FROM pos_files WHERE venue_id = ? AND report_day = ? AND status = 'ok' "
                                 "ORDER BY file_id DESC", (venue, day)):
        denied = [f for f, in conn.execute("SELECT finding_id FROM findings WHERE venue_id = ? AND kind = 'changed_file' "
                                           "AND file_id = ?", (venue, file_id)) if _finding_status(conn, f) == "denied"]
        if not denied:
            return file_id
    return None


def receive_pos_reports(settings: CatalogSettings, now: Optional[datetime] = None,
                        settle_seconds: int = _pos.SETTLE_SECONDS) -> Dict[str, Any]:
    """Read new, complete POS reports into the Catalog. The drop folder, the map and RAW are never touched.

    The first run only records what is already there (baseline, no findings), so old days do not flood the list.
    """
    if not settings.pos_dir or not settings.pos_adapter:
        raise CatalogError("POS folder is not configured (catalog_pos_dir, catalog_pos_adapter)")
    if settings.media_dir is None:
        raise CatalogError("Catalog media folder is not configured")
    now = now or datetime.now()
    venue = settings.venue_id
    conn = _open_for_write(settings)
    report: Dict[str, Any] = {"received": [], "waiting": [], "findings": 0, "baseline": False}
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT OR IGNORE INTO venues (venue_id, name, created_at) VALUES (?, ?, ?)",
                     (venue, settings.venue_name or venue, _now()))
        known = {sha for sha, in conn.execute("SELECT sha256 FROM pos_files WHERE venue_id = ?", (venue,))}
        baseline = not known
        report["baseline"] = baseline
        pairs_known = _known_pairs(conn, venue)
        seen_findings = {(k, d.get("code"), _articles.norm(d.get("name", ""))) for k, d in
                         ((k, json.loads(d)) for k, d in conn.execute("SELECT kind, data FROM findings WHERE venue_id = ?", (venue,)))}
        current_view: Optional[Dict[int, _decisions.Variation]] = None
        ready, report["waiting"] = _pos.find_new(settings.pos_dir, settings.pos_adapter, known, now, settle_seconds)
        copies = []
        for path in ready:
            item = _pos.read_report(path, settings.pos_adapter)
            if item.sha256 in known:  # same bytes under another name
                continue
            known.add(item.sha256)
            at = _now()
            first, last = item.ring_span
            previous = _active_file(conn, venue, item.report_day) if item.report_day else None
            file_id = conn.execute(
                "INSERT INTO pos_files (venue_id, file_name, sha256, size, received_at, report_day, name_day, ring_first, ring_last, "
                "rows_total, rows_paid, rows_unpaid, bad_rows, status, baseline) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (venue, item.name, item.sha256, item.size, at, item.report_day, item.name_day, first, last, item.rows_total,
                 item.rows_paid, item.rows_unpaid, len(item.bad_rows), "bad" if item.error else "ok", int(baseline))).lastrowid
            conn.executemany("INSERT INTO pos_file_rows (file_id, row_key, count, code, name, ring_time, quantity, price) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             [(file_id, json.dumps(list(key), ensure_ascii=False), count, r.code, r.name,
                               r.ring_time.isoformat(sep=" "), r.quantity, r.price) for key, (count, r) in item.rows.items()])
            copies.append((path, settings.media_dir / "pos" / f"{item.sha256}{path.suffix}"))
            report["received"].append({"file": item.name, "day": item.report_day, "rows": item.rows_paid,
                                       "status": "bad" if item.error else "ok"})
            if baseline:
                continue
            day_text = _pos.fmt_day(item.report_day)
            if item.error:
                _add_finding(conn, venue, "bad_file", item.report_day, file_id, f"Neispravan POS izveštaj: {item.name}",
                             item.error + " Ne koristi se.", {"file": item.name}, at)
            if item.name_day and item.report_day and item.name_day != item.report_day:
                _add_finding(conn, venue, "name_mismatch", item.report_day, file_id,
                             f"Ime fajla i datum se ne slažu: {item.name}",
                             f"Ime fajla kaže {_pos.fmt_day(item.name_day)}, a u fajlu piše {day_text} Važi datum iz fajla.",
                             {"file": item.name, "name_day": item.name_day}, at)
            if previous is not None and not item.error:
                old_name, = conn.execute("SELECT file_name FROM pos_files WHERE file_id = ?", (previous,)).fetchone()
                old_keys = [tuple(json.loads(k)) for k, in conn.execute("SELECT row_key FROM pos_file_rows WHERE file_id = ?", (previous,))]
                diff = _pos.compare_versions(old_keys, item)
                _add_finding(conn, venue, "changed_file", item.report_day, file_id, f"Izmenjen izveštaj za {day_text}",
                             f"Stigla je nova verzija ({item.name}) za dan koji već postoji ({old_name}): "
                             f"{diff['only_new']} novih redova, {diff['only_old']} redova kojih više nema, {diff['both']} istih. "
                             "Važi nova verzija dok ne odlučiš drugačije; obe verzije su sačuvane.",
                             dict(diff, previous_file_id=previous, file=item.name, previous_file=old_name), at)
            if not item.error:
                sales: Dict[Tuple[str, str], List[float]] = {}
                for count, r in item.rows.values():
                    if r.quantity > 0:
                        sales.setdefault((r.code, r.name.strip()), []).extend([r.price] * count)
                for f in _articles.detect(pairs_known, sales):
                    key = (f["kind"], f["code"], _articles.norm(f["name"]))
                    if key in seen_findings:
                        continue
                    seen_findings.add(key)
                    if f["kind"] == "rename":
                        if current_view is None:
                            current_view = _current(conn, venue)
                        old = next((v for v in current_view.values() if (v.code or "") == f["code"]
                                    and _articles.norm(v.name) == _articles.norm(f["old_name"])), None)
                        f["classes"] = [list(link) for link in old.links] if old and old.state == "tracked" else []
                    title, text = _article_finding_text(f)
                    _add_finding(conn, venue, f["kind"], item.report_day, file_id, title, text, f, at)
                for (code, name), prices in sales.items():
                    pairs_known.add(code, name, sorted(prices)[len(prices) // 2])
            if item.bad_rows and not item.error:
                _add_finding(conn, venue, "bad_file", item.report_day, file_id, f"Neki redovi nisu pročitani: {item.name}",
                             f"{len(item.bad_rows)} redova ne može da se pročita i ne računa se. Ostatak izveštaja važi.",
                             {"file": item.name, "rows": item.bad_rows[:20]}, at)
        report["findings"] = conn.execute("SELECT COUNT(*) FROM findings WHERE venue_id = ?", (venue,)).fetchone()[0]
        for source, target in copies:  # every received version is kept in the Catalog's own folder
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                import shutil
                shutil.copy2(source, target)
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return report


def _known_pairs(conn: sqlite3.Connection, venue: str) -> "_articles.Known":
    known = _articles.Known()
    for code, name, price in conn.execute("SELECT code, name, last_price FROM pos_identities WHERE venue_id = ? AND code IS NOT NULL",
                                          (venue,)):
        known.add(code, name, price)
    for code, name, price in conn.execute(
            "SELECT r.code, r.name, r.price FROM pos_file_rows r JOIN pos_files f ON f.file_id = r.file_id "
            "WHERE f.venue_id = ? AND r.quantity > 0 ORDER BY r.ring_time", (venue,)):
        known.add(code, name, price)
    return known


def _article_finding_text(f: Mapping[str, Any]) -> Tuple[str, str]:
    name, code = f["name"], f["code"]
    if f["kind"] == "rename":
        classes = ", ".join(f"{m}.{c}" for m, c in f.get("classes") or []) or "bez klase"
        return (f"Preimenovano u kasi: {f['old_name']} → {name}",
                f"Šifra {code} se sada zove \"{name}\". Prati se kao i do sada ({classes}), sa oznakom \"na proveri\". "
                "Potvrdi ako je isto piće; ako nije, novo ime se ne prati.")
    if f["kind"] == "code_reused":
        return (f"Šifra {code} dobila potpuno drugo ime: {name}",
                f"Ranije: {', '.join(f['old_names'])}. Sada: \"{name}\". Možda je šifra data drugom artiklu. "
                "Novo ime se ne prati dok ne odlučiš.")
    if f["kind"] == "second_code":
        return (f"Nova šifra za poznat artikal: {name}",
                f"\"{name}\" se sada kuca i pod šifrom {code} (ranije {', '.join(f['other_codes'])}). "
                "Prati se kao isti artikal, sa oznakom \"na proveri\".")
    if f["kind"] == "price_jump":
        return (f"Nagla promena cene: {name}",
                f"Cena po komadu za šifru {code} sa {f['old_price']:g} na {f['new_price']:g} din. Proveri da šifra nije data drugom artiklu.")
    return (f"Novi artikal u kasi: {name}", f"Šifra {code}, prodato {f['sold']} u ovom izveštaju. Čeka razvrstavanje.")


def list_findings(settings: CatalogSettings, status: Optional[str] = "open") -> Dict[str, Any]:
    venue = settings.venue_id
    with _open_for_read(settings) as conn:
        rows = conn.execute("SELECT finding_id, kind, day, file_id, title, text, data, created_at FROM findings "
                            "WHERE venue_id = ? ORDER BY finding_id DESC", (venue,)).fetchall()
        items = []
        for finding_id, kind, day, file_id, title, text, data, created_at in rows:
            state = _finding_status(conn, finding_id)
            if status and status != "all" and state != status:
                continue
            history = [{"action": a, "actor": who, "role": r, "reason": why, "at": at} for a, who, r, why, at in conn.execute(
                "SELECT action, actor, role, reason, at FROM finding_events WHERE finding_id = ? ORDER BY event_id", (finding_id,))]
            apply_label, deny_label = FINDING_ACTIONS.get(kind, ("Primeni", "Odbij"))
            items.append({"finding_id": finding_id, "kind": kind, "day": day, "title": title, "text": text,
                          "data": json.loads(data), "created_at": created_at, "status": state,
                          "status_label": FINDING_STATUS_LABELS[state], "apply_label": apply_label, "deny_label": deny_label,
                          "history": history})
        counts = Counter(_finding_status(conn, f) for f, in conn.execute("SELECT finding_id FROM findings WHERE venue_id = ?", (venue,)))
    return {"items": items, "counts": {k: counts.get(k, 0) for k in FINDING_STATUS_LABELS}}


def decide_finding(settings: CatalogSettings, finding_id: int, action: str, actor: Any, role: Any, reason: Any = "") -> Dict[str, Any]:
    """Apply, deny or reopen (change later) a finding. Admin only; every step stays in its history."""
    person, role = _clean_person(actor, role)
    if role != "admin":
        raise NotAllowed("O nalazima odlučuje admin.")
    if action not in ("applied", "denied", "reopened"):
        raise CatalogError("Odluka mora biti primeni, odbij ili vrati na odluku.")
    venue = settings.venue_id
    conn = _open_for_write(settings)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM findings WHERE venue_id = ? AND finding_id = ?", (venue, finding_id)).fetchone() is None:
            raise UnknownFinding(finding_id)
        state = _finding_status(conn, finding_id)
        if (action == "reopened") == (state == "open"):
            raise CatalogError("Nalaz već čeka odluku." if action == "reopened" else "O ovom nalazu je već odlučeno; prvo ga vrati na odluku.")
        conn.execute("INSERT INTO finding_events (venue_id, finding_id, action, actor, role, reason, at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (venue, finding_id, action, person, role, " ".join(str(reason or "").split())[:500] or None, _now()))
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return next(i for i in list_findings(settings, "all")["items"] if i["finding_id"] == finding_id)


def list_pos_days(settings: CatalogSettings) -> Dict[str, Any]:
    """Received reports per day and which version counts (base for the day table, ticket 13)."""
    venue = settings.venue_id
    with _open_for_read(settings) as conn:
        days: Dict[str, Dict[str, Any]] = {}
        for file_id, name, day, rows, status, received, baseline in conn.execute(
                "SELECT file_id, file_name, report_day, rows_paid, status, received_at, baseline FROM pos_files "
                "WHERE venue_id = ? ORDER BY file_id", (venue,)):
            entry = days.setdefault(day or "?", {"day": day, "versions": []})
            entry["versions"].append({"file_id": file_id, "file": name, "rows": rows, "status": status,
                                      "received_at": received, "baseline": bool(baseline)})
        for entry in days.values():
            entry["active_file_id"] = _active_file(conn, venue, entry["day"]) if entry["day"] else None
    return {"days": sorted(days.values(), key=lambda d: d["day"] or "", reverse=True)}


# ---------------- Day table (ticket 13) ----------------

DAY_STATUS_LABELS = {"ok": "Stiglo u redu", "suspicious": "Stiglo sumnjivo", "late": "Kasni", "closed": "Nije radilo",
                     "waiting": "Čeka (rok je kraj sutrašnjeg dana)", "before": "Pre uključivanja"}


def _day_closed(conn: sqlite3.Connection, venue: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for day, action, actor, reason, at in conn.execute(
            "SELECT day, action, actor, reason, at FROM pos_day_marks WHERE venue_id = ? ORDER BY mark_id", (venue,)):
        if action == "closed":
            out[day] = {"actor": actor, "reason": reason, "at": at}
        else:
            out.pop(day, None)
    return out


def day_table(settings: CatalogSettings, today: Optional[str] = None, days: int = 45) -> Dict[str, Any]:
    """One status per day. A report for day D is on time during all of D+1 (D4). Days before the module was switched on
    are never late (they are shown as they are)."""
    from datetime import date, timedelta
    import statistics
    today_d = date.fromisoformat(today) if today else date.today()
    venue = settings.venue_id
    with _open_for_read(settings) as conn:
        first = conn.execute("SELECT MIN(received_at) FROM pos_files WHERE venue_id = ?", (venue,)).fetchone()[0]
        enabled = first[:10] if first else None
        files = {}
        for file_id, day, rows, last, status in conn.execute(
                "SELECT file_id, report_day, rows_paid, ring_last, status FROM pos_files WHERE venue_id = ? AND report_day IS NOT NULL",
                (venue,)):
            files.setdefault(day, []).append((file_id, rows, last, status))
        active = {day: _active_file(conn, venue, day) for day in files}
        info = {f: (rows, last) for day in files for f, rows, last, _ in files[day]}
        mismatch = {f for f, in conn.execute("SELECT file_id FROM findings WHERE venue_id = ? AND kind = 'name_mismatch'", (venue,))}
        closed = _day_closed(conn, venue)
    by_weekday: Dict[int, List[Tuple[str, int, str]]] = {}
    for day, f in active.items():
        if f is not None:
            rows, last = info[f]
            by_weekday.setdefault(date.fromisoformat(day).weekday(), []).append((day, rows, last or ""))

    def usual(day: str) -> Tuple[Optional[float], Optional[str]]:
        same = [(r, l) for d, r, l in by_weekday.get(date.fromisoformat(day).weekday(), []) if d != day]
        if len(same) < 3:
            return None, None
        lasts = sorted(l[11:16] for _, l in same if l)
        return statistics.median(r for r, _ in same), lasts[len(lasts) // 2] if lasts else None

    out = []
    for back in range(1, days + 1):
        d = today_d - timedelta(days=back)
        day = d.isoformat()
        f = active.get(day)
        notes: List[str] = []
        rows = None
        if day in closed:
            status = "closed"
            c = closed[day]
            notes.append(f"{c['actor']}: {c['reason'] or 'bez razloga'}")
        elif f is not None:
            rows, last = info[f]
            status = "ok"
            typical_rows, typical_last = usual(day)
            if f in mismatch:
                notes.append("ime fajla i datum se ne slažu")
            if typical_rows and rows < typical_rows / 2:
                notes.append(f"{rows} redova, obično oko {int(typical_rows)}")
            if typical_last and last and last[11:16] < f"{max(0, int(typical_last[:2]) - 3):02d}:{typical_last[3:5]}":
                notes.append(f"poslednji račun u {last[11:16]}, obično oko {typical_last}")
            if notes:
                status = "suspicious"
        elif files.get(day):
            status = "suspicious" if enabled and day >= enabled else "before"
            notes.append("stigao je samo neispravan izveštaj")
        elif enabled is None or day < enabled:
            status = "before"
        elif back >= 2:
            status = "late"
        else:
            status = "waiting"
        out.append({"day": day, "status": status, "label": DAY_STATUS_LABELS[status], "rows": rows, "notes": notes,
                    "can_close": status in ("late", "waiting", "suspicious", "before"), "can_reopen": status == "closed"})
    return {"enabled": enabled, "days": out, "counts": dict(Counter(d["status"] for d in out))}


def mark_day(settings: CatalogSettings, day: str, action: str, actor: Any, role: Any, reason: Any = "") -> Dict[str, Any]:
    """'Lokal nije radio' (closed) or undo (reopened). Labeler or admin; nothing is erased."""
    from datetime import date
    person, role = _clean_person(actor, role)
    try:
        date.fromisoformat(day)
    except ValueError:
        raise CatalogError("Neispravan datum.")
    if action not in ("closed", "reopened"):
        raise CatalogError("Nepoznata akcija.")
    why = " ".join(str(reason or "").split())[:300] or None
    if action == "closed" and not why:
        raise CatalogError("Upiši razlog (npr. praznik, renoviranje).")
    venue = settings.venue_id
    conn = _open_for_write(settings)
    try:
        conn.execute("BEGIN IMMEDIATE")
        is_closed = day in _day_closed(conn, venue)
        if (action == "closed") == is_closed:
            raise CatalogError("Dan je već označen." if is_closed else "Dan nije označen kao neradni.")
        conn.execute("INSERT INTO pos_day_marks (venue_id, day, action, actor, role, reason, at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (venue, day, action, person, role, why, _now()))
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return {"day": day, "action": action}


# ---------------- Inbox "čeka te" (ticket 14) ----------------

SNOOZE_DAYS = 14


def _snoozed(conn: sqlite3.Connection, venue: str, current: Mapping[int, _decisions.Variation], today: str) -> Dict[int, str]:
    """Snoozed until 14 days pass or the article is sold again, whichever is first."""
    out: Dict[int, str] = {}
    for identity_id, until, at in conn.execute(
            "SELECT identity_id, until_day, at FROM inbox_snoozes WHERE venue_id = ? ORDER BY snooze_id", (venue,)):
        v = current.get(identity_id)
        sold_again = v is not None and v.last_sale and v.last_sale[:10] > at[:10]
        if until > today and not sold_again:
            out[identity_id] = until
        else:
            out.pop(identity_id, None)
    return out


def list_inbox(settings: CatalogSettings, today: Optional[str] = None) -> Dict[str, Any]:
    """What waits for the labeler: sold articles without a class, nobody proposed anything yet, not snoozed."""
    today = today or _today()
    venue = settings.venue_id
    with _open_for_read(settings) as conn:
        current = _current(conn, venue)
        decisions = _load_decisions(conn, venue)
        snoozed = _snoozed(conn, venue, current, today)
        evidence = _evidence_by_identity(conn, venue)
    waiting_ids, waiting_articles, new_class = set(), set(), set()
    for d in decisions:
        if d.kind == "propose" and _proposal_status(d.decision_id, decisions) == "pending":
            if d.identity_id:
                waiting_ids.add(d.identity_id)
            if d.payload.get("kind") == "new_class" and d.identity_id:
                new_class.add(d.identity_id)
            elif d.article_id and not d.identity_id:
                waiting_articles.add(d.article_id)
    flows = _flows(settings, current, decisions)
    items, counts = [], Counter()
    for v in _sorted(x for x in current.values() if x.code is not None and x.state == "pending" and x.sold > 0):
        if v.target:
            continue  # has its class, waits for labeling or training: listed below with its step
        if v.identity_id in new_class:
            counts["waiting_class"] += 1
        elif v.identity_id in waiting_ids or v.article_id in waiting_articles:
            counts["proposed"] += 1
        elif v.identity_id in snoozed:
            counts["snoozed"] += 1
        else:
            items.append(_variation(v, evidence))
    path = [_variation(v, evidence, flows[v.identity_id]) for v in _sorted(current.values())
            if v.identity_id in flows and flows[v.identity_id]["stage"] != "tracked"]
    clip_status = _clips.read_status(settings.db_path.parent) if settings.clips and settings.db_path else {}
    for item in items + path:
        st = clip_status.get(item["identity_id"])
        item["clip"] = {"status": st, "label": _clips.CLIP_STATUS_LABELS[st]} if st in _clips.CLIP_STATUS_LABELS else None
    order = list(FLOW_LABELS)
    path.sort(key=lambda i: order.index(i["flow"]["stage"]))
    return {"items": items, "new_class": path, "flow_labels": FLOW_LABELS,
            "counts": {"waiting": len(items), "proposed": counts["proposed"], "waiting_class": counts["waiting_class"],
                       "snoozed": counts["snoozed"], "new_class": len(path)}}


def snooze(settings: CatalogSettings, identity_id: int, actor: Any, role: Any, reason: Any = "") -> Dict[str, Any]:
    from datetime import date, timedelta
    person, role = _clean_person(actor, role)
    venue = settings.venue_id
    until = (date.fromisoformat(_today()) + timedelta(days=SNOOZE_DAYS)).isoformat()
    conn = _open_for_write(settings)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM pos_identities WHERE venue_id = ? AND identity_id = ?", (venue, identity_id)).fetchone() is None:
            raise UnknownArticle(identity_id)
        conn.execute("INSERT INTO inbox_snoozes (venue_id, identity_id, until_day, actor, role, reason, at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (venue, identity_id, until, person, role, " ".join(str(reason or "").split())[:300] or None, _now()))
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return {"identity_id": identity_id, "until": until}


# ---------------- Reminder for a missing POS report (ticket 15) ----------------

REMINDER_HOUR = 9


def missing_report_reminders(settings: CatalogSettings, now: Optional[datetime] = None,
                             send: Optional[Any] = None) -> Dict[str, Any]:
    """For every late day (not 'lokal nije radio'), once, after 09:00: a finding in the tool and the email text in
    <db dir>/reminders/. An email goes out only when catalog_reminder_emails is set AND a sender is given; nothing is
    sent until the owner decides addresses and the way of sending."""
    now = now or datetime.now()
    report: Dict[str, Any] = {"created": [], "sent": [], "skipped_before_hour": now.hour < REMINDER_HOUR}
    if now.hour < REMINDER_HOUR:
        return report
    table = day_table(settings, today=now.date().isoformat())
    late = [d["day"] for d in table["days"] if d["status"] == "late"]
    venue = settings.venue_id
    conn = _open_for_write(settings)
    try:
        conn.execute("BEGIN IMMEDIATE")
        done = {day for day, in conn.execute("SELECT day FROM findings WHERE venue_id = ? AND kind = 'pos_missing'", (venue,))}
        outbox = settings.db_path.parent / "reminders"
        for day in sorted(set(late) - done):
            text = (f"Podsetnik: POS izveštaj za {_pos.fmt_day(day)} nije stigao. Molimo pošaljite ga kao i obično, "
                    "ili u alatu (Katalog → Dani) označite da lokal nije radio.")
            _add_finding(conn, venue, "pos_missing", day, None, f"Nije stigao POS izveštaj za {_pos.fmt_day(day)}", text,
                         {"day": day}, _now())
            outbox.mkdir(parents=True, exist_ok=True)
            (outbox / f"{day}.txt").write_text(text + "\n", encoding="utf-8")
            report["created"].append(day)
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    recipients = list(getattr(settings, "reminder_emails", None) or [])
    if send is not None and recipients:
        for day in report["created"]:
            send(recipients, f"Nije stigao POS izveštaj za {_pos.fmt_day(day)}",
                 (settings.db_path.parent / "reminders" / f"{day}.txt").read_text(encoding="utf-8"))
            report["sent"].append(day)
    return report


# ---------------- Footage of new articles (ticket 16) ----------------

def _clip_candidates(settings: CatalogSettings, conn: sqlite3.Connection, today: str) -> List[Dict[str, Any]]:
    """Articles that need footage: sold, waiting for a class (not X, not snoozed), or on the new class path before
    labeling is done (Čeka klasu / U označavanju)."""
    venue = settings.venue_id
    current = _current(conn, venue)
    decisions = _load_decisions(conn, venue)
    snoozed = _snoozed(conn, venue, current, today)
    flows = _flows(settings, current, decisions)
    # someone already proposed X (food, cigarettes): no footage needed unless the admin turns the proposal down
    x_ids, x_articles = set(), set()
    for d in decisions:
        if d.kind == "propose" and d.payload.get("kind") == "ignore" and _proposal_status(d.decision_id, decisions) == "pending":
            (x_ids.add(d.identity_id) if d.identity_id else x_articles.add(d.article_id))
    out = []
    for v in _sorted(current.values()):
        if v.code is None or v.sold <= 0 or v.state == "ignored":
            continue
        if v.identity_id in x_ids or v.article_id in x_articles or v.source_article_id in x_articles:
            continue
        stage = (flows.get(v.identity_id) or {}).get("stage")
        waiting = v.state == "pending" and not v.target and v.identity_id not in snoozed
        if waiting or stage in ("waiting_class", "labeling"):
            out.append({"identity_id": v.identity_id, "article_id": v.article_id, "code": v.code, "name": v.name,
                        "last_sale": v.last_sale, "stage": stage})
    return out


def _clip_events(conn: sqlite3.Connection, venue: str, identity_id: Optional[int] = None) -> List[Dict[str, Any]]:
    where, args = ("AND identity_id = ? ", (venue, identity_id)) if identity_id is not None else ("", (venue,))
    return [{"event_id": e, "identity_id": i, "action": a, "sale_time": t, "day": d, "bytes": b, "folder": f,
             "detail": json.loads(x or "{}"), "actor": who, "role": r, "at": at}
            for e, i, a, t, d, b, f, x, who, r, at in conn.execute(
                "SELECT event_id, identity_id, action, sale_time, day, bytes, folder, detail, actor, role, at FROM clip_events "
                f"WHERE venue_id = ? {where}ORDER BY event_id", args)]


def clip_context(settings: CatalogSettings, today: str) -> Dict[str, Any]:
    """Everything the footage job needs, read in one go: candidates, clip history, when each article was last decided."""
    venue = settings.venue_id
    with _open_for_read(settings) as conn:
        candidates = _clip_candidates(settings, conn, today)
        events = _clip_events(conn, venue)
        idents = {i: a for i, a in conn.execute("SELECT identity_id, article_id FROM pos_identities WHERE venue_id = ?", (venue,))}
        by_identity: Dict[int, str] = {}
        by_article: Dict[int, str] = {}
        for i, a, at in conn.execute("SELECT identity_id, article_id, created_at FROM decisions WHERE venue_id = ?", (venue,)):
            if i is not None:
                by_identity[i] = max(by_identity.get(i, ""), at)
            if a is not None:
                by_article[a] = max(by_article.get(a, ""), at)
    decided = {}
    for iid in {e["identity_id"] for e in events}:
        at = max(by_identity.get(iid, ""), by_article.get(idents.get(iid), ""))
        if at:
            decided[iid] = at
    return {"candidates": candidates, "events": events, "decided_at": decided}


def add_clip_events(settings: CatalogSettings, events: Sequence[Mapping[str, Any]],
                    findings: Sequence[Mapping[str, Any]]) -> None:
    """Append clip events (and the job's findings, once per kind+day+article) in one transaction."""
    venue = settings.venue_id
    conn = _open_for_write(settings)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for e in events:
            conn.execute("INSERT INTO clip_events (venue_id, identity_id, action, sale_time, day, bytes, folder, detail, actor, "
                         "role, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (venue, int(e["identity_id"]), e["action"], e.get("sale_time"), e.get("day"), e.get("bytes"),
                          e.get("folder"), json.dumps(dict(e.get("detail") or {}), ensure_ascii=False), e.get("actor"),
                          e.get("role"), e.get("at") or _now()))
        for f in findings:
            same = [json.loads(d) for d, in conn.execute("SELECT data FROM findings WHERE venue_id = ? AND kind = ? AND day = ?",
                                                         (venue, f["kind"], f["day"]))]
            if any(d.get("identity_id") == f["data"].get("identity_id") for d in same):
                continue
            _add_finding(conn, venue, f["kind"], f["day"], None, f["title"], f["text"], f["data"], _now())
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


# ---------------- Card with the cameras (ticket 17) ----------------

class UnknownClip(Exception):
    pass


def _clip_settings(settings: CatalogSettings) -> "_clips.ClipSettings":
    if settings.clips is None:
        raise CatalogError("Snimci nisu podešeni.")
    return settings.clips


def _fmt_sale(key: str) -> str:
    day, hm = key.split(" ")
    y, m, d = day.split("-")
    return f"{int(d)}.{int(m)}.{y}. u {hm}"


def sale_slug(key: str) -> str:
    return key.replace("-", "").replace(":", "").replace(" ", "_")


def _sales_for(settings: CatalogSettings, conn: sqlite3.Connection, identity_id: int) -> Dict[str, "_clips.SaleState"]:
    clips = _clip_settings(settings)
    folded = _clips.fold_events(_clip_events(conn, settings.venue_id, identity_id), clips.attempts_before_missing)
    return (folded.get(identity_id) or {}).get("sales", {})


def clip_card(settings: CatalogSettings, identity_id: int) -> Dict[str, Any]:
    """What the card plays: downloaded sales nobody marked "Nije na snimku", newest first, each with its cameras."""
    clips = _clip_settings(settings)
    with _open_for_read(settings) as conn:
        row = conn.execute("SELECT name, code FROM pos_identities WHERE venue_id = ? AND identity_id = ?",
                           (settings.venue_id, identity_id)).fetchone()
        if row is None:
            raise UnknownArticle(identity_id)
        sales = _sales_for(settings, conn, identity_id)
        target = _labeling_target(settings, conn, identity_id)
        sent = {e["detail"].get("seen_event") for e in _clip_events(conn, settings.venue_id, identity_id) if e["action"] == "sent"}
    status = _clips.read_status(settings.db_path.parent).get(identity_id) if settings.db_path else None
    shown = []
    for sale in sorted(sales.values(), key=lambda x: x.key, reverse=True):
        if sale.status != "downloaded" or sale.not_visible:
            continue
        cams = [{"channel": int(ch), "name": clips.camera_name(int(ch)), "ready": ch in sale.previews,
                 "url": f"/api/catalog/clips/{identity_id}/video/{sale_slug(sale.key)}/{ch}" if ch in sale.previews else None}
                for ch in sorted(sale.channels, key=int)]
        shown.append({"sale": sale.key, "slug": sale_slug(sale.key), "label": f"Kucano {_fmt_sale(sale.key)}",
                      "cameras": cams, "seen": [{"camera": clips.camera_name(int(x["channel"])), "second": x["second"],
                                                 "evidence_id": x.get("evidence_id"), "sent": x.get("event_id") in sent}
                                                for x in sale.seen]})
    if shown:
        status = "seen" if any(x["seen"] for x in shown) else "ready"
    elif any(x.not_visible for x in sales.values()):
        status = "missing"  # every downloaded sale was "Nije na snimku"; the job looks for another sale
    elif status not in ("waiting", "late", "missing", "deleted"):
        status = "waiting"
    to_send = sum(1 for x in shown for y in x["seen"] if not y["sent"])
    return {"identity_id": identity_id, "name": row[0].strip(), "code": row[1], "sales": shown,
            "status": status, "status_label": _clips.CLIP_STATUS_LABELS[status],
            "target": {"model": target[0], "class": target[1]} if target else None, "to_send": to_send}


def _find_sale(settings: CatalogSettings, conn: sqlite3.Connection, identity_id: int, slug: str) -> "_clips.SaleState":
    for sale in _sales_for(settings, conn, identity_id).values():
        if sale_slug(sale.key) == slug and sale.status == "downloaded":
            return sale
    raise UnknownClip(slug)


def clip_video(settings: CatalogSettings, identity_id: int, slug: str, channel: int) -> Path:
    clips = _clip_settings(settings)
    with _open_for_read(settings) as conn:
        sale = _find_sale(settings, conn, identity_id, slug)
    rel = sale.previews.get(str(channel))
    if not rel:
        raise UnknownClip(slug)
    path = (clips.root / rel).resolve()
    if clips.root.resolve() not in path.parents or not path.is_file():
        raise UnknownClip(slug)
    return path


def _frame_with_box(clips: "_clips.ClipSettings", sale: "_clips.SaleState", channel: int, second: float,
                    box: Sequence[float]) -> bytes:
    """The frame at that second from the full camera file (preview as fallback), the drawn box on it, as JPEG."""
    import io
    import subprocess
    from PIL import Image, ImageDraw
    sources = [clips.root / sale.channels[str(channel)]] if str(channel) in sale.channels else []
    if str(channel) in sale.previews:
        sources.append(clips.root / sale.previews[str(channel)])
    for src in sources:
        if not src.is_file():
            continue
        try:
            data = subprocess.run([clips.ffmpeg, "-v", "error", "-ss", f"{second:.2f}", "-i", str(src), "-frames:v", "1",
                                   "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, timeout=120).stdout
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            continue
        w, h = image.size
        x, y, bw, bh = box
        draw = ImageDraw.Draw(image)
        draw.rectangle([x * w, y * h, (x + bw) * w, (y + bh) * h], outline=(255, 210, 0), width=max(3, w // 400))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=88)
        return out.getvalue()
    raise CatalogError("Kadar sa snimka ne može da se pročita.")


def mark_seen(settings: CatalogSettings, identity_id: int, slug: str, channel: int, second: Any, box: Any,
              actor: Any, role: Any) -> Dict[str, Any]:
    """"Vidim ga": one box on one camera. Saved only as Catalog evidence (waits for the admin, like a pasted
    screenshot); never RAW, never a dataset."""
    clips = _clip_settings(settings)
    person, role = _clean_person(actor, role)
    try:
        second = float(second)
        box = [float(v) for v in box]
    except (TypeError, ValueError):
        raise CatalogError("Nacrtaj jedan pravougaonik oko pića.")
    if len(box) != 4 or box[2] <= 0.005 or box[3] <= 0.005 or box[0] < 0 or box[1] < 0 \
            or box[0] + box[2] > 1.001 or box[1] + box[3] > 1.001:
        raise CatalogError("Nacrtaj jedan pravougaonik oko pića.")
    if not 0 <= second <= clips.window_minutes * 60 + 5:
        raise CatalogError("To vreme nije na ovom snimku.")
    with _open_for_read(settings) as conn:
        sale = _find_sale(settings, conn, identity_id, slug)
    if str(channel) not in sale.channels:
        raise UnknownClip(slug)
    image = _frame_with_box(clips, sale, int(channel), second, box)
    evidence = add_evidence(settings, identity_id, image, person, role, source="clip")
    add_clip_events(settings, [{"identity_id": identity_id, "action": "seen", "sale_time": sale.key, "actor": person,
                                "role": role, "detail": {"channel": int(channel), "second": round(second, 2),
                                                         "box": [round(v, 4) for v in box],
                                                         "evidence_id": evidence["evidence_id"]}}], [])
    return {"evidence": evidence, "card": clip_card(settings, identity_id)}


def mark_not_visible(settings: CatalogSettings, identity_id: int, slug: str, actor: Any, role: Any) -> Dict[str, Any]:
    """"Nije na snimku": this sale is skipped; the card shows the next one, after the last the article says "Nema snimka"
    and the job fetches another sale if there is one."""
    person, role = _clean_person(actor, role)
    with _open_for_read(settings) as conn:
        sale = _find_sale(settings, conn, identity_id, slug)
    add_clip_events(settings, [{"identity_id": identity_id, "action": "not_visible", "sale_time": sale.key,
                                "actor": person, "role": role}], [])
    return clip_card(settings, identity_id)


# ---------------- "Pošalji na označavanje" (ticket 18) ----------------

def _labeling_target(settings: CatalogSettings, conn: sqlite3.Connection, identity_id: int) -> Optional[Tuple[str, str]]:
    """The class the article goes to (new class or existing), as the tool names its model (its class list file)."""
    current = _current(conn, settings.venue_id)
    v = current.get(identity_id)
    if v is None:
        return None
    pair = v.target or (v.links[0] if v.links else None)
    if not pair:
        return None
    model, cls = pair
    for path in list(settings.source_models_dir.glob("*.yaml")) + list(settings.source_models_dir.glob("*.yml")):
        if path.stem.lower() == str(model).lower():
            return path.stem, cls
    return str(model), cls


def send_to_labeling(settings: CatalogSettings, identity_id: int, actor: Any, role: Any) -> Dict[str, Any]:
    """Cut 10 s before and after every "Vidim ga" (that camera only) into the tool's video list, batch "Za označavanje",
    with the model and class already chosen. From there the normal labeling, export and ingest path; nothing goes to RAW
    any other way."""
    clips = _clip_settings(settings)
    person, role = _clean_person(actor, role)
    folder_root = clips.labeling_videos_dir
    if folder_root is None:
        raise CatalogError("Folder sa videima za označavanje nije podešen.")
    for root in settings.forbidden_roots:
        if _is_within(folder_root.resolve(), Path(root).resolve()):
            raise CatalogError(f"Isečci ne smeju u {root}")
    with _open_for_read(settings) as conn:
        row = conn.execute("SELECT name, code FROM pos_identities WHERE venue_id = ? AND identity_id = ?",
                           (settings.venue_id, identity_id)).fetchone()
        if row is None:
            raise UnknownArticle(identity_id)
        target = _labeling_target(settings, conn, identity_id)
        events = _clip_events(conn, settings.venue_id, identity_id)
        sales = _sales_for(settings, conn, identity_id)
    if not target:
        raise CatalogError("Artikal još nema klasu. Prvo izaberi postojeću ili otvori novu klasu.")
    sent = {e["detail"].get("seen_event") for e in events if e["action"] == "sent"}
    todo = [(sale, seen) for sale in sales.values() if sale.status == "downloaded" for seen in sale.seen
            if seen.get("event_id") not in sent]
    if not todo:
        raise CatalogError("Nema novih \"Vidim ga\" za slanje. Prvo na snimku označi gde se vidi piće.")
    batch = folder_root / _clips.LABELING_BATCH
    batch.mkdir(parents=True, exist_ok=True)
    session = batch / "_session.json"
    if not session.exists():
        session.write_text(json.dumps({"session_id": "katalog (za označavanje)", "opened_at": _now()}, ensure_ascii=False),
                           encoding="utf-8")
    model, cls = target
    name = row[0].strip()
    code = re.sub(r"[^0-9A-Za-z]+", "", str(row[1] or "")) or "X"
    made, new_events = [], []
    for sale, seen in todo:
        ch = int(seen["channel"])
        src = clips.root / sale.channels[str(ch)]
        start = max(0.0, float(seen["second"]) - clips.cut_seconds)
        ring = datetime.strptime(sale.key, "%Y-%m-%d %H:%M")
        stamp = (ring + timedelta(seconds=start)).strftime("%Y%m%d%H%M%S")
        stem = f"BLAZNAVAC_KATALOG_{code}_{clips.camera_token(ch)}_{stamp}"
        video = batch / f"{stem}.mp4"
        _clips.cut_clip(clips, src, start, 2 * clips.cut_seconds, video)
        at = round(float(seen["second"]) - start, 2)
        hint = {"detected_class": cls, "model": model, "tlwh_norm": seen.get("box"), "at_second": at, "window_seconds": 6,
                "catalog": True, "preselect": {"model": model, "class": cls}}
        info = {"case_type": "KATALOG", "target_class": cls,
                # the tool shows this text as HTML: escape what comes from the till
                "instruction": f"Novi artikal iz kase: {html.escape(name)}. Označi piće na kadrovima; model "
                               f"{html.escape(model)} i klasa {html.escape(cls)} su već izabrani."}
        (batch / f"{stem}_hints.json").write_text(json.dumps(hint, ensure_ascii=False), encoding="utf-8")
        (batch / f"{stem}_info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        made.append({"video": video.name, "camera": clips.camera_name(ch)})
        new_events.append({"identity_id": identity_id, "action": "sent", "sale_time": sale.key, "actor": person, "role": role,
                           "detail": {"seen_event": seen.get("event_id"), "video": video.name, "batch": batch.name,
                                      "model": model, "class": cls}})
    add_clip_events(settings, new_events, [])
    return {"sent": len(made), "videos": [m["camera"] for m in made], "model": model, "class": cls,
            "where": "Labeler → lista videa → katalog (za označavanje)"}
