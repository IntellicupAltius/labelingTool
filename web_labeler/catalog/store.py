"""Catalog core: the only code that reads or changes the Catalog database."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from web_labeler.catalog import assortment as _assortment
from web_labeler.catalog import class_rules as _rules
from web_labeler.catalog import examples as _examples
from web_labeler.catalog import miro as _miro
from web_labeler.yolo import load_yolo_names_from_yaml

SCHEMA_VERSION = 4
_TABLES = {"venues", "models", "classes", "import_runs", "articles", "pos_identities", "map_entries",
           "map_keys", "class_links", "class_training", "class_rule_texts", "source_findings",
           "miro_pages", "miro_suggested", "miro_reviews", "miro_links"}

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
        return cls(db_path, venue_id, venue_name, Path(models_dir).resolve(), tuple(roots),
                   optional_path("catalog_pos_dir"), str(cfg.get("catalog_pos_adapter") or "").strip() or None,
                   optional_path("article_map_path"), optional_path("catalog_blacklist_path"),
                   int(cfg.get("catalog_active_days") or 30), baseline if isinstance(baseline, dict) else None,
                   rule_sources, optional_path("catalog_miro_dir"),
                   {str(k): str(v) for k, v in miro_files.items()} if isinstance(miro_files, dict) else None,
                   optional_path("catalog_miro_suggestions_path"), media_dir, dataset_dir)

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
            conn.executescript(f"BEGIN;{_SCHEMA}{_SCHEMA_V2}{_SCHEMA_V3}{_SCHEMA_V4}PRAGMA user_version = {SCHEMA_VERSION};COMMIT;")
        elif version in (1, 2, 3):
            steps = [_SCHEMA_V2, _SCHEMA_V3, _SCHEMA_V4][version - 1:]
            conn.executescript(f"BEGIN;{''.join(steps)}PRAGMA user_version = {SCHEMA_VERSION};COMMIT;")
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
    for table in ("class_training", "class_rule_texts", "source_findings"):
        conn.execute(f"DELETE FROM {table} WHERE venue_id = ?", (venue,))
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


def _variation(row: Sequence[Any], classes: Optional[List[str]] = None) -> Dict[str, Any]:
    article_id, code, name, state, sold, last_sale, not_verified, provisional, mode = row
    tags: List[str] = []
    if not_verified:
        tags.append("nije provereno na snimku")
    if provisional:
        tags.append("privremeno")
    if code is None:
        tags.append("nije prodavano")
    elif sold <= 0:
        tags.append("sve stornirano")
    return {"article_id": article_id, "code": code, "name": name.strip(), "state": state,
            "state_label": STATE_LABELS[state], "sold": sold, "last_sale": last_sale, "tags": tags,
            "provisional_reason": provisional, "mapping_mode": mode, "classes": classes or [],
            "image": None}


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


def _identity_ids(conn: sqlite3.Connection, venue: str, model: str, names: Sequence[str]) -> set:
    marks = ",".join("?" for _ in names)
    return {r[0] for r in conn.execute(f"SELECT i.identity_id {_LINKED_IDENTITIES}"
                                       f"WHERE l.model_key = ? AND l.class_name IN ({marks})",
                                       (venue, model.lower(), *names))}


def list_model_view(settings: CatalogSettings, model: str) -> Dict[str, Any]:
    """Classes as labelers draw them, how training treats them, and map classes with no home."""
    classes = list_classes(settings, model)
    venue = settings.venue_id
    with _open_for_read(settings) as conn:
        training = _training_rows(conn, venue, model)
        for item in classes:
            info = training[item["name"]]
            item["variation_count"] = len(_identity_ids(conn, venue, model, _class_scope(item["name"], training)))
            item["trained_name"] = info["trained_name"]
            item["training_state"] = info["state"]
            item["training_label"] = TRAINING_LABELS[info["state"]]
        known = set(training) | {i["trained_name"] for i in training.values() if i["trained_name"]}
        linked = [r[0] for r in conn.execute(f"SELECT DISTINCT l.class_name {_LINKED_IDENTITIES}WHERE l.model_key = ? "
                                             "ORDER BY l.class_name", (venue, model.lower()))]
        others = [{"name": n, "variation_count": len(_identity_ids(conn, venue, model, [n]))}
                  for n in linked if n not in known]
    return {"model": model, "classes": classes, "map_only_classes": others}


def list_variations(settings: CatalogSettings, model: str, class_name: str) -> List[Dict[str, Any]]:
    view = list_model_view(settings, model)
    if class_name not in {c["name"] for c in view["classes"]} | {c["name"] for c in view["map_only_classes"]}:
        raise UnknownModel(f"{model}.{class_name}")
    venue = settings.venue_id
    with _open_for_read(settings) as conn:
        scope = _class_scope(class_name, _training_rows(conn, venue, model))
        marks = ",".join("?" for _ in scope)
        rows = conn.execute(
            "SELECT DISTINCT i.identity_id, i.article_id, i.code, i.name, i.state, i.sold_qty, i.last_sale, i.not_verified, "
            f"e.provisional, e.mapping_mode, e.entry_id {_LINKED_IDENTITIES}"
            f"WHERE l.model_key = ? AND l.class_name IN ({marks}) ORDER BY i.sold_qty DESC, i.name, i.code",
            (venue, model.lower(), *scope)).fetchall()
        result = []
        for row in rows:
            links = conn.execute("SELECT model_key, class_name FROM class_links WHERE entry_id = ? ORDER BY position",
                                 (row[10],)).fetchall()
            item = _variation(row[1:10], [f"{m}.{c}" for m, c in links])
            own = {c for m, c in links if m == model.lower()}
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
    """POS pairs without a class, grouped by state, best sellers first."""
    with _open_for_read(settings) as conn:
        rows = conn.execute(
            "SELECT i.article_id, i.code, i.name, i.state, i.sold_qty, i.last_sale, i.not_verified, NULL, NULL "
            "FROM pos_identities i WHERE i.venue_id = ? AND i.code IS NOT NULL AND i.state != 'tracked' "
            "ORDER BY i.sold_qty DESC, i.name, i.code", (settings.venue_id,)).fetchall()
    groups = {state: [] for state in ("pending", "dormant", "ignored")}
    for row in rows:
        groups[row[3]].append(_variation(row))
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
