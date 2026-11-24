"""Nightly copy of the Catalog (database and its images) to another disk, kept 30 days, and how to bring it back.

The database is the only source of truth for decisions and pasted images, so a copy must be checked
before it counts: every copy is opened and passes SQLite's integrity check.
"""
from __future__ import annotations

import os
import re
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

_NAME = re.compile(r"^catalog_(?P<venue>[A-Za-z0-9_-]+)_(?P<day>\d{4}-\d{2}-\d{2})\.db$")


class BackupError(Exception):
    pass


def check_database(path: Path) -> int:
    """Open read-only, run the integrity check, return the schema version."""
    try:
        conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()[0]
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error as e:
        raise BackupError(f"{path.name}: cannot be read ({e})")
    if result != "ok":
        raise BackupError(f"{path.name}: integrity check failed ({result})")
    if version <= 0:
        raise BackupError(f"{path.name}: not a Catalog database")
    return version


def list_backups(dest_dir: Path, venue: str) -> List[Dict[str, str]]:
    found = []
    if dest_dir.is_dir():
        for p in dest_dir.iterdir():
            m = _NAME.match(p.name)
            if m and m.group("venue") == venue:
                found.append({"day": m.group("day"), "path": str(p)})
    return sorted(found, key=lambda b: b["day"])


def _copy_media(media_dir: Optional[Path], target_root: Path) -> int:
    """Images are stored by content and never change, so only missing files are copied."""
    if media_dir is None or not media_dir.is_dir():
        return 0
    copied = 0
    for src in media_dir.rglob("*"):
        if not src.is_file() or src.suffix == ".part":
            continue
        target = target_root / src.relative_to(media_dir)
        if target.is_file() and target.stat().st_size == src.stat().st_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        shutil.copy2(src, tmp)
        os.replace(tmp, target)
        copied += 1
    return copied


def on_mounted_disk(path: Path) -> bool:
    """True if path lies on a mounted disk other than the system root (a removable disk that is plugged in)."""
    current = Path(path).resolve()
    while True:
        if current.exists() and os.path.ismount(current) and current != Path("/"):
            return True
        if current.parent == current:
            return False
        current = current.parent


def backup(db_path: Path, dest_dir: Path, venue: str, media_dir: Optional[Path] = None, keep_days: int = 30,
           today: Optional[datetime] = None, require_mount: bool = False) -> Dict[str, object]:
    if not db_path.is_file():
        raise BackupError(f"Catalog database does not exist: {db_path}")
    if require_mount and not on_mounted_disk(dest_dir):
        raise BackupError(f"Backup disk is not plugged in: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    day = (today or datetime.now()).strftime("%Y-%m-%d")
    target = dest_dir / f"catalog_{venue}_{day}.db"
    tmp = target.with_name(target.name + ".part")
    if tmp.exists():
        tmp.unlink()
    src = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        out = sqlite3.connect(str(tmp))
        try:
            src.backup(out)
        finally:
            out.close()
    finally:
        src.close()
    version = check_database(tmp)
    os.replace(tmp, target)
    copied = _copy_media(media_dir, dest_dir / "media")
    cutoff = ((today or datetime.now()) - timedelta(days=keep_days)).strftime("%Y-%m-%d")
    removed = []
    for b in list_backups(dest_dir, venue):
        if b["day"] < cutoff:
            Path(b["path"]).unlink()
            removed.append(b["day"])
    return {"backup": str(target), "schema_version": version, "media_copied": copied, "removed": removed,
            "kept": [b["day"] for b in list_backups(dest_dir, venue)]}


def restore(backup_file: Path, db_path: Path, dest_media: Optional[Path] = None,
            media_dir: Optional[Path] = None) -> Dict[str, object]:
    """Put a checked copy back. The current database is kept next to it, never deleted. Stop the tool first."""
    version = check_database(backup_file)
    kept = None
    if db_path.exists():
        kept = db_path.with_name(f"{db_path.name}.before_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.replace(db_path, kept)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(db_path.name + ".part")
    shutil.copy2(backup_file, tmp)
    os.replace(tmp, db_path)
    copied = _copy_media(dest_media, media_dir) if dest_media and media_dir else 0
    return {"restored": str(db_path), "from": str(backup_file), "schema_version": version,
            "previous_kept_as": str(kept) if kept else None, "media_copied": copied}
