"""Carry manual work from one Catalog database into another (e.g. the :8002 trial database into the real one).

What moves: decisions (with revokes, rejects and accepted proposals kept linked), pasted images with their history
and files, and the admin's Miro page reviews. What never moves: releases and locks (the real Catalog publishes its
own), anything by a skipped person (test entries), and imported data (the target imports the sources itself).

POS variations are matched by code + name, articles through one of their variations, Miro pages by page key.
The target must be imported and still without manual work, so a transfer is done once and cannot double anything.
Original authors, times and "valid from" dates are kept.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from web_labeler.catalog import store


class TransferError(Exception):
    pass


def _image_path(media_root: Path, sha256: str, media_type: str) -> Path:
    ext = next(e for t, e in store._EVIDENCE_FORMATS.values() if t == media_type)
    return Path(media_root) / "evidence" / f"{sha256}.{ext}"


def _identity_keys(conn: sqlite3.Connection, venue: str) -> Tuple[Dict[int, Tuple[str, str]], Dict[Tuple[str, str], int],
                                                                    Dict[int, int], Dict[int, List[int]]]:
    by_id, by_key, article_of, members = {}, {}, {}, {}
    for identity_id, article_id, code, name in conn.execute(
            "SELECT identity_id, article_id, IFNULL(code, ''), name FROM pos_identities WHERE venue_id = ?", (venue,)):
        by_id[identity_id] = (code, name)
        by_key[(code, name)] = identity_id
        article_of[identity_id] = article_id
        members.setdefault(article_id, []).append(identity_id)
    return by_id, by_key, article_of, members


def transfer(source_db: Path, settings: store.CatalogSettings, source_media: Optional[Path] = None,
             skip_people: Sequence[str] = (), dry_run: bool = False) -> Dict[str, Any]:
    venue = settings.venue_id
    if not Path(source_db).is_file():
        raise TransferError(f"Izvorna baza ne postoji: {source_db}")
    if settings.db_path and Path(source_db).resolve() == settings.db_path.resolve():
        raise TransferError("Izvor i cilj su ista baza.")
    src = sqlite3.connect(Path(source_db).resolve().as_uri() + "?mode=ro", uri=True)
    skip = {p.strip() for p in skip_people if p.strip()}
    report: Dict[str, Any] = {"decisions": 0, "images": 0, "miro_pages": 0, "skipped": [], "not_found": []}
    conn = store._open_for_write(settings)
    try:
        conn.execute("BEGIN IMMEDIATE")
        busy = [t for t in ("decisions", "evidence", "miro_reviews")
                if conn.execute(f"SELECT COUNT(*) FROM {t} WHERE venue_id = ?", (venue,)).fetchone()[0]]
        if busy:
            raise TransferError("Ciljna baza već ima ručni rad (" + ", ".join(busy) + "). Prenos ide samo u novu bazu.")
        if not conn.execute("SELECT COUNT(*) FROM pos_identities WHERE venue_id = ?", (venue,)).fetchone()[0]:
            raise TransferError("Ciljna baza još nema uvoz. Prvo pokreni import.")
        s_by_id, _, s_article_of, s_members = _identity_keys(src, venue)
        _, t_by_key, t_article_of, _ = _identity_keys(conn, venue)

        def target_identity(identity_id: Optional[int]) -> Optional[int]:
            return None if identity_id is None else t_by_key.get(s_by_id.get(identity_id, ("", "")))

        def target_article(article_id: Optional[int]) -> Optional[int]:
            for identity_id in s_members.get(article_id, []):
                t = target_identity(identity_id)
                if t is not None:
                    return t_article_of[t]
            return None

        def label(identity_id: Optional[int], article_id: Optional[int]) -> str:
            ids = [identity_id] if identity_id else s_members.get(article_id, [])
            return ", ".join(f"{s_by_id[i][1]} ({s_by_id[i][0] or 'bez šifre'})" for i in ids[:2] if i in s_by_id) or "?"

        # Decisions, in their original order so links to earlier decisions can be renumbered.
        id_map: Dict[int, int] = {}
        rows = src.execute("SELECT decision_id, kind, article_id, identity_id, target_id, payload, reason, evidence, actor, role, "
                           "created_at, valid_from FROM decisions WHERE venue_id = ? ORDER BY decision_id", (venue,)).fetchall()
        for d_id, kind, article_id, identity_id, target_id, payload, reason, evidence, actor, role, created_at, valid_from in rows:
            data = json.loads(payload)
            if actor in skip:
                report["skipped"].append(f"odluka #{d_id} ({actor})")
                continue
            if kind == "class_target":  # it opened a class in the trial tool's lists; open it again in the real tool
                report["skipped"].append(f"odluka #{d_id}: klasa za novi artikal se ne prenosi, otvori je ponovo")
                continue
            if target_id is not None and target_id not in id_map:
                report["skipped"].append(f"odluka #{d_id}: odnosi se na odluku koja se ne prenosi")
                continue
            if "from_proposal" in data and data["from_proposal"] not in id_map:
                data.pop("from_proposal")
            elif "from_proposal" in data:
                data["from_proposal"] = id_map[data["from_proposal"]]
            new_identity = target_identity(identity_id)
            new_article = target_article(article_id) if article_id is not None else None
            if (identity_id is not None and new_identity is None) or (article_id is not None and new_article is None):
                report["not_found"].append(f"odluka #{d_id}: {label(identity_id, article_id)}")
                continue
            if kind == "merge" or (kind == "propose" and data.get("kind") == "merge"):
                change = data if kind == "merge" else data.get("change", {})
                into = target_article(change.get("into"))
                if into is None:
                    report["not_found"].append(f"odluka #{d_id}: artikal za spajanje")
                    continue
                change["into"] = into
            new_id = conn.execute(
                "INSERT INTO decisions (venue_id, kind, article_id, identity_id, target_id, payload, reason, evidence, actor, role, "
                "created_at, valid_from) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (venue, kind, new_article, new_identity, id_map.get(target_id) if target_id is not None else None,
                 json.dumps(data, ensure_ascii=False, sort_keys=True), reason, evidence, actor, role, created_at, valid_from)).lastrowid
            id_map[d_id] = new_id
            report["decisions"] += 1

        # Pasted images and their history; the files move by checksum.
        media_from = Path(source_media) if source_media else Path(source_db).resolve().parent / "media"
        rows = src.execute("SELECT evidence_id, identity_id, model_key, class_name, sha256, media_type, width, height, size_bytes, "
                           "source, added_by, added_role, added_at, status, decided_by, decided_at, reason FROM evidence "
                           "WHERE venue_id = ? ORDER BY evidence_id", (venue,)).fetchall()
        files: List[Tuple[Path, Path]] = []
        for row in rows:
            e_id, identity_id, sha, media_type, added_by = row[0], row[1], row[4], row[5], row[10]
            if added_by in skip:
                report["skipped"].append(f"slika #{e_id} ({added_by})")
                continue
            new_identity = target_identity(identity_id)
            if new_identity is None:
                report["not_found"].append(f"slika #{e_id}: {label(identity_id, None)}")
                continue
            source_file = _image_path(media_from, sha, media_type)
            if not source_file.is_file():
                report["not_found"].append(f"slika #{e_id}: fajl {source_file.name} ne postoji")
                continue
            new_id = conn.execute(
                "INSERT INTO evidence (venue_id, identity_id, article_id, model_key, class_name, sha256, media_type, width, height, "
                "size_bytes, source, added_by, added_role, added_at, status, decided_by, decided_at, reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (venue, new_identity, t_article_of[new_identity]) + tuple(row[2:])).lastrowid
            for at, actor, role, action, note in src.execute(
                    "SELECT at, actor, role, action, note FROM evidence_events WHERE venue_id = ? AND evidence_id = ? ORDER BY event_id",
                    (venue, e_id)):
                conn.execute("INSERT INTO evidence_events (venue_id, evidence_id, at, actor, role, action, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (venue, new_id, at, actor, role, action, note))
            files.append((source_file, _image_path(settings.media_dir, sha, media_type)))
            report["images"] += 1

        # Admin's Miro page reviews.
        known_pages = {k for k, in conn.execute("SELECT page_key FROM miro_pages WHERE venue_id = ?", (venue,))}
        for page_key, decided_at in src.execute("SELECT page_key, decided_at FROM miro_reviews WHERE venue_id = ?", (venue,)):
            if page_key not in known_pages:
                report["not_found"].append(f"Miro strana {page_key}")
                continue
            conn.execute("INSERT INTO miro_reviews (venue_id, page_key, decided_at) VALUES (?, ?, ?)", (venue, page_key, decided_at))
            conn.executemany("INSERT INTO miro_links (venue_id, page_key, model_key, class_name) VALUES (?, ?, ?, ?)",
                             [(venue, page_key, m, c) for m, c in src.execute(
                                 "SELECT model_key, class_name FROM miro_links WHERE venue_id = ? AND page_key = ?", (venue, page_key))])
            report["miro_pages"] += 1

        if dry_run:
            conn.execute("ROLLBACK")
            report["dry_run"] = True
            return report
        for a, b in files:
            b.parent.mkdir(parents=True, exist_ok=True)
            if not b.exists():
                shutil.copy2(a, b)
        conn.execute("COMMIT")
        return report
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
        src.close()
