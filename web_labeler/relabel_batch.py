#!/usr/bin/env python3
"""
relabel_batch.py — MDQ-15c-2: export the owner's ``relabel`` decisions into a relabel batch.

A relabel batch is a small dataset (``images/`` + ``labels/`` + ``manifest.json`` + ``data.yaml``)
of COPIES of RAW images and their current label files, which Goca opens in the Dataset Fixer
("Load dataset") and corrects there. RAW is only ever READ here; the batch lives outside RAW:

    /opt/intellicup/datasets/relabel_batches/<pool>/<batch_id>/

(sibling of ``raw/`` and ``raw_review/``; ``batch_id`` = ``RELABEL_<pool>_<YYYYmmdd_HHMMSS>``).
A batch is written to a hidden ``.tmp_*`` folder and renamed into place, so an interrupted export
never leaves a half batch that the Dataset Fixer or a re-run could pick up.

Which entries: every sidecar entry with ``owner_decision == "relabel"`` (MDQ-15c-1) that is not
already in an OPEN batch (a manifest with ``"status": "open"``; MDQ-15c-3 closes it when applied).
Per entry, in this order:
  - ``forward``     a ``_forward/`` key (video frame, not in RAW yet) — nothing to relabel.
  - ``no_snapshot`` the entry carries no ``relabel_snapshot`` — cannot prove the original is unchanged.
  - ``in_open_batch`` the key, or another key with the same file name, is already in an open batch.
  - ``stale``       the RAW image/label no longer matches the ``relabel_snapshot`` (or is gone).
  - ``conflict``    the same file name exists in several class folders and the copies differ
                    (image bytes, or label content — e.g. class A in one folder, background in
                    another; a missing label file counts as an empty one).
  - ``name_clash``  two different file names with the same stem (labels are keyed by stem).
Skipped entries are reported by name and never exported.

The same image name may live in several class folders of RAW. ONE copy goes into the batch
(``images/<name>``, ``labels/<stem>.txt``; an empty label file if RAW has none) and the manifest
lists every RAW path. ``--execute`` copies with ``shutil.copy2`` (never a link: the Dataset Fixer's
Save overwrites label files in place) and re-checks the sha256 of every copy.

Manifest (``format_version`` 1)::

    {"format_version": 1, "batch_id": "...", "pool": "cups", "created_at": "<ISO>", "status": "open",
     "raw_base_path": "/opt/.../raw/blaznavac", "class_names": ["CAJ", ...],   # RAW <pool>/data.yaml order
     "items": [{
        "item_name": "x.jpg", "label_name": "x.txt",          # file names inside images/ and labels/
        "image_key": "CAJ/x.jpg",                             # primary key (first flagged key, sorted)
        "image_keys": ["CAJ/x.jpg", "background/x.jpg"],      # every flagged sidecar key of this item
        "raw_paths": [{"image": "cups/images/CAJ/x.jpg",      # relative to raw_base_path; every copy
                       "label": "cups/labels/CAJ/x.txt" | null}],
        "relabel_snapshot": {"image_sha256": "...", "label_sha256": "..." | null},   # original, at decision time
        "comment": "<flag comment(s), ' | ' joined>", "flag_source": "manual_goca",
        "flags": [{"image_key", "source", "category", "comment", "flagged_by", "flagged_at",
                   "owner_decision_at", "relabel_snapshot"}]}]}

``--dry-run`` (the default) prints exactly what would be exported and skipped and writes nothing;
``--execute`` writes the batch. Stdlib only (plus PyYAML for ``data.yaml``).

    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python web_labeler/relabel_batch.py --dry-run
    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python web_labeler/relabel_batch.py --pool cups --execute
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flag_store  # noqa: E402  (sibling module; see sys.path insert above)

FORMAT_VERSION = 1
STATUS_OPEN = "open"
MANIFEST = "manifest.json"
BATCH_PREFIX = "RELABEL_"
DEFAULT_RAW_BASE = Path("/opt/intellicup/datasets/raw/blaznavac")
DEFAULT_REVIEW_DIR = Path("/opt/intellicup/datasets/raw_review")
DEFAULT_BATCHES_DIR = Path("/opt/intellicup/datasets/relabel_batches")


class RelabelBatchError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_EMPTY_SHA = hashlib.sha256(b"").hexdigest()


@dataclass
class Skip:
    image_key: str
    reason: str
    detail: str = ""


@dataclass
class Plan:
    pool: str
    items: List[dict] = field(default_factory=list)
    skipped: List[Skip] = field(default_factory=list)
    class_names: Optional[List[str]] = None


def _inside(child: Path, parent: Path) -> bool:
    child, parent = child.resolve(), parent.resolve()
    return child == parent or parent in child.parents


def check_batches_dir(batches_dir: Path, raw_base: Path) -> None:
    """The batches folder must be neither inside RAW nor contain RAW."""
    if _inside(batches_dir, raw_base) or _inside(raw_base, batches_dir):
        raise RelabelBatchError(f"batches dir {batches_dir} must be outside the RAW tree {raw_base}")


def read_manifest(batch_dir: Path) -> dict:
    p = batch_dir / MANIFEST
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RelabelBatchError(f"{p} is unreadable ({e})") from e
    if not isinstance(doc, dict) or doc.get("format_version") != FORMAT_VERSION or not isinstance(doc.get("items"), list):
        raise RelabelBatchError(f"{p} has an unsupported manifest format")
    return doc


def list_batches(batches_dir: Path, pool: Optional[str] = None, only_open: bool = True) -> List[tuple]:
    """``[(pool, batch_id, batch_dir, manifest)]``; hidden ``.tmp_*`` folders are never batches."""
    out = []
    if not batches_dir.is_dir():
        return out
    for pdir in sorted(batches_dir.iterdir()):
        if not pdir.is_dir() or pdir.name.startswith(".") or (pool and pdir.name != pool):
            continue
        for bdir in sorted(pdir.iterdir()):
            if not bdir.is_dir() or bdir.name.startswith(".") or not (bdir / MANIFEST).is_file():
                continue
            man = read_manifest(bdir)
            if only_open and man.get("status") != STATUS_OPEN:
                continue
            out.append((pdir.name, bdir.name, bdir, man))
    return out


def resolve_batch(batches_dir: Path, pool: str, batch_id: str, raw_base: Optional[Path] = None) -> Path:
    """Validated path of one open batch, for the Dataset Fixer. Raises RelabelBatchError."""
    if pool not in flag_store.POOLS or not batch_id or batch_id.startswith(".") \
            or "/" in batch_id or "\\" in batch_id or batch_id in (".", ".."):
        raise RelabelBatchError("invalid relabel batch name")
    bdir = (batches_dir / pool / batch_id).resolve()
    if bdir.parent != (batches_dir / pool).resolve() or not (bdir / MANIFEST).is_file():
        raise RelabelBatchError("relabel batch not found")
    if raw_base is not None and (_inside(bdir, raw_base) or _inside(raw_base, bdir)):
        raise RelabelBatchError("relabel batch must never be inside the RAW tree")
    return bdir


FIXER_DECISIONS = "fixer_decisions.json"
FIXER_BACKGROUND = "background"
FIXER_DELETE = "delete"


def read_fixer_decisions(batch_dir: Path) -> Dict[str, str]:
    """Dataset Fixer marks of a relabel batch: ``{item_name: "background" | "delete"}`` ({} if none saved yet).

    Written by the Dataset Fixer's Save. ``apply_relabel.py`` reads it: an image left with zero boxes goes
    to ``background/`` ONLY when it is marked background here; otherwise (marked delete, or no choice made)
    every RAW copy of it is archived.
    """
    p = batch_dir / FIXER_DECISIONS
    if not p.is_file():
        return {}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RelabelBatchError(f"{p} is unreadable ({e})") from e
    items = doc.get("items") if isinstance(doc, dict) else None
    if not isinstance(items, dict) or any(v not in (FIXER_BACKGROUND, FIXER_DELETE) for v in items.values()):
        raise RelabelBatchError(f"{p} has an unsupported format")
    return {str(k): v for k, v in items.items()}


def write_fixer_decisions(batch_dir: Path, items: Dict[str, str]) -> None:
    if any(v not in (FIXER_BACKGROUND, FIXER_DELETE) for v in items.values()):
        raise RelabelBatchError("fixer decision must be 'background' or 'delete'")
    doc = {"format_version": 1, "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
           "items": dict(sorted(items.items()))}
    p = batch_dir / FIXER_DECISIONS
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def _raw_class_names(raw_base: Path, pool: str) -> List[str]:
    y = raw_base / pool / "data.yaml"
    if not y.is_file():
        raise RelabelBatchError(f"{y} is missing; cannot record the class order")
    try:
        import yaml  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RelabelBatchError("PyYAML is required (use the INTELLICUP_LABELING_TOOL interpreter)") from e
    names = (yaml.safe_load(y.read_text(encoding="utf-8")) or {}).get("names")
    if isinstance(names, dict):
        names = [v for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
    if not isinstance(names, list) or not names:
        raise RelabelBatchError(f"{y} has no usable 'names'")
    return [str(n) for n in names]


def _label_path(raw_base: Path, pool: str, cls: str, fname: str) -> Path:
    return raw_base / pool / "labels" / cls / f"{Path(fname).stem}.txt"


def _label_sha(p: Path) -> Optional[str]:
    return sha256_file(p) if p.is_file() else None


def plan_pool(pool: str, raw_base: Path, review_dir: Path, batches_dir: Path) -> Plan:
    """Read-only: decide what would be exported and what skipped."""
    pool = flag_store.normalize_pool(pool)
    plan = Plan(pool=pool)
    entries = flag_store.FlagStore(review_dir).list_entries(pool)
    relabel = {k: e for k, e in entries.items() if e.get("owner_decision") == flag_store.DECISION_RELABEL}
    if not relabel:
        return plan
    plan.class_names = _raw_class_names(raw_base, pool)

    open_keys, open_names = set(), set()
    for _p, _id, _d, man in list_batches(batches_dir, pool):
        for it in man["items"]:
            open_names.add(it.get("item_name"))
            open_keys.update(it.get("image_keys") or [it.get("image_key")])

    groups: Dict[str, List[str]] = {}
    for key in sorted(relabel):
        if key.startswith(flag_store.FORWARD_DIR + "/"):
            plan.skipped.append(Skip(key, "forward", "video frame not in RAW yet"))
            continue
        try:
            flag_store.validate_image_key(key)
        except flag_store.FlagError as e:
            plan.skipped.append(Skip(key, "conflict", f"malformed key: {e}"))
            continue
        groups.setdefault(key.split("/")[1], []).append(key)

    classes = sorted(p.name for p in (raw_base / pool / "images").iterdir() if p.is_dir()) \
        if (raw_base / pool / "images").is_dir() else []
    stems: Dict[str, str] = {}
    for fname, keys in sorted(groups.items()):
        def skip_all(reason: str, detail: str = "") -> None:
            plan.skipped.extend(Skip(k, reason, detail) for k in keys)

        if any(k in open_keys for k in keys) or fname in open_names:
            skip_all("in_open_batch", "already exported to an open relabel batch")
            continue
        snaps = {k: relabel[k].get("relabel_snapshot") for k in keys}
        if any(not isinstance(s, dict) or not s.get("image_sha256") for s in snaps.values()):
            skip_all("no_snapshot", "entry has no relabel_snapshot")
            continue

        stale = None
        for k in keys:
            cls = k.split("/")[0]
            img = raw_base / pool / "images" / cls / fname
            if not img.is_file():
                stale = f"{k}: RAW image is gone"
                break
            if sha256_file(img) != snaps[k]["image_sha256"]:
                stale = f"{k}: RAW image changed since the decision"
                break
            if _label_sha(_label_path(raw_base, pool, cls, fname)) != snaps[k].get("label_sha256"):
                stale = f"{k}: RAW label changed since the decision"
                break
        if stale:
            skip_all("stale", stale)
            continue

        copies = []          # every RAW copy of this file name, across class folders
        for cls in classes:
            img = raw_base / pool / "images" / cls / fname
            if img.is_file():
                lab = _label_path(raw_base, pool, cls, fname)
                copies.append({"cls": cls, "img": img, "lab": lab if lab.is_file() else None})
        img_shas = {sha256_file(c["img"]) for c in copies}
        lab_shas = {sha256_file(c["lab"]) if c["lab"] else _EMPTY_SHA for c in copies}
        if len(img_shas) > 1 or len(lab_shas) > 1:
            what = "image bytes" if len(img_shas) > 1 else "label content"
            skip_all("conflict", f"copies in {', '.join(c['cls'] for c in copies)} differ in {what}")
            continue
        stem = Path(fname).stem
        if stem in stems and stems[stem] != fname:
            skip_all("name_clash", f"same stem as {stems[stem]}")
            continue
        stems[stem] = fname

        prim = keys[0]
        comments = []
        for k in keys:
            c = (relabel[k].get("comment") or "").strip()
            if c and c not in comments:
                comments.append(c)
        plan.items.append({
            "item_name": fname,
            "label_name": f"{stem}.txt",
            "image_key": prim,
            "image_keys": list(keys),
            "raw_paths": [{"image": f"{pool}/images/{c['cls']}/{fname}",
                           "label": f"{pool}/labels/{c['cls']}/{stem}.txt" if c["lab"] else None}
                          for c in copies],
            "relabel_snapshot": {"image_sha256": snaps[prim]["image_sha256"],
                                 "label_sha256": snaps[prim].get("label_sha256")},
            "comment": " | ".join(comments),
            "flag_source": relabel[prim].get("source"),
            "flags": [{"image_key": k, "source": relabel[k].get("source"), "category": relabel[k].get("category"),
                       "comment": relabel[k].get("comment") or "", "flagged_by": relabel[k].get("flagged_by"),
                       "flagged_at": relabel[k].get("flagged_at"),
                       "owner_decision_at": relabel[k].get("owner_decision_at"),
                       "relabel_snapshot": snaps[k]} for k in keys],
            "_copy": next(c for c in copies if c["cls"] == prim.split("/")[0]),
        })
    return plan


def write_batch(plan: Plan, raw_base: Path, batches_dir: Path, now: Optional[datetime] = None) -> Optional[Path]:
    """Copy the planned items into a new batch folder. Returns it, or None if nothing to export."""
    if not plan.items:
        return None
    check_batches_dir(batches_dir, raw_base)
    now = now or datetime.now()
    pool_dir = batches_dir / plan.pool
    pool_dir.mkdir(parents=True, exist_ok=True)
    batch_id = f"{BATCH_PREFIX}{plan.pool}_{now.strftime('%Y%m%d_%H%M%S')}"
    n = 1
    while (pool_dir / batch_id).exists():
        n += 1
        batch_id = f"{BATCH_PREFIX}{plan.pool}_{now.strftime('%Y%m%d_%H%M%S')}_{n}"
    tmp = pool_dir / f".tmp_{batch_id}_{os.getpid()}"
    (tmp / "images").mkdir(parents=True)
    (tmp / "labels").mkdir()
    try:
        items = []
        for it in plan.items:
            c = it["_copy"]
            dst_img = tmp / "images" / it["item_name"]
            dst_lab = tmp / "labels" / it["label_name"]
            shutil.copy2(c["img"], dst_img)                 # a real copy, never a link
            if c["lab"]:
                shutil.copy2(c["lab"], dst_lab)
            else:
                dst_lab.write_bytes(b"")
            if sha256_file(dst_img) != it["relabel_snapshot"]["image_sha256"]:
                raise RelabelBatchError(f"copy of {it['item_name']} does not match the original")
            if sha256_file(dst_lab) != (it["relabel_snapshot"]["label_sha256"] or _EMPTY_SHA):
                raise RelabelBatchError(f"label copy of {it['item_name']} does not match the original")
            items.append({k: v for k, v in it.items() if k != "_copy"})
        manifest = {
            "format_version": FORMAT_VERSION, "batch_id": batch_id, "pool": plan.pool,
            "created_at": now.astimezone().isoformat(timespec="seconds"), "status": STATUS_OPEN,
            "raw_base_path": str(raw_base), "class_names": plan.class_names, "items": items,
        }
        (tmp / "data.yaml").write_text(
            f"nc: {len(plan.class_names)}\nnames: {json.dumps(plan.class_names, ensure_ascii=False)}\n", encoding="utf-8")
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        final = pool_dir / batch_id
        os.rename(tmp, final)
        return final
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MDQ-15c-2: export owner 'relabel' decisions into a relabel batch.")
    ap.add_argument("--pool", action="append", choices=flag_store.POOLS, help="limit to a pool (repeatable)")
    ap.add_argument("--raw-base-path", type=Path, default=DEFAULT_RAW_BASE)
    ap.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    ap.add_argument("--batches-dir", type=Path, default=DEFAULT_BATCHES_DIR)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print what would be exported/skipped; write nothing (default)")
    mode.add_argument("--execute", action="store_true", help="actually write the batch(es)")
    args = ap.parse_args(argv)

    try:
        check_batches_dir(args.batches_dir, args.raw_base_path)
        plans = [plan_pool(p, args.raw_base_path, args.review_dir, args.batches_dir) for p in (args.pool or flag_store.POOLS)]
    except (RelabelBatchError, flag_store.FlagError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"{'EXECUTE' if args.execute else 'DRY RUN (nothing is written)'} — raw={args.raw_base_path} batches={args.batches_dir}")
    total = 0
    for pl in plans:
        print(f"\n[{pl.pool}] would export {len(pl.items)} item(s), skip {len(pl.skipped)}")
        for it in pl.items:
            print(f"  + {it['image_key']}  copies={len(it['raw_paths'])}  comment={it['comment']!r}")
        for s in pl.skipped:
            label = {"stale": "zastarela (stale)", "conflict": "konflikt (conflict)"}.get(s.reason, s.reason)
            print(f"  - {s.image_key}  [{label}] {s.detail}")
        total += len(pl.items)
        if args.execute:
            try:
                out = write_batch(pl, args.raw_base_path, args.batches_dir)
            except (RelabelBatchError, OSError) as e:
                print(f"ERROR: {pl.pool}: {e}", file=sys.stderr)
                return 1
            if out:
                print(f"  => wrote {out}")
    print(f"\ntotal items {'exported' if args.execute else 'that would be exported'}: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
