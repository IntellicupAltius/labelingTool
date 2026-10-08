#!/usr/bin/env python3
"""
apply_relabel.py — MDQ-15c-3: apply a corrected relabel batch (MDQ-15c-2) to RAW.

Separate script on purpose: ``archive_raw.py`` (MDQ-8) is untouched. This one reuses its archive
mechanism (same ``raw_archive/<stamp>/<pool>/{images,labels}/<class>/`` layout, same
``_unique_dest`` no-overwrite rule, move never delete) and its path defaults.

Input: an OPEN batch ``/opt/intellicup/datasets/relabel_batches/<pool>/<batch_id>/`` (``manifest.json``,
``images/``, ``labels/``) whose label copies Goca corrected in the Dataset Fixer. Default is a
DRY RUN: the plan is printed per item and nothing is changed. ``--apply`` really does it.

Per item, in this order (the first failing check skips the item and reports it by name):
  1. already done      ``apply_state.json`` of the batch says applied/unchanged -> nothing is done twice
                       (a sidecar mark that is still missing is repaired).
  2. corrected copy    ``labels/<stem>.txt`` exists, parses, class ids exist in RAW ``data.yaml``, coords in
                       [0,1]; ``images/<name>`` sha256 == the snapshot's ``image_sha256`` (image not edited).
  3. original current  every RAW copy of the file name (all class folders, also new ones) still equals the
                       snapshot (image sha256; label sha256, a missing label counts as empty) and the set
                       of copies equals the manifest's ``raw_paths``; else ``stale`` (zastarela).
  3b. zero boxes      the corrected label has NO box: if the Dataset Fixer marked it "background"
                       (``fixer_decisions.json`` in the batch) it goes on as a normal item (into ``background/``);
                       otherwise (marked "delete", or NO choice made) it is a DELETE: step 5 archives every copy,
                       nothing is inserted, sidecar ``relabel_result.status = "deleted"``. Delete is the default
                       on purpose: a skipped image may still show drinks and must never silently become background.
                       A "delete"/"background" mark on an image that still has boxes is ``invalid`` (skipped).
  4. not corrected     Dataset Fixer Save re-quantizes ALL labels (~1 px drift), so labels are compared by
                       BOXES: same number of boxes, same class per box, every corner (x1,y1,x2,y2 in
                       pixels) within ``--box-tol-px`` (default 2.0). No difference over the tolerance ->
                       NOT applied, reported "nije ispravljeno", sidecar marked ``unchanged``.
  5. archive           EVERY copy of the original (image + label, same name in all class folders) is MOVED
                       into ``raw_archive/<stamp>_relabel/`` (never deleted), BEFORE ingest: ingest
                       silently drops an image whose name already exists.
  6. insert            the corrected image + label go in through the ingest distributor
                       (``ingestion_pipeline.distributor.distribute_samples``), i.e. into the class folder of
                       every class in the corrected label (``background`` if it has none), the same
                       label in all of them. That is the RAW invariant "all copies of an image carry all its
                       labels"; if the correction adds/removes a class the folders therefore differ from the
                       original ones (reported in the plan).
  7. verify            every target folder really holds the image (sha256 == snapshot) and the label
                       (sha256 == corrected label), and no unexpected copy remains. Else step 8.
  8. rollback          any failure in 5-7 puts the item back: inserted files are moved to
                       ``<archive>/_rolled_back/``, the originals moved back. Reported.
After success: ``apply_state.json`` in the batch, the apply manifest ``<archive>/apply_manifest.json``
(archived paths, inserted paths, sha256), sidecar ``relabel_result`` (via FlagStore). The batch manifest
``status`` becomes ``applied`` once every item is resolved; otherwise it stays ``open``.

Rollback of a whole run:  ``--rollback <stamp>`` (dry-run unless ``--apply``): moves the inserted files
away, restores the archived originals, clears the sidecar marks, re-opens the batch.

    python web_labeler/apply_relabel.py                      # dry run, every open batch
    python web_labeler/apply_relabel.py --batch shots/RELABEL_shots_20261005_101500 --apply
    python web_labeler/apply_relabel.py --rollback 20261005_110000_relabel --apply

Run with /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python (needs PyYAML + Pillow).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive_raw  # noqa: E402  (MDQ-8 archive mechanism: layout, _unique_dest, defaults)
import flag_store  # noqa: E402
import relabel_batch  # noqa: E402  (manifest format of 15c-2)

try:  # POSIX only
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore

DEFAULT_INGEST_ROOT = Path(__file__).resolve().parents[2] / "IntelliCup"
DEFAULT_BOX_TOL_PX = 2.0
APPLY_STATE = "apply_state.json"
APPLY_MANIFEST = "apply_manifest.json"
STATUS_APPLIED = "applied"
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


class ApplyError(Exception):
    pass


def sha256_file(p: Path) -> str:
    return relabel_batch.sha256_file(p)


def _atomic_json(path: Path, doc: dict) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_ingest(ingest_root: Path):
    """Import the ingest distributor/utils (IntelliCup repo); RAW_BASE is patched by the caller."""
    if str(ingest_root) not in sys.path:
        sys.path.insert(0, str(ingest_root))
    try:
        import ingestion_pipeline.distributor as dist  # type: ignore
        import ingestion_pipeline.utils as iutils  # type: ignore
    except ImportError as e:
        raise ApplyError(f"cannot import the ingest pipeline from {ingest_root}: {e}") from e
    return dist, iutils


# --------------------------------------------------------------------------- label comparison

def read_label_boxes(path: Path, iutils) -> Optional[List[dict]]:
    """Parsed YOLO boxes; ``[]`` for an empty file; None when unparsable."""
    return iutils.parse_yolo_label(iutils.read_label(path))


def _corners(b: dict, w: int, h: int) -> Tuple[float, float, float, float]:
    return ((b["x"] - b["w"] / 2) * w, (b["y"] - b["h"] / 2) * h,
            (b["x"] + b["w"] / 2) * w, (b["y"] + b["h"] / 2) * h)


def boxes_equal(a: List[dict], b: List[dict], img_wh: Tuple[int, int], tol_px: float) -> bool:
    """Same boxes within ``tol_px`` (max abs difference of any corner, in pixels), class must match."""
    if len(a) != len(b):
        return False
    w, h = img_wh
    left = list(b)
    for box in sorted(a, key=lambda x: (x["class_id"], x["x"], x["y"])):
        ca = _corners(box, w, h)
        best, best_d = None, None
        for cand in left:
            if cand["class_id"] != box["class_id"]:
                continue
            d = max(abs(p - q) for p, q in zip(ca, _corners(cand, w, h)))
            if best_d is None or d < best_d:
                best, best_d = cand, d
        if best is None or best_d > tol_px:
            return False
        left.remove(best)
    return True


# --------------------------------------------------------------------------- planning

@dataclass
class ItemPlan:
    pool: str
    batch_id: str
    item: dict
    action: str                      # apply | delete | unchanged | done | stale | invalid | missing_corrected
    detail: str = ""
    copies: List[dict] = field(default_factory=list)   # [{"cls","img","lab"}] current RAW copies
    corrected_img: Optional[Path] = None
    corrected_lab: Optional[Path] = None
    corrected_boxes: Optional[List[dict]] = None
    target_classes: List[str] = field(default_factory=list)
    marked: bool = True              # sidecar already marked (for "done" repair)


def _copies(raw_base: Path, pool: str, fname: str) -> List[dict]:
    out = []
    imgs = raw_base / pool / "images"
    if not imgs.is_dir():
        return out
    for cdir in sorted(p for p in imgs.iterdir() if p.is_dir()):
        img = cdir / fname
        if img.is_file():
            lab = raw_base / pool / "labels" / cdir.name / f"{Path(fname).stem}.txt"
            out.append({"cls": cdir.name, "img": img, "lab": lab if lab.is_file() else None})
    return out


def plan_item(pool: str, bdir: Path, man: dict, item: dict, state: dict, store: Optional[flag_store.FlagStore],
              raw_base: Path, class_names: List[str], iutils, tol_px: float,
              decisions: Optional[Dict[str, str]] = None) -> ItemPlan:
    from PIL import Image  # local import: only needed here
    name = item["item_name"]
    p = ItemPlan(pool, man["batch_id"], item, "invalid")
    snap = item.get("relabel_snapshot") or {}

    done = state.get(name)
    if done:
        p.action = "done"
        p.detail = f"already {done.get('status')} in run {done.get('run')}"
        if store is not None:
            for k in item.get("image_keys") or [item["image_key"]]:
                ent = store.get(pool, k)
                if ent is not None and not ent.get("relabel_result"):
                    p.marked = False
        return p

    c_img = bdir / "images" / name
    c_lab = bdir / "labels" / item["label_name"]
    if not c_img.is_file() or not c_lab.is_file():
        p.action = "missing_corrected"
        p.detail = f"corrected copy missing ({'image' if not c_img.is_file() else 'label'} not in the batch)"
        return p
    if not snap.get("image_sha256"):
        p.detail = "manifest item has no relabel_snapshot"
        return p
    if sha256_file(c_img) != snap["image_sha256"]:
        p.detail = "corrected image differs from the snapshot (image was edited or replaced)"
        return p
    boxes = read_label_boxes(c_lab, iutils)
    if boxes is None:
        p.detail = "corrected label does not parse"
        return p
    if not iutils.is_bbox_in_range(boxes):
        p.detail = "corrected label has coordinates outside [0,1]"
        return p
    if not iutils.validate_class_ids(boxes, {"nc": len(class_names)}):
        p.detail = "corrected label has a class id that is not in RAW data.yaml"
        return p

    copies = _copies(raw_base, pool, name)
    want = sorted(rp["image"] for rp in item.get("raw_paths") or [])
    have = sorted(f"{pool}/images/{c['cls']}/{name}" for c in copies)
    p.copies = copies
    if not copies or want != have:
        p.action = "stale"
        p.detail = f"RAW copies changed since the export (manifest {want}, now {have})"
        return p
    want_lab = snap.get("label_sha256") or EMPTY_SHA
    for c in copies:
        if sha256_file(c["img"]) != snap["image_sha256"]:
            p.action, p.detail = "stale", f"{c['cls']}: RAW image changed since the decision"
            return p
        if (sha256_file(c["lab"]) if c["lab"] else EMPTY_SHA) != want_lab:
            p.action, p.detail = "stale", f"{c['cls']}: RAW label changed since the decision"
            return p

    dec = (decisions or {}).get(name)
    if boxes and dec in (relabel_batch.FIXER_DELETE, relabel_batch.FIXER_BACKGROUND):
        p.detail = f"marked {dec!r} in the Dataset Fixer but the label still has {len(boxes)} box(es)"
        return p
    if not boxes and dec != relabel_batch.FIXER_BACKGROUND:
        p.corrected_img, p.corrected_lab, p.corrected_boxes = c_img, c_lab, boxes
        p.action = "delete"
        p.detail = ("marked Delete" if dec == relabel_batch.FIXER_DELETE
                    else "0 boxes and no background mark -> Delete (default)")
        return p

    orig_lab = next((c["lab"] for c in copies if c["lab"]), None)
    orig_boxes = read_label_boxes(orig_lab, iutils) if orig_lab else []
    if orig_boxes is None:
        p.action, p.detail = "stale", "original RAW label does not parse"
        return p
    with Image.open(c_img) as im:
        wh = im.size
    p.corrected_img, p.corrected_lab, p.corrected_boxes = c_img, c_lab, boxes
    if boxes_equal(orig_boxes, boxes, wh, tol_px):
        p.action = "unchanged"
        p.detail = f"nije ispravljeno: boxes equal within {tol_px} px"
        return p
    p.target_classes = [class_names[i] for i in sorted({b["class_id"] for b in boxes})] or ["background"]
    p.action = "apply"
    return p


# --------------------------------------------------------------------------- execution

def _move(src: Path, dst: Path) -> Path:
    dst = archive_raw._unique_dest(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def apply_item(p: ItemPlan, raw_base: Path, archive_dir: Path, dist, class_names: List[str],
               fail_after: Optional[str] = None) -> dict:
    """Archive -> ingest -> verify, transactional. Returns the manifest record; raises ApplyError after rolling back.

    ``fail_after`` ("archive" | "ingest" | "verify") is a test hook that raises at that step.
    """
    pool, name, item = p.pool, p.item["item_name"], p.item
    stem = Path(name).stem
    record = {"item_name": name, "image_keys": item.get("image_keys") or [item["image_key"]],
              "image_sha256": item["relabel_snapshot"]["image_sha256"],
              "corrected_label_sha256": sha256_file(p.corrected_lab),
              "archived": [], "inserted": [], "ingest_index": None, "target_classes": p.target_classes}
    archived: List[Tuple[Path, Path]] = []   # (original path, archive path)
    inserted: List[Path] = []
    index_file: Optional[Path] = None
    batch_name = f"{p.batch_id}__apply__{archive_dir.name}__{stem}"
    try:
        for c in p.copies:
            for src, kind in ((c["img"], "images"), (c["lab"], "labels")):
                if src is None:
                    continue
                dst = _move(src, archive_dir / pool / kind / c["cls"] / src.name)
                archived.append((src, dst))
                record["archived"].append({"from": str(src), "to": str(dst), "sha256": sha256_file(dst)})
        if fail_after == "archive":
            raise ApplyError("test hook: failure after archive")
        if p.action == "delete":
            left = [c["cls"] for c in _copies(raw_base, pool, name)]
            if left:
                raise ApplyError(f"copies remain after archiving in {left}")
            if fail_after == "verify":
                raise ApplyError("test hook: failure after verify")
            return record

        yaml_data = {"nc": len(class_names), "names": class_names}
        dist.RAW_BASE = str(raw_base.parent)
        index_file = raw_base / "_ingest_index" / f"{batch_name}.json"
        sample = (p.corrected_img, p.corrected_lab, p.corrected_boxes)
        stats = dist.distribute_samples([sample], raw_base.name, pool, yaml_data, batch_name)
        for cls in p.target_classes:
            inserted += [raw_base / pool / "images" / cls / name, raw_base / pool / "labels" / cls / f"{stem}.txt"]
        if fail_after == "ingest":
            raise ApplyError("test hook: failure after ingest")

        # The ingest drops a same-name file silently (conflicts_existing_diff): check that it really landed.
        if stats.get("raw_files_conflicts_existing_diff"):
            raise ApplyError(f"ingest reported conflicts: {stats.get('raw_files_conflict_examples')}")
        want_img, want_lab = record["image_sha256"], record["corrected_label_sha256"]
        for cls in p.target_classes:
            ip = raw_base / pool / "images" / cls / name
            lp = raw_base / pool / "labels" / cls / f"{stem}.txt"
            if not ip.is_file() or sha256_file(ip) != want_img:
                raise ApplyError(f"after ingest the image is not in {cls}/ (or differs): {ip}")
            if not lp.is_file() or sha256_file(lp) != want_lab:
                raise ApplyError(f"after ingest the label is not in {cls}/ (or differs): {lp}")
            record["inserted"].append({"path": str(ip), "sha256": want_img})
            record["inserted"].append({"path": str(lp), "sha256": want_lab})
        stray = [c["cls"] for c in _copies(raw_base, pool, name) if c["cls"] not in p.target_classes]
        if stray:
            raise ApplyError(f"unexpected copies remain in {stray}")
        if fail_after == "verify":
            raise ApplyError("test hook: failure after verify")
        record["ingest_index"] = str(index_file) if index_file.is_file() else None
        return record
    except BaseException as exc:
        problems = _undo(archived, inserted, index_file, archive_dir)
        msg = f"{pool}/{name}: {exc}; rolled back" + (f" WITH PROBLEMS: {problems}" if problems else "")
        raise ApplyError(msg) from exc


def _undo(archived: List[Tuple[Path, Path]], inserted: List[Path], index_file: Optional[Path],
          archive_dir: Path) -> List[str]:
    """Inserted files -> ``_rolled_back/``; archived originals -> back. Never deletes. Returns problems."""
    problems: List[str] = []
    for f in [*inserted, *([index_file] if index_file else [])]:
        if f.is_file():
            try:
                _move(f, archive_dir / "_rolled_back" / f.parent.parent.name / f.parent.name / f.name)
            except OSError as e:
                problems.append(f"could not move away {f}: {e}")
    for orig, arch in reversed(archived):
        try:
            if orig.exists():
                problems.append(f"original path {orig} is occupied, archived file stays at {arch}")
                continue
            orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(arch), str(orig))
        except OSError as e:
            problems.append(f"could not restore {orig} from {arch}: {e}")
    return problems


# --------------------------------------------------------------------------- batch driver

@contextlib.contextmanager
def _batch_lock(bdir: Path):
    with open(bdir / ".apply.lock", "a+") as fp:
        if fcntl is not None:
            fcntl.flock(fp, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fp, fcntl.LOCK_UN)


def _read_state(bdir: Path) -> dict:
    f = bdir / APPLY_STATE
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8")).get("items", {})
    except (OSError, ValueError) as e:
        raise ApplyError(f"{f} is unreadable ({e}); not continuing") from e


def _write_state(bdir: Path, items: dict) -> None:
    _atomic_json(bdir / APPLY_STATE, {"format_version": 1, "items": items})


def _mark(store: Optional[flag_store.FlagStore], pool: str, item: dict, status: str, batch_id: str, run: str) -> None:
    if store is None:
        return
    for k in item.get("image_keys") or [item["image_key"]]:
        try:
            store.set_relabel_result(pool, k, {"status": status, "batch_id": batch_id, "apply_run": run})
        except flag_store.FlagNotFound:
            pass


def process_batch(pool: str, bdir: Path, raw_base: Path, store: Optional[flag_store.FlagStore], archive_root: Path,
                  run_stamp: str, tol_px: float, ingest_root: Path, apply: bool, out=print,
                  fail_after: Optional[Dict[str, str]] = None) -> dict:
    """Returns counts ``{applied, unchanged, done, skipped, failed}`` plus ``errors``."""
    man = relabel_batch.read_manifest(bdir)
    res = {"applied": 0, "deleted": 0, "unchanged": 0, "done": 0, "skipped": 0, "failed": 0, "errors": []}
    if man.get("pool") != pool:
        raise ApplyError(f"manifest pool {man.get('pool')!r} != folder pool {pool!r}")
    class_names = relabel_batch._raw_class_names(raw_base, pool)
    # RAW may have classes APPENDED since the export (ids of existing classes unchanged): allowed, and the
    # corrected labels may then use the new ids. Rename / reorder / removal is refused.
    if not relabel_batch.class_names_compatible(man.get("class_names"), class_names):
        raise ApplyError("batch classes are not the start of RAW data.yaml's class list (renamed/reordered/removed); "
                         "boxes would get wrong classes")
    dist, iutils = _load_ingest(ingest_root)
    decisions = relabel_batch.read_fixer_decisions(bdir)
    archive_dir = archive_root / run_stamp
    with _batch_lock(bdir):
        state = _read_state(bdir)
        out(f"\n[{pool}] batch {man['batch_id']}: {len(man['items'])} item(s)")
        plans = [plan_item(pool, bdir, man, it, state, store, raw_base, class_names, iutils, tol_px, decisions) for it in man["items"]]
        manifest_rec = {"format_version": 1, "run": run_stamp, "pool": pool, "batch_id": man["batch_id"],
                        "box_tol_px": tol_px, "raw_base_path": str(raw_base), "archive_dir": str(archive_dir),
                        "applied_at": datetime.now().astimezone().isoformat(timespec="seconds"), "items": []}
        for p in plans:
            name = p.item["item_name"]
            if p.action == "done":
                res["done"] += 1
                out(f"  = {name}: {p.detail}" + ("" if p.marked else " (sidecar mark missing, repaired)" if apply else " (sidecar mark missing)"))
                if apply and not p.marked:
                    _mark(store, pool, p.item, state[name]["status"], man["batch_id"], state[name]["run"])
                continue
            if p.action == "unchanged":
                out(f"  ~ {name}: NIJE ISPRAVLJENO ({p.detail}); marked done without change")
                res["unchanged"] += 1
                if apply:
                    state[name] = {"status": "unchanged", "run": run_stamp, "at": datetime.now().astimezone().isoformat(timespec="seconds")}
                    _write_state(bdir, state)
                    _mark(store, pool, p.item, "unchanged", man["batch_id"], run_stamp)
                    manifest_rec["items"].append({"item_name": name, "status": "unchanged"})
                continue
            if p.action not in ("apply", "delete"):
                label = {"stale": "zastarela (stale)"}.get(p.action, p.action)
                out(f"  - {name}: SKIP [{label}] {p.detail}")
                res["skipped"] += 1
                continue
            orig_cls = [c["cls"] for c in p.copies]
            is_del = p.action == "delete"
            kind = "deleted" if is_del else "applied"
            if is_del:
                out(f"  x {name}: DELETE ({p.detail}): ARCHIVE {len(p.copies)} copy(ies) in {orig_cls} -> {archive_dir}, nothing inserted")
            else:
                out(f"  + {name}: ARCHIVE {len(p.copies)} copy(ies) in {orig_cls} -> {archive_dir}")
                out(f"      INSERT corrected image+label into {p.target_classes}"
                    + ("" if sorted(orig_cls) == sorted(p.target_classes) else f"  (folders differ from the original {orig_cls})"))
            if not apply:
                res[kind] += 1
                continue
            try:
                rec = apply_item(p, raw_base, archive_dir, dist, class_names, (fail_after or {}).get(name))
            except ApplyError as e:
                out(f"    ! FAILED, item rolled back: {e}")
                res["failed"] += 1
                res["errors"].append(str(e))
                manifest_rec["items"].append({"item_name": name, "status": "failed", "error": str(e)})
                continue
            res[kind] += 1
            rec["status"] = kind
            if is_del:
                rec["delete_reason"] = p.detail
            manifest_rec["items"].append(rec)
            archive_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(archive_dir / f"{APPLY_MANIFEST}", _merge_manifest(archive_dir, manifest_rec))
            state[name] = {"status": kind, "run": run_stamp, "at": rec_time()}
            _write_state(bdir, state)
            _mark(store, pool, p.item, kind, man["batch_id"], run_stamp)
            out(f"    ok: archived {len(rec['archived'])} file(s), inserted {len(rec['inserted'])} file(s)")
        if apply and manifest_rec["items"]:
            archive_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(archive_dir / APPLY_MANIFEST, _merge_manifest(archive_dir, manifest_rec))
            _close_if_resolved(bdir, man, state, run_stamp)
    return res


def rec_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _merge_manifest(archive_dir: Path, rec: dict) -> dict:
    """Several batches/pools can share one run stamp: ``runs`` keeps one record per batch."""
    f = archive_dir / APPLY_MANIFEST
    doc = {"format_version": 1, "run": archive_dir.name, "batches": []}
    if f.is_file():
        doc = json.loads(f.read_text(encoding="utf-8"))
    doc["batches"] = [b for b in doc["batches"] if b["batch_id"] != rec["batch_id"]] + [rec]
    return doc


def _close_if_resolved(bdir: Path, man: dict, state: dict, run_stamp: str) -> None:
    if all(it["item_name"] in state for it in man["items"]):
        man = dict(man)
        man["status"] = STATUS_APPLIED
        man["applied_at"] = rec_time()
        man["apply_runs"] = sorted({s["run"] for s in state.values()})
        _atomic_json(bdir / relabel_batch.MANIFEST, man)


# --------------------------------------------------------------------------- rollback

def rollback_run(run_stamp: str, archive_root: Path, batches_dir: Path, store: Optional[flag_store.FlagStore],
                 apply: bool, out=print) -> int:
    archive_dir = archive_root / run_stamp
    mf = archive_dir / APPLY_MANIFEST
    if not mf.is_file():
        raise ApplyError(f"no apply manifest at {mf}")
    doc = json.loads(mf.read_text(encoding="utf-8"))
    problems = 0
    for b in doc["batches"]:
        pool, batch_id = b["pool"], b["batch_id"]
        bdir = batches_dir / pool / batch_id
        out(f"\n[{pool}] rollback of {batch_id} (run {run_stamp})")
        state = _read_state(bdir) if bdir.is_dir() else {}
        for rec in b["items"]:
            if rec.get("status") not in ("applied", "unchanged", "deleted"):
                continue
            name = rec["item_name"]
            if rec["status"] == "unchanged":
                out(f"  ~ {name}: unchanged item, only the sidecar mark is cleared")
                if apply:
                    _clear(store, pool, rec, bdir, state, name)
                continue
            # Every inserted file must still be what we wrote, every original slot must be free.
            bad = [i["path"] for i in rec["inserted"] if not Path(i["path"]).is_file() or sha256_file(Path(i["path"])) != i["sha256"]]
            ins = {i["path"] for i in rec["inserted"]}
            busy = [a["from"] for a in rec["archived"] if Path(a["from"]).exists() and a["from"] not in ins]
            lost = [a["to"] for a in rec["archived"] if not Path(a["to"]).is_file()]
            if bad or busy or lost:
                out(f"  ! {name}: cannot roll back (inserted changed/missing {bad}, original slots occupied {busy}, archive missing {lost})")
                problems += 1
                continue
            out(f"  + {name}: move {len(rec['inserted'])} inserted file(s) to {archive_dir}/_rolled_back, restore {len(rec['archived'])} original(s)")
            if not apply:
                continue
            probs = _undo([(Path(a["from"]), Path(a["to"])) for a in rec["archived"]],
                          [Path(i["path"]) for i in rec["inserted"]],
                          Path(rec["ingest_index"]) if rec.get("ingest_index") else None, archive_dir)
            if probs:
                out(f"    ! problems: {probs}")
                problems += 1
                continue
            rec["status"] = "rolled_back"
            _clear(store, pool, rec, bdir, state, name)
        if apply:
            _atomic_json(mf, doc)
    return 1 if problems else 0


def _clear(store, pool, rec, bdir: Path, state: dict, name: str) -> None:
    if store is not None:
        for k in rec.get("image_keys", []):
            store.clear_relabel_result(pool, k)
    if bdir.is_dir():
        state.pop(name, None)
        _write_state(bdir, state)
        man = relabel_batch.read_manifest(bdir)
        if man.get("status") != relabel_batch.STATUS_OPEN:
            man["status"] = relabel_batch.STATUS_OPEN
            man.pop("applied_at", None)
            _atomic_json(bdir / relabel_batch.MANIFEST, man)


# --------------------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MDQ-15c-3: apply a corrected relabel batch to RAW (dry-run by default).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", action="append", choices=flag_store.POOLS, help="limit to a pool (repeatable)")
    ap.add_argument("--batch", action="append", help="<pool>/<batch_id> (repeatable); default: every open batch")
    ap.add_argument("--raw-base-path", type=Path, default=relabel_batch.DEFAULT_RAW_BASE)
    ap.add_argument("--review-dir", type=Path, default=relabel_batch.DEFAULT_REVIEW_DIR)
    ap.add_argument("--batches-dir", type=Path, default=relabel_batch.DEFAULT_BATCHES_DIR)
    ap.add_argument("--archive-root", type=Path, default=archive_raw.DEFAULT_ARCHIVE_ROOT)
    ap.add_argument("--ingest-root", type=Path, default=DEFAULT_INGEST_ROOT, help="IntelliCup repo (ingestion_pipeline)")
    ap.add_argument("--box-tol-px", type=float, default=DEFAULT_BOX_TOL_PX,
                    help=f"max corner difference in pixels for 'not corrected' (default {DEFAULT_BOX_TOL_PX})")
    ap.add_argument("--rollback", metavar="STAMP", help="roll back the run <archive-root>/<STAMP>")
    ap.add_argument("--apply", "--execute", dest="apply", action="store_true", help="really change RAW (default: dry run)")
    args = ap.parse_args(argv)

    raw_base = args.raw_base_path.expanduser().resolve()
    try:
        relabel_batch.check_batches_dir(args.batches_dir, raw_base)
        store = flag_store.FlagStore(args.review_dir, forbidden_roots=[raw_base])
        if not store.available:
            raise ApplyError(f"review dir not available: {args.review_dir}")
        if args.rollback:
            return rollback_run(args.rollback, args.archive_root, args.batches_dir, store, args.apply)
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_relabel"
        print(f"{'APPLY' if args.apply else 'DRY RUN (nothing is changed)'} — raw={raw_base} archive={args.archive_root}/{run_stamp} tol={args.box_tol_px}px")
        todo = []
        if args.batch:
            for b in args.batch:
                pool, _, bid = b.partition("/")
                todo.append((pool, relabel_batch.resolve_batch(args.batches_dir, pool, bid, raw_base)))
        else:
            todo = [(p, d) for p, _i, d, _m in relabel_batch.list_batches(args.batches_dir)
                    if not args.pool or p in args.pool]
        if not todo:
            print("no open relabel batch")
            return 0
        failed = 0
        for pool, bdir in todo:
            r = process_batch(pool, bdir, raw_base, store, args.archive_root, run_stamp, args.box_tol_px,
                              args.ingest_root, args.apply)
            failed += r["failed"]
            print(f"  => {'applied' if args.apply else 'would apply'} {r['applied']}, "
                  f"{'deleted' if args.apply else 'would delete'} {r['deleted']}, unchanged {r['unchanged']}, "
                  f"already done {r['done']}, skipped {r['skipped']}, failed {r['failed']}")
        return 1 if failed else 0
    except (ApplyError, relabel_batch.RelabelBatchError, flag_store.FlagError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
