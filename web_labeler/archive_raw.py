#!/usr/bin/env python3
"""
archive_raw.py — MDQ-8: RAW dataset archive script.

The only code ever allowed to physically move files out of
``<raw_base_path>/<pool>/{images,labels}/<class>/``. Reads every pool's flag sidecar (see
``flag_store.py``, MDQ-3/MDQ-6), filters to entries the owner has decided
``owner_decision == "approve_delete"`` for, and MOVES (never deletes) each image+label pair
into a timestamped archive directory outside the RAW tree, preserving pool/class structure.

Never deletes anything and never touches ``active_classes.yaml``/``data.yaml``/
``_ingest_index``/anything else under RAW — it only ever moves the exact two files (image +
label) named by an entry.

Not written by this script: the sidecar entry itself is left exactly as it was (per
``flag_store.set_owner_decision``'s own docstring, moving the files is this script's job, not
the sidecar's). This makes the script naturally idempotent — an entry whose image is already
gone (already archived by a prior run, or removed by hand) is reported and skipped, not an
error, so a re-run after a partial failure is safe.

Skipped, always reported by name rather than silently dropped:
  - ``_forward/`` keys — a video frame flagged before ingest has no RAW file yet, nothing to
    move.
  - entries whose RAW image or label file is already gone.
  - entries whose ``owner_decision`` is not exactly ``"approve_delete"`` at read time (pending,
    ``"keep"``, or reverted back to null — read live from disk, never cached, so an owner's undo
    genuinely stops the archive).
  - malformed ``<class>/<filename>`` keys (defensive — the sidecar's own writer validates this,
    but a hand-edited or legacy file is not trusted blindly).

``--dry-run`` is the default: prints the exact plan, moves nothing. Pass ``--execute`` to
actually perform the moves.

Usage:
    # Dry run (default) — prints what WOULD move, touches nothing:
    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python web_labeler/archive_raw.py

    # Actually move the approved files:
    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python web_labeler/archive_raw.py --execute

    # Limit to one pool, custom locations:
    ... archive_raw.py --pool glasses --raw-base-path /path/to/raw \\
        --review-dir /path/to/raw_review --archive-root /path/to/raw_archive --execute

No third-party dependencies — runs under any Python 3.8+, including the labeler's own
interpreter or the system one.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flag_store  # noqa: E402  (sibling module; see sys.path insert above)

DEFAULT_RAW_BASE_PATH = Path("/opt/intellicup/datasets/raw/blaznavac")
DEFAULT_REVIEW_DIR = Path("/opt/intellicup/datasets/raw_review")
DEFAULT_ARCHIVE_ROOT = Path("/opt/intellicup/datasets/raw_archive")
APPROVE_DELETE = "approve_delete"


@dataclass
class PlanEntry:
    pool: str
    image_key: str
    image_src: Path
    label_src: Path
    image_dst: Path
    label_dst: Path


@dataclass
class SkipEntry:
    pool: str
    image_key: str
    reason: str


@dataclass
class Plan:
    to_move: List[PlanEntry] = field(default_factory=list)
    skipped: List[SkipEntry] = field(default_factory=list)
    pool_errors: List[str] = field(default_factory=list)


def _resolve_raw_base_path(cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.getenv("ANALYZER_RAW_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_RAW_BASE_PATH


def _resolve_review_dir(cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.getenv("LABELER_RAW_REVIEW_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_REVIEW_DIR


def _resolve_archive_root(cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.getenv("LABELER_RAW_ARCHIVE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_ARCHIVE_ROOT


def build_plan(
    store: flag_store.FlagStore,
    raw_base_path: Path,
    archive_root: Path,
    run_stamp: str,
    pools: List[str],
) -> Plan:
    """Read every requested pool's sidecar and work out exactly what would move where.

    Pure planning — touches nothing on disk beyond reading the sidecars and stat()-ing
    candidate RAW paths to see whether they still exist.
    """
    plan = Plan()
    all_entries = store.list_all()
    for pool in pools:
        pool_result = all_entries.get(pool, {})
        if "error" in pool_result:
            plan.pool_errors.append(f"{pool}: {pool_result['error']}")
            continue
        for image_key, entry in sorted(pool_result.get("entries", {}).items()):
            if entry.get("owner_decision") != APPROVE_DELETE:
                continue  # pending / "keep" / reverted — not this script's business
            if image_key.startswith(flag_store.FORWARD_DIR + "/"):
                plan.skipped.append(SkipEntry(pool, image_key, "forward-flagged frame, no RAW file"))
                continue
            parts = image_key.split("/")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                plan.skipped.append(SkipEntry(pool, image_key, "malformed image_key, not '<class>/<filename>'"))
                continue
            class_name, filename = parts
            image_src = raw_base_path / pool / "images" / class_name / filename
            label_src = raw_base_path / pool / "labels" / class_name / f"{Path(filename).stem}.txt"
            if not image_src.is_file():
                plan.skipped.append(SkipEntry(pool, image_key, f"RAW image not found: {image_src}"))
                continue
            if not label_src.is_file():
                plan.skipped.append(SkipEntry(pool, image_key, f"RAW label not found: {label_src}"))
                continue
            image_dst = archive_root / run_stamp / pool / "images" / class_name / filename
            label_dst = archive_root / run_stamp / pool / "labels" / class_name / label_src.name
            plan.to_move.append(PlanEntry(pool, image_key, image_src, label_src, image_dst, label_dst))
    return plan


def _unique_dest(dst: Path) -> Path:
    """Never silently overwrite an existing destination — append a numeric suffix instead."""
    if not dst.exists():
        return dst
    stem, suffix, parent, n = dst.stem, dst.suffix, dst.parent, 1
    while True:
        candidate = parent / f"{stem}__dup{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def execute_plan(plan: Plan) -> List[str]:
    """Perform the moves. Returns a list of human-readable error strings (never raises).

    Each entry's directory creation AND move are scoped inside that entry's own try/except —
    a failure creating the destination directory is just as reportable-and-skippable as a
    failed move itself, and must never abort the whole run or leave an earlier entry's
    successful move unrecorded.
    """
    errors: List[str] = []
    for e in plan.to_move:
        image_dst = _unique_dest(e.image_dst)
        try:
            image_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(e.image_src), str(image_dst))
        except OSError as exc:
            errors.append(f"{e.pool}/{e.image_key}: failed to move image ({exc}); label left in place")
            continue
        label_dst = _unique_dest(e.label_dst)
        try:
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(e.label_src), str(label_dst))
        except OSError as exc:
            # Best-effort rollback so a half-moved pair doesn't silently desync RAW's
            # image/label pairing — report loudly either way, never pretend it's fine.
            try:
                shutil.move(str(image_dst), str(e.image_src))
                errors.append(
                    f"{e.pool}/{e.image_key}: failed to move label ({exc}); "
                    f"image moved back to {e.image_src}"
                )
            except OSError as rollback_exc:
                errors.append(
                    f"{e.pool}/{e.image_key}: failed to move label ({exc}); "
                    f"ALSO FAILED to move image back to {e.image_src} ({rollback_exc}) — "
                    f"image now sits at {image_dst} without its label, needs manual fix"
                )
    return errors


def _print_plan(plan: Plan, raw_base_path: Path, review_dir: Path, archive_root: Path, execute: bool) -> None:
    print(f"RAW base path : {raw_base_path}")
    print(f"Review dir    : {review_dir}")
    print(f"Archive root  : {archive_root}")
    print(f"Mode          : {'EXECUTE — files will be moved' if execute else 'DRY RUN — nothing will be moved'}")
    print()
    if plan.pool_errors:
        print(f"Pool sidecars that could not be read (skipped, other pools unaffected):")
        for msg in plan.pool_errors:
            print(f"  ! {msg}")
        print()
    print(f"{len(plan.to_move)} pair(s) to archive:")
    for e in plan.to_move:
        print(f"  {e.pool}/{e.image_key}")
        print(f"    {e.image_src} -> {e.image_dst}")
        print(f"    {e.label_src} -> {e.label_dst}")
    print()
    print(f"{len(plan.skipped)} entrie(s) skipped:")
    for s in plan.skipped:
        print(f"  {s.pool}/{s.image_key}: {s.reason}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", action="append", choices=flag_store.POOLS,
                         help="limit to one pool (repeatable); default: all 5 pools")
    parser.add_argument("--raw-base-path", default=None, help=f"default: {DEFAULT_RAW_BASE_PATH} (or $ANALYZER_RAW_ROOT)")
    parser.add_argument("--review-dir", default=None, help=f"default: {DEFAULT_REVIEW_DIR} (or $LABELER_RAW_REVIEW_DIR)")
    parser.add_argument("--archive-root", default=None, help=f"default: {DEFAULT_ARCHIVE_ROOT} (or $LABELER_RAW_ARCHIVE_ROOT)")
    parser.add_argument("--execute", action="store_true",
                         help="actually move files (default is dry-run: print the plan only)")
    args = parser.parse_args(argv)

    raw_base_path = _resolve_raw_base_path(args.raw_base_path)
    review_dir = _resolve_review_dir(args.review_dir)
    archive_root = _resolve_archive_root(args.archive_root)
    pools = args.pool or list(flag_store.POOLS)

    if not raw_base_path.is_dir():
        print(f"ERROR: raw base path not found: {raw_base_path}", file=sys.stderr)
        return 1

    try:
        store = flag_store.FlagStore(review_dir, forbidden_roots=[raw_base_path])
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if not store.available:
        print(f"ERROR: review dir not available: {review_dir}", file=sys.stderr)
        return 1

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = build_plan(store, raw_base_path, archive_root, run_stamp, pools)
    _print_plan(plan, raw_base_path, review_dir, archive_root, args.execute)

    if not args.execute:
        print("\n[dry-run] No files were moved. Re-run with --execute to perform the moves above.")
        return 0

    if not plan.to_move:
        print("\nNothing to move.")
        return 0

    errors = execute_plan(plan)
    print(f"\nMoved {len(plan.to_move) - len(errors)}/{len(plan.to_move)} pair(s).")
    if errors:
        print(f"{len(errors)} error(s):")
        for msg in errors:
            print(f"  ! {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
