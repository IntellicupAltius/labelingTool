"""Catalog commands, run with the labeling interpreter from the labelingTool folder:

    python -m web_labeler.catalog import              # read the sources into the Catalog
    python -m web_labeler.catalog backup              # nightly copy to catalog_backup_dir, kept 30 days
    python -m web_labeler.catalog backups             # list the copies
    python -m web_labeler.catalog restore <file.db>   # put a copy back (stop the tool first)
    python -m web_labeler.catalog model-cards         # rebuild the read-only model cards (the tool also starts it)
    python -m web_labeler.catalog transfer <old.db> [--skip "Proba (Claude)"] [--dry-run]
                                                      # carry manual work from another Catalog DB into this one, once
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from web_labeler.catalog import backup as _backup
from web_labeler.catalog import model_cards as _model_cards
from web_labeler.catalog import transfer as _transfer
from web_labeler.catalog.store import CatalogError, CatalogSettings, import_catalog, model_cards_snapshot_path


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m web_labeler.catalog")
    parser.add_argument("command", choices=["import", "backup", "backups", "restore", "model-cards", "transfer"])
    parser.add_argument("file", nargs="?")
    parser.add_argument("--skip", action="append", default=[], help="transfer: leave out everything by this person (test entries)")
    parser.add_argument("--dry-run", action="store_true", help="transfer: show what would move, change nothing")
    args = parser.parse_args()
    default_cfg = Path(__file__).resolve().parents[2] / "labeler_config.json"
    config_path = Path(os.getenv("LABELER_CONFIG_PATH") or default_cfg)
    try:
        settings = CatalogSettings.from_config_file(config_path)
        if args.command == "import":
            report = import_catalog(settings)
        elif args.command == "transfer":
            if not args.file:
                raise CatalogError("Name the Catalog database to carry the work from")
            report = _transfer.transfer(Path(args.file), settings, skip_people=args.skip, dry_run=args.dry_run)
        elif args.command == "model-cards":
            target = model_cards_snapshot_path(settings)
            if target is None or settings.model_cards is None:
                raise CatalogError("Set catalog_live_models_dir and the Catalog database first")
            report = _model_cards.write_snapshot(settings.model_cards, target)
        else:
            if settings.backup_dir is None or settings.db_path is None or not settings.venue_id:
                raise CatalogError("Set catalog_backup_dir, catalog_venue_id and the Catalog database first")
            if args.command == "backup":
                report = _backup.backup(settings.db_path, settings.backup_dir, settings.venue_id, settings.media_dir,
                                        require_mount=str(settings.backup_dir).startswith("/media/"))
            elif args.command == "backups":
                report = {"backups": _backup.list_backups(settings.backup_dir, settings.venue_id)}
            else:
                if not args.file:
                    raise CatalogError("Name the backup file to restore")
                report = _backup.restore(Path(args.file), settings.db_path, settings.backup_dir / "media", settings.media_dir)
    except (CatalogError, _backup.BackupError, _transfer.TransferError, OSError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
