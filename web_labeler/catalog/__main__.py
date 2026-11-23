"""Run the class list import:  python -m web_labeler.catalog import"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from web_labeler.catalog.store import CatalogError, CatalogSettings, import_class_lists


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m web_labeler.catalog")
    parser.add_argument("command", choices=["import"])
    args = parser.parse_args()
    default_cfg = Path(__file__).resolve().parents[2] / "labeler_config.json"
    config_path = Path(os.getenv("LABELER_CONFIG_PATH") or default_cfg)
    try:
        report = import_class_lists(CatalogSettings.from_config_file(config_path))
    except (CatalogError, OSError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
