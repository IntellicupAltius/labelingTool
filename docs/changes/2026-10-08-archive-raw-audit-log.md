# 2026-10-08 — archive_raw.py writes its own audit log

Owner request: a record of what was archived, when and how much. Backup: `web_labeler/archive_raw.py.bak_pre_archivelog_20261008`.

`archive_raw.py --execute` now writes into its run folder `<archive_root>/<run_stamp>/`:
- `archive_log.txt` — the run's console output (plan, skips, result, errors);
- `archive_manifest.json` — `format_version`, `run`, `finished_at`, `raw_base_path`, `review_dir`, `pools`,
  `counts {planned, moved, errors, skipped}`, `moved` (per pair: pool, image_key, image/label from → to, sha256 of the
  archived files), `skipped`, `errors`, `pool_errors`.

Written after the moves, also when some failed; never overwrites (same `_unique_dest` rule). A dry run writes nothing.
The move logic, sidecar handling and exit codes are unchanged (`execute_plan` got an optional `moved` list).

First real run (2026-10-08, shots, 201 pairs) happened just before this change; its output was saved by hand with `tee`
as `raw_archive/archive_raw_shots_20261008_143905.log`.

Verified: unit test `test_execute_writes_audit_log_and_manifest_dry_run_does_not`; full suite OK; dry run on real RAW
(0 to move, 201 "already gone") writes nothing. No server restart needed (the server does not import this script).
Owner manual: `/opt/intellicup/datasets/raw_archive/00_PROCITAJ_ME_uputstvo.md` updated.
