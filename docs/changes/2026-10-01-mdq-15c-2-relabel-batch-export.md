# 2026-10-01 — MDQ-15c-2: relabel batch export

Export of every sidecar entry with `owner_decision == "relabel"` (MDQ-15c-1) into a **relabel batch**:
a small dataset of COPIES of the RAW image and its current label file, which Goca opens in the Dataset
Fixer ("Load dataset") and corrects. RAW is only read. Done manually at the owner's request, outside
`next_ticket.sh` (same as 15c-1). Commit: see `git log` for "MDQ-15c-2" (on `main`).

## 1. Where the batch lives

`/opt/intellicup/datasets/relabel_batches/<pool>/<batch_id>/`, `batch_id = RELABEL_<pool>_<YYYYmmdd_HHMMSS>`.

- Sibling of `raw/` and `raw_review/`, so it is outside RAW, on the same disk as RAW (no cross-disk surprises), and next to the other owner-side artifacts. The ticket text proposed `~/LabelingToolData/datasets/RELABEL_<pool>_<ts>/`; not used because that folder is the labelers' own working area on every machine and is not the server's owner data.
- Layout: `images/`, `labels/` (flat, what `dataset.py::load_dataset_session()` expects), `data.yaml` (RAW class order), `manifest.json`.
- Built in a hidden `.tmp_*` folder and renamed into place; hidden folders are never listed as batches.
- `check_batches_dir()` refuses a batches dir inside RAW or containing RAW (CLI and server).

## 2. How the Dataset Fixer loads it (smallest change)

No change to `datasets_dir`. `GET /api/datasets` now also lists open batches as `relabel/<pool>/<batch_id>`; `POST /api/datasets/load` accepts those names (`server.py`).

- New optional config key `relabel_batches_dir` / env `LABELER_RELABEL_BATCHES_DIR`; default `/opt/intellicup/datasets/relabel_batches` when `/opt/intellicup/datasets` exists (same rule as `raw_review_dir`), otherwise off.
- Load guards: batch name is validated (`resolve_batch`, no `..`/slashes/hidden); the model must equal the pool; the manifest `class_names` must equal the model's class list (else 400, boxes would get wrong names); the path is refused if it is inside or contains the RAW tree. The last guard applies to every dataset load, so the Dataset Fixer can never be left open over RAW.
- Comment: `GET /api/datasets/annotations` returns `relabel_note` (owner-flag comment, image key, number of RAW copies) and the existing `#flagBadge` shows "RELABEL — <comment>". `DatasetSession.relabel_items` is a new optional field.

## 3. Export rules (`web_labeler/relabel_batch.py`)

CLI only (simpler than an endpoint; no sidecar write is needed). `--dry-run` is the default and writes nothing; `--execute` writes.

```
/opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python web_labeler/relabel_batch.py --dry-run [--pool shots]
/opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python web_labeler/relabel_batch.py --pool shots --execute
```

Per entry, in order; a skipped entry is reported by name and never exported:

| reason | meaning |
|---|---|
| `forward` | `_forward/` key, not in RAW yet |
| `no_snapshot` | entry has no `relabel_snapshot` |
| `in_open_batch` | the key, or another key with the same file name, is in an open batch (manifest `status == "open"`) → idempotent |
| `stale` (zastarela) | RAW image or label no longer matches the `relabel_snapshot`, or the image is gone |
| `conflict` (konflikt) | same file name in several class folders and the copies differ (image bytes or label content, e.g. class vs background). A missing label counts as an empty one |
| `name_clash` | two different file names with the same stem (labels are keyed by stem) |

The same file name in several class folders goes into the batch ONCE; `raw_paths` lists every RAW copy. Copies are made with `shutil.copy2` (never a link: Dataset Fixer Save writes labels in place) and their sha256 is re-checked against the snapshot. No label file in RAW → empty label file in the batch. The sidecar is never written.

## 4. `manifest.json` format (`format_version` 1; MDQ-15c-3 reads this)

```
{"format_version": 1, "batch_id": "...", "pool": "cups", "created_at": "<ISO>", "status": "open",
 "raw_base_path": "/opt/intellicup/datasets/raw/blaznavac",
 "class_names": ["CAJ", ...],                       // RAW <pool>/data.yaml order
 "items": [{
   "item_name": "x.jpg", "label_name": "x.txt",      // names inside images/ and labels/
   "image_key": "CAJ/x.jpg",                         // primary key
   "image_keys": ["CAJ/x.jpg", "background/x.jpg"],  // every flagged sidecar key of this item
   "raw_paths": [{"image": "cups/images/CAJ/x.jpg", "label": "cups/labels/CAJ/x.txt" | null}],   // relative to raw_base_path, every copy
   "relabel_snapshot": {"image_sha256": "...", "label_sha256": "..." | null},   // the ORIGINAL at decision time
   "comment": "<flag comment(s) joined ' | '>", "flag_source": "manual_goca",
   "flags": [{"image_key", "source", "category", "comment", "flagged_by", "flagged_at", "owner_decision_at", "relabel_snapshot"}]}]}
```

For 15c-3: `status` is `open` until applied; 15c-3 must set it to a non-`open` value (e.g. `applied`) so the entries are not blocked/re-exported. Corrected labels are `labels/<label_name>`; the image must still equal `relabel_snapshot.image_sha256`.

## 5. Compatibility

`flag_store.py` is not touched: `FlagStore(...)`, `add_auto_flag()`, `schema_version` 1 unchanged, no new sidecar fields. `IntelliCup/tests/data4_audit.py --pool shots --dry-run` on a scratch sidecar copy: ok, 0 errors.

## 6. Verification

- `web_labeler/tests/test_relabel_batch.py` (17 tests, temp dirs only): export + manifest, copy is not a link / RAW unchanged, stale (image, label, gone), conflict (labels, class vs background, image bytes), missing==empty label, same name in two folders (one copy, all paths), idempotence (+ closed batch does not block), non-relabel/forward ignored, stem clash, dry-run (default) writes nothing and never touches the sidecar, batches dir inside RAW refused, `resolve_batch` traversal guards, old sidecar. Run: `INTELLICUP_LABELING_TOOL/bin/python -m unittest discover -s web_labeler/tests`.
- Live test instance :8010 with a temp batch: listing, load, `relabel_note`, wrong model 400, class-order mismatch 400, `..` name 404. Not checked in a browser: the badge text (one JS branch, `node --check` ok).

## 7. Known limits

- Dataset Fixer Save (overwrite) re-quantizes every label of the batch (`issues/dataset_fixer_save_requantizes_labels.md`). Harmless for the corrected boxes, but 15c-3 must compare the boxes, not the bytes, of untouched items.
- Save also deletes images marked deleted, inside the batch copy only.
- Nothing marks a sidecar entry as "exported"; idempotence comes from open manifests only.
