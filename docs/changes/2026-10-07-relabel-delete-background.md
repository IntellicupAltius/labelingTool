# 2026-10-07 — Relabel batch: Delete / Background in the Dataset Fixer, Delete as the default

Owner request after the first shots pilot (`RELABEL_shots_20261006_170836`). Backups: `*.bak_pre_relabeldelete_20261007`.

## Problem
- In a relabel batch the Fixer's Delete went through a `window.confirm` dialog; the click never reached the server
  (log: no `POST /api/datasets/delete`), so nothing showed as marked.
- Even when it worked, Save (overwrite) erased the batch copy and `apply_relabel.py` then skipped the item
  ("corrected copy missing"): RAW unchanged, batch open forever.
- The Background mark was UI-only. An image left with 0 boxes was saved as an empty label and applied into
  `background/`, also when nobody chose anything (a skipped image could still show drinks).

## Behaviour now (relabel batch only; normal datasets unchanged)
- Delete: no confirm dialog, shows "MARKED FOR DELETION" + Unmark. Save never erases files.
- Save (overwrite) writes the labels and `<batch>/fixer_decisions.json`
  (`{"format_version": 1, "saved_at", "items": {"<image name>": "background" | "delete"}}`, only images with 0 boxes).
  Marks are restored when the batch is loaded again. No zip. Save (duplicate as _fixed) is hidden and refused (400).
- An image with 0 boxes and no choice shows "NO BOXES, NO CHOICE → DELETE".
- `apply_relabel.py`, new step 3b: corrected label with 0 boxes → `background/` only if marked background;
  marked delete or no choice → DELETE: every RAW copy is archived (`raw_archive/<stamp>_relabel/`), nothing inserted,
  sidecar `relabel_result.status = "deleted"`, apply manifest item `status: "deleted"` + `delete_reason`.
  A delete/background mark on an image that still has boxes → `invalid`, skipped. `--rollback` restores deleted items too.
- `flag_store.set_relabel_result` accepts `"deleted"`.
- Button text: "Save as _fixed" → "Save (duplicate as _fixed)".

## Verified
- `python -m unittest discover -s web_labeler/tests` (INTELLICUP_LABELING_TOOL): OK, incl. new `ZeroBoxDecisionTests`
  (default delete, marked delete, marked background, mark on boxed image, rollback, failure rollback) and
  `FixerSaveRelabelTests`.
- Second server on :8011 over a scratch copy of the pilot: load → Delete/Background → Save → reload restores marks → apply dry-run reads them.
- Real pilot applied by the owner 2026-10-07 (`raw_archive/20261007_225556_relabel`): 10 applied, 2 deleted, RAW checked file by file.
- Server :8000 restarted for this change.
