# MDQ-17c — model proposals in a relabel batch (2026-10-20)

Why: about half of the glasses RAW has drinks with no box. Fair (cross-fold) models (MDQ-17a/b) suggest the
missing boxes, so the labelers mostly confirm, re-class or delete instead of drawing every box.

## Files in a batch
- `proposals.json` (written by a script, never by the tool):
  `{"format_version": 1, "created_at", "source", "items": {item_name: [{"pid", "class_name", "box": [x1,y1,x2,y2] 0-1, "conf"?, "agreement"?, "decoy"?}]}}`
- `proposal_decisions.json` (written by Save): `{item_name: {pid: confirmed | reclassed | rejected}}`.
- `labels/` stays the exact RAW copy until Save. Save writes only real and decided boxes, never an undecided proposal or a decoy.

## Dataset Fixer (relabel batch)
- A proposal is a dashed pink box with a "P" label. Its sidebar row says "model proposal · conf". The first undecided proposal of an image is selected automatically.
- **A / Enter / OK button** = confirm as it is. **C / Class** = choose a class, which also decides the proposal. **Del / Delete** = reject it. After each action the next proposal is selected.
- **N / Shift+N** = next / previous image that still has undecided proposals.
- The badge shows `P: <here>, <in batch>`. The Save dialog reports undecided proposals.
- Background/Delete marks are blocked while an image has an undecided proposal, because the proposal counts as a box.
- A wrong box is deleted and redrawn, because the Fixer has no box resize.

## apply_relabel.py
- New step 1b: an image with any undecided proposal is skipped (`pending_proposals`) and RAW is untouched. This also covers the 0-box default Delete: an image whose only boxes are undecided proposals is never deleted.

## QA decoys
- `decoy: true` is a deliberately wrong proposal. The browser cannot tell it apart (the field is dropped from the API). It is never written to labels/.
- Its decision is recorded in `proposal_decisions.json`; the correct answer is `rejected`.
- Decoys must only go on images that already have a real box. On an otherwise empty image, a confirmed decoy leaves 0 boxes, which means the default Delete.

## Verification
- `python -m unittest discover -s web_labeler/tests`: 99 OK, 5 of them new in `test_proposals.py`.
- Browser test on an isolated instance (:8012, a copy of `RELABEL_glasses_20261019_233404` + 427 test proposals from current.pt): A, Del, C, the OK button, N, Save (decisions + labels), reopen (no duplicates), and an apply dry-run (320 pending skipped, 3 apply).
- The production :8000 was not restarted.
