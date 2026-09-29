# 2026-09-29 — Hold B to hide bboxes (review viewer + Dataset Fixer) and the crop zone in the Dataset Fixer

Two front-end features, `static/` only. Branch `mdq-peek-and-zone-fixer` (worktree `../labelingTool_wt_peek`), branched from `main` at `4d22982`. **Not merged**; the owner approves the merge. Anything not known is marked **nepoznato**.

| | commit | subject |
|---|---|---|
| C | this commit on `mdq-peek-and-zone-fixer` | Hold B hides bboxes (review viewer, Dataset Fixer, Video Labeler) with mouse editing locked; crop zone layer + O in the Dataset Fixer |

## 1. Task 1 — B in the review viewer

- Holding **B** hides the `.reviewBox` bboxes (green/red labels included). The zone (crop rectangle + `bar_roi`/`pickup_roi`, key O) is untouched. A blue badge "Boxes hidden — hold B" sits on the image while it is held.
- Implementation: `setPeek()` toggles class `peek` on `#reviewStage`; CSS `.reviewStage.peek .reviewBox { visibility: hidden }`. The boxes stay in the DOM, so the existing token / `image_key` lifecycle of `showReviewItem` needs no change: a new image loaded while B is held simply gets hidden boxes (no ghost of the old image, no flash of visible boxes).
- Keys (review branch of `installHotkeys`, `app.js` ~2679): B is handled after the `isTypingTarget` / modifier check, so it is ignored in input/select/textarea and with Ctrl/Meta/Alt; `e.repeat` is ignored (no flicker). K, D, R, O, arrows and Esc are unchanged. Release: `keyup` B, window `blur`, `visibilitychange` (tab hidden), Esc / closing the panel (`closeReviewPanel` calls `setPeek(false)`).

## 2. Task 2 — Dataset Fixer

**a. B in the Dataset Fixer (done).** Same key, same indicator (`#peekBadge` over the canvas). Because the canvas is shared, it works in the Video Labeler too (same `draw()`); it is **not** active in Background mode, where B already means "background".
- `draw()` skips the annotation boxes and the orange hint box while `state.peek` (2 one-line guards).
- **Mouse editing is locked while B is held.** Editing in this tool happens in exactly three places: drawing on `#canvas` (mousedown → drag → modal), and the box lists `#currentList` / `#globalList` (Delete, class change, tags, background). A capture-phase listener on `document` (`installPeek`) swallows `pointerdown / mousedown / mouseup / click / dblclick / contextmenu / change / input` whose target is inside those three, with `preventDefault` + `stopImmediatePropagation`. If a drag was already in progress when B goes down it is cancelled (`state.dragging=false`), so releasing the mouse cannot open the modal. Lists dim and show a not-allowed cursor.
- Deliberately **not** `pointer-events: none` on the canvas: the first attempt did that, and the drag then started a text selection on the parent, after which the next real drag on the canvas lost its `mouseup` (found in testing, fixed).
- `setMode()` calls `setPeek(false)`.

**b. Crop zone in the Dataset Fixer (done).** A separate `<svg id="fixerZoneSvg">` (`pointer-events: none`) inside `.canvasWrap`, positioned from the canvas' own transform (`state.offsetX/Y`, `state.scale`, `dpr`), plus a bar above the canvas with the toggle button and legend (`#fixerZoneBar`) and a badge for the "unknown" note. Same data and drawing code as the review viewer: `GET /api/cameras/roi` via `fetchReviewRoi`, shapes via the new shared `roiShapes()`, legend via the new shared `fillRoiLegend()` (both refactored out of `drawReviewRoi` / `renderReviewRoiLegend`, behaviour unchanged). The preference (O / button) is the same flag as in the review viewer (`review.showRoi`).
- `draw()` is **not edited inside**: it is renamed `drawCanvas()` and a new `draw()` calls `drawCanvas()` then `syncFixerZone()`, so the zone layer follows every redraw (image change, resize, drag) with no change to the editing code. `syncFixerZone` skips work when nothing it depends on changed.
- No ghost zones: `state.zoneName` is cleared when navigation starts and set to the new image name only right before the final `draw()` of `renderDatasetImage` / `renderRawImage` (2 lines each). The zone therefore always belongs to the image that is on the canvas.
- No image / camera: an image without a camera name (or a failed request) shows the legend warning + orange badge "Camera unknown — crop zone not shown (…)"; no zone, no error. Before an image is drawn the bar says "Zone not available". The bar is hidden outside the Dataset Fixer (Video Labeler: no zone).
- A `ResizeObserver` on `.canvasWrap` redraws (deferred with `requestAnimationFrame`; without it the bar toggling inside the callback produced the "ResizeObserver loop" warning).
- Key **O** in Dataset Fixer mode toggles the zone (`installHotkeys`, `app.js` ~2727). It is not bound in Video or Background mode.

**c. Key collisions checked (all handlers).** Review branch: arrows, K, D, R, O, Esc, + new B. Modal branch (class picker): Esc, Enter, ArrowUp/Down — B/O never reach it (it returns first). Flag panel branch: Esc, Enter, 1-4 — returns first. Then: X (video+dataset), new B (video+dataset, not typing, no modifiers), then dataset: new O, ArrowLeft/Right; bg mode: B/S/arrows (unchanged, B stays "background"); video: P, Home, End, Shift+arrows. No overlap.

**d. Not touched:** `flag_store.py`, `server.py`, `camera_roi.py` are byte-identical to `main` (`git diff main` empty for them) — so `FlagStore(...)`, `add_auto_flag()`, schema v1, `owner_decision`, `replaced_auto`, `relabel_snapshot`, and the shape of every review endpoint response are unchanged. Save / export / label code untouched. `IntelliCup/tests/data4_audit.py` imports only `flag_store.py`, which did not change.

**Files:** `web_labeler/static/app.js` (+134/−5: `setMode` 625, `draw`/`drawCanvas` 729-779, render fns 1363-1411, `closeReviewPanel` 1966/1974, shared helpers 2229-2268, new peek + zone block 2577-2666, hotkeys 2679/2727-2739, init 3466), `index.html` (+14: fixer zone bar 180, canvas overlays 188-190, hints 194/358/405), `styles.css` (+11, 678-688).

## 3. Testing

Test instance from the worktree on `:8011` (`LABELER_RAW_REVIEW_DIR` = copy of the production sidecars, `LABELER_VIDEOS_DIR` / `LABELER_OUTPUT_DIR` = scratch, worktree `labeler_config.json` temporarily pointed at a scratch datasets dir and restored with `git checkout` before the commit). Browser: headless Firefox via geckodriver (raw WebDriver HTTP, real key and pointer actions). Production `:8000` (PID 3239295, started 19:43:50) was not touched.

**Editing tests only on a copy:** a scratch dataset `bottles_tiny` = 10 image+label files *copied* from RAW (9 real, 3 cameras, plus one copy renamed `zz_noname_check.jpg`). No real dataset or RAW file was ever opened for saving; `find /opt/intellicup/datasets/raw -newer <marker>` returned 0.

- **Dataset Fixer (34/34):** B hides boxes and restores the canvas pixel-identically; zone SVG unchanged while B held; auto-repeat keeps it on; drag on the canvas and clicks on every list control while B held do nothing; after release drag + modal + save + list delete work; window blur restores; B keydown from a select ignored; zone matches `cameras.yaml` on all three cameras (rect and 2 polygons, ≤ 2.1 px incl. the 2 px stroke; legend camera correct) for all 9 images; O hides/shows; navigating with B held keeps peek on and the zone follows.
- **Byte comparison of labels (14/14):** the label files after "B episode (blocked drag + blocked list clicks) + Save (overwrite)" are byte-identical to those after a plain Save from the same start, all 10 files. An intentional edit after release changes exactly one label file. Delete via list works after release. No-camera image → note, no zone, boxes still drawn. Zone layer hidden in Video mode.
- **Review viewer + regression (46/46):** boxes hidden/visible per B; zone unchanged; indicator; ArrowRight and 12 fast arrows with B held → never more boxes than the current image has; blur; select; K / D / R each wrote the right `owner_decision` to the sidecar copy (R also `relabel_snapshot`) and Undo cleared it; filters (all, pending, relabel, source, pool); sort newest/oldest (+ source, class run); Show metrics; Esc while B held closes and resets; O toggles the zone. In RAW mode: X on an unflagged image → `manual_goca` + badge, Unflag removes it; X on an `analyzer_auto` and on a `data4_audit` image replaces it with `replaced_auto`, Unflag restores the automatic flag; zone drawn in RAW mode; B works there.
- **Video Labeler (10/10):** loads a copied mp4, canvas draws a frame, B/indicator, drag locked while B held and opens the modal after release, Esc closes it, O does nothing, no Fixer zone, P play/pause; no console errors (`error`, `unhandledrejection`, `console.error` hooked) in any of the four runs.
- Production sidecar md5 identical before and after (below).

**NOT tested:**
- Chrome, Windows, a real human's hands (Firefox headless only); a physically held B on a real keyboard (WebDriver `keyDown` without `keyUp`).
- Dataset Fixer overwrite Save on a **real** dataset / RAW: by design never done. RAW mode has no canvas editing at all (canvas mousedown needs `datasetLoaded`), so the lock is only meaningful in Load Dataset mode and the Video Labeler.
- Editing inside the Video Labeler beyond "drag opens the modal / locked while B" (no annotation was saved from video), the Background Labeler and the Analyzer tab (B stays "background" there; untouched code).
- `tests/data4_audit.py` was not run (it runs YOLO and writes flags); its dependency `flag_store.py` is byte-identical to `main`.
- A missing/unreadable `cameras.yaml` in the Fixer UI (same code path as the review viewer's, which reported `unknown` earlier; here only the no-camera-in-filename case was exercised).
- Zone accuracy in the Fixer for `sank_desno` was measured on 1 image only (the copy has 1).

**Pre-existing finding (not caused by this change):** Dataset Fixer *Save (overwrite)* rewrites every label of the loaded dataset through pixel coordinates, so even with no edit the label bytes change by up to 1 px (e.g. `0.539453 0.463426 0.040365 0.129630` → `0.539323 0.463194 0.040625 0.130093`) and a trailing newline differs; a second plain save is also not always identical to the first. That is why the byte comparison above is against a plain Save, not against the original files. Written up in `issues/dataset_fixer_save_requantizes_labels.md`.

**Production sidecar md5 (`/opt/intellicup/datasets/raw_review/*_flagged.json`), identical before and after:**

```
01684ad3079a9eab970ea28eda5c3cb0  bottles_flagged.json
6c0ba9c76a5ee884144696b0f0a49490  cups_flagged.json
e4be7b94a76ab7e6757f0fc136889d92  glasses_flagged.json
518744e8e9d9343c050cc2d3fc25d049  pitchers_flagged.json
f89aff8af4b8b5cfee87ab8aa130caec  shots_flagged.json
```

## 4. Merge and restart (when approved)

```
cd /home/intellicup/Projects/labelingTool
git merge --ff-only mdq-peek-and-zone-fixer      # main must still be at 4d22982
# static files only: no server restart is needed (index.html/app.js/styles.css are served no-store).
# In the browser: Ctrl+Shift+R.
git worktree remove ../labelingTool_wt_peek && git branch -d mdq-peek-and-zone-fixer
```

Nothing here changes `server.py`, so restarting `:8000` is **not** required (a restart is harmless but unnecessary). Rollback: `git revert --no-edit <this commit>` and Ctrl+Shift+R.
