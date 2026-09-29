# 2026-09-29 — Review queue: camera crop zone overlay

A read-only overlay in the review queue viewer showing, for each image, the rectangle that
IntelliCup training crops the image to, plus the camera's `bar_roi` / `pickup_roi` polygons.
Written from the session's own work and `git diff`. Anything not known is marked **nepoznato**.

| | commit | subject |
|---|---|---|
| C | this commit on branch `mdq-roi-overlay` (worktree `../labelingTool_wt_roi`) | Review queue: camera crop zone overlay (crop rectangle + bar/pickup ROI) with O toggle |

**Not merged into `main`.** The owner approves the merge. Branched from `main` at `b5c5178`.
Line numbers below are from the branch.

## 1. Context

- IntelliCup training crops every RAW image to the bounding rectangle of the camera's `bar_roi` + `pickup_roi` polygons. Labels whose centre falls outside that rectangle are dropped.
  - Camera from filename: `IntelliCup/training_pipeline/tmp_training_builder/dataset_builder.py` `_detect_camera` 46-55.
  - Rectangle: `_camera_crop_rect` 58-67.
  - Crop and label drop: `_apply_roi_crop` 564-596, called from `write_split` 637-645.
- Production inference crops every frame to the same rectangle plus 20 px (`intellicup_deep_sort/tracking/deep_sort_tracker_simplified_v2.py` 382-388).
- So an object outside the rectangle never reaches training and is never seen by the detector in production.
- The coordinates come from `/opt/intellicup/config/cameras.yaml` (a symlink to `IntelliCup/config/cameras.yaml`).
- Until now the owner and Goca had no way to see where that line is while reviewing. This matters most for shots put down outside the bar (on the counter extension).

## 2. Change

Files: new `web_labeler/camera_roi.py`, plus `web_labeler/server.py`, `static/app.js`, `static/index.html`, `static/styles.css`.

**`camera_roi.py` (new, 116 lines):**
- The camera detection and rectangle logic is **replicated** here, not imported. The module docstring cites every source file and line.
  - Why not import: `dataset_builder.py` is a training module. At import time it pulls in its augmentation engine, and its config path is hardcoded.
  - This repo has no code dependency on IntelliCup, only on-disk data files (same as `article_map_path`).
- `detect_camera`: first camera in `cameras.yaml` order whose lowercased name is a substring of the lowercased file name. This is the same rule as training and as production `get_roi_points` (v2 135-139).
- `camera_roi` never raises. If the camera is not recognised, the config is missing or unreadable, or the camera has no polygons, it returns `status: "unknown"` with a `reason`.
- `cameras.yaml` is cached and re-read when its mtime changes.

**`server.py` (+10 lines, additions only):**
- 63: `from web_labeler import camera_roi as _camroi`.
- 1019-1027: new `GET /api/cameras/roi?image_key=<class/filename>`. It returns:
  - `camera`, `status` (`ok` or `unknown`), `reason`;
  - `crop_rect` `{x0, y0, x1, y1}`;
  - `bar_roi`, `pickup_roi` (normalized lists of points);
  - `inference_pad_px` (20).
- The config path is overridable with `LABELER_CAMERAS_YAML` and defaults to `/opt/intellicup/config/cameras.yaml`.
- Only the file name inside `image_key` is used. The route touches no RAW file and no sidecar.
- Unchanged: `GET /api/review/queue`, `POST /api/review/decision`, `/api/review/image`, `/api/review/metrics`.

**`index.html`:**
- 349: the key hint now includes "O zones".
- 385-390: `#reviewRoiBar`, which holds the toggle button `#reviewRoiBtn` and the legend `#reviewRoiLegend`.
- 393-394: `<svg id="reviewRoiSvg">` and `#reviewRoiBadge` inside `#reviewStage`.

**`styles.css` (678-694):** overlay, badge, legend and swatch styles. Colours:
- crop rectangle: white dashed line;
- outside the rectangle: dimmed to about 45 % black;
- `bar_roi`: green `#4ade80`;
- `pickup_roi`: orange `#fb923c`;
- bboxes: blue, as before.

**`app.js`:**
- `review` gets three new fields: `showRoi` (default `true`), `roiReq` and `roiData` (caches per `image_key`) (1916-1918). Refresh clears both caches (1965-1966). Closing the panel clears the overlay (1958).
- `reviewLetterbox` (2131): the letterbox calculation from `drawReviewBoxes`, pulled out so the boxes and the zone share it. `drawReviewBoxes` (2140) now calls it and places boxes exactly as before.
- New functions:
  - `fetchReviewRoi` (2162): a failed request is shown to the user and re-fetched next time.
  - `clearReviewRoi` (2182), `reviewImageReady` (2189), `svgEl` (2194).
  - `drawReviewRoi` (2200): SVG placed over the letterbox with an even-odd dim path, both polygons and the dashed rectangle. On top of that come the bbox divs (the SVG sits under them).
  - `renderReviewRoiLegend` (2239): camera name, legend and the note "Van isečenog pravougaonika objekti ne ulaze u trening ni u produkciju", or the unknown warning.
  - `toggleReviewRoi` (2263).
- How `showReviewItem` avoids ghost zones (same pattern as the bboxes):
  - The zone is cleared the instant the item changes.
  - It is drawn from the new image's `onload`, and from the zone fetch's callback only while `review.token` still matches and the image has finished loading.
- `ResizeObserver` (`installReviewPanel`) redraws the zone together with the boxes. The button handler is at 2544.
- Hotkey **O** (2565) is handled in the review branch of `installHotkeys`. It is ignored while typing or with focus in a `<select>`, with modifiers held, and on key auto-repeat.

**Not touched:**
- `flag_store.py`: byte-identical to `main`. So the `FlagStore` constructor, `add_auto_flag()`, sidecar `schema_version` 1, `owner_decision`, `replaced_auto` and `relabel_snapshot` are all unchanged.
- The export format: no change to naming, layout or `data.yaml`.

## 3. Dataset Fixer: not added (by decision)

- The Dataset Fixer RAW view (`renderRawImage`, `app.js` ~1376) draws on the shared editable `<canvas>` through `draw()`.
- That same canvas and `draw()` are used by the Video Labeler for interactive bbox drawing and editing.
- It is **not** the review viewer's `<img>` + DOM overlay, so the change there is not low-risk. It was not made.
- Where it would go: in `draw()`, after the image and before the annotations, when `state.mode === "dataset" && state.rawLoaded`. Use `state.datasetImageName` for `/api/cameras/roi` and the canvas image transform for the coordinates.

## 4. Testing

Test instance on `:8010`, run from the worktree, with `LABELER_RAW_REVIEW_DIR` pointing at a **copy** of the production sidecars (session scratchpad). Browser: headless Firefox 156 via `geckodriver`, raw WebDriver HTTP.
- Firefox's WebDriver cannot see the page's top-level `const`s. The test scripts injected a `<script>` that sets `window.review`/`window.state`; app code was not changed for this.
- Production `:8000` was never used or restarted: PID 3223569, started 18:11:56, before and after.

**Numerical checks:**
- API against `cameras.yaml` for `sank_levo`, `sank_desno` and `sank_tocilica`:
  - `crop_rect` equals min/max over `bar_roi` + `pickup_roi`, and the polygons equal the yaml.
  - levo 0.2731/0.1389/0.8656/0.9956, desno 0.1756/0.3244/0.5981/0.9856, tocilica 0.2631/0.2744/0.6356/0.9867.
- Parity with IntelliCup's own functions: `dataset_builder._detect_camera` and `_camera_crop_rect`, imported read-only under `INTELLICUP_MODELS`, compared with `camera_roi.py`.
  - 974 RAW filenames (every 50th): 0 mismatches.
  - All 5 cameras: identical.
- On screen: the dashed rectangle's client rect was measured against the image's letterbox, computed independently from the `<img>` rect and its natural size.
  - All three cameras: ≤ 0.10 px off at 1090 px displayed width.
  - Polygon vertices equal the yaml to 1e-6.

**Browser: 40/40 checks passed, plus unknown-camera 5/5 and X-flag 6/6:**
- Toggle: O and the button hide/show the zone, bboxes are unaffected, and the choice persists to the next image.
- Rapid browsing: 30 fast ←/→ presses, sampled after each one. The zone and boxes never appeared on a loading image or with the wrong camera's rectangle, and exactly one rectangle and 2 polygons remain afterwards.
- Resize to 1100x780 and 1900x1100: re-placed within 0.09 px.
- Full res (3840 px natural width): still aligned.
- K / D / R each wrote to the copy and auto-advanced; Undo after each.
- Pool filter (shots, 513 items): zone correct. Decision filter "All" (690).
- Show metrics renders.
- O with focus in a `<select>` does nothing. Esc closes the panel and clears the overlay.
- Unknown camera: the test instance was restarted with `LABELER_CAMERAS_YAML` pointing at a copy without `sank_desno`.
  - Desno images show the orange badge "Camera unknown — crop zone not shown (…)" and a warning in the legend. No zone is drawn and the bboxes are still drawn.
  - Levo is still drawn.
  - API: unknown file name → `status: unknown`, HTTP 200. Missing `image_key` → 422.
- X flag in RAW mode (Dataset Fixer, cups/CAFFE_LATTE):
  - ←/→ navigation works.
  - X → category → comment → save wrote one `manual_goca` entry to the copy, and the badge showed.
  - The review queue listed it with its zone (`sank_desno`).
  - K + Undo worked.
  - Unflag returned the copy sidecar to exactly its previous entries.
- FlagStore: loaded with the same importlib pattern `IntelliCup/tests/data4_audit.py` uses, on a scratch copy. Constructor and `add_auto_flag` signatures unchanged; `list_all` → 35/33/30/14/578.

**Production sidecar md5 (`/opt/intellicup/datasets/raw_review/*_flagged.json`), identical before and after:**

```
01684ad3079a9eab970ea28eda5c3cb0  bottles_flagged.json
6c0ba9c76a5ee884144696b0f0a49490  cups_flagged.json
e4be7b94a76ab7e6757f0fc136889d92  glasses_flagged.json
518744e8e9d9343c050cc2d3fc25d049  pitchers_flagged.json
f89aff8af4b8b5cfee87ab8aa130caec  shots_flagged.json
```

**NOT tested:**
- `tests/data4_audit.py` itself was not run: it runs YOLO inference and writes auto flags. Only the `flag_store.py` API it uses was checked, and that file is unchanged.
- Chrome and Windows. The owner's Chrome session was deliberately not used.
- Human use on production data.
- A missing or unreadable `cameras.yaml` in the browser. The code path returns `unknown` with a reason; it was not exercised through the UI.
- Video mode and Background Labeler: untouched code, not re-tested.
- The repo has no automated test suite. The scripts were ad hoc (scratchpad, not committed).

## 5. Finding during testing (no action taken)

- `shots` / `SHOT_TRANSPARENT/sank_tocilica_20260316203800_f008849.jpg` (manual_goca, "Gibberish"): the box centre is at y = 0.2595, which is above the crop top y0 = 0.2744 for `sank_tocilica`.
- Per `_apply_roi_crop`, this label is **dropped from training**, whatever the review decision.
- The overlay makes this visible. How many RAW labels are in this situation: **nepoznato** (not counted).

## 6. Behaviour changes users should know

- The review queue now shows the camera crop zone by default:
  - dashed white line = what training (and, +20 px, production) keeps;
  - dimmed area outside it = never used;
  - green = `bar_roi`, orange = `pickup_roi`.
- **O** or the "Zones" button toggles it. The setting is not remembered across page reloads (always on after a reload).
- If the camera can't be derived from the file name, an orange "Camera unknown" note appears instead of the zone.

## 7. Merge / restart / rollback

After the owner approves:

```bash
cd /home/intellicup/Projects/labelingTool
git merge --ff-only mdq-roi-overlay        # main is at b5c5178, the branch is 1 commit ahead
# restart :8000 right after the merge (the new app.js calls /api/cameras/roi; an old server answers 404)
ps -o pid,lstart,args -p "$(ss -ltnp | grep ':8000 ' | grep -oP 'pid=\K[0-9]+')"   # confirm it is run_web_labeler.py
kill <that PID>
nohup /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python run_web_labeler.py >> labeler_8000.log 2>&1 &
# then reload the browser tab (Ctrl+Shift+R)
git worktree remove ../labelingTool_wt_roi && git branch -d mdq-roi-overlay
```

- The production process was started without `LABELER_*` variables (checked in `/proc/<pid>/environ`), so the plain command above matches it.
- Between the merge and the restart, the viewer just shows "Camera unknown — could not load the zone: Not Found". Nothing breaks.

Rollback: `git revert --no-edit <commit>`, then restart `:8000`. No data or sidecar change to undo.
