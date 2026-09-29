# 2026-09-29 — Review queue redesign + X on auto-flagged images

Two changes to the labelingTool web labeler, both on 2026-09-29. Written from the session's own
work and `git log`/`git diff`; anything not known is marked **nepoznato**.

| | commit | subject |
|---|---|---|
| A | `97b6ce7` | Review queue: Dataset-Fixer-style viewer (big image + bboxes, arrows, K/D, virtualized side list) |
| B | `1bc63f8` | Flags: Goca's manual flag can replace an undecided analyzer_auto/data4_audit flag |

Both are on `main` (fast-forward merge from branch `mdq-review-viewer-redesign`, which points at the same commit).
Line numbers below are from `main` at `1bc63f8`.

## 1. Context

- **Review queue did not render normally.** With ~690 flagged items (566 of them `manual_goca` shots flags added on 2026-09-29 between ~11:14 and ~14:09) the queue looked glitchy.
  - *Confirmed (code + logs + data):* the server was healthy (`GET /api/review/queue` 200, ~546 KB, ~0.07 s, valid JSON; no 5xx/tracebacks in `labeler_8000.log`), the sidecars were clean (all 690 entries well-formed, all RAW images present). Source images are 3840x2160 JPEGs.
  - *Hypothesis, never confirmed in a browser:* the old UI built one card (with its own `<img>`) for every item, and every action (filter, sort, Refresh, each decision — twice per decision) tore down and rebuilt all ~677 cards, which resets scroll and makes images reload/reflow. The UI code had not changed since 2026-09-25; only the data volume had.
  - The redesign removes that pattern, but whether it was *the* cause of the visible glitch was not separately measured before/after. **nepoznato.**
- **Goca could not flag auto-flagged images.** Pressing X on an image already flagged by `analyzer_auto` or `data4_audit` opened the panel, but Flag/categories were disabled ("can no longer be changed here") and the server would have answered 409 (`FlagConflict`). Owner's requirement: Goca must be able to set her manual flag anyway.

## 2. Change A — review queue redesign (`97b6ce7`)

Files: `web_labeler/server.py`, `static/app.js`, `static/index.html`, `static/styles.css` (4 files, +463/−222).

**Layout** (`index.html` `#reviewPanel` and children; `styles.css` "Owner review queue (MDQ-6) — viewer layout" block):
- Left: 260 px column with the four filters (source, pool, decision, sort) and the item list. Row = 96x54 thumbnail, class, source badge, coloured left edge by decision (pending / keep / approve_delete).
- Middle: one big image (`#reviewStage`, fits the window without scrolling) with bboxes, prev/next arrow buttons and a "n / N" counter for the *filtered* list.
- Under the image (`#reviewInfo`): pool · class, source, filename, Goca's comment in full (large), short reason (full text in tooltip), flagged-at/by, decision text, **Approve delete (D)**, **Keep (K)**, **Undo** (only when a decision exists), **Full res** toggle, **Show metrics** (MDQ-7, unchanged behaviour).
- No Relabel (waits for MDQ-15c-1).

**Behaviour** (`app.js`):
- List virtualization: fixed 64 px rows absolutely positioned by index, only a window (visible rows ± `REVIEW_WINDOW_PAD`=40) exists in the DOM (`renderReviewWindow`, ~2054). Observed 52–92 rows in the DOM for 689 items.
- The filtered list `review.view` is rebuilt only on open / Refresh / filter change (`rebuildReviewView`, ~2024). A decision changes one row's class (`updateReviewRow`), never rebuilds the list, and the decided item stays in the list until a filter change / Refresh.
- After a successful decision the viewer auto-advances (`setReviewDecision`, ~2283; not after Undo, not past the last item). K/D on an item already decided that way just moves on (no re-write). A failed POST shows the error and does not advance.
- Preload of the next and previous image (`preloadReviewImage`).
- Ghost-bbox protection (`showReviewItem` ~2148, `drawReviewBoxes` ~2124): boxes are cleared the instant the item changes; drawn only from the `onload` of the *new* image; `onload` is ignored unless `review.token` and the current `image_key` still match; scaling uses `naturalWidth/naturalHeight` inside the `object-fit: contain` letterbox; a `ResizeObserver` on the stage redraws on resize (`installReviewPanel`, ~2391).
- Hotkeys (`installHotkeys`, review branch, comment "Its own keys"): Left/Right = prev/next, K = keep, D = approve delete, Esc = close panel (writes nothing). Ignored while typing / focus in a `<select>`, with Ctrl/Meta/Alt; K/D also ignored on key auto-repeat. Filters blur after change so arrows page instead of changing the select. The review branch returns early, so no RAW/video hotkey acts behind the panel.

**Server** (`server.py`, `review_image` ~875, cache ~869–912): optional `w` query parameter (64–3840 px target width). Default (`w` omitted) is unchanged: 480 px thumbnail; `thumb=0` still returns the original file. Resized JPEGs are cached in an in-process LRU (`OrderedDict`, keyed path + mtime + width, bounded to 200 MB, lock-protected). The big image uses `w=1600`, list thumbnails `w=160`. Added `from collections import OrderedDict`.

**Removed (MDQ-14 lightbox and old grid), and nothing else references them:**
- `app.js` functions: `openReviewLightbox`, `closeReviewLightbox`, `renderBoxesOverImage` (replaced by `drawReviewBoxes`), `renderReviewCard`, `renderReviewGrid`; field `review.lightbox` and the lightbox branch in `installHotkeys`.
- `index.html`: `#reviewLightbox`, `#reviewLightboxTitle`, `#reviewLightboxClose`, `#reviewLightboxImg` (and the old `#reviewGrid`).
- `styles.css`: `.reviewLightbox*`, `.reviewCard*`, `.reviewGrid`, `.reviewThumb*`, `.reviewBody`.
- The lightbox was only ever used inside the review queue (checked against the `*.bak_pre_reviewviewer_20260929` backups); no use in RAW view, Dataset Fixer or Video mode. Every `$("id")` in `app.js` exists in `index.html`.
- The "full resolution" capability lives on as the **Full res** button (loads `thumb=0`).

## 3. Change B — X on auto-flagged images (`1bc63f8`)

Files: `web_labeler/flag_store.py`, `server.py`, `static/app.js` (3 files, +74/−16).

**`flag_store.py`:**
- `_is_replaceable_by_manual` (143): a manual flag may be written over an entry that has no `owner_decision` and whose source is `manual_goca`, `analyzer_auto` or `data4_audit`. `_is_mutable_by_manual` is unchanged and still governs Unflag.
- `add_manual_flag` (238): when it overwrites an automatic entry it stores that entry in the new entry's **`replaced_auto`** (`source, category, signal, comment, flagged_at, flagged_by, flag_context`). Re-saving her own flag keeps the existing `replaced_auto`.
- `unflag_manual` (288, new): removes the manual flag; if `replaced_auto` exists the automatic flag is written back exactly as it was (`owner_decision` null), otherwise the entry is deleted as before. Returns `{"removed": bool, "restored": entry|None}`.
- `remove_manual_flag` (315): same signature and bool return, now calls `unflag_manual`.
- Module docstring updated.
- **Why `replaced_auto` is a top-level key and not inside `flag_context`:** `flag_context` is a string (`retroactive_review` / `forward_labeling`) that other code reads; turning it into an object would change the schema. A top-level optional key is the same pattern as the existing `video` / `frame_idx` / `annotation_classes` extras, so `schema_version` stays 1.

**`server.py` (~751–756):** `DELETE /api/raw/flag` uses `unflag_manual` and returns `removed` plus the new `restored`. `GET /api/review/queue` shape untouched. `/api/frame/flag` (video frames) still goes through `remove_manual_flag`.

**`app.js`:**
- `flagCanSave` (1702, new): may save over no flag, own undecided flag, or undecided `analyzer_auto`/`data4_audit`. `flagIsMutable` (1706) is kept and now only decides whether Unflag is shown.
- `renderFlagPanel` (1714): for an auto flag it says "Automatically flagged by … Pick a reason and save to replace it with your manual flag (the automatic one is kept and comes back if you Unflag)"; for her own replaced flag it notes that Unflag brings the automatic one back. `selectFlagCategory`/`saveFlag` use `flagCanSave`.
- `removeFlag` (1838): sets `state.rawFlag` from `restored`, so the badge shows the restored automatic flag; status text says so.
- Review panel (`renderReviewInfo`, ~2248): extra line "Replaced an automatic <source> flag (<metric>: <value>)" (full `replaced_auto` JSON in tooltip).

**Still locked:** any entry with an `owner_decision` (auto or manual) — Flag button disabled, server answers `FlagConflict`; the decision is never overwritten or lost.

## 4. Compatibility

- `schema_version` stays 1; `replaced_auto` is optional, old sidecars are read by the same code with no migration (`FlagStore.list_all()` on the production sidecars returned 35/33/30/14/578 entries).
- Unchanged: `FlagStore` constructor, `add_auto_flag()` signature and behaviour, `set_owner_decision`, `list_all`, `list_entries`, `get`, `describe_reason`, `_raw_image_path`, `_raw_label_boxes`, `GET /api/review/queue`, `POST /api/review/decision`, `GET /api/review/metrics`.
- `add_auto_flag` returns `None` (no-op) when the key already has any entry — verified on a copy: re-running it over a manual (replaced) flag changes nothing.
- `IntelliCup/tests/data4_audit.py` loads `flag_store.py` from file and only calls `add_auto_flag`/`list_all`-style APIs; `flag_store.py` public API is unchanged. **The script itself was not run.**
- RAW was never written; `archive_raw.py` was not run.

## 5. Behaviour changes users should know

- Review queue is now a viewer, not a card grid: arrows (or ←/→ keys) browse, **K** = Keep, **D** = Approve delete, **Undo** clears a decision (does not advance), Esc closes the panel.
- After K/D the queue moves to the next item automatically; the just-decided item stays in the list, marked by colour, until Refresh or a filter change.
- **Full res** button loads the original image; default is 1600 px wide.
- The old click-thumbnail lightbox is gone.
- X on an image flagged by `analyzer_auto`/`data4_audit` now works and **replaces** that flag with Goca's manual one; Unflag on it restores the automatic flag. X on an owner-decided image still does nothing.
- X does not work while the review panel is open (unchanged: the panel swallows keys).

## 6. Testing

Test instance on `:8010` with `LABELER_RAW_REVIEW_DIR` pointing at a **copy** of the production sidecars in a temp dir (scratchpad), production `:8000` never used. Headless Firefox 156 driven through `geckodriver` (raw WebDriver HTTP; Playwright/Chromium not installed). Production sidecar md5s were identical before and after each test run.

- *Change A:* list renders 689 pending items, 52–92 DOM rows; 25 rapid → presses land on 26/689 with boxes matching the API; 30 rapid (→ → ←) bursts never showed boxes while the new image was loading; resize re-places boxes; K/D wrote only to the copy (2 entries), auto-advance works, DOM rows not rebuilt (all 52 rows kept), K in a `<select>` does nothing, keep-filter and Esc work; Show metrics works; `review_image`: `w=160` ≈ 6.6 KB, `w=1600` ≈ 308 KB, cached repeat ~1 ms.
- *X flag in RAW mode (Dataset Fixer):* X → category → comment → save wrote a `manual_goca` entry; badge shown; Unflag removed it; review queue showed it, and after removal no longer; with the panel closed K/D changed nothing and ←/→ only did normal RAW navigation.
- *Change B:* X on a `data4_audit` image (glasses/NES_KAFA) and on an `analyzer_auto` image (glasses/VISOKA_USKA_PROVIDNO) replaced them, `replaced_auto` held the original signal, review queue showed the manual flag and the "Replaced…" line, Unflag restored both to exactly the original entries; an image with `owner_decision` (set by hand in the copy) stayed locked and unchanged.
- *FlagStore unit checks on a copy:* replace, `add_auto_flag` no-op over manual, re-save keeps `replaced_auto`, unflag restores original, decided auto locked, decided manual locked with `replaced_auto` kept, plain manual unflag deletes.

**NOT tested:**
- `tests/data4_audit.py` was not run.
- X in Video mode (frame flags); that path only received the shared `flagCanSave` change.
- Chrome, Windows, any browser other than headless Firefox.
- Rapid browsing by a human on production data; live use by Goca; the original glitch was never reproduced in a browser, so the before/after improvement is unmeasured.
- The repo has no automated test suite; these were ad-hoc scripts (in the session scratchpad, not committed).

## 7. Known issues not touched

- `analyzer_auto` flags from `kartice_review` have a signal without `value`, so the RAW badge reads e.g. `kartice_review_color_mismatch: undefined` (display only; `describe_reason` and the badge use `metric: value`).
- Goca's comments: 344 of 567 manual flags have an empty comment (analysis only, no code change).
- Video-frame (`_forward/…`) flags have no review image (unchanged behaviour: "not in RAW yet").

## 8. Production state

- `main` = `1bc63f8` (contains `97b6ce7`). Working tree has unrelated uncommitted `.claude/rules/analyzer.md`, `.claude/rules/export.md` and untracked files (backups, logs, `.claude/agents/`) — deliberately not committed.
- Server restarts: I (Claude) did **not** restart `:8000`. At documentation time the process on `:8000` was PID 3219979, started **2026-09-29 17:51:33** (from `ps`), i.e. after commit `1bc63f8` (17:48:33), so it was started with the current code. The previous process (PID 2637903) had been started 2026-09-25 16:58. Who restarted it, and any restarts in between: **nepoznato**.
- Backups (untracked, next to the originals in `labelingTool/web_labeler/`):
  - Before change A: `server.py.bak_pre_reviewviewer_20260929`, `static/app.js.bak_pre_reviewviewer_20260929`, `static/index.html.bak_pre_reviewviewer_20260929`, `static/styles.css.bak_pre_reviewviewer_20260929`.
  - Before change B: `flag_store.py.bak_pre_replaceauto_20260929`, `server.py.bak_pre_replaceauto_20260929`, `static/app.js.bak_pre_replaceauto_20260929`.
- Screenshots (15 files, temporary, session scratchpad — may be deleted): `/tmp/claude-1000/-home-intellicup-Projects/2805e03d-3e79-425b-9ebe-aa379352c907/scratchpad/shots/` (`01`–`07` review viewer, `10`–`13` RAW flag, `20`–`22` replaced auto flags).
- Production sidecars (`/opt/intellicup/datasets/raw_review/*_flagged.json`) were only read by me; the owner's own review decisions changed `shots_flagged.json` during the day (not by these tests).

## 9. Rollback

```bash
cd /home/intellicup/Projects/labelingTool
git revert --no-edit 1bc63f8 97b6ce7      # newest first; creates two revert commits on main
# then restart the labeler on :8000 (server.py changes)
```

Reverting only B: `git revert --no-edit 1bc63f8`. Sidecar entries that already carry `replaced_auto` stay valid: the old code ignores unknown keys, `schema_version` is still 1. After a revert, Unflag on such an entry simply deletes it (the saved automatic flag would remain visible in the file but is no longer restored), and X on auto-flagged images is locked again.
