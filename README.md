## Labeling Tool (browser version)

This repo contains a **local browser app** that replicates the workflow of `labelingMvpV6_1.py`:

- Load a video
- Play/Pause + jump (Start/End, ±50)
- Draw boxes
- Pick **Model + Class** (type to filter)
- Export frames as `.jpg` + YOLO `.txt`

### Quick start (Windows)

By default, the app stores everything **outside the project** in your home folder:

- **Windows**: `C:\Users\<you>\LabelingToolData\`
- **Linux**: `~/LabelingToolData/`

1. Put your YOLO `*.yaml` files in the configured `Models/` folder
2. Put your videos in the configured `videos/` folder (or use **Upload…** in the browser UI)
3. Install Python 3.10+ from python.org (check **“Add Python to PATH”**)
4. Double-click `run_web_labeler.bat` (it creates a local `.venv` automatically)
4. Open the browser at `http://127.0.0.1:8000`

### Quick start (Linux)

1. Put your YOLO `*.yaml` files in the configured `Models/` folder
2. Put your videos in the configured `videos/` folder (or use **Upload…** in the browser UI)
3. Run:

```bash
chmod +x run_web_labeler.sh
./run_web_labeler.sh
```

4. Open the browser at `http://127.0.0.1:8000`

**If you see “No module named pip” on Linux**: install pip first (example for Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y python3-pip
```

### Output layout (simple)

Exports go to:

- `output/<VIDEO_PREFIX>/<MODEL>/images/*.jpg`
- `output/<VIDEO_PREFIX>/<MODEL>/labels/*.txt`

**Important**: For any labeled frame, the image and label always share the same base filename:

- `.../images/<base>.jpg`
- `.../labels/<base>.txt`

### Export filename format (important)

Export filenames are designed to be robust and pipeline-friendly:

`<TIMESTAMP>_<VIDEO_PREFIX>_f<FRAME>.jpg/.txt`

Example:

- `20251205090046_BLAZNAVAC_NVR_01_G_SANK_LEVO_f001036.jpg`

Rules:

- The `{GUID}` part is removed (to avoid Windows path limits).
- **BAR_COUNTER_INFO** (e.g. `SANK_LEVO` / `SANK_DESNO`) is required.
  - If it’s missing in the video name, Export will ask you to choose one.

## Dataset Fixer (second tab)

Use this when you already have a YOLO dataset and want labelers to **fix** boxes/classes.

### Dataset folder structure

Put datasets into the configured datasets folder (default: `~/LabelingToolData/datasets`):

- `<datasets_dir>/<dataset_name>/images/*`
- `<datasets_dir>/<dataset_name>/labels/*.txt`

Image and label filenames must match by stem (e.g. `img_001.jpg` ↔ `img_001.txt`).

### Model naming warning (recommended)

It’s recommended that the dataset folder name contains the model name (e.g. `Bottles_run1`), so labelers load the correct model.
If it doesn’t, the UI warns before loading.

### Saving

From the Dataset Fixer tab:

- **Save (overwrite)**: writes fixed labels into the same dataset folder
- **Save as _fixed**: writes into `<dataset_name>_fixed/` (copies/hardlinks images + writes labels)

## Background Labeler (third tab)

This module is for generating **high-quality negative/background samples fast** (no boxes).

### Flow

1. Select source dataset (`<dataset>/images`)
2. Select target model (the model you’re generating negatives for)
3. Optional: camera filter (e.g. `SANK_DESNO`)
4. Rapid decisions:
   - **B** = background
   - **S** = skip / contains target object

### Output structure

Only images marked as background are copied to:

`<datasets_dir>/<dataset_name>_background/<model_name>/`

- `images/` (copied/hardlinked)
- `labels/` (empty `.txt` files with the same stem as the image)

Nothing overwrites your original dataset.

### Hotkeys

- `P`: play/pause
- `Shift + Left/Right`: step -50 / +50 frames
- `Home/End`: jump to start / end

### Optional configuration (advanced)

You can change folders via environment variables:

- `LABELER_MODELS_DIR`
- `LABELER_VIDEOS_DIR`
- `LABELER_OUTPUT_DIR`
- `LABELER_HOST` (default `127.0.0.1`)
- `LABELER_PORT` (default `8000`)

Or edit `labeler_config.json` in the project root (created automatically on first run).

**Tip for Windows installs**: `labeler_config.json` uses `~/LabelingToolData` so it works on both Windows and Linux.


