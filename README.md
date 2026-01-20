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
3. Double-click `run_web_labeler.bat`
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

- `output/<video_stem>/images/*.jpg`
- `output/<video_stem>/labels/*.txt`

If multiple models are used on the same frame, labels are written as separate files:

- `labels/<base>__<model>.txt`

### Export filename format (important)

Export filenames are designed to be robust and pipeline-friendly:

`<TIMESTAMP>_<VIDEO_PREFIX>_f<FRAME>.jpg/.txt`

Example:

- `20251205090046_BLAZNAVAC_NVR_01_G_SANK_LEVO_f001036.jpg`

Rules:

- The `{GUID}` part is removed (to avoid Windows path limits).
- **BAR_COUNTER_INFO** (e.g. `SANK_LEVO` / `SANK_DESNO`) is required.
  - If it’s missing in the video name, Export will ask you to choose one.

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


