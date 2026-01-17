#!/usr/bin/env python3
"""
TakeLabeler v6.8 — robust type-to-jump (no auto-open) + fast playback (up to ~x16)

- Model/Class comboboxes are readonly.
- While focused, typing builds a small buffer and jumps to the first item
  whose text CONTAINS that buffer (substring, case-insensitive).
- Buffer resets after a short idle (800 ms). Backspace works.
- Enter = Save.

Keeps:
- Export ALL labeled frames (top bar)
- Speed slider (now 1–8, up to ~x16)
- Start/End buttons (and Home/End)
- HEVC-friendly _safe_seek; P play/pause; Shift+Left/Right = -50/+50
- Per-frame annotation memory + sidebar (current-frame & all)
- No File menu
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
from PIL import Image, ImageTk
import yaml

HARDCODED_MODELS_DIR = Path(
    os.getenv("LABELER_MODELS_DIR", str(Path(__file__).parent / "Models"))
)

@dataclass
class Annotation:
    model: str
    class_name: str
    class_id: int
    x1: int; y1: int; x2: int; y2: int

def clamp(v, lo, hi): return max(lo, min(hi, v))
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_yolo_names_from_yaml(yaml_path: Path) -> List[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names")
    if isinstance(names, dict):
        items = sorted(names.items(), key=lambda kv: int(kv[0]))
        return [v for _, v in items]
    if isinstance(names, list):
        return names
    raise ValueError(f"'names' not found or invalid in {yaml_path}")

def yolo_line(class_id: int, x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> str:
    box_w = x2 - x1; box_h = y2 - y1
    cx = x1 + box_w / 2.0; cy = y1 + box_h / 2.0
    return f"{class_id} {cx/img_w:.6f} {cy/img_h:.6f} {box_w/img_w:.6f} {box_h/img_h:.6f}"

class TakeLabeler(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Yolo Labeler v6.8 — Video Labeler for YOLOv8")
        self.geometry("1320x840"); self.minsize(1150, 760)

        # Video
        self.video_path: Optional[Path] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.fps = 0.0
        self.frame_idx = -1
        self.current_frame_rgb = None
        self.playing = False
        self.play_after_id = None
        self.speed_var = tk.IntVar(value=4)  # default middle
        self.play_step = 3
        self.play_delay_ms = 25
        self._apply_speed(self.speed_var.get())
        self.end_reached = False

        # Display
        self.canvas: Optional[tk.Canvas] = None
        self.canvas_img_tk = None
        self.img_w = 0; self.img_h = 0
        self.scale_x = 1.0; self.scale_y = 1.0
        self.offset_x = 0; self.offset_y = 0

        # Models/classes
        self.models_dir: Optional[Path] = None
        self.model_to_names: Dict[str, List[str]] = {}

        # Output (auto)
        self.output_root: Optional[Path] = None

        # Annotations
        self.ann_by_frame: Dict[int, List[Annotation]] = {}
        self.annotations: List[Annotation] = []

        # Drawing
        self.dragging = False
        self.start_x = 0; self.start_y = 0
        self.temp_rect_id = None

        self._build_ui()
        self._bind_keys()
        self._show_quick_start()

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self); top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Button(top, text="1) Load Models Folder", command=self.on_load_models, width=22).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="2) Load ONE Model", command=self.on_load_one_model, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="3) Load Video", command=self.on_load_video, width=16).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Export ALL", command=self._export_all_labeled, width=14).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Help", command=self.show_help, width=10).pack(side=tk.RIGHT, padx=3)

        mid = ttk.Frame(self); mid.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        self.btn_start = ttk.Button(mid, text="⟵⟵ Start", command=self.goto_start, width=12); self.btn_start.pack(side=tk.LEFT, padx=(2,2))
        self.play_btn = ttk.Button(mid, text="Play [P]", command=self.toggle_play, width=14); self.play_btn.pack(side=tk.LEFT, padx=4)
        self.btn_back = ttk.Button(mid, text="-50", command=lambda: self.step_seek(-50), width=10); self.btn_back.pack(side=tk.LEFT, padx=2)
        self.btn_fwd = ttk.Button(mid, text="+50", command=lambda: self.step_seek(+50), width=10); self.btn_fwd.pack(side=tk.LEFT, padx=2)
        self.btn_end = ttk.Button(mid, text="End ⟶⟶", command=self.goto_end, width=12); self.btn_end.pack(side=tk.LEFT, padx=(2,8))

        ttk.Label(mid, text="Speed").pack(side=tk.LEFT)
        speed_scale = ttk.Scale(mid, from_=1, to=8, orient="horizontal",
                                command=lambda _v: self._on_speed_change(), variable=self.speed_var)
        speed_scale.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        ttk.Label(mid, text="slow").pack(side=tk.LEFT, padx=(6,2))
        ttk.Label(mid, text="fast").pack(side=tk.LEFT, padx=(2,6))

        main = ttk.Frame(self); main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.canvas = tk.Canvas(main, bg="#111", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", lambda e: self.canvas.focus_set())
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        sidebar = ttk.Frame(main, width=360); sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(sidebar, text="Current frame boxes:").pack(anchor="w", padx=6, pady=(2,2))
        self.ann_list_frame = tk.Listbox(sidebar, height=10)
        self.ann_list_frame.pack(fill=tk.X, padx=6)
        ttk.Button(sidebar, text="Delete selected (current frame)", command=self.delete_selected_ann_current).pack(fill=tk.X, padx=6, pady=4)

        ttk.Separator(sidebar).pack(fill=tk.X, padx=6, pady=8)
        ttk.Label(sidebar, text="All annotations (double-click to jump):").pack(anchor="w", padx=6, pady=(2,2))
        self.ann_list_all = tk.Listbox(sidebar, height=14)
        self.ann_list_all.pack(fill=tk.BOTH, expand=True, padx=6)
        self.ann_list_all.bind("<Double-Button-1>", self.on_all_double_click)
        btns = ttk.Frame(sidebar); btns.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(btns, text="Go to selected", command=self.goto_selected_global).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,4))
        ttk.Button(btns, text="Delete selected", command=self.delete_selected_global).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4,0))

        self.status = tk.StringVar(value="Load models and a video to begin.")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)

        self._update_transport_enabled()

    def _apply_speed(self, s: int):
        # approx multipliers by (play_step, delay_ms)
        # ~ 1x, 2x, 4x, 6x, 8x, 10x, 12x, 16x (subjective, depends on codec/CPU)
        mapping = {
            1: (1, 35),   # ~x1
            2: (2, 30),   # ~x2
            3: (4, 25),   # ~x4
            4: (6, 22),   # ~x6
            5: (8, 20),   # ~x8
            6: (10, 18),  # ~x10
            7: (12, 16),  # ~x12
            8: (16, 14),  # ~x16
        }
        step, delay = mapping.get(int(s), (4, 25))
        self.play_step, self.play_delay_ms = step, delay

    def _display_speed_multiplier(self) -> int:
        # purely for UI text – map step to friendly multiplier
        table = {1:1, 2:2, 4:4, 6:6, 8:8, 10:10, 12:12, 16:16}
        return table.get(self.play_step, self.play_step)

    def _on_speed_change(self):
        self._apply_speed(self.speed_var.get())

    def _bind_keys(self):
        self.bind("p", lambda e: self.toggle_play()); self.bind("P", lambda e: self.toggle_play())
        self.bind("<Shift-Left>", lambda e: self.step_seek(-50)); self.bind("<Shift-Right>", lambda e: self.step_seek(+50))
        self.bind("<Home>", lambda e: self.goto_start()); self.bind("<End>", lambda e: self.goto_end())
        self.canvas.bind("<space>", lambda e: self.toggle_play())

    def _show_quick_start(self):
        self.after(400, lambda: messagebox.showinfo(
            "Quick start",
            "1) Load Models Folder or just one model.\n"
            "2) Load Video. Output is created automatically: <video_dir>/output/<video_stem>/\n\n"
            "Draw a box → popup for Model/Class → Save/Delete.\n"
            "P = play/pause, Shift+Left/Right = -50/+50, Home/End = start/end.\n"
            "Use the Speed slider to adjust playback (now up to ~x16).\n"
            "Boxes persist per frame and reappear when you revisit that frame."
        ))

    def show_help(self):
        messagebox.showinfo("Help",
            "Shortcuts:\n"
            "  P .............. Play/Pause\n"
            "  Shift+Left/Right  Step -50/+50 frames\n"
            "  Home/End ........ Jump to start/end\n\n"
            "Tip in assign popup: while a dropdown has focus, just start typing to jump\n"
            "to the first matching item (substring). Backspace edits the buffer. Enter = Save."
        )

    # ---------- Load ----------
    def _load_models_from_dir(self, directory: Path) -> int:
        if not directory or not directory.exists(): return 0
        self.models_dir = directory
        self.model_to_names.clear()
        count = 0
        for yaml_path in list(directory.glob("*.yaml")) + list(directory.glob("*.yml")):
            try:
                names = load_yolo_names_from_yaml(yaml_path)
                self.model_to_names[yaml_path.stem] = names; count += 1
            except Exception as e:
                messagebox.showwarning("YAML skipped", f"{yaml_path.name}: {e}")
        return count

    def on_load_models(self):
        # Try hardcoded first
        cnt = self._load_models_from_dir(HARDCODED_MODELS_DIR)
        if cnt == 0:
            chosen = self._open_fixed_folder_picker("Select folder with YOLO data.yaml files",
                                                    start_dir=HARDCODED_MODELS_DIR if HARDCODED_MODELS_DIR.exists() else None)
            if not chosen:
                return
            cnt = self._load_models_from_dir(chosen)
        if cnt == 0:
            messagebox.showerror("No models", "No valid data.yaml files found.")
            return
        self.status.set(f"Loaded {cnt} model(s) from: {self.models_dir}")
        self._update_transport_enabled()

    def _load_single_model_from_yaml(self, yaml_path: Path) -> bool:
        try:
            names = load_yolo_names_from_yaml(yaml_path)
            self.model_to_names[yaml_path.stem] = names
            return True
        except Exception as e:
            messagebox.showwarning("YAML skipped", f"{yaml_path.name}: {e}")
            return False

    def on_load_one_model(self):
        start_dir = HARDCODED_MODELS_DIR if HARDCODED_MODELS_DIR.exists() else None
        chosen = self._open_fixed_yaml_picker("Select one YOLO data.yaml", start_dir=start_dir)
        if not chosen:
            return
        if self.models_dir is None:
            self.models_dir = chosen.parent
        ok = self._load_single_model_from_yaml(chosen)
        if not ok:
            return
        self.status.set(f"Loaded 1 model: {chosen.stem}")
        self._update_transport_enabled()

    # ---- fixed-size pickers (folder/yaml/video) ----
    def _open_fixed_folder_picker(self, title="Select a folder", start_dir: Optional[Path] = None) -> Optional[Path]:
        start_dir = Path(start_dir or getattr(self, "_model_last_dir", str(Path.home()))).resolve()
        dlg = tk.Toplevel(self)
        dlg.title(title); dlg.transient(self); dlg.grab_set()
        dlg.geometry("880x560+120+90"); dlg.resizable(False, False)

        cur_dir = [start_dir]; history = [cur_dir[0]]; hpos = [0]

        top = ttk.Frame(dlg); top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)
        b_back = ttk.Button(top, text="⟵ Back", width=10)
        b_fwd = ttk.Button(top, text="Forward ⟶", width=12)
        b_up = ttk.Button(top, text="Up ⤴", width=8)
        b_home = ttk.Button(top, text="Home ~", width=9)
        path_var = tk.StringVar(value=str(cur_dir[0]))
        ent_path = ttk.Entry(top, textvariable=path_var, width=80)
        b_back.pack(side=tk.LEFT, padx=(0, 5)); b_fwd.pack(side=tk.LEFT, padx=(0, 10))
        b_up.pack(side=tk.LEFT, padx=(0, 5)); b_home.pack(side=tk.LEFT, padx=(0, 10))
        ent_path.pack(side=tk.LEFT, fill=tk.X, expand=True)

        mid = ttk.Frame(dlg); mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        ttk.Label(mid, text="Folders").pack(anchor="w")
        lb_dirs = tk.Listbox(mid, height=20); lb_dirs.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        bot = ttk.Frame(dlg); bot.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        b_select = ttk.Button(bot, text="Select", width=12)
        b_cancel = ttk.Button(bot, text="Cancel", width=12)
        b_select.pack(side=tk.RIGHT, padx=(8, 0)); b_cancel.pack(side=tk.RIGHT)

        chosen = {"dir": None}

        def list_dirs(d: Path):
            try: return sorted([p for p in d.iterdir() if p.is_dir()])
            except Exception: return []

        def refresh():
            path_var.set(str(cur_dir[0])); lb_dirs.delete(0, tk.END)
            for p in list_dirs(cur_dir[0]): lb_dirs.insert(tk.END, p.name)
            b_back.configure(state=("normal" if hpos[0] > 0 else "disabled"))
            b_fwd.configure(state=("normal" if hpos[0] < len(history) - 1 else "disabled"))

        def go(newdir: Path, add_hist=True):
            if newdir.exists() and newdir.is_dir():
                cur_dir[0] = newdir.resolve()
                if add_hist:
                    del history[hpos[0] + 1:]; history.append(cur_dir[0]); hpos[0] = len(history) - 1
                refresh()

        def on_back():
            if hpos[0] > 0: hpos[0] -= 1; go(history[hpos[0]], add_hist=False)

        def on_fwd():
            if hpos[0] < len(history) - 1: hpos[0] += 1; go(history[hpos[0]], add_hist=False)

        def on_up(): go(cur_dir[0].parent)
        def on_home(): go(Path.home())
        def on_dir_double(_=None):
            sel = lb_dirs.curselection();
            if sel: go(cur_dir[0] / lb_dirs.get(sel[0]))
        def on_enter_path(_=None):
            p = Path(path_var.get());
            if p.is_dir(): go(p)

        def on_select(): chosen["dir"] = cur_dir[0]; close()
        def close():
            dlg.grab_release(); dlg.destroy()
        def on_cancel(): chosen["dir"] = None; close()

        b_back.configure(command=on_back); b_fwd.configure(command=on_fwd)
        b_up.configure(command=on_up); b_home.configure(command=on_home)
        b_select.configure(command=on_select); b_cancel.configure(command=on_cancel)
        lb_dirs.bind("<Double-Button-1>", on_dir_double)
        ent_path.bind("<Return>", on_enter_path); ent_path.bind("<KP_Enter>", on_enter_path)
        dlg.bind("<Escape>", lambda _e: on_cancel())

        refresh(); dlg.wait_window()
        if chosen["dir"]: self._model_last_dir = str(chosen["dir"])
        return chosen["dir"]

    def _open_fixed_yaml_picker(self, title="Select a YOLO data.yaml", start_dir: Optional[Path] = None) -> Optional[Path]:
        start_dir = Path(start_dir or getattr(self, "_model_last_dir", str(Path.home()))).resolve()
        dlg = tk.Toplevel(self)
        dlg.title(title); dlg.transient(self); dlg.grab_set()
        dlg.geometry("880x560+140+110"); dlg.resizable(False, False)

        cur_dir = [start_dir]; history = [cur_dir[0]]; hpos = [0]

        top = ttk.Frame(dlg); top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)
        b_back = ttk.Button(top, text="⟵ Back", width=10)
        b_fwd = ttk.Button(top, text="Forward ⟶", width=12)
        b_up = ttk.Button(top, text="Up ⤴", width=8)
        b_home = ttk.Button(top, text="Home ~", width=9)
        path_var = tk.StringVar(value=str(cur_dir[0]))
        ent_path = ttk.Entry(top, textvariable=path_var, width=80)
        b_back.pack(side=tk.LEFT, padx=(0, 5)); b_fwd.pack(side=tk.LEFT, padx=(0, 10))
        b_up.pack(side=tk.LEFT, padx=(0, 5)); b_home.pack(side=tk.LEFT, padx=(0, 10))
        ent_path.pack(side=tk.LEFT, fill=tk.X, expand=True)

        mid = ttk.Frame(dlg); mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        left = ttk.Frame(mid, width=320); left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="Folders").pack(anchor="w")
        lb_dirs = tk.Listbox(left, height=20); lb_dirs.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        right = ttk.Frame(mid); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        ttk.Label(right, text="data.yaml / .yml").pack(anchor="w")
        lb_files = tk.Listbox(right, height=20); lb_files.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        bot = ttk.Frame(dlg); bot.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        b_open = ttk.Button(bot, text="Open", width=12)
        b_cancel = ttk.Button(bot, text="Cancel", width=12)
        b_open.pack(side=tk.RIGHT, padx=(8, 0)); b_cancel.pack(side=tk.RIGHT)

        chosen = {"file": None}

        def list_dir(d: Path):
            try: items = list(d.iterdir())
            except Exception: return [], []
            dirs = sorted([p for p in items if p.is_dir()])
            files = sorted([p for p in items if p.is_file() and p.suffix.lower() in (".yaml", ".yml")])
            return dirs, files

        def refresh():
            path_var.set(str(cur_dir[0])); lb_dirs.delete(0, tk.END); lb_files.delete(0, tk.END)
            dirs, files = list_dir(cur_dir[0])
            for p in dirs: lb_dirs.insert(tk.END, p.name)
            for p in files: lb_files.insert(tk.END, p.name)
            b_back.configure(state=("normal" if hpos[0] > 0 else "disabled"))
            b_fwd.configure(state=("normal" if hpos[0] < len(history) - 1 else "disabled"))

        def go(newdir: Path, add_hist=True):
            if newdir.exists() and newdir.is_dir():
                cur_dir[0] = newdir.resolve()
                if add_hist:
                    del history[hpos[0] + 1:]; history.append(cur_dir[0]); hpos[0] = len(history) - 1
                refresh()

        def on_back():
            if hpos[0] > 0: hpos[0] -= 1; go(history[hpos[0]], add_hist=False)

        def on_fwd():
            if hpos[0] < len(history) - 1: hpos[0] += 1; go(history[hpos[0]], add_hist=False)

        def on_up(): go(cur_dir[0].parent)
        def on_home(): go(Path.home())
        def on_dir_double(_=None):
            sel = lb_dirs.curselection()
            if sel: go(cur_dir[0] / lb_dirs.get(sel[0]))
        def on_file_double(_=None):
            sel = lb_files.curselection()
            if sel: chosen["file"] = cur_dir[0] / lb_files.get(sel[0]); close_ok()
        def on_open():
            sel = lb_files.curselection()
            if not sel:
                messagebox.showinfo("Pick a model", "Select a data.yaml file from the list."); return
            chosen["file"] = cur_dir[0] / lb_files.get(sel[0]); close_ok()

        def close_ok():
            dlg.grab_release(); dlg.destroy()
        def on_cancel():
            chosen["file"] = None; close_ok()

        b_back.configure(command=on_back); b_fwd.configure(command=on_fwd)
        b_up.configure(command=on_up); b_home.configure(command=on_home)
        b_open.configure(command=on_open); b_cancel.configure(command=on_cancel)
        lb_dirs.bind("<Double-Button-1>", on_dir_double)
        lb_files.bind("<Double-Button-1>", on_file_double)
        dlg.bind("<Escape>", lambda _e: on_cancel())

        refresh(); dlg.wait_window()
        if chosen["file"]: self._model_last_dir = str(chosen["file"].parent)
        return chosen["file"]

    def _open_fixed_video_picker(self, title="Select a video file") -> Optional[Path]:
        VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg")
        start_dir = getattr(self, "_video_last_dir", str(Path.home()))

        dlg = tk.Toplevel(self)
        dlg.title(title); dlg.transient(self); dlg.grab_set()
        dlg.geometry("880x560+100+80"); dlg.resizable(False, False)

        cur_dir = [Path(start_dir).resolve()]
        history: List[Path] = [cur_dir[0]]
        hpos = [0]

        frm_top = ttk.Frame(dlg); frm_top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)
        btn_back = ttk.Button(frm_top, text="⟵ Back", width=10)
        btn_fwd = ttk.Button(frm_top, text="Forward ⟶", width=12)
        btn_up = ttk.Button(frm_top, text="Up ⤴", width=8)
        btn_home = ttk.Button(frm_top, text="Home ~", width=9)
        path_var = tk.StringVar(value=str(cur_dir[0]))
        ent_path = ttk.Entry(frm_top, textvariable=path_var, width=80)
        btn_back.pack(side=tk.LEFT, padx=(0, 5))
        btn_fwd.pack(side=tk.LEFT, padx=(0, 10))
        btn_up.pack(side=tk.LEFT, padx=(0, 5))
        btn_home.pack(side=tk.LEFT, padx=(0, 10))
        ent_path.pack(side=tk.LEFT, fill=tk.X, expand=True)

        frm_mid = ttk.Frame(dlg); frm_mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        frm_left = ttk.Frame(frm_mid, width=320); frm_left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(frm_left, text="Folders").pack(anchor="w")
        lb_dirs = tk.Listbox(frm_left, height=20); lb_dirs.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        frm_right = ttk.Frame(frm_mid); frm_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        ttk.Label(frm_right, text="Video files").pack(anchor="w")
        lb_files = tk.Listbox(frm_right, height=20); lb_files.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        frm_bot = ttk.Frame(dlg); frm_bot.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        btn_open = ttk.Button(frm_bot, text="Open", width=12)
        btn_cancel = ttk.Button(frm_bot, text="Cancel", width=12)
        btn_open.pack(side=tk.RIGHT, padx=(8, 0))
        btn_cancel.pack(side=tk.RIGHT)

        selection: Dict[str, Optional[Path]] = {"file": None}

        def list_dir(d: Path):
            try: items = list(d.iterdir())
            except Exception: return [], []
            dirs = sorted([p for p in items if p.is_dir()])
            files = sorted([p for p in items if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
            return dirs, files

        def refresh_lists():
            path_var.set(str(cur_dir[0])); lb_dirs.delete(0, tk.END); lb_files.delete(0, tk.END)
            dirs, files = list_dir(cur_dir[0])
            for p in dirs: lb_dirs.insert(tk.END, p.name)
            for p in files: lb_files.insert(tk.END, p.name)
            btn_back.configure(state=("normal" if hpos[0] > 0 else "disabled"))
            btn_fwd.configure(state=("normal" if hpos[0] < len(history) - 1 else "disabled"))

        def go_dir(newdir: Path, add_hist=True):
            if not newdir.exists() or not newdir.is_dir(): return
            cur_dir[0] = newdir.resolve()
            if add_hist:
                del history[hpos[0] + 1:]; history.append(cur_dir[0]); hpos[0] = len(history) - 1
            refresh_lists()

        def on_back():
            if hpos[0] > 0:
                hpos[0] -= 1; cur_dir[0] = history[hpos[0]]; refresh_lists()

        def on_fwd():
            if hpos[0] < len(history) - 1:
                hpos[0] += 1; cur_dir[0] = history[hpos[0]]; refresh_lists()

        def on_up(): go_dir = cur_dir[0].parent if cur_dir[0].parent != cur_dir[0] else cur_dir[0]; go_dir and refresh_lists()
        def on_home(): go_dir(Path.home())

        def on_enter_path(_evt=None):
            p = Path(path_var.get())
            if p.is_dir():
                go_dir(p)
            elif p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                selection["file"] = p; close_ok()
            else:
                messagebox.showwarning("Path", "Enter a folder path or a supported video file.")

        def on_dir_double(_evt=None):
            sel = lb_dirs.curselection();
            if not sel: return
            name = lb_dirs.get(sel[0]); go_dir(cur_dir[0] / name)

        def on_file_double(_evt=None):
            sel = lb_files.curselection()
            if not sel: return
            name = lb_files.get(sel[0]); selection["file"] = cur_dir[0] / name; close_ok()

        def on_open():
            sel = lb_files.curselection()
            if sel:
                name = lb_files.get(sel[0]); selection["file"] = cur_dir[0] / name; close_ok()
            else:
                on_enter_path()

        def close_ok():
            dlg.grab_release(); dlg.destroy()

        def on_cancel():
            selection["file"] = None; close_ok()

        btn_back.configure(command=on_back); btn_fwd.configure(command=on_fwd)
        btn_up.configure(command=lambda: on_up()); btn_home.configure(command=on_home)
        btn_open.configure(command=on_open); btn_cancel.configure(command=on_cancel)
        ent_path.bind("<Return>", on_enter_path); ent_path.bind("<KP_Enter>", on_enter_path)
        lb_dirs.bind("<Double-Button-1>", on_dir_double)
        lb_files.bind("<Double-Button-1>", on_file_double)
        dlg.bind("<Escape>", lambda _e: on_cancel())

        refresh_lists(); dlg.wait_window()
        chosen = selection["file"]
        if chosen: self._video_last_dir = str(chosen.parent)
        else: self._video_last_dir = str(cur_dir[0])
        return chosen

    def on_load_video(self):
        p = self._open_fixed_video_picker("Select a video file")
        if not p: return
        self.video_path = Path(p)
        if self.cap: self.cap.release()

        cap = cv2.VideoCapture(str(self.video_path), cv2.CAP_FFMPEG) if hasattr(cv2, "CAP_FFMPEG") else cv2.VideoCapture(str(self.video_path))
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                messagebox.showerror("Failed", f"Could not open video: {self.video_path}"); return
        self.cap = cap

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.frame_idx = -1; self.end_reached = False
        self.playing = False; self._cancel_play_timer()
        self.play_btn.configure(text="Play [P]")
        self.output_root = self.video_path.parent / "output" / self.video_path.stem
        self._seek_and_show(0)
        self._update_transport_enabled()

    def _update_transport_enabled(self):
        ok = (self.cap is not None) and (len(self.model_to_names) > 0)
        state = ("normal" if ok else "disabled")
        self.play_btn.configure(state=state)
        self.btn_back.configure(state=state); self.btn_fwd.configure(state=state)
        self.btn_start.configure(state=state); self.btn_end.configure(state=state)

    # ---------- Playback / Seeking ----------
    def toggle_play(self):
        if not self.cap:
            messagebox.showinfo("No video", "Load a video first."); return
        if not self.model_to_names:
            messagebox.showinfo("Models required", "Load the Models Folder before playing."); return
        if self.end_reached:
            messagebox.showinfo("End of video", "Reached the end."); return
        if self.playing:
            self.playing = False; self.play_btn.configure(text="Play [P]")
            self._cancel_play_timer()
            self._render_from_cache(draw_boxes=True)
        else:
            self._cancel_play_timer()
            self.playing = True; self.play_btn.configure(text="Pause [P]")
            self._play_loop()

    def _cancel_play_timer(self):
        if self.play_after_id is not None:
            try: self.after_cancel(self.play_after_id)
            except Exception: pass
            self.play_after_id = None

    def _freeze_at_end(self):
        target = self.total_frames - 1 if self.total_frames > 0 else max(0, self.frame_idx)
        for back in range(0, 6):
            idx = max(0, target - back)
            if self._safe_seek(idx):
                break
        self.playing = False
        self.play_btn.configure(text="Play [P]")
        self.end_reached = True

    def _play_loop(self):
        if not self.playing: return

        # Fast skipping: grab N-1 frames, decode only the last
        if self.play_step > 1:
            for _ in range(self.play_step - 1):
                ok = self.cap.grab()
                if not ok:
                    self._freeze_at_end()
                    self._cancel_play_timer()
                    # keep quiet here to avoid modal popup spam at high speed
                    return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._freeze_at_end()
            self._cancel_play_timer()
            return

        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.frame_idx = max(0, pos - 1)
        self._show_frame(frame, draw_boxes=False)

        self.play_after_id = self.after(self.play_delay_ms, self._play_loop)

    def step_seek(self, delta: int):
        if not self.cap: return
        if not self.model_to_names:
            messagebox.showinfo("Models required", "Load the Models Folder before stepping."); return
        if self.playing: self.toggle_play()
        target = clamp((self.frame_idx if self.frame_idx >= 0 else 0) + delta, 0, max(0, (self.total_frames or 1) - 1))
        if not self._safe_seek(target):
            self._seek_and_show(target)
        self.end_reached = False

    def goto_start(self):
        if not self.cap or not self.model_to_names: return
        if self.playing: self.toggle_play()
        if not self._safe_seek(0):
            self._seek_and_show(0)
        self.end_reached = False

    def goto_end(self):
        if not self.cap or not self.model_to_names: return
        if self.playing: self.toggle_play()
        last = max(0, (self.total_frames or 1) - 1)
        for back in (0, 2, 4, 6, 10, 15, 20):
            idx = max(0, last - back)
            if self._safe_seek(idx):
                return
        self._seek_and_show(last)
        self.end_reached = True

    def _safe_seek(self, target_idx: int) -> bool:
        if not self.cap: return False
        self._cancel_play_timer()
        self.playing = False; self.play_btn.configure(text="Play [P]")

        if self.fps > 0:
            target_ms = int((target_idx / self.fps) * 1000)
            back_ms = 600
            start_ms = max(0, target_ms - back_ms)
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        else:
            start_idx = max(0, target_idx - 30)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        max_reads = 120
        last_frame = None
        for _ in range(max_reads):
            ok, frame = self.cap.read()
            if not ok or frame is None: break
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.frame_idx = max(0, pos - 1)
            last_frame = frame
            if self.frame_idx >= target_idx:
                self._show_frame(frame, draw_boxes=True)
                return True

        if last_frame is not None:
            self._show_frame(last_frame, draw_boxes=True)
            return True
        return False

    def _seek_and_show(self, idx: int):
        self._cancel_play_timer()
        self.playing = False; self.play_btn.configure(text="Play [P]")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None: return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.frame_idx = max(0, pos - 1)
        self._show_frame(frame, draw_boxes=True)

    # ---------- Rendering ----------
    def _show_frame(self, frame_bgr, draw_boxes: bool):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.current_frame_rgb = frame_rgb
        self.img_h, self.img_w = frame_rgb.shape[:2]
        self.annotations = self.ann_by_frame.setdefault(self.frame_idx, [])
        self._render_from_cache(draw_boxes=draw_boxes)
        self._refresh_lists()

        if self.fps > 0:
            t_seconds = self.frame_idx / self.fps
            mm = int(t_seconds // 60); ss = int(t_seconds % 60)
        else:
            mm, ss = 0, 0
        self.status.set(
            f"{self.video_path.name if self.video_path else ''} | Frame {self.frame_idx+1} "
            f"| {mm:02d}:{ss:02d} | speed x{self._display_speed_multiplier()}"
        )

    def _render_from_cache(self, draw_boxes: bool=True):
        if self.current_frame_rgb is None: return
        c_w = max(100, self.canvas.winfo_width()); c_h = max(100, self.canvas.winfo_height())
        scale = min(c_w / self.img_w, c_h / self.img_h)
        new_w = int(self.img_w * scale); new_h = int(self.img_h * scale)
        self.scale_x = scale; self.scale_y = scale
        self.offset_x = (c_w - new_w) // 2; self.offset_y = (c_h - new_h) // 2
        pil_img = Image.fromarray(cv2.resize(self.current_frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA))
        self.canvas_img_tk = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.canvas_img_tk)
        if draw_boxes and not self.playing:
            self._redraw_boxes()

    def _redraw_boxes(self):
        for ann in self.annotations:
            c1 = self.image_to_canvas_coords(ann.x1, ann.y1)
            c2 = self.image_to_canvas_coords(ann.x2, ann.y2)
            self.canvas.create_rectangle(c1[0], c1[1], c2[0], c2[1], outline="#00ff88", width=2)
            self.canvas.create_text(c1[0]+4, c1[1]+12,
                                    text=f"{ann.model}:{ann.class_name}",
                                    anchor="nw", fill="#00ff88",
                                    font=("TkDefaultFont", 9, "bold"))

    def on_canvas_resize(self, _evt=None):
        self._render_from_cache(draw_boxes=True)

    # ---------- Coordinate helpers ----------
    def canvas_to_image_coords(self, cx: int, cy: int) -> Tuple[int,int]:
        x = int((cx - self.offset_x) / (self.scale_x or 1.0))
        y = int((cy - self.offset_y) / (self.scale_y or 1.0))
        return x, y

    def image_to_canvas_coords(self, ix: int, iy: int) -> Tuple[int,int]:
        x = int(ix * (self.scale_x or 1.0) + self.offset_x)
        y = int(iy * (self.scale_y or 1.0) + self.offset_y)
        return x, y

    # ---------- Assign dialog with type-to-jump ----------
    def _attach_typeahead(self, combo: ttk.Combobox, get_values_callable, substring=True, reset_ms=800):
        """
        Readonly combobox 'type ahead' that works whether the dropdown is closed or OPEN.
        - Maintains a small buffer per widget.
        - Substring (case-insensitive) search; Backspace edits the buffer.
        - Buffer resets after `reset_ms` ms idle.
        """
        import time
        combo["state"] = "readonly"
        combo["values"] = list(get_values_callable())
        combo._type_buf = ""
        combo._type_last = 0.0

        def now(): return time.time()

        def select_match(buf: str):
            vals = list(get_values_callable())
            if not vals: return
            target = buf.lower()
            idx = None
            if substring:
                for i, v in enumerate(vals):
                    if target in str(v).lower(): idx = i; break
            else:
                for i, v in enumerate(vals):
                    if str(v).lower().startswith(target): idx = i; break
            if idx is not None:
                combo.current(idx)
                try:
                    pop_path = combo.tk.eval(f'ttk::combobox::PopdownWindow {str(combo)}')
                    lb = combo.nametowidget(pop_path + '.f.l')
                    lb.selection_clear(0, 'end'); lb.selection_set(idx); lb.see(idx); lb.activate(idx)
                except Exception:
                    pass

        def handle_key(evt):
            keys_pass = {"Up", "Down", "Prior", "Next", "Home", "End", "Escape", "Tab", "Return"}
            if evt.keysym in keys_pass: return
            if now() - combo._type_last > (reset_ms / 1000.0):
                combo._type_buf = ""
            if evt.keysym == "BackSpace":
                combo._type_buf = combo._type_buf[:-1]
            elif len(evt.char) == 1 and (evt.char.isalnum() or evt.char.isspace() or evt.char in "-_./"):
                combo._type_buf += evt.char
            else:
                return
            combo._type_last = now()
            if combo._type_buf:
                select_match(combo._type_buf)
            else:
                vals = list(get_values_callable())
                if vals: combo.current(0)

        def on_focus_in(_e=None):
            combo["values"] = list(get_values_callable())
            if combo.get() not in combo["values"]:
                if combo["values"]: combo.current(0)
            combo._type_buf = ""; combo._type_last = 0.0
            combo.after(50, bind_popdown_keys)

        def bind_popdown_keys():
            try:
                pop_path = combo.tk.eval(f'ttk::combobox::PopdownWindow {str(combo)}')
                lb = combo.nametowidget(pop_path + '.f.l')
                lb.unbind("<KeyPress>")
                lb.bind("<KeyPress>", handle_key)
            except Exception:
                combo.after(100, bind_popdown_keys)

        combo.bind("<KeyPress>", handle_key)
        combo.bind("<FocusIn>", on_focus_in)
        combo.bind("<Button-1>", lambda _e: combo.after(50, bind_popdown_keys))

    def _open_assign_dialog(self, x1, y1, x2, y2):
        if not self.model_to_names:
            messagebox.showinfo("Models required", "Load the Models Folder first."); return
        dlg = tk.Toplevel(self); dlg.title("Assign label"); dlg.transient(self); dlg.grab_set()

        ttk.Label(dlg, text="Model:").grid(row=0, column=0, padx=8, pady=8, sticky="e")
        model_var = tk.StringVar(value=(list(self.model_to_names.keys())[0]))
        model_combo = ttk.Combobox(dlg, textvariable=model_var, width=28)
        model_combo.grid(row=0, column=1, padx=8, pady=8, sticky="w")

        ttk.Label(dlg, text="Class:").grid(row=1, column=0, padx=8, pady=8, sticky="e")
        class_var = tk.StringVar()
        class_combo = ttk.Combobox(dlg, textvariable=class_var, width=28)
        class_combo.grid(row=1, column=1, padx=8, pady=8, sticky="w")

        def get_all_models(): return list(self.model_to_names.keys())
        def get_all_classes(): return self.model_to_names.get(model_var.get(), [])

        self._attach_typeahead(model_combo, get_all_models, substring=True)
        self._attach_typeahead(class_combo, get_all_classes, substring=True)

        def refresh_classes(*_):
            class_combo.configure(values=get_all_classes())
            vals = get_all_classes()
            if vals:
                class_var.set(vals[0]); class_combo.current(0)
            class_combo.after(50, lambda: class_combo.event_generate("<FocusIn>"))

        model_combo.bind("<<ComboboxSelected>>", refresh_classes)
        model_combo.bind("<FocusOut>", refresh_classes)
        refresh_classes()

        btns = ttk.Frame(dlg); btns.grid(row=2, column=0, columnspan=2, pady=(4, 10))

        def do_save(_evt=None):
            names = self.model_to_names.get(model_var.get(), [])
            if not names:
                messagebox.showerror("Class", "Pick a model."); return
            try:
                class_id = names.index(class_var.get())
            except ValueError:
                messagebox.showerror("Class", "Pick a class (type to jump, then Enter)."); return
            ann = Annotation(model=model_var.get(), class_name=class_var.get(), class_id=class_id,
                             x1=x1, y1=y1, x2=x2, y2=y2)
            self.annotations.append(ann)
            self._render_from_cache(draw_boxes=True)
            self._refresh_lists()
            dlg.destroy()

        ttk.Button(btns, text="Save (Enter)", command=do_save, width=14).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Cancel", command=dlg.destroy, width=12).pack(side=tk.LEFT, padx=6)
        dlg.bind("<Return>", do_save)
        model_combo.focus_set()

    # ---------- Drawing flow ----------
    def on_mouse_down(self, evt):
        if not self.cap: return
        if not self.model_to_names:
            messagebox.showinfo("Models required", "Load the Models Folder before labeling."); return
        if self.playing: self.toggle_play()
        self.dragging = True
        self.start_x, self.start_y = self.canvas_to_image_coords(evt.x, evt.y)
        self.start_x = clamp(self.start_x, 0, self.img_w - 1)
        self.start_y = clamp(self.start_y, 0, self.img_h - 1)
        if self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id); self.temp_rect_id = None

    def on_mouse_drag(self, evt):
        if not self.dragging or not self.cap: return
        x2, y2 = self.canvas_to_image_coords(evt.x, evt.y)
        x2 = clamp(x2, 0, self.img_w - 1); y2 = clamp(y2, 0, self.img_h - 1)
        c1 = self.image_to_canvas_coords(self.start_x, self.start_y)
        c2 = self.image_to_canvas_coords(x2, y2)
        if self.temp_rect_id is not None:
            self.canvas.coords(self.temp_rect_id, c1[0], c1[1], c2[0], c2[1])
        else:
            self.temp_rect_id = self.canvas.create_rectangle(c1[0], c1[1], c2[0], c2[1], outline="#00ff88", width=2)

    def on_mouse_up(self, evt):
        if not self.dragging or not self.cap: return
        self.dragging = False
        if self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id); self.temp_rect_id = None
        x2, y2 = self.canvas_to_image_coords(evt.x, evt.y)
        x2 = clamp(x2, 0, self.img_w - 1); y2 = clamp(y2, 0, self.img_h - 1)
        x1, y1 = self.start_x, self.start_y
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        if (x2 - x1) < 4 or (y2 - y1) < 4: return
        self._open_assign_dialog(x1, y1, x2, y2)

    # ---------- Lists / Deletion / Navigation ----------
    def _refresh_lists(self):
        self.ann_list_frame.delete(0, tk.END)
        for i, ann in enumerate(self.annotations, start=1):
            self.ann_list_frame.insert(tk.END, f"{i}. {ann.model}/{ann.class_name} [{ann.x1},{ann.y1}]→[{ann.x2},{ann.y2}]")
        self._rebuild_global_index()

    def _rebuild_global_index(self):
        self.global_index: List[Tuple[int, int]] = []
        self.ann_list_all.delete(0, tk.END)
        for f in sorted(self.ann_by_frame.keys()):
            for j, ann in enumerate(self.ann_by_frame[f]):
                self.global_index.append((f, j))
                self.ann_list_all.insert(tk.END, f"f{f:06d}  {ann.model}/{ann.class_name}  [{ann.x1},{ann.y1}]→[{ann.x2},{ann.y2}]")

    def delete_selected_ann_current(self):
        sel = self.ann_list_frame.curselection()
        if not sel: return
        idx = sel[0]
        if 0 <= idx < len(self.annotations):
            del self.annotations[idx]
            self._render_from_cache(draw_boxes=True)
            self._refresh_lists()

    def goto_selected_global(self):
        sel = self.ann_list_all.curselection()
        if not sel: return
        f, _ = self.global_index[sel[0]]
        if not self._safe_seek(f):
            self._seek_and_show(f)

    def on_all_double_click(self, _evt):
        self.goto_selected_global()

    def delete_selected_global(self):
        sel = self.ann_list_all.curselection()
        if not sel: return
        f, j = self.global_index[sel[0]]
        anns = self.ann_by_frame.get(f, [])
        if 0 <= j < len(anns):
            del anns[j]
            if f == self.frame_idx:
                self.annotations = self.ann_by_frame.setdefault(self.frame_idx, [])
                self._render_from_cache(draw_boxes=True)
                self._refresh_lists()
            else:
                self._rebuild_global_index()

    # ---------- Export ----------
    def _export_all_labeled(self):
        if not self.cap or self.video_path is None:
            messagebox.showinfo("No video", "Load a video first."); return
        if not self.ann_by_frame:
            messagebox.showinfo("Nothing to export", "No labeled frames yet."); return

        total_frames_to_write = sum(1 for f, anns in self.ann_by_frame.items() if anns)
        if total_frames_to_write == 0:
            messagebox.showinfo("Nothing to export", "No labeled frames yet."); return

        ensure_dir(self.output_root)
        written_count = 0; failures = 0

        for f in sorted(self.ann_by_frame.keys()):
            anns = self.ann_by_frame[f]
            if not anns: continue

            if not self._safe_seek(f):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    failures += 1; continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = self.current_frame_rgb

            base = f"{self.video_path.stem}_f{f:06d}"
            by_model: Dict[str, List[Annotation]] = {}
            for ann in anns:
                by_model.setdefault(ann.model, []).append(ann)

            for model, model_anns in by_model.items():
                model_root = self.output_root / model
                images_dir = model_root / "images"; labels_dir = model_root / "labels"
                ensure_dir(images_dir); ensure_dir(labels_dir)
                img_out = images_dir / f"{base}.jpg"; txt_out = labels_dir / f"{base}.txt"
                cv2.imwrite(str(img_out), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                with open(txt_out, "w", encoding="utf-8") as ftxt:
                    for a in model_anns:
                        ftxt.write(yolo_line(a.class_id, a.x1, a.y1, a.x2, a.y2, self.img_w, self.img_h) + "\n")

            written_count += 1

        messagebox.showinfo(
            "Export complete",
            f"Wrote {written_count} labeled frame(s) into:\n{self.output_root}\n"
            + ("" if failures == 0 else f"\nSkipped {failures} frame(s) due to read errors.")
        )

# ---- App entrypoint ----
def main():
    app = TakeLabeler()
    app.mainloop()

if __name__ == "__main__":
    main()
