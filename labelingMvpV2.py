#!/usr/bin/env python3
"""
TakeLabeler v6.3 — START/END jump buttons, HEVC-safe seeks, per-frame memory, global nav

Adds:
- Buttons: Start (⟵⟵) and End (⟶⟶)
- Hotkeys: Home = start, End = end
- Uses _safe_seek to land cleanly (HEVC-friendly)

Keeps:
- P to play/pause, Shift+Left/Right = -50/+50 (pauses first)
- Freeze cleanly on last frame + popup at natural end
- Right sidebar: current frame + all annotations (double-click to jump)
- Annotations persist by frame and reappear on revisit
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
from PIL import Image, ImageTk
import yaml

# ====== OPTIONAL: your default YOLO models folder. Fallbacks to picker if missing. ======
HARDCODED_MODELS_DIR = Path("/home/aleksanovevski/Documents/Blaznavac/Testovi/DataYamls")

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
        self.title("TakeLabeler v6.3 — Video Labeler for YOLOv8")
        self.geometry("1320x820"); self.minsize(1150, 740)

        # Video
        self.video_path: Optional[Path] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.fps = 0.0
        self.frame_idx = -1
        self.current_frame_rgb = None
        self.playing = False
        self.play_after_id = None
        self.play_step = 2          # gentle speed
        self.play_delay_ms = 20
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
        self.ann_by_frame: Dict[int, List[Annotation]] = {}   # persistent store
        self.annotations: List[Annotation] = []               # alias to current frame’s list

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
        ttk.Button(top, text="2) Load Video", command=self.on_load_video, width=16).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Help", command=self.show_help, width=14).pack(side=tk.RIGHT, padx=3)

        mid = ttk.Frame(self); mid.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        self.btn_start = ttk.Button(mid, text="⟵⟵ Start", command=self.goto_start, width=12)
        self.btn_start.pack(side=tk.LEFT, padx=(2,2))
        self.play_btn = ttk.Button(mid, text="Play [P]", command=self.toggle_play, width=14)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        self.btn_back = ttk.Button(mid, text="-50", command=lambda: self.step_seek(-50), width=10)
        self.btn_back.pack(side=tk.LEFT, padx=2)
        self.btn_fwd = ttk.Button(mid, text="+50", command=lambda: self.step_seek(+50), width=10)
        self.btn_fwd.pack(side=tk.LEFT, padx=2)
        self.btn_end = ttk.Button(mid, text="End ⟶⟶", command=self.goto_end, width=12)
        self.btn_end.pack(side=tk.LEFT, padx=(2,2))

        main = ttk.Frame(self); main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.canvas = tk.Canvas(main, bg="#111", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", lambda e: self.canvas.focus_set())
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Sidebar
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

        # Menu
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Export current frame", command=self._export_current_frame)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

        self._update_transport_enabled()

    def _bind_keys(self):
        self.bind("p", lambda e: self.toggle_play())
        self.bind("P", lambda e: self.toggle_play())
        self.bind("<Shift-Left>", lambda e: self.step_seek(-50))
        self.bind("<Shift-Right>", lambda e: self.step_seek(+50))
        self.bind("<Home>", lambda e: self.goto_start())
        self.bind("<End>", lambda e: self.goto_end())
        self.canvas.bind("<space>", lambda e: self.toggle_play())

    def _show_quick_start(self):
        self.after(400, lambda: messagebox.showinfo(
            "Quick start",
            "1) Load Models Folder (your data.yaml files).\n"
            "2) Load Video. Output auto: <video_dir>/output/<video_stem>/\n\n"
            "Draw a box → popup for Model/Class → Save/Delete.\n"
            "P = play/pause, Shift+Left/Right = -50/+50, Home/End = start/end.\n"
            "Boxes persist per frame and reappear when you revisit that frame."
        ))

    def show_help(self):
        messagebox.showinfo("Help",
            "Shortcuts:\n"
            "  P .............. Play/Pause\n"
            "  Shift+Left/Right  Step -50/+50 frames\n"
            "  Home/End ........ Jump to start/end\n\n"
            "Boxes are hidden while playing, and shown when paused on their frame.\n"
            "Use the 'All annotations' list to jump anywhere."
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
        cnt = self._load_models_from_dir(HARDCODED_MODELS_DIR)
        if cnt == 0:
            d = filedialog.askdirectory(title="Select folder with YOLO data.yaml files")
            if not d: return
            cnt = self._load_models_from_dir(Path(d))
        if cnt == 0:
            messagebox.showerror("No models", "No valid data.yaml files found.")
            return
        self.status.set(f"Loaded {cnt} model(s) from: {self.models_dir}")
        self._update_transport_enabled()

    def on_load_video(self):
        p = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")]
        )
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
        last = None
        for _ in range(self.play_step):
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._freeze_at_end()
                messagebox.showinfo("End of video", "Reached the end of the video.")
                self._cancel_play_timer()
                return
            last = frame
        if last is not None:
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.frame_idx = max(0, pos - 1)
            self._show_frame(last, draw_boxes=False)
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
        # try a safe landing near the end; walk back a few frames if needed
        for back in (0, 2, 4, 6, 10, 15, 20):
            idx = max(0, last - back)
            if self._safe_seek(idx):
                return
        self._seek_and_show(last)
        self.end_reached = True  # we're at end because user asked for it

    def _safe_seek(self, target_idx: int) -> bool:
        """HEVC-friendly: seek slightly earlier (by time if possible), read forward."""
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

        # Status: Frame N and mm:ss (no /total)
        if self.fps > 0:
            t_seconds = self.frame_idx / self.fps
            mm = int(t_seconds // 60); ss = int(t_seconds % 60)
        else:
            mm, ss = 0, 0
        self.status.set(f"{self.video_path.name if self.video_path else ''} | Frame {self.frame_idx+1} | {mm:02d}:{ss:02d}")

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

    # ---------- Drawing flow ----------
    def on_mouse_down(self, evt):
        if not self.cap: return
        if not self.model_to_names:
            messagebox.showinfo("Models required", "Load the Models Folder before labeling."); return
        if self.playing: self.toggle_play()  # auto-pause
        self.dragging = True
        self.start_x, self.start_y = self.canvas_to_image_coords(evt.x, evt.y)
        self.start_x = clamp(self.start_x, 0, self.img_w-1)
        self.start_y = clamp(self.start_y, 0, self.img_h-1)
        if self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id); self.temp_rect_id = None

    def on_mouse_drag(self, evt):
        if not self.dragging or not self.cap: return
        x2, y2 = self.canvas_to_image_coords(evt.x, evt.y)
        x2 = clamp(x2, 0, self.img_w-1); y2 = clamp(y2, 0, self.img_h-1)
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
        x2 = clamp(x2, 0, self.img_w-1); y2 = clamp(y2, 0, self.img_h-1)
        x1, y1 = self.start_x, self.start_y
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        if (x2 - x1) < 4 or (y2 - y1) < 4: return
        self._open_assign_dialog(x1, y1, x2, y2)

    def _open_assign_dialog(self, x1, y1, x2, y2):
        if not self.model_to_names:
            messagebox.showinfo("Models required", "Load the Models Folder first."); return
        dlg = tk.Toplevel(self); dlg.title("Assign label"); dlg.transient(self); dlg.grab_set()
        ttk.Label(dlg, text="Model:").grid(row=0, column=0, padx=8, pady=8, sticky="e")
        model_var = tk.StringVar(value=list(self.model_to_names.keys())[0])
        model_combo = ttk.Combobox(dlg, textvariable=model_var, state="readonly",
                                   values=list(self.model_to_names.keys()), width=28)
        model_combo.grid(row=0, column=1, padx=8, pady=8, sticky="w")
        ttk.Label(dlg, text="Class:").grid(row=1, column=0, padx=8, pady=8, sticky="e")
        class_var = tk.StringVar(); class_combo = ttk.Combobox(dlg, textvariable=class_var, state="readonly", width=28)
        def refresh_classes(*_):
            names = self.model_to_names.get(model_var.get(), [])
            class_combo["values"] = names
            class_var.set(names[0] if names else "")
        model_combo.bind("<<ComboboxSelected>>", refresh_classes); refresh_classes()
        class_combo.grid(row=1, column=1, padx=8, pady=8, sticky="w")

        btns = ttk.Frame(dlg); btns.grid(row=2, column=0, columnspan=2, pady=(4,10))
        def do_save():
            names = self.model_to_names.get(model_var.get(), [])
            try:
                class_id = names.index(class_var.get())
            except ValueError:
                messagebox.showerror("Class", "Pick a class."); return
            ann = Annotation(model=model_var.get(), class_name=class_var.get(), class_id=class_id,
                             x1=x1, y1=y1, x2=x2, y2=y2)
            self.annotations.append(ann)
            self._render_from_cache(draw_boxes=True)
            self._refresh_lists()
            dlg.destroy()
        def do_delete(): dlg.destroy()
        ttk.Button(btns, text="Save", command=do_save, width=12).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Delete", command=do_delete, width=12).pack(side=tk.LEFT, padx=6)

    # ---------- Lists / Deletion / Navigation ----------
    def _refresh_lists(self):
        self.ann_list_frame.delete(0, tk.END)
        for i, ann in enumerate(self.annotations, start=1):
            self.ann_list_frame.insert(tk.END, f"{i}. {ann.model}/{ann.class_name} [{ann.x1},{ann.y1}]→[{ann.x2},{ann.y2}]")
        self._rebuild_global_index()

    def _rebuild_global_index(self):
        self.global_index: List[Tuple[int, int]] = []  # (frame_idx, idx_in_frame)
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
    def _export_current_frame(self):
        if not self.cap or self.video_path is None:
            messagebox.showinfo("No video", "Load a video first."); return
        if not self.annotations:
            messagebox.showinfo("No boxes", "Draw and Save boxes first."); return
        idx = self.frame_idx if self.frame_idx >= 0 else 0
        if not self._safe_seek(idx):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                messagebox.showerror("Error", "Couldn't read current frame."); return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = self.current_frame_rgb

        base = f"{self.video_path.stem}_f{idx:06d}"
        by_model: Dict[str, List[Annotation]] = {}
        for ann in self.annotations:
            by_model.setdefault(ann.model, []).append(ann)
        written = []
        for model, anns in by_model.items():
            model_root = self.output_root / model
            images_dir = model_root / "images"; labels_dir = model_root / "labels"
            ensure_dir(images_dir); ensure_dir(labels_dir)
            img_out = images_dir / f"{base}.jpg"; txt_out = labels_dir / f"{base}.txt"
            cv2.imwrite(str(img_out), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            with open(txt_out, "w", encoding="utf-8") as f:
                for a in anns:
                    f.write(yolo_line(a.class_id, a.x1, a.y1, a.x2, a.y2, self.img_w, self.img_h) + "\n")
            written.append((model, img_out, txt_out))
        messagebox.showinfo("Saved", "Exported:\n" + "\n".join([str(p) for (_,p,_) in written]))

# ---- App entrypoint ----
def main():
    app = TakeLabeler()
    app.mainloop()

if __name__ == "__main__":
    main()
