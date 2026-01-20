/* global window, document */

const api = {
  async getConfig() {
    return fetch("/api/config").then(r => r.json());
  },
  async loadVideo(videoName) {
    return fetch("/api/video/load", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({video_name: videoName}),
    }).then(async r => {
      if (!r.ok) throw new Error((await r.json()).detail || "Failed to load video");
      return r.json();
    });
  },
  async getClasses(modelName) {
    return fetch(`/api/model/${encodeURIComponent(modelName)}/classes`).then(async r => {
      if (!r.ok) throw new Error((await r.json()).detail || "Failed to load classes");
      return r.json();
    });
  },
  async getFrame(index) {
    const r = await fetch(`/api/frame?index=${index}`);
    if (!r.ok) throw new Error((await r.json()).detail || "Frame fetch failed");
    const blob = await r.blob();
    const realIdx = parseInt(r.headers.get("X-Frame-Index") || String(index), 10);
    const endReached = (r.headers.get("X-End-Reached") === "1");
    const requestedIdx = parseInt(r.headers.get("X-Requested-Index") || String(index), 10);
    return {blob, frameIdx: realIdx, endReached, requestedIdx};
  },
  async getNextFrame(step) {
    const r = await fetch(`/api/frame/next?step=${step}`);
    if (!r.ok) throw new Error((await r.json()).detail || "Next frame fetch failed");
    const blob = await r.blob();
    const frameIdx = parseInt(r.headers.get("X-Frame-Index") || "0", 10);
    const endReached = (r.headers.get("X-End-Reached") === "1");
    return {blob, frameIdx, endReached};
  },
  async getFrameAnnotations(frameIdx) {
    return fetch(`/api/annotations?frame=${frameIdx}`).then(r => r.json());
  },
  async getAllAnnotations() {
    return fetch(`/api/annotations`).then(r => r.json());
  },
  async addAnnotation(payload) {
    return fetch(`/api/annotations`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    }).then(async r => {
      if (!r.ok) throw new Error((await r.json()).detail || "Failed to add annotation");
      return r.json();
    });
  },
  async deleteAnnotation(annId) {
    return fetch(`/api/annotations/${encodeURIComponent(annId)}`, {method: "DELETE"}).then(r => r.json());
  },
  async exportAll() {
    return fetch(`/api/export`, {method: "POST"}).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        if (d && typeof d === "object" && d.error) {
          const err = new Error(d.error);
          err.code = d.error;
          err.options = d.options || [];
          err.video_name = d.video_name;
          err.provided = d.provided;
          throw err;
        }
        throw new Error(d || "Export failed");
      }
      return data;
    });
  },
  async exportAllWithBarCounter(barCounter) {
    return fetch(`/api/export`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({bar_counter: barCounter})
    }).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        if (d && typeof d === "object" && d.error) {
          const err = new Error(d.error);
          err.code = d.error;
          err.options = d.options || [];
          err.video_name = d.video_name;
          err.provided = d.provided;
          throw err;
        }
        throw new Error(d || "Export failed");
      }
      return data;
    });
  }
};

const speedMap = {
  1: {step: 1, delay: 35, mult: 1},
  2: {step: 2, delay: 30, mult: 2},
  3: {step: 4, delay: 25, mult: 4},
  4: {step: 6, delay: 22, mult: 6},
  5: {step: 8, delay: 20, mult: 8},
  6: {step: 10, delay: 18, mult: 10},
  7: {step: 12, delay: 16, mult: 12},
  8: {step: 16, delay: 14, mult: 16},
};

const state = {
  config: null,
  videoLoaded: false,
  videoName: null,
  totalFrames: 0,
  fps: 0,
  imgW: 0,
  imgH: 0,
  frameIdx: 0,

  // rendering
  canvas: null,
  ctx: null,
  img: new Image(),
  scale: 1,
  offsetX: 0,
  offsetY: 0,

  // annotations
  frameAnnotations: [],
  allAnnotations: [],

  // playback
  playing: false,
  playTimer: null,
  speed: 4,
  dirty: false, // changes since last Export
  navSeq: 0, // increments to invalidate in-flight frame renders (fixes End/Play fighting)
  playCursor: 0,

  // drawing
  dragging: false,
  dragStart: null,
  dragRect: null,

  // modal
  modalOpen: false,
  modalRect: null,
  modalModel: null,
  modalClasses: [],
  modalSelectedClass: null,
};

function $(id) { return document.getElementById(id); }

function setStatus(text) {
  $("status").textContent = text;
}

function normalizeText(s) {
  return String(s || "").toLowerCase().replaceAll(" ", "").replaceAll("-", "").replaceAll("_", "");
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function canvasSizeToDisplaySize(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(100, Math.floor(rect.width * dpr));
  const h = Math.max(100, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    return true;
  }
  return false;
}

function computeScaleAndOffset() {
  const cW = state.canvas.width;
  const cH = state.canvas.height;
  const scale = Math.min(cW / state.imgW, cH / state.imgH);
  const newW = Math.floor(state.imgW * scale);
  const newH = Math.floor(state.imgH * scale);
  state.scale = scale;
  state.offsetX = Math.floor((cW - newW) / 2);
  state.offsetY = Math.floor((cH - newH) / 2);
}

function canvasToImage(cx, cy) {
  const dpr = window.devicePixelRatio || 1;
  const rect = state.canvas.getBoundingClientRect();
  const x = (cx - rect.left) * dpr;
  const y = (cy - rect.top) * dpr;
  const ix = Math.floor((x - state.offsetX) / (state.scale || 1));
  const iy = Math.floor((y - state.offsetY) / (state.scale || 1));
  return {x: clamp(ix, 0, state.imgW - 1), y: clamp(iy, 0, state.imgH - 1)};
}

function imageToCanvas(ix, iy) {
  const x = Math.floor(ix * (state.scale || 1) + state.offsetX);
  const y = Math.floor(iy * (state.scale || 1) + state.offsetY);
  return {x, y};
}

function draw() {
  if (!state.videoLoaded) return;

  canvasSizeToDisplaySize(state.canvas);
  computeScaleAndOffset();
  const ctx = state.ctx;
  ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);

  // image
  const drawW = Math.floor(state.imgW * state.scale);
  const drawH = Math.floor(state.imgH * state.scale);
  ctx.drawImage(state.img, state.offsetX, state.offsetY, drawW, drawH);

  // boxes (only when paused, matching v6_1-ish)
  ctx.save();
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.strokeStyle = "#00ff88";
  ctx.fillStyle = "#00ff88";
  ctx.font = `${Math.floor(12 * (window.devicePixelRatio || 1))}px ui-sans-serif`;

  const drawAnn = (a) => {
    const c1 = imageToCanvas(a.x1, a.y1);
    const c2 = imageToCanvas(a.x2, a.y2);
    ctx.strokeRect(c1.x, c1.y, c2.x - c1.x, c2.y - c1.y);
    ctx.fillText(`${a.model}:${a.class_name}`, c1.x + 6, c1.y + 16);
  };
  for (const a of state.frameAnnotations) drawAnn(a);

  // temporary drag rect
  if (state.dragging && state.dragRect) {
    const r = state.dragRect;
    const c1 = imageToCanvas(r.x1, r.y1);
    const c2 = imageToCanvas(r.x2, r.y2);
    ctx.strokeStyle = "#60a5fa";
    ctx.strokeRect(c1.x, c1.y, c2.x - c1.x, c2.y - c1.y);
  }

  ctx.restore();
}

async function refreshLists() {
  if (!state.videoLoaded) return;
  const frameData = await api.getFrameAnnotations(state.frameIdx);
  state.frameAnnotations = frameData.annotations || [];
  const allData = await api.getAllAnnotations();
  state.allAnnotations = allData.annotations || [];

  // current frame
  const ul = $("currentList");
  ul.innerHTML = "";
  for (const a of state.frameAnnotations) {
    const li = document.createElement("li");
    li.className = "item";
    const main = document.createElement("div");
    main.className = "itemMain";
    const title = document.createElement("div");
    title.className = "itemTitle";
    title.textContent = `${a.model}/${a.class_name}`;
    const sub = document.createElement("div");
    sub.className = "itemSub";
    sub.textContent = `[${a.x1},${a.y1}] → [${a.x2},${a.y2}]`;
    main.appendChild(title); main.appendChild(sub);

    const btns = document.createElement("div");
    btns.className = "itemBtns";
    const del = document.createElement("button");
    del.className = "btn danger";
    del.textContent = "Delete";
    del.onclick = async (e) => {
      e.stopPropagation();
      await api.deleteAnnotation(a.id);
      state.dirty = true;
      await refreshLists();
      draw();
    };
    btns.appendChild(del);
    li.appendChild(main);
    li.appendChild(btns);
    ul.appendChild(li);
  }

  // global list
  const ul2 = $("globalList");
  ul2.innerHTML = "";
  for (const a of state.allAnnotations) {
    const li = document.createElement("li");
    li.className = "item";
    li.onclick = async () => {
      await gotoFrame(a.frame_idx);
    };
    const main = document.createElement("div");
    main.className = "itemMain";
    const title = document.createElement("div");
    title.className = "itemTitle";
    title.textContent = `f${String(a.frame_idx).padStart(6, "0")}  ${a.model}/${a.class_name}`;
    const sub = document.createElement("div");
    sub.className = "itemSub";
    sub.textContent = `[${a.x1},${a.y1}] → [${a.x2},${a.y2}]`;
    main.appendChild(title); main.appendChild(sub);

    const btns = document.createElement("div");
    btns.className = "itemBtns";
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `f${a.frame_idx + 1}`;
    const del = document.createElement("button");
    del.className = "btn danger";
    del.textContent = "Delete";
    del.onclick = async (e) => {
      e.stopPropagation();
      await api.deleteAnnotation(a.id);
      state.dirty = true;
      await refreshLists();
      draw();
    };
    btns.appendChild(chip);
    btns.appendChild(del);

    li.appendChild(main);
    li.appendChild(btns);
    ul2.appendChild(li);
  }
}

async function renderFrame(idx) {
  // Invalidate any in-flight render; only the newest navigation may update the UI.
  const mySeq = ++state.navSeq;

  const {blob, frameIdx, endReached, requestedIdx} = await api.getFrame(idx);
  if (mySeq !== state.navSeq) return;

  // Use the real index the server returned (esp. near end-of-video).
  state.frameIdx = frameIdx;
  state.playCursor = state.frameIdx;
  const url = URL.createObjectURL(blob);
  await new Promise((resolve, reject) => {
    state.img.onload = () => resolve();
    state.img.onerror = reject;
    state.img.src = url;
  });
  URL.revokeObjectURL(url);
  if (mySeq !== state.navSeq) return;

  await refreshLists();
  if (mySeq !== state.navSeq) return;
  draw();

  const mm = state.fps > 0 ? Math.floor((state.frameIdx / state.fps) / 60) : 0;
  const ss = state.fps > 0 ? Math.floor((state.frameIdx / state.fps) % 60) : 0;
  setStatus(`${state.videoName} | Frame ${state.frameIdx + 1} | ${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")} | speed x${speedMap[state.speed].mult}`);

  // Stop playback at end, or if the server couldn't reach the requested frame.
  if (endReached || (Number.isFinite(requestedIdx) && frameIdx < requestedIdx)) {
    stopPlayback();
  }
}

async function gotoFrame(idx) {
  if (!state.videoLoaded) return;
  if (!Number.isFinite(state.totalFrames) || state.totalFrames <= 0) {
    setStatus("No video loaded (invalid frame count). Click Load.");
    return;
  }
  if (!Number.isFinite(idx)) return;
  const clamped = clamp(idx, 0, Math.max(0, state.totalFrames - 1));
  try {
    await renderFrame(clamped);
  } catch (e) {
    stopPlayback();
    setStatus(`Server disconnected or crashed. Restart server, then refresh page. (${e?.message || e})`);
  }
}

function stopPlayback() {
  state.playing = false;
  if (state.playTimer) window.clearTimeout(state.playTimer);
  state.playTimer = null;
  state.navSeq++; // cancel any in-flight render that would overwrite a jump
  $("playBtn").textContent = "Play [P]";
}

function startPlayback() {
  if (!state.videoLoaded) return;
  state.playing = true;
  $("playBtn").textContent = "Pause [P]";
  state.playCursor = state.frameIdx;

  const loop = async () => {
    if (!state.playing) return;
    const {step, delay} = speedMap[state.speed];
    try {
      const mySeq = ++state.navSeq;
      const {blob, frameIdx, endReached} = await api.getNextFrame(step);
      if (mySeq !== state.navSeq) return;

      // update image
      state.frameIdx = frameIdx;
      state.playCursor = frameIdx;
      const url = URL.createObjectURL(blob);
      await new Promise((resolve, reject) => {
        state.img.onload = () => resolve();
        state.img.onerror = reject;
        state.img.src = url;
      });
      URL.revokeObjectURL(url);
      if (mySeq !== state.navSeq) return;

      await refreshLists();
      if (mySeq !== state.navSeq) return;
      draw();

      const mm = state.fps > 0 ? Math.floor((state.frameIdx / state.fps) / 60) : 0;
      const ss = state.fps > 0 ? Math.floor((state.frameIdx / state.fps) % 60) : 0;
      setStatus(`${state.videoName} | Frame ${state.frameIdx + 1} | ${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")} | speed x${speedMap[state.speed].mult}`);

      if (endReached) {
        stopPlayback();
        return;
      }
    } catch (e) {
      stopPlayback();
      return;
    }
    state.playTimer = window.setTimeout(loop, delay);
  };

  state.playTimer = window.setTimeout(loop, speedMap[state.speed].delay);
}

function togglePlay() {
  if (!state.videoLoaded) return;
  if (state.playing) stopPlayback();
  else startPlayback();
}

function openModalForRect(rect) {
  state.modalOpen = true;
  state.modalRect = rect;
  state.modalSelectedClass = null;
  $("modalError").textContent = "";
  $("classFilter").value = "";
  $("modal").classList.remove("hidden");
  $("classFilter").focus();
  refreshModalClasses();
}

function closeModal() {
  state.modalOpen = false;
  state.modalRect = null;
  state.modalSelectedClass = null;
  $("modal").classList.add("hidden");
  draw();
}

async function refreshModalClasses() {
  const model = $("modalModelSelect").value;
  state.modalModel = model;
  try {
    const data = await api.getClasses(model);
    state.modalClasses = data.classes || [];
    renderClassList();
  } catch (e) {
    state.modalClasses = [];
    renderClassList();
  }
}

function renderClassList() {
  const ul = $("classList");
  ul.innerHTML = "";
  const filter = normalizeText($("classFilter").value);
  let items = state.modalClasses;
  if (filter) items = state.modalClasses.filter(c => normalizeText(c).includes(filter));

  // Always keep a valid selection in the current filtered list.
  if (items.length > 0 && (!state.modalSelectedClass || !items.includes(state.modalSelectedClass))) {
    state.modalSelectedClass = items[0];
  }
  if (items.length === 0) {
    const li = document.createElement("li");
    li.className = "classItem";
    li.textContent = "(no matches)";
    ul.appendChild(li);
    return;
  }

  for (const cls of items) {
    const li = document.createElement("li");
    li.className = "classItem" + (state.modalSelectedClass === cls ? " selected" : "");
    li.textContent = cls;
    li.onclick = () => {
      state.modalSelectedClass = cls;
      renderClassList();
      $("classFilter").focus();
    };
    li.ondblclick = () => {
      state.modalSelectedClass = cls;
      onModalSave();
    };
    ul.appendChild(li);
  }
}

async function onModalSave() {
  const model = $("modalModelSelect").value;
  const cls = state.modalSelectedClass;
  if (!model) {
    $("modalError").textContent = "Pick a model.";
    return;
  }
  if (!cls) {
    $("modalError").textContent = "Pick a class.";
    return;
  }
  const r = state.modalRect;
  try {
    await api.addAnnotation({
      frame_idx: state.frameIdx,
      model,
      class_name: cls,
      x1: r.x1, y1: r.y1, x2: r.x2, y2: r.y2
    });
    state.dirty = true;
    closeModal();
    await refreshLists();
    draw();
  } catch (e) {
    $("modalError").textContent = e.message || String(e);
  }
}

function installCanvasHandlers() {
  state.canvas.addEventListener("mousedown", (evt) => {
    if (!state.videoLoaded) return;
    if (state.playing) stopPlayback();
    state.dragging = true;
    const p = canvasToImage(evt.clientX, evt.clientY);
    state.dragStart = p;
    state.dragRect = {x1: p.x, y1: p.y, x2: p.x, y2: p.y};
    draw();
  });

  state.canvas.addEventListener("mousemove", (evt) => {
    if (!state.dragging) return;
    const p = canvasToImage(evt.clientX, evt.clientY);
    state.dragRect.x2 = p.x;
    state.dragRect.y2 = p.y;
    draw();
  });

  window.addEventListener("mouseup", (evt) => {
    if (!state.dragging) return;
    state.dragging = false;
    const p = canvasToImage(evt.clientX, evt.clientY);
    let x1 = state.dragStart.x, y1 = state.dragStart.y;
    let x2 = p.x, y2 = p.y;
    if (x2 < x1) [x1, x2] = [x2, x1];
    if (y2 < y1) [y1, y2] = [y2, y1];
    state.dragRect = null;
    if ((x2 - x1) < 4 || (y2 - y1) < 4) {
      draw();
      return;
    }
    openModalForRect({x1, y1, x2, y2});
  });

  window.addEventListener("resize", () => draw());
}

function installHotkeys() {
  window.addEventListener("keydown", async (e) => {
    if (state.modalOpen) {
      if (e.key === "Escape") { e.preventDefault(); closeModal(); }
      if (e.key === "Enter") { e.preventDefault(); onModalSave(); }
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        // move selection in class list
        const ul = $("classList");
        const items = Array.from(ul.querySelectorAll(".classItem")).filter(li => li.textContent !== "(no matches)");
        if (items.length === 0) return;
        let idx = items.findIndex(li => li.classList.contains("selected"));
        if (idx < 0) idx = 0;
        idx = e.key === "ArrowDown" ? Math.min(items.length - 1, idx + 1) : Math.max(0, idx - 1);
        items.forEach(li => li.classList.remove("selected"));
        items[idx].classList.add("selected");
        state.modalSelectedClass = items[idx].textContent;
        items[idx].scrollIntoView({block: "nearest"});
        e.preventDefault();
      }
      return;
    }

    if (e.key === "p" || e.key === "P") { e.preventDefault(); togglePlay(); }
    if (e.key === "Home") { e.preventDefault(); await gotoFrame(0); }
    if (e.key === "End") { e.preventDefault(); await gotoFrame(state.totalFrames - 1); }
    if (e.shiftKey && e.key === "ArrowLeft") { e.preventDefault(); await gotoFrame(state.frameIdx - 50); }
    if (e.shiftKey && e.key === "ArrowRight") { e.preventDefault(); await gotoFrame(state.frameIdx + 50); }
  });
}

async function init() {
  state.canvas = $("canvas");
  state.ctx = state.canvas.getContext("2d");

  setStatus("Loading config…");
  const cfg = await api.getConfig();
  state.config = cfg;

  function populateVideos(videos) {
    const vs = $("videoSelect");
    const prev = vs.value;
    vs.innerHTML = "";
    for (const v of videos) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      vs.appendChild(opt);
    }
    if (prev && videos.includes(prev)) vs.value = prev;
  }

  async function refreshConfigAndVideos() {
    const cfg2 = await api.getConfig();
    state.config = cfg2;
    populateVideos(cfg2.videos);
    return cfg2;
  }

  async function doLoadVideo(videoName) {
    setStatus("Loading video…");
    const res = await api.loadVideo(videoName);
    state.videoLoaded = true;
    state.videoName = res.video_name;
    state.totalFrames = res.total_frames;
    state.fps = res.fps;
    state.imgW = res.width;
    state.imgH = res.height;
    state.frameIdx = 0;
    state.dirty = false;
    stopPlayback();
    await renderFrame(0);
    if ((res.models_loaded || 0) === 0) {
      setStatus(`Video loaded. Now add model *.yaml in: ${state.config?.models_dir || ""}`);
      window.alert(`Video loaded.\n\nNo models found yet.\n\nPut YOLO *.yaml into:\n${state.config?.models_dir || ""}\n\n(You can still play/jump frames, but labeling needs models.)`);
    }
  }

  // populate selects
  populateVideos(cfg.videos);

  const ms = $("modelSelect");
  const mms = $("modalModelSelect");
  ms.innerHTML = "";
  mms.innerHTML = "";
  for (const m of cfg.models) {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = `${m.name} (${m.class_count})`;
    ms.appendChild(opt);
    mms.appendChild(opt.cloneNode(true));
  }

  // default modal model tracks top model
  mms.value = ms.value;

  $("loadVideoBtn").onclick = async () => {
    const v = $("videoSelect").value;
    if (!v) {
      setStatus(`No video found in: ${state.config?.videos_dir || "(unknown)"}`);
      window.alert(`No videos found.\n\nPut videos into:\n${state.config?.videos_dir || ""}`);
      return;
    }
    try {
      if (state.videoLoaded && state.videoName && v !== state.videoName && state.dirty) {
        const ok = window.confirm("You have un-exported annotations. Switch video and lose them?");
        if (!ok) return;
      }
      await doLoadVideo(v);
    } catch (e) {
      setStatus(`Error: ${e.message || e}`);
      window.alert(`Load failed: ${e.message || e}`);
    }
  };

  // Upload is disabled for now; keep UI simple/reliable: copy videos into videos_dir.
  $("uploadVideoBtn").onclick = () => {
    window.alert(`Upload is disabled for now.\n\nCopy videos into:\n${state.config?.videos_dir || ""}`);
  };

  $("exportBtn").onclick = async () => {
    if (!state.videoLoaded) return;
    stopPlayback();
    try {
      setStatus("Exporting… (this may take a bit)");
      let res;
      try {
        res = await api.exportAll();
      } catch (e) {
        if (e && (e.code === "BAR_COUNTER_MISSING" || e.code === "BAR_COUNTER_INVALID")) {
          const opts = (e.options || []).join(", ");
          const chosen = window.prompt(
            `BAR COUNTER is missing/invalid for this video.\n\nType one of: ${opts}\n\nExample: SANK_LEVO`,
            (e.options && e.options[0]) ? e.options[0] : "SANK_LEVO"
          );
          if (!chosen) throw e;
          res = await api.exportAllWithBarCounter(String(chosen).trim().toUpperCase());
        } else {
          throw e;
        }
      }
      state.dirty = false;
      setStatus(`Export complete: ${res.output_root}`);
      window.alert(`Export complete.\n\nImages: ${res.images_dir}\nLabels: ${res.labels_dir}\n\nFrames: ${res.frames_labeled}\nLabel files: ${res.written_label_files}`);
    } catch (e) {
      setStatus(`Export error: ${e.message || e}`);
      window.alert(`Export failed: ${e.message || e}`);
    }
  };

  $("playBtn").onclick = () => togglePlay();
  $("startBtn").onclick = async () => { stopPlayback(); await gotoFrame(0); };
  $("endBtn").onclick = async () => { stopPlayback(); await gotoFrame(state.totalFrames - 1); };
  $("backBtn").onclick = async () => { stopPlayback(); await gotoFrame(state.frameIdx - 50); };
  $("fwdBtn").onclick = async () => { stopPlayback(); await gotoFrame(state.frameIdx + 50); };

  $("speedSlider").oninput = (e) => {
    state.speed = parseInt(e.target.value, 10);
    $("speedText").textContent = `x${speedMap[state.speed].mult}`;
    if (state.playing) {
      stopPlayback();
      startPlayback();
    }
  };
  $("speedText").textContent = `x${speedMap[state.speed].mult}`;

  ms.onchange = () => {
    mms.value = ms.value;
  };
  mms.onchange = () => refreshModalClasses();
  $("classFilter").oninput = () => renderClassList();

  $("modalSaveBtn").onclick = () => onModalSave();
  $("modalCancelBtn").onclick = () => closeModal();
  $("modal").addEventListener("mousedown", (e) => {
    if (e.target === $("modal")) closeModal();
  });

  installCanvasHandlers();
  installHotkeys();

  if (cfg.videos.length === 0) {
    setStatus(`Put videos into: ${cfg.videos_dir}`);
  } else if (cfg.models.length === 0) {
    setStatus(`Put model *.yaml into: ${cfg.models_dir}`);
  } else {
    setStatus(`Ready. Select a video and click Load. (Videos: ${cfg.videos_dir})`);
  }

  window.addEventListener("beforeunload", (e) => {
    if (!state.dirty) return;
    e.preventDefault();
    e.returnValue = "";
  });
}

window.addEventListener("DOMContentLoaded", () => {
  init().catch((e) => setStatus(`Init error: ${e.message || e}`));
});


