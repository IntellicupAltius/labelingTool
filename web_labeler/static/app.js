/* global window, document */

const api = {
  async getConfig() {
    return fetch("/api/config").then(r => r.json());
  },
  async loadVideo(videoName, loadExistingExports=false) {
    return fetch("/api/video/load", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({video_name: videoName, load_existing_exports: !!loadExistingExports}),
    }).then(async r => {
      if (!r.ok) throw new Error((await r.json()).detail || "Failed to load video");
      return r.json();
    });
  },
  async getVideoInfo(videoName) {
    return fetch(`/api/video/info?video_name=${encodeURIComponent(videoName)}`).then(async r => {
      if (!r.ok) throw new Error((await r.json()).detail || "Failed to get video info");
      return r.json();
    });
  },
  async getBgConfig() {
    return fetch(`/api/background_labeler/config`).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error((data.detail && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))) || "Failed to load bg config");
      return data;
    });
  },
  async bgStart(payload) {
    return fetch(`/api/background_labeler/start`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    }).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error((data.detail && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))) || "Failed to start bg session");
      return data;
    });
  },
  async bgGetImage() {
    // Add a cache buster; some browsers otherwise keep showing the first image.
    const r = await fetch(`/api/background_labeler/image?ts=${Date.now()}`);
    if (!r.ok) throw new Error((await r.json()).detail || "Failed to get bg image");
    const blob = await r.blob();
    const idx = parseInt(r.headers.get("X-Index") || "0", 10);
    const name = r.headers.get("X-Name") || "";
    const decision = r.headers.get("X-Decision") || "skip";
    return {blob, idx, name, decision};
  },
  async bgSetIndex(index) {
    return fetch(`/api/background_labeler/set_index?index=${encodeURIComponent(index)}`, {method: "POST"}).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error((data.detail && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))) || "Failed to set index");
      return data;
    });
  },
  async bgDecide(action) {
    return fetch(`/api/background_labeler/decide`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action}),
    }).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error((data.detail && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))) || "Failed to decide");
      return data;
    });
  },
  async bgStatus() {
    return fetch(`/api/background_labeler/status`).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error((data.detail && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))) || "Failed to get bg status");
      return data;
    });
  },
  async bgFinish() {
    return fetch(`/api/background_labeler/finish`, {method: "POST"}).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error((data.detail && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))) || "Failed to finish");
      return data;
    });
  },
  async listDatasets() {
    return fetch(`/api/datasets`).then(async r => {
      if (!r.ok) throw new Error((await r.json()).detail || "Failed to list datasets");
      return r.json();
    });
  },
  async loadDataset(datasetName, model) {
    return fetch(`/api/datasets/load`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({dataset_name: datasetName, model})
    }).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        const msg = (typeof d === "string") ? d : JSON.stringify(d || data);
        throw new Error(msg || "Failed to load dataset");
      }
      return data;
    });
  },
  async getDatasetImage(index) {
    const r = await fetch(`/api/datasets/image?index=${index}`);
    if (!r.ok) throw new Error((await r.json()).detail || "Dataset image fetch failed");
    const blob = await r.blob();
    const imageIdx = parseInt(r.headers.get("X-Image-Index") || String(index), 10);
    const imageName = r.headers.get("X-Image-Name") || "";
    return {blob, imageIdx, imageName};
  },
  async getDatasetAnnotations(index) {
    return fetch(`/api/datasets/annotations?index=${index}`).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        const msg = (typeof d === "string") ? d : JSON.stringify(d || data);
        throw new Error(msg || "Failed to get dataset annotations");
      }
      return data;
    });
  },
  async addDatasetAnnotation(payload) {
    return fetch(`/api/datasets/annotations`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    }).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        const msg = (typeof d === "string") ? d : JSON.stringify(d || data);
        throw new Error(msg || "Failed to add dataset annotation");
      }
      return data;
    });
  },
  async deleteDatasetAnnotation(annId) {
    return fetch(`/api/datasets/annotations/${encodeURIComponent(annId)}`, {method: "DELETE"}).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        const msg = (typeof d === "string") ? d : JSON.stringify(d || data);
        throw new Error(msg || "Failed to delete dataset annotation");
      }
      return data;
    });
  },
  async saveDataset(strategy) {
    return fetch(`/api/datasets/save`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({strategy})
    }).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        const msg = (typeof d === "string") ? d : JSON.stringify(d || data);
        throw new Error(msg || "Failed to save dataset");
      }
      return data;
    });
  },
  async markDatasetBackground(imageIdx) {
    return fetch(`/api/datasets/background`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({image_idx: imageIdx})
    }).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        const msg = (typeof d === "string") ? d : JSON.stringify(d || data);
        throw new Error(msg || "Failed to mark dataset background");
      }
      return data;
    });
  },
  async unmarkDatasetBackground(imageIdx) {
    return fetch(`/api/datasets/background?image_idx=${encodeURIComponent(imageIdx)}`, {method: "DELETE"}).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        const msg = (typeof d === "string") ? d : JSON.stringify(d || data);
        throw new Error(msg || "Failed to unmark dataset background");
      }
      return data;
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
  async markBackground(frameIdx, model) {
    return fetch(`/api/background`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({frame_idx: frameIdx, model})
    }).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        const msg = (typeof d === "string") ? d : JSON.stringify(d || data);
        throw new Error(msg || "Failed to mark background");
      }
      return data;
    });
  },
  async unmarkBackground(frameIdx, model) {
    return fetch(`/api/background?frame_idx=${encodeURIComponent(frameIdx)}&model=${encodeURIComponent(model)}`, {method: "DELETE"}).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        const msg = (typeof d === "string") ? d : JSON.stringify(d || data);
        throw new Error(msg || "Failed to unmark background");
      }
      return data;
    });
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

  // mode
  mode: "video", // "video" | "dataset"

  // dataset fixer
  datasetLoaded: false,
  datasetName: null,
  datasetModel: null,
  datasetImageCount: 0,
  datasetImageIdx: 0,
  datasetImageName: "",

  // background labeler
  bgLoaded: false,
  bgDataset: null,
  bgModel: null,
  bgTotal: 0,
  bgIdx: 0,
  bgSelected: 0,
  bgSkipped: 0,
  bgOutRoot: "",

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

function resetWorkspaceUI(message) {
  stopPlayback();
  // close modal if open
  if (state.modalOpen) {
    try { closeModal(); } catch (_e) {}
  }

  state.videoLoaded = false;
  state.videoName = null;
  state.totalFrames = 0;
  state.fps = 0;
  state.imgW = 0;
  state.imgH = 0;
  state.frameIdx = 0;
  state.playCursor = 0;
  state.dirty = false;

  state.datasetLoaded = false;
  state.datasetName = null;
  state.datasetModel = null;
  state.datasetImageCount = 0;
  state.datasetImageIdx = 0;
  state.datasetImageName = "";

  state.bgLoaded = false;
  state.bgDataset = null;
  state.bgModel = null;
  state.bgTotal = 0;
  state.bgIdx = 0;
  state.bgSelected = 0;
  state.bgSkipped = 0;
  state.bgOutRoot = "";

  state.dragging = false;
  state.dragStart = null;
  state.dragRect = null;

  state.frameAnnotations = [];
  state.allAnnotations = [];
  state.navSeq++;

  $("currentList").innerHTML = "";
  $("globalList").innerHTML = "";

  try {
    // Clear any previously displayed image (prevents "ghost frame" staying visible)
    state.img.src = "";
  } catch (_e) {}

  if (state.ctx && state.canvas) {
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
  }

  if (message) setStatus(message);
}

function setMode(mode) {
  state.mode = mode;
  const isVideo = (mode === "video");
  const isDataset = (mode === "dataset");
  const isBg = (mode === "bg");

  $("tabVideo").classList.toggle("tabActive", isVideo);
  $("tabDataset").classList.toggle("tabActive", isDataset);
  $("tabBg").classList.toggle("tabActive", isBg);

  // Header controls
  $("videoControls").classList.toggle("hidden", !isVideo);

  // Viewer controls
  $("datasetControls").classList.toggle("hidden", !isDataset);
  $("bgControls").classList.toggle("hidden", !isBg);
  $("startBtn").closest(".transport").classList.toggle("hidden", !isVideo);

  // Sidebar titles + global list
  if (isVideo) {
    $("currentTitle").textContent = "Current frame boxes";
    $("globalTitle").textContent = "All annotations (click to jump)";
  } else if (isDataset) {
    $("currentTitle").textContent = "Current image boxes";
    $("globalTitle").textContent = "All annotations";
  } else {
    $("currentTitle").textContent = "Background Labeler";
    $("globalTitle").textContent = "All annotations";
  }
  $("globalList").classList.toggle("hidden", !isVideo);
  if (!isVideo) {
    $("globalList").innerHTML = "";
  }
}

function detectModelsFromDatasetName(datasetName) {
  const ds = normalizeText(datasetName);
  const models = (state.config?.models || []).map(m => m.name).filter(Boolean);
  const matches = [];
  for (const m of models) {
    const nm = normalizeText(m);
    if (!nm) continue;
    if (ds.includes(nm)) matches.push(m);
  }
  return matches;
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
  if (state.mode === "video") {
    if (!state.videoLoaded) return;
  } else if (state.mode === "dataset") {
    if (!state.datasetLoaded) return;
  } else {
    if (!state.bgLoaded) return;
  }

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
  if (state.mode === "dataset") {
    if (!state.datasetLoaded) return;
    const data = await api.getDatasetAnnotations(state.datasetImageIdx);
    state.frameAnnotations = data.annotations || [];
    state.allAnnotations = [];

    // current image list
    const ul = $("currentList");
    ul.innerHTML = "";

    // Background marker entry (dataset)
    if (state.frameAnnotations.length === 0) {
      const liBg = document.createElement("li");
      liBg.className = "item";
      const mainBg = document.createElement("div");
      mainBg.className = "itemMain";
      const titleBg = document.createElement("div");
      titleBg.className = "itemTitle";
      titleBg.textContent = `BACKGROUND (${state.datasetModel || ""})`;
      const subBg = document.createElement("div");
      subBg.className = "itemSub";
      subBg.textContent = "Empty label file will be saved";
      mainBg.appendChild(titleBg); mainBg.appendChild(subBg);
      const btnsBg = document.createElement("div");
      btnsBg.className = "itemBtns";
      const setBg = document.createElement("button");
      setBg.className = "btn";
      setBg.textContent = "Mark";
      setBg.onclick = async (e) => {
        e.stopPropagation();
        await api.markDatasetBackground(state.datasetImageIdx);
        await refreshLists();
      };
      btnsBg.appendChild(setBg);
      liBg.appendChild(mainBg);
      liBg.appendChild(btnsBg);
      ul.appendChild(liBg);
    }

    for (const a of state.frameAnnotations) {
      const li = document.createElement("li");
      li.className = "item";
      const main = document.createElement("div");
      main.className = "itemMain";
      const title = document.createElement("div");
      title.className = "itemTitle";
      title.textContent = `${a.class_name}`;
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
        await api.deleteDatasetAnnotation(a.id);
        await refreshLists();
        draw();
      };
      btns.appendChild(del);
      li.appendChild(main);
      li.appendChild(btns);
      ul.appendChild(li);
    }

    $("globalList").innerHTML = "";
    return;
  }

  // video mode
  if (!state.videoLoaded) return;
  const frameData = await api.getFrameAnnotations(state.frameIdx);
  state.frameAnnotations = frameData.annotations || [];
  const backgroundModels = frameData.background_models || [];
  const allData = await api.getAllAnnotations();
  state.allAnnotations = allData.annotations || [];

  // current frame
  const ul = $("currentList");
  ul.innerHTML = "";

  // Background marker entries (video)
  for (const m of backgroundModels) {
    const liBg = document.createElement("li");
    liBg.className = "item";
    const mainBg = document.createElement("div");
    mainBg.className = "itemMain";
    const titleBg = document.createElement("div");
    titleBg.className = "itemTitle";
    titleBg.textContent = `BACKGROUND (${m})`;
    const subBg = document.createElement("div");
    subBg.className = "itemSub";
    subBg.textContent = "Empty label file will be exported";
    mainBg.appendChild(titleBg); mainBg.appendChild(subBg);
    const btnsBg = document.createElement("div");
    btnsBg.className = "itemBtns";
    const delBg = document.createElement("button");
    delBg.className = "btn danger";
    delBg.textContent = "Delete";
    delBg.onclick = async (e) => {
      e.stopPropagation();
      await api.unmarkBackground(state.frameIdx, m);
      await refreshLists();
    };
    btnsBg.appendChild(delBg);
    liBg.appendChild(mainBg);
    liBg.appendChild(btnsBg);
    ul.appendChild(liBg);
  }

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
    if (a.kind === "background" || a.class_name === "BACKGROUND") {
      title.textContent = `f${String(a.frame_idx).padStart(6, "0")}  BACKGROUND (${a.model})`;
    } else {
      title.textContent = `f${String(a.frame_idx).padStart(6, "0")}  ${a.model}/${a.class_name}`;
    }
    const sub = document.createElement("div");
    sub.className = "itemSub";
    sub.textContent = (a.kind === "background" || a.class_name === "BACKGROUND")
      ? "empty label file"
      : `[${a.x1},${a.y1}] → [${a.x2},${a.y2}]`;
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
      if (a.kind === "background" || a.class_name === "BACKGROUND") {
        await api.unmarkBackground(a.frame_idx, a.model);
      } else {
        await api.deleteAnnotation(a.id);
      }
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

async function renderDatasetImage(idx) {
  if (!state.datasetLoaded) return;
  const mySeq = ++state.navSeq;
  const clamped = clamp(idx, 0, Math.max(0, state.datasetImageCount - 1));

  const {blob, imageIdx, imageName} = await api.getDatasetImage(clamped);
  if (mySeq !== state.navSeq) return;

  state.datasetImageIdx = imageIdx;
  state.datasetImageName = imageName;

  const url = URL.createObjectURL(blob);
  await new Promise((resolve, reject) => {
    state.img.onload = () => resolve();
    state.img.onerror = reject;
    state.img.src = url;
  });
  URL.revokeObjectURL(url);
  if (mySeq !== state.navSeq) return;

  state.imgW = state.img.naturalWidth || state.imgW;
  state.imgH = state.img.naturalHeight || state.imgH;

  await refreshLists();
  if (mySeq !== state.navSeq) return;
  draw();

  setStatus(`Dataset ${state.datasetName} | ${state.datasetImageIdx + 1}/${state.datasetImageCount} | ${state.datasetImageName}`);
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
  $("backgroundCheck").checked = false;
  $("modal").classList.remove("hidden");
  $("classFilter").focus();

  // In dataset mode we fix the model to the dataset model (one-model-at-a-time).
  if (state.mode === "dataset") {
    $("modalModelSelect").value = state.datasetModel || $("modalModelSelect").value;
    $("modalModelSelect").disabled = true;
  } else {
    $("modalModelSelect").disabled = false;
  }
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
  const asBackground = $("backgroundCheck").checked;
  if (!model) {
    $("modalError").textContent = "Pick a model.";
    return;
  }
  if (!asBackground && !cls) {
    $("modalError").textContent = "Pick a class.";
    return;
  }
  const r = state.modalRect;
  try {
    if (state.mode === "dataset") {
      if (asBackground) {
        if ((state.frameAnnotations || []).length > 0) {
          $("modalError").textContent = "Please remove all annotations from current image to select it as background.";
          return;
        }
        await api.markDatasetBackground(state.datasetImageIdx);
      } else {
        await api.addDatasetAnnotation({
          image_idx: state.datasetImageIdx,
          class_name: cls,
          x1: r.x1, y1: r.y1, x2: r.x2, y2: r.y2
        });
      }
    } else {
      if (asBackground) {
        if ((state.frameAnnotations || []).length > 0) {
          $("modalError").textContent = "Please remove all annotations from current frame to select it as background.";
          return;
        }
        await api.markBackground(state.frameIdx, model);
        state.dirty = true;
      } else {
        await api.addAnnotation({
          frame_idx: state.frameIdx,
          model,
          class_name: cls,
          x1: r.x1, y1: r.y1, x2: r.x2, y2: r.y2
        });
        state.dirty = true;
      }
    }
    closeModal();
    await refreshLists();
    draw();
  } catch (e) {
    $("modalError").textContent = e.message || String(e);
    // If server cleared the workspace (e.g., after export) but UI still had an old frame,
    // reset UI so user can't keep drawing on a stale image.
    if ((e.message || "").toLowerCase().includes("no video loaded")) {
      resetWorkspaceUI("No video loaded. Load a video to continue.");
    }
  }
}

function installCanvasHandlers() {
  state.canvas.addEventListener("mousedown", (evt) => {
    if (state.mode === "video") {
      if (!state.videoLoaded) return;
      if (state.playing) stopPlayback();
    } else {
      if (!state.datasetLoaded) return;
    }
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

    if (state.mode === "dataset") {
      if (e.key === "ArrowLeft") { e.preventDefault(); await renderDatasetImage(state.datasetImageIdx - 1); }
      if (e.key === "ArrowRight") { e.preventDefault(); await renderDatasetImage(state.datasetImageIdx + 1); }
      return;
    }

    if (state.mode === "bg") {
      if (e.key === "b" || e.key === "B") { e.preventDefault(); await window.__bgBackground?.(); }
      if (e.key === "s" || e.key === "S") { e.preventDefault(); await window.__bgSkip?.(); }
      if (e.key === "ArrowLeft") { e.preventDefault(); await window.__bgPrev?.(); }
      if (e.key === "ArrowRight") { e.preventDefault(); await window.__bgNext?.(); }
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
  if (cfg.app_version) {
    setStatus(`Loaded (v${cfg.app_version}). Ready.`);
  }

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

  async function doLoadVideo(videoName, loadExistingExports=false) {
    // Clear any previous UI state before loading a new video.
    resetWorkspaceUI("Loading video…");
    setStatus("Loading video…");
    const res = await api.loadVideo(videoName, !!loadExistingExports);
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
    if ((res.loaded_existing || 0) > 0) {
      setStatus(`Loaded existing labels: ${res.loaded_existing}`);
    }
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

  // ---- Tabs ----
  setMode("video");
  $("tabVideo").onclick = () => {
    setMode("video");
    setStatus(`Video Labeler. Videos: ${state.config?.videos_dir || ""}`);
  };
  $("tabDataset").onclick = async () => {
    if (state.dirty) {
      const ok = window.confirm("You have un-exported video annotations. Switch to Dataset Fixer and lose them?");
      if (!ok) return;
    }
    resetWorkspaceUI("Dataset Fixer");
    setMode("dataset");
    try {
      const ds = await api.listDatasets();
      const sel = $("datasetSelect");
      sel.innerHTML = "";
      for (const name of (ds.datasets || [])) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
      }
      sel.onchange = () => {
        const name = sel.value || "";
        const matches = detectModelsFromDatasetName(name);
        if (matches.length === 1) {
          setStatus(`Dataset Fixer. Detected model: ${matches[0]}`);
        } else if (matches.length > 1) {
          setStatus(`Dataset Fixer. Multiple model matches: ${matches.join(", ")}`);
        } else if (name) {
          setStatus(`Dataset Fixer. No model detected from dataset name.`);
        }
      };
      if ((ds.datasets || []).length === 0) {
        setStatus(`No datasets found in: ${ds.datasets_dir}`);
      } else {
        setStatus(`Dataset Fixer. Datasets: ${ds.datasets_dir}`);
      }
    } catch (e) {
      setStatus(`Dataset list failed: ${e.message || e}`);
    }
  };

  // ---- Background Labeler ----
  $("tabBg").onclick = async () => {
    if (state.dirty) {
      const ok = window.confirm("You have un-exported video annotations. Switch and lose them?");
      if (!ok) return;
    }
    resetWorkspaceUI("Background Labeler");
    setMode("bg");
    try {
      const cfgBg = await api.getBgConfig();
      // Populate folder select (Mode 1)
      const folderSel = $("bgFolderSelect");
      folderSel.innerHTML = "";
      const optAll = document.createElement("option");
      optAll.value = "";
      optAll.textContent = "-- Select folder --";
      folderSel.appendChild(optAll);
      for (const name of (cfgBg.datasets || [])) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        folderSel.appendChild(opt);
      }
      // Populate model select
      const mSel = $("bgModelSelect");
      mSel.innerHTML = "";
      for (const name of (cfgBg.models || [])) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        mSel.appendChild(opt);
      }
      // Populate camera filter
      const cSel = $("bgCameraSelect");
      cSel.innerHTML = "";
      for (const name of (cfgBg.camera_filters || ["ALL"])) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        cSel.appendChild(opt);
      }
      // Show existing dir path label (Mode 2) - display path from config
      const existingLabel = $("bgExistingDirLabel");
      if (cfgBg.has_existing && cfgBg.existing_path) {
        existingLabel.textContent = cfgBg.existing_path;
      } else {
        // Default: show relative path even if folder doesn't exist yet
        const datasetsDir = cfgBg.datasets_dir.replace(/\\/g, "/");
        existingLabel.textContent = `${datasetsDir}/existing`;
      }
      // Mode selector change handler
      $("bgModeSelect").onchange = () => {
        const mode = $("bgModeSelect").value;
        $("bgModeFolder").classList.toggle("hidden", mode !== "folder");
        $("bgModeExisting").classList.toggle("hidden", mode !== "existing");
      };
      $("bgModeSelect").onchange(); // initial toggle
      setStatus(`Background Labeler. Datasets: ${cfgBg.datasets_dir}`);
    } catch (e) {
      setStatus(`BG config failed: ${e.message || e}`);
    }
  };

  async function bgRenderCurrent() {
    const mySeq = ++state.navSeq;
    const {blob, idx, name, decision} = await api.bgGetImage();
    if (mySeq !== state.navSeq) return;
    state.bgIdx = idx;
    state.bgLoaded = true;

    const url = URL.createObjectURL(blob);
    await new Promise((resolve, reject) => {
      state.img.onload = () => resolve();
      state.img.onerror = reject;
      state.img.src = url;
    });
    URL.revokeObjectURL(url);
    if (mySeq !== state.navSeq) return;

    state.imgW = state.img.naturalWidth || state.imgW;
    state.imgH = state.img.naturalHeight || state.imgH;
    state.frameAnnotations = [];
    $("currentList").innerHTML = "";
    // UI: highlight current decision (default skip)
    $("bgBtn").classList.toggle("active", decision === "background");
    $("skipBtn").classList.toggle("active", decision !== "background");
    draw();
    setStatus(`BG ${state.bgDataset} | ${state.bgIdx + 1}/${state.bgTotal} | selected=${state.bgSelected} skipped=${state.bgSkipped} | ${name}`);
  }

  async function bgDecide(action) {
    if (!state.bgLoaded) return;
    const res = await api.bgDecide(action);
    state.bgSelected = res.selected || state.bgSelected;
    state.bgSkipped = res.skipped || state.bgSkipped;
    if (res.done) {
      setStatus(`Done. Output: ${state.bgOutRoot}`);
      return;
    }
    await bgRenderCurrent();
  }

  window.__bgBackground = async () => {
    try { await bgDecide("background"); } catch (e) { setStatus(`BG error: ${e.message || e}`); }
  };
  window.__bgSkip = async () => {
    try { await bgDecide("skip"); } catch (e) { setStatus(`BG error: ${e.message || e}`); }
  };

  $("bgStartBtn").onclick = async () => {
    const mode = $("bgModeSelect").value;
    const model = $("bgModelSelect").value;
    const cam = $("bgCameraSelect").value || "ALL";
    const shuffled = $("bgShuffleSelect").value === "1";
    if (!model) {
      window.alert("Pick target model.");
      return;
    }
    let req = {mode, target_model: model, camera_filter: cam, shuffled};
    if (mode === "folder") {
      const folder = $("bgFolderSelect").value;
      if (!folder) {
        window.alert("Pick source folder.");
        return;
      }
      req.folder_path = folder;
    } else if (mode === "existing") {
      // Always use "existing" relative to datasets_dir (from config)
      req.existing_datasets_dir = "existing";
    }
    try {
      resetWorkspaceUI("Starting background session…");
      setMode("bg");
      const res = await api.bgStart(req);
      state.bgLoaded = true;
      state.bgDataset = res.dataset_name || "";
      state.bgModel = model;
      state.bgTotal = res.total || 0;
      state.bgSelected = 0;
      state.bgSkipped = 0;
      state.bgOutRoot = res.out_root || "";
      await bgRenderCurrent();
    } catch (e) {
      setStatus(`BG start failed: ${e.message || e}`);
      window.alert(`BG start failed: ${e.message || e}`);
    }
  };
  $("bgBtn").onclick = async () => { await window.__bgBackground(); };
  $("skipBtn").onclick = async () => { await window.__bgSkip(); };
  $("bgPrevBtn").onclick = async () => {
    if (!state.bgLoaded) return;
    await api.bgSetIndex(Math.max(0, state.bgIdx - 1));
    await bgRenderCurrent();
  };
  $("bgNextBtn").onclick = async () => {
    if (!state.bgLoaded) return;
    await api.bgSetIndex(Math.min(state.bgTotal - 1, state.bgIdx + 1));
    await bgRenderCurrent();
  };

  window.__bgPrev = async () => { await $("bgPrevBtn").onclick(); };
  window.__bgNext = async () => { await $("bgNextBtn").onclick(); };
  $("bgFinishBtn").onclick = async () => {
    try {
      const res = await api.bgFinish();
      resetWorkspaceUI("Background session finished.");
      setMode("bg");
      if (res.out_root) window.alert(`Background session finished.\n\nOutput:\n${res.out_root}`);
    } catch (e) {
      window.alert(`Finish failed: ${e.message || e}`);
    }
  };

  $("loadDatasetBtn").onclick = async () => {
    const dsName = $("datasetSelect").value;
    if (!dsName) {
      window.alert(`No datasets found.\n\nPut datasets into:\n${state.config?.datasets_dir || ""}\n\nEach dataset must have images/ and labels/.`);
      return;
    }
    const matches = detectModelsFromDatasetName(dsName);
    let model = null;
    if (matches.length === 1) {
      model = matches[0];
    } else if (matches.length > 1) {
      const chosen = window.prompt(
        `Multiple models match this dataset name.\n\nDataset: ${dsName}\nMatches: ${matches.join(", ")}\n\nType the correct model name:`,
        matches[0]
      );
      if (!chosen) return;
      model = String(chosen).trim();
    } else {
      const allModels = (state.config?.models || []).map(m => m.name).filter(Boolean);
      const chosen = window.prompt(
        `WARNING: Could not detect model from dataset name.\n\nDataset: ${dsName}\n\nType model name to use:\n${allModels.join(", ")}`,
        allModels[0] || ""
      );
      if (!chosen) return;
      model = String(chosen).trim();
    }
    if (!model) {
      window.alert("Model is empty. Cannot load dataset.");
      return;
    }
    try {
      resetWorkspaceUI("Loading dataset…");
      setMode("dataset");
      const res = await api.loadDataset(dsName, model);
      state.datasetLoaded = true;
      state.datasetName = dsName;
      state.datasetModel = model;
      state.datasetImageCount = res.image_count || 0;
      state.datasetImageIdx = 0;
      if (matches.length === 0) {
        setStatus(`Dataset loaded: ${dsName} (${state.datasetImageCount} images) | Model: ${model} (manual)`);
      } else {
        setStatus(`Dataset loaded: ${dsName} (${state.datasetImageCount} images) | Model: ${model}`);
      }
      await renderDatasetImage(0);
    } catch (e) {
      setStatus(`Load dataset failed: ${e.message || e}`);
      window.alert(`Load dataset failed: ${e.message || e}`);
    }
  };

  $("prevImgBtn").onclick = async () => {
    if (!state.datasetLoaded) return;
    await renderDatasetImage(state.datasetImageIdx - 1);
  };
  $("nextImgBtn").onclick = async () => {
    if (!state.datasetLoaded) return;
    await renderDatasetImage(state.datasetImageIdx + 1);
  };
  $("saveOverwriteBtn").onclick = async () => {
    if (!state.datasetLoaded) return;
    try {
      const res = await api.saveDataset("overwrite");
      window.alert(`Saved (overwrite).\n\n${res.output_dataset_path}`);
      if (res.cleared) {
        resetWorkspaceUI("Dataset saved. Dataset unloaded.");
        setMode("dataset");
      }
    } catch (e) {
      window.alert(`Save failed: ${e.message || e}`);
    }
  };
  $("saveFixedBtn").onclick = async () => {
    if (!state.datasetLoaded) return;
    try {
      const res = await api.saveDataset("create_new");
      window.alert(`Saved as _fixed.\n\n${res.output_dataset_path}`);
      if (res.cleared) {
        resetWorkspaceUI("Dataset saved. Dataset unloaded.");
        setMode("dataset");
      }
    } catch (e) {
      window.alert(`Save failed: ${e.message || e}`);
    }
  };

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

      // Warn if this video already has exported files on disk.
      let loadExisting = false;
      try {
        const info = await api.getVideoInfo(v);
        if (info.has_exports) {
          const ok2 = window.confirm(
            `This video already has exports on disk:\n\n${info.output_root}\n\nImages: ${info.image_files}\nLabels: ${info.label_files}\n\nLoad anyway?`
          );
          if (!ok2) return;
          loadExisting = true; // user said "yes" => load exported labels back into workspace
        }
      } catch (e) {
        // Don't silently ignore; this is important UX.
        setStatus(`Warning check failed: ${e.message || e}`);
      }

      await doLoadVideo(v, loadExisting);
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

      // After export, backend clears workspace; mirror that in the UI.
      if (res.cleared) {
        resetWorkspaceUI("Export complete. Video closed. Load a new video to continue.");
      }
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


