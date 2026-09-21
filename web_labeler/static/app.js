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
  async getVideoHints(videoName) {
    return fetch(`/api/video/hints?video_name=${encodeURIComponent(videoName)}`).then(r => r.ok ? r.json() : {hints: null});
  },
  async getVideoCaseInfo(videoName) {
    return fetch(`/api/video/case_info?video_name=${encodeURIComponent(videoName)}`).then(r => r.ok ? r.json() : {instruction: null});
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
  async getDatasetStatus() {
    return fetch(`/api/datasets/status`).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        const msg = (typeof d === "string") ? d : JSON.stringify(d || data);
        throw new Error(msg || "Failed to get dataset status");
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
  async getTestMode() {
    return fetch("/api/datasets/test_mode").then(r => r.json());
  },
  async setTestMode(testMode) {
    return fetch("/api/datasets/test_mode", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({test_mode: testMode}),
    }).then(r => r.json());
  },
  async getImageTags(imageIdx) {
    return fetch(`/api/datasets/tags?image_idx=${encodeURIComponent(imageIdx)}`).then(r => r.json());
  },
  async setImageTags(imageIdx, tags) {
    return fetch("/api/datasets/tags", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({image_idx: imageIdx, tags}),
    }).then(r => r.json());
  },
  async getFrameTags(frameIdx) {
    return fetch(`/api/frame/tags?frame_idx=${frameIdx}`).then(r => r.json());
  },
  async setFrameTags(frameIdx, frameTags, bboxTags, bboxVariations) {
    return fetch("/api/frame/tags", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({frame_idx: frameIdx, frame_tags: frameTags, bbox_tags: bboxTags, bbox_variations: bboxVariations || {}}),
    }).then(r => r.json());
  },
  async getDatasetTags(imageIdx) {
    return fetch(`/api/datasets/tags?image_idx=${imageIdx}`).then(r => r.json());
  },
  async setDatasetTags(imageIdx, frameTags, bboxTags, bboxVariations) {
    return fetch("/api/datasets/tags", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({image_idx: imageIdx, frame_tags: frameTags, bbox_tags: bboxTags, bbox_variations: bboxVariations || {}}),
    }).then(r => r.json());
  },
  async getClassVariations() {
    return fetch("/api/class_variations").then(r => r.json());
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
  },
  async rawListModels() {
    return fetch(`/api/raw/models`).then(async r => {
      if (!r.ok) throw new Error((await r.json()).detail || "Failed to list raw models");
      return r.json();
    });
  },
  async rawListClasses(model) {
    return fetch(`/api/raw/classes?model=${encodeURIComponent(model)}`).then(async r => {
      if (!r.ok) throw new Error((await r.json()).detail || "Failed to list raw classes");
      return r.json();
    });
  },
  async rawLoad(model, className, camera) {
    return fetch(`/api/raw/load`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({model, class_name: className, camera: camera || "ALL"}),
    }).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || "Failed to load raw dataset");
      return data;
    });
  },
  async getRawImage(index) {
    const r = await fetch(`/api/raw/image?index=${index}`);
    if (!r.ok) throw new Error((await r.json()).detail || "Raw image fetch failed");
    const blob = await r.blob();
    const imageIdx = parseInt(r.headers.get("X-Image-Index") || String(index), 10);
    const imageName = r.headers.get("X-Image-Name") || "";
    return {blob, imageIdx, imageName};
  },
  async getRawAnnotations(index) {
    return fetch(`/api/raw/annotations?index=${index}`).then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || "Failed to get raw annotations");
      return data;
    });
  },
  // MDQ-3 flags: write a sidecar entry only, never touch the image/label.
  async _flagFetch(url, method, body) {
    const r = await fetch(url, {
      method,
      headers: body ? {"Content-Type": "application/json"} : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `Flag request failed (${r.status})`);
    return data;
  },
  async rawFlagSet(imageKey, pool, category, comment) {
    return this._flagFetch("/api/raw/flag", "POST", {image_key: imageKey, pool, category, comment});
  },
  async rawFlagRemove(imageKey, pool) {
    return this._flagFetch(`/api/raw/flag?image_key=${encodeURIComponent(imageKey)}&pool=${encodeURIComponent(pool)}`, "DELETE");
  },
  async frameFlagGet(model, frameIdx) {
    return this._flagFetch(`/api/frame/flag?model=${encodeURIComponent(model)}&frame_idx=${frameIdx}`, "GET");
  },
  async frameFlagSet(model, frameIdx, category, comment) {
    return this._flagFetch("/api/frame/flag", "POST", {model, frame_idx: frameIdx, category, comment});
  },
  async frameFlagRemove(model, frameIdx) {
    return this._flagFetch(`/api/frame/flag?model=${encodeURIComponent(model)}&frame_idx=${frameIdx}`, "DELETE");
  }
};

function buildSpeedMap() {
  return {
    1: {step: 1,  delay: 50, label: "x1"},
    2: {step: 2,  delay: 40, label: "x2"},
    3: {step: 4,  delay: 30, label: "x4"},
    4: {step: 8,  delay: 22, label: "x8"},
    5: {step: 16, delay: 18, label: "x16"},
    6: {step: 32, delay: 14, label: "x32"},
  };
}
let speedMap = buildSpeedMap();

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
  rawLoaded: false,
  datasetName: null,
  datasetModel: null,
  datasetImageCount: 0,
  datasetImageIdx: 0,
  datasetImageName: "",

  testMode: false,
  imageTags: [],
  videoFrameTags: {},  // {frame_idx: {frame_tags: [], bbox_tags: {ann_id: []}, bbox_variations: {ann_id: "LABEL"}}}
  classVariations: {},  // {model: {class: [variation_labels]}} — loaded from server on init

  // labeling hints (UV/MPI sidecar)
  hintData: null,   // parsed _hints.json for current video (UV ghost bbox)
  caseInfo: null,   // parsed _info.json or _hints.json for current video (instruction banner)

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

  // flag panel (MDQ-3)
  flagPanelOpen: false,
  flagTarget: null,      // {kind: "raw"|"frame", pool, imageKey?, frameIdx?, label}
  flagCategory: null,
  flagExisting: null,
  lastFlagPool: null,
  rawImageKey: null,     // "<class>/<filename>" of the RAW image on screen
  rawFlag: null,         // existing flag entry of that image, if any
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
  state.rawLoaded = false;
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

  state.hintData = null;
  state.caseInfo = null;
  renderCaseInfoBanner();
  state.rawImageKey = null;
  state.rawFlag = null;
  closeFlagPanel();
  updateFlagBadge();

  if (message) setStatus(message);
}

function renderCaseInfoBanner() {
  const banner = document.getElementById("caseInfoBanner");
  if (!banner) return;
  const info = state.caseInfo || state.hintData;
  if (!info || !info.instruction) {
    banner.style.display = "none";
    return;
  }
  const isUV  = (info.case_type === "UV");
  const color = isUV ? "#f97316" : "#3b82f6";
  const bg    = isUV ? "rgba(249,115,22,0.12)" : "rgba(59,130,246,0.12)";
  const badge = isUV ? "UV hallucination" : "MPI missing";
  const cls   = info.detected_class || info.target_class || "";
  banner.style.display = "block";
  banner.style.cssText = `display:block; padding:6px 12px; background:${bg}; border-left:3px solid ${color}; font-size:12px; color:#e2e8f0; line-height:1.5;`;
  banner.innerHTML =
    `<span style="background:${color};color:#fff;border-radius:3px;padding:1px 6px;font-weight:700;margin-right:8px;">${badge}</span>` +
    (cls ? `<strong>${cls}</strong> — ` : "") +
    info.instruction;
}

function setMode(mode) {
  state.mode = mode;
  const isVideo    = (mode === "video");
  const isDataset  = (mode === "dataset");
  const isBg       = (mode === "bg");
  const isAnalyzer = (mode === "analyzer");

  $("tabVideo").classList.toggle("tabActive", isVideo);
  $("tabDataset").classList.toggle("tabActive", isDataset);
  $("tabBg").classList.toggle("tabActive", isBg);
  $("tabAnalyzer").classList.toggle("tabActive", isAnalyzer);

  // Modestrip
  $("videoControls").classList.toggle("hidden", !isVideo);
  $("datasetControls").classList.toggle("hidden", !isDataset);
  $("bgControls").classList.toggle("hidden", !isBg);
  $("analyzerControls").classList.toggle("hidden", !isAnalyzer);

  // Main area: hide normal viewer+sidebar for analyzer, show analyzer panel
  const mainEl = document.querySelector(".main");
  if (mainEl) mainEl.style.display = isAnalyzer ? "none" : "";
  $("analyzerPanel").classList.toggle("hidden", !isAnalyzer);
  if (isAnalyzer && typeof window.__setAnalyzerSubMode === "function") {
    window.__setAnalyzerSubMode(analyzerState ? analyzerState.subMode : "inspector");
  }

  $("startBtn").closest(".transport").classList.toggle("hidden", !isVideo);

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
    if (!state.datasetLoaded && !state.rawLoaded) return;
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

  // UV ghost bbox hint (visual only — never stored or exported)
  if (state.hintData && Array.isArray(state.hintData.tlwh_norm) && state.hintData.at_second != null) {
    const currentSec = state.frameIdx / (state.fps || 25);
    const halfWindow = (state.hintData.window_seconds || 120) / 2;
    if (Math.abs(currentSec - state.hintData.at_second) <= halfWindow) {
      const [tx, ty, tw, th] = state.hintData.tlwh_norm;
      const c1 = imageToCanvas(tx * state.imgW, ty * state.imgH);
      const c2 = imageToCanvas((tx + tw) * state.imgW, (ty + th) * state.imgH);
      ctx.save();
      ctx.setLineDash([8, 4]);
      ctx.strokeStyle = "rgba(255,165,0,0.85)";
      ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
      ctx.strokeRect(c1.x, c1.y, c2.x - c1.x, c2.y - c1.y);
      ctx.setLineDash([]);
      ctx.font = `${Math.floor(11 * (window.devicePixelRatio || 1))}px ui-sans-serif`;
      ctx.fillStyle = "rgba(255,165,0,0.9)";
      ctx.fillText(`HINT: ${state.hintData.detected_class}`, c1.x + 4, c1.y > 14 ? c1.y - 4 : c1.y + 14);
      ctx.restore();
    }
  }
}

function renderFrameTagsPanel(frameIdx, frameTagsData) {
  const panel = $("frameTagsPanel");
  if (!state.testMode) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");

  const FRAME_TAGS = ["low_light", "busy", "force_day", "force_night"];
  const OVERRIDE_TAGS = ["force_day", "force_night"];
  const currentFrameTags = (frameTagsData && frameTagsData.frame_tags) ? [...frameTagsData.frame_tags] : [];

  const content = $("frameTagsContent");
  content.innerHTML = "";
  const grid = document.createElement("div");
  grid.style.cssText = "display:flex; flex-wrap:wrap; gap:6px;";

  for (const tag of FRAME_TAGS) {
    const isOverride = OVERRIDE_TAGS.includes(tag);
    const lbl = document.createElement("label");
    lbl.style.cssText = `display:flex; align-items:center; gap:3px; font-size:12px; cursor:pointer; padding:2px 6px; border-radius:4px; background:${isOverride ? "#fef3c7" : "#ede9fe"}; border:1px solid ${isOverride ? "#f59e0b" : "#a78bfa"}; color:${isOverride ? "#713f12" : "#3b0764"};`;
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.dataset.tag = tag;
    cb.checked = currentFrameTags.includes(tag);
    cb.addEventListener("change", async () => {
      let tags = [...currentFrameTags];
      if (cb.checked) {
        if (tag === "force_day") tags = tags.filter(t => t !== "force_night");
        if (tag === "force_night") tags = tags.filter(t => t !== "force_day");
        if (!tags.includes(tag)) tags.push(tag);
      } else {
        tags = tags.filter(t => t !== tag);
      }
      // update currentFrameTags in place
      currentFrameTags.length = 0;
      tags.forEach(t => currentFrameTags.push(t));
      // get current bbox_tags from state and save
      const existing = state.videoFrameTags[frameIdx] || {};
      await api.setFrameTags(frameIdx, tags, existing.bbox_tags || {}, existing.bbox_variations || {});
      state.videoFrameTags[frameIdx] = {frame_tags: tags, bbox_tags: existing.bbox_tags || {}, bbox_variations: existing.bbox_variations || {}};
      // re-render checkboxes to update visual state
      renderFrameTagsPanel(frameIdx, state.videoFrameTags[frameIdx]);
    });
    const span = document.createElement("span");
    span.textContent = tag;
    lbl.appendChild(cb); lbl.appendChild(span);
    grid.appendChild(lbl);
  }
  content.appendChild(grid);
}

async function refreshLists() {
  if (state.mode === "dataset") {
    if (!state.datasetLoaded && !state.rawLoaded) return;
    const data = state.rawLoaded
      ? await api.getRawAnnotations(state.datasetImageIdx)
      : await api.getDatasetAnnotations(state.datasetImageIdx);
    state.frameAnnotations = data.annotations || [];
    state.allAnnotations = [];
    if (state.rawLoaded) {
      state.rawImageKey = data.image_key || null;
      state.rawFlag = data.flag || null;
      updateFlagBadge();
    }
    let tagsData = {frame_tags: [], bbox_tags: {}, bbox_variations: {}};
    if (state.testMode) {
      tagsData = await api.getDatasetTags(state.datasetImageIdx);
      state.imageTags = tagsData.frame_tags || [];
    }
    const isBackground = data.is_background || false;
    const isDeleted = data.is_deleted || false;

    // current image list
    const ul = $("currentList");
    ul.innerHTML = "";

    // Tag panel (test mode only) — frame-level tags only
    if (state.testMode) {
      const liTags = document.createElement("li");
      liTags.className = "item";
      liTags.style.backgroundColor = "rgba(124,58,237,0.12)";
      liTags.style.borderLeft = "3px solid #7c3aed";
      liTags.style.padding = "8px 10px";
      liTags.style.flexWrap = "wrap";
      liTags.style.gap = "4px";

      const tagHeader = document.createElement("div");
      tagHeader.style.cssText = "font-weight:600; color:#c4b5fd; font-size:11px; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px; width:100%;";
      tagHeader.textContent = "🧪 Frame Tags";
      liTags.appendChild(tagHeader);

      const FRAME_TAGS = ["low_light", "busy", "force_day", "force_night"];
      const OVERRIDE_TAGS = ["force_day", "force_night"];
      const currentFrameTags = [...(tagsData.frame_tags || [])];

      const tagGrid = document.createElement("div");
      tagGrid.style.cssText = "display:flex; flex-wrap:wrap; gap:6px; align-items:center;";

      for (const tag of FRAME_TAGS) {
        const isOverride = OVERRIDE_TAGS.includes(tag);
        const lbl = document.createElement("label");
        lbl.style.cssText = `display:flex; align-items:center; gap:3px; font-size:12px; cursor:pointer; padding:2px 6px; border-radius:4px; background:${isOverride ? "#fef3c7" : "#ede9fe"}; border:1px solid ${isOverride ? "#f59e0b" : "#a78bfa"}; color:${isOverride ? "#713f12" : "#3b0764"};`;
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.dataset.tag = tag;
        cb.checked = currentFrameTags.includes(tag);
        cb.addEventListener("change", async () => {
          let tags = [...currentFrameTags];
          if (cb.checked) {
            // Mutual exclusion for force_day/force_night
            if (tag === "force_day") tags = tags.filter(t => t !== "force_night");
            if (tag === "force_night") tags = tags.filter(t => t !== "force_day");
            if (!tags.includes(tag)) tags.push(tag);
          } else {
            tags = tags.filter(t => t !== tag);
          }
          currentFrameTags.length = 0;
          tags.forEach(t => currentFrameTags.push(t));
          state.imageTags = tags;
          await api.setDatasetTags(state.datasetImageIdx, tags, tagsData.bbox_tags || {}, tagsData.bbox_variations || {});
          await refreshLists();
        });
        const span = document.createElement("span");
        span.textContent = tag;
        lbl.appendChild(cb);
        lbl.appendChild(span);
        tagGrid.appendChild(lbl);
      }

      liTags.appendChild(tagGrid);
      ul.appendChild(liTags);
    }

    // Deletion marker entry (always shown if marked for deletion)
    if (isDeleted) {
      const liDel = document.createElement("li");
      liDel.className = "item";
      liDel.style.backgroundColor = "#fee";
      liDel.style.borderLeft = "3px solid #c00";
      const mainDel = document.createElement("div");
      mainDel.className = "itemMain";
      const titleDel = document.createElement("div");
      titleDel.className = "itemTitle";
      titleDel.textContent = "🗑️ MARKED FOR DELETION";
      titleDel.style.color = "#c00";
      titleDel.style.fontWeight = "bold";
      const subDel = document.createElement("div");
      subDel.className = "itemSub";
      subDel.textContent = "Image and label will be deleted on save (overwrite)";
      mainDel.appendChild(titleDel); mainDel.appendChild(subDel);
      const btnsDel = document.createElement("div");
      btnsDel.className = "itemBtns";
      const unsetDel = document.createElement("button");
      unsetDel.className = "btn";
      unsetDel.textContent = "Unmark";
      unsetDel.onclick = async (e) => {
        e.stopPropagation();
        await api.unmarkDatasetDelete(state.datasetImageIdx);
        await refreshLists();
      };
      btnsDel.appendChild(unsetDel);
      liDel.appendChild(mainDel);
      liDel.appendChild(btnsDel);
      ul.appendChild(liDel);
    }

    // Background marker entry (dataset) - only if no annotations and not deleted
    if (state.frameAnnotations.length === 0 && !isDeleted) {
      const liBg = document.createElement("li");
      liBg.className = "item";
      if (isBackground) {
        liBg.style.backgroundColor = "#efe";
        liBg.style.borderLeft = "3px solid #0a0";
      }
      const mainBg = document.createElement("div");
      mainBg.className = "itemMain";
      const titleBg = document.createElement("div");
      titleBg.className = "itemTitle";
      titleBg.textContent = isBackground ? `✅ BACKGROUND (${state.datasetModel || ""})` : `BACKGROUND (${state.datasetModel || ""})`;
      const subBg = document.createElement("div");
      subBg.className = "itemSub";
      subBg.textContent = isBackground ? "Empty label file will be saved" : "Click Mark to set as background";
      mainBg.appendChild(titleBg); mainBg.appendChild(subBg);
      const btnsBg = document.createElement("div");
      btnsBg.className = "itemBtns";
      if (!isBackground) {
        const setBg = document.createElement("button");
        setBg.className = "btn";
        setBg.textContent = "Mark";
        setBg.onclick = async (e) => {
          e.stopPropagation();
          await api.markDatasetBackground(state.datasetImageIdx);
          await refreshLists();
        };
        btnsBg.appendChild(setBg);
      } else {
        const unsetBg = document.createElement("button");
        unsetBg.className = "btn";
        unsetBg.textContent = "Unmark";
        unsetBg.onclick = async (e) => {
          e.stopPropagation();
          await api.unmarkDatasetBackground(state.datasetImageIdx);
          await refreshLists();
        };
        btnsBg.appendChild(unsetBg);
      }
      // Delete button (only if no annotations and not background)
      if (!isBackground) {
        const delBtn = document.createElement("button");
        delBtn.className = "btn";
        delBtn.style.backgroundColor = "#c00";
        delBtn.style.color = "#fff";
        delBtn.textContent = "Delete";
        delBtn.onclick = async (e) => {
          e.stopPropagation();
          if (window.confirm("Mark this image for deletion? It will be removed from the dataset when you save (overwrite).")) {
            await api.markDatasetDelete(state.datasetImageIdx);
            await refreshLists();
          }
        };
        btnsBg.appendChild(delBtn);
      }
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

      // BBox-level tags + variation (test mode only)
      if (state.testMode) {
        const BBOX_TAGS = ["occlusion", "partial", "blurry"];
        const bboxTagRow = document.createElement("div");
        bboxTagRow.style.cssText = "display:flex; gap:4px; flex-wrap:wrap; margin-top:4px; padding-top:4px; border-top:1px solid #e0c9ff; align-items:center;";
        const currentBboxTags = [...((tagsData.bbox_tags && tagsData.bbox_tags[a.id]) || [])];

        for (const tag of BBOX_TAGS) {
          const lbl = document.createElement("label");
          lbl.style.cssText = "display:flex; align-items:center; gap:2px; font-size:11px; cursor:pointer; padding:1px 5px; border-radius:3px; background:#dbeafe; border:1px solid #93c5fd; color:#1e3a5f;";
          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.checked = currentBboxTags.includes(tag);
          cb.addEventListener("change", async () => {
            let tags = [...currentBboxTags];
            if (cb.checked) { if (!tags.includes(tag)) tags.push(tag); }
            else { tags = tags.filter(t => t !== tag); }
            currentBboxTags.length = 0;
            tags.forEach(t => currentBboxTags.push(t));
            const updatedBboxTags = {...(tagsData.bbox_tags || {})};
            updatedBboxTags[a.id] = tags;
            tagsData.bbox_tags = updatedBboxTags;
            await api.setDatasetTags(state.datasetImageIdx, tagsData.frame_tags || [], updatedBboxTags, tagsData.bbox_variations || {});
          });
          const span = document.createElement("span");
          span.textContent = tag;
          lbl.appendChild(cb); lbl.appendChild(span);
          bboxTagRow.appendChild(lbl);
        }

        // Variation dropdown — only for classes with known subclasses
        // In dataset mode a.model is absent; fall back to state.datasetModel
        const _dsModel = (a.model || state.datasetModel || '').toLowerCase();
        const modelVars = state.classVariations[_dsModel] || {};
        const variations = modelVars[a.class_name] || [];
        if (variations.length > 0) {
          const sel = document.createElement("select");
          sel.style.cssText = "font-size:11px; padding:1px 4px; border-radius:3px; border:1px solid #a78bfa; background:#f5f3ff; color:#3b0764; cursor:pointer; max-width:160px;";
          sel.title = "Variation (subclass)";
          const blank = document.createElement("option");
          blank.value = ""; blank.textContent = "— variation —";
          sel.appendChild(blank);
          for (const v of variations) {
            const opt = document.createElement("option");
            opt.value = v; opt.textContent = v;
            sel.appendChild(opt);
          }
          sel.value = (tagsData.bbox_variations && tagsData.bbox_variations[a.id]) || "";
          sel.addEventListener("change", async () => {
            const updatedVariations = {...(tagsData.bbox_variations || {})};
            if (sel.value) updatedVariations[a.id] = sel.value;
            else delete updatedVariations[a.id];
            tagsData.bbox_variations = updatedVariations;
            await api.setDatasetTags(state.datasetImageIdx, tagsData.frame_tags || [], tagsData.bbox_tags || {}, updatedVariations);
          });
          bboxTagRow.appendChild(sel);
        }

        li.appendChild(bboxTagRow);
      }

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

  if (state.testMode && state.videoLoaded) {
    const tagsData = await api.getFrameTags(state.frameIdx);
    state.videoFrameTags[state.frameIdx] = {frame_tags: tagsData.frame_tags || [], bbox_tags: tagsData.bbox_tags || {}, bbox_variations: tagsData.bbox_variations || {}};
    renderFrameTagsPanel(state.frameIdx, state.videoFrameTags[state.frameIdx]);
  } else {
    renderFrameTagsPanel(state.frameIdx, null);
  }

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

    // BBox-level tags + variation (test mode only)
    if (state.testMode) {
      const BBOX_TAGS = ["occlusion", "partial", "blurry"];
      const bboxTagRow = document.createElement("div");
      bboxTagRow.style.cssText = "display:flex; gap:4px; flex-wrap:wrap; margin-top:4px; padding-top:4px; border-top:1px solid #e0c9ff; align-items:center;";
      const currentBboxTags = (state.videoFrameTags[state.frameIdx] && state.videoFrameTags[state.frameIdx].bbox_tags)
        ? [...(state.videoFrameTags[state.frameIdx].bbox_tags[a.id] || [])]
        : [];

      for (const tag of BBOX_TAGS) {
        const lbl = document.createElement("label");
        lbl.style.cssText = "display:flex; align-items:center; gap:2px; font-size:11px; cursor:pointer; padding:1px 5px; border-radius:3px; background:#dbeafe; border:1px solid #93c5fd; color:#1e3a5f;";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = currentBboxTags.includes(tag);
        cb.addEventListener("change", async () => {
          let tags = [...currentBboxTags];
          if (cb.checked) { if (!tags.includes(tag)) tags.push(tag); }
          else { tags = tags.filter(t => t !== tag); }
          currentBboxTags.length = 0;
          tags.forEach(t => currentBboxTags.push(t));
          const frameEntry = state.videoFrameTags[state.frameIdx] || {frame_tags: [], bbox_tags: {}, bbox_variations: {}};
          frameEntry.bbox_tags = {...(frameEntry.bbox_tags || {})};
          frameEntry.bbox_tags[a.id] = tags;
          state.videoFrameTags[state.frameIdx] = frameEntry;
          await api.setFrameTags(state.frameIdx, frameEntry.frame_tags || [], frameEntry.bbox_tags, frameEntry.bbox_variations || {});
        });
        const span = document.createElement("span");
        span.textContent = tag;
        lbl.appendChild(cb); lbl.appendChild(span);
        bboxTagRow.appendChild(lbl);
      }

      // Variation dropdown — only for classes with known subclasses
      const modelVars = state.classVariations[(a.model || '').toLowerCase()] || {};
      const variations = modelVars[a.class_name] || [];
      if (variations.length > 0) {
        const frameEntry = state.videoFrameTags[state.frameIdx] || {};
        const currentVariation = (frameEntry.bbox_variations && frameEntry.bbox_variations[a.id]) || "";
        const sel = document.createElement("select");
        sel.style.cssText = "font-size:11px; padding:1px 4px; border-radius:3px; border:1px solid #a78bfa; background:#f5f3ff; color:#3b0764; cursor:pointer; max-width:160px;";
        sel.title = "Variation (subclass)";
        const blank = document.createElement("option");
        blank.value = ""; blank.textContent = "— variation —";
        sel.appendChild(blank);
        for (const v of variations) {
          const opt = document.createElement("option");
          opt.value = v; opt.textContent = v;
          sel.appendChild(opt);
        }
        sel.value = currentVariation;
        sel.addEventListener("change", async () => {
          const fe = state.videoFrameTags[state.frameIdx] || {frame_tags: [], bbox_tags: {}, bbox_variations: {}};
          fe.bbox_variations = {...(fe.bbox_variations || {})};
          if (sel.value) fe.bbox_variations[a.id] = sel.value;
          else delete fe.bbox_variations[a.id];
          state.videoFrameTags[state.frameIdx] = fe;
          await api.setFrameTags(state.frameIdx, fe.frame_tags || [], fe.bbox_tags || {}, fe.bbox_variations);
        });
        bboxTagRow.appendChild(sel);
      }

      li.appendChild(bboxTagRow);
    }

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
  setStatus(`${state.videoName} | Frame ${state.frameIdx + 1} | ${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")} | speed ${speedMap[state.speed].label}`);

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

async function renderRawImage(idx) {
  if (!state.rawLoaded) return;
  const mySeq = ++state.navSeq;
  const clamped = clamp(idx, 0, Math.max(0, state.datasetImageCount - 1));

  const {blob, imageIdx, imageName} = await api.getRawImage(clamped);
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

  setStatus(`Raw ${state.datasetName} | ${state.datasetImageIdx + 1}/${state.datasetImageCount} | ${state.datasetImageName}`);
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
      setStatus(`${state.videoName} | Frame ${state.frameIdx + 1} | ${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")} | speed ${speedMap[state.speed].label}`);

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

// ── Flag panel (MDQ-3) ─────────────────────────────────────────────────────
const FLAG_POOLS = ["glasses", "bottles", "cups", "pitchers", "shots"];
const FLAG_CATEGORIES = [
  {key: "gibberish", label: "Gibberish",
   hint: "Blur / out of focus so strong that even a person cannot tell the class."},
  {key: "wrong_frame_wrong_class", label: "Wrong frame, wrong class",
   hint: "The image is clearly NOT the declared class, regardless of image quality."},
  {key: "near_duplicate", label: "Near-duplicate",
   hint: "Visually almost identical to another example of this class — adds no new information."},
  {key: "mislabeled_background", label: "Mislabeled background",
   hint: "A background image that actually contains a visible example of some class."},
];

function isTypingTarget(t) {
  if (!t || !t.tagName) return false;
  const tag = t.tagName.toUpperCase();
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || !!t.isContentEditable;
}

function updateFlagBadge() {
  const el = document.getElementById("flagBadge");
  if (!el) return;
  const f = (state.mode === "dataset" && state.rawLoaded) ? state.rawFlag : null;
  if (!f) {
    el.style.display = "none";
    el.textContent = "";
    return;
  }
  const cat = FLAG_CATEGORIES.find(c => c.key === f.category);
  const what = f.category ? (cat ? cat.label : f.category)
    : (f.signal ? `${f.signal.metric}: ${f.signal.value}` : "flagged");
  const dec = f.owner_decision ? ` · owner: ${f.owner_decision}` : "";
  el.textContent = `⚑ FLAGGED — ${what}${dec}${f.comment ? " — " + f.comment : ""}`;
  el.style.display = "block";
}

function flagIsMutable(entry) {
  return !entry || (entry.source === "manual_goca" && !entry.owner_decision);
}

function showFlagError(msg) {
  $("flagError").textContent = msg || "";
}

function renderFlagPanel() {
  const t = state.flagTarget;
  if (!t) return;
  $("flagTarget").textContent = `${t.pool} · ${t.label}`;
  $("flagPoolRow").style.display = t.kind === "frame" ? "" : "none";
  const ex = state.flagExisting;
  const box = $("flagExisting");
  if (ex) {
    const what = ex.category ? ex.category : (ex.signal ? `${ex.signal.metric}: ${ex.signal.value}` : "flagged");
    if (flagIsMutable(ex)) {
      box.textContent = `Already flagged by you (${what}${ex.comment ? " — " + ex.comment : ""}). Pick a reason and save to change it, or Unflag.`;
    } else {
      const dec = ex.owner_decision ? `, owner decided: ${ex.owner_decision}` : "";
      box.textContent = `Already flagged by ${ex.source}${dec} (${what}). It can no longer be changed here.`;
    }
    box.style.display = "";
  } else {
    box.style.display = "none";
  }
  const editable = flagIsMutable(ex);
  $("flagSaveBtn").disabled = !editable;
  $("flagRemoveBtn").style.display = (ex && editable) ? "" : "none";
  for (const b of $("flagCats").children) {
    b.classList.toggle("selected", b.dataset.cat === state.flagCategory);
    b.disabled = !editable;
  }
}

async function loadFlagExisting() {
  const t = state.flagTarget;
  state.flagExisting = null;
  showFlagError("");
  try {
    if (t.kind === "frame") {
      const res = await api.frameFlagGet(t.pool, t.frameIdx);
      state.flagExisting = res.entry || null;
    } else {
      state.flagExisting = state.rawFlag || null;
    }
  } catch (e) {
    showFlagError(e.message || String(e));
  }
  if (state.flagExisting && state.flagExisting.category) {
    state.flagCategory = state.flagExisting.category;
    $("flagComment").value = state.flagExisting.comment || "";
  }
  renderFlagPanel();
}

async function openFlagPanel() {
  if (state.flagPanelOpen || state.modalOpen) return;
  if (!state.config || !state.config.flagging_available) {
    setStatus("Flagging is not available on this machine (it needs the server's raw_review folder).");
    return;
  }
  let target = null;
  if (state.mode === "dataset" && state.rawLoaded) {
    if (!state.rawImageKey) { setStatus("Nothing to flag: no RAW image is loaded."); return; }
    target = {kind: "raw", pool: String(state.datasetModel || "").toLowerCase(),
              imageKey: state.rawImageKey, label: state.rawImageKey};
  } else if (state.mode === "video" && state.videoLoaded) {
    stopPlayback();
    const fa = state.frameAnnotations || [];
    const fromAnn = fa.length ? String(fa[fa.length - 1].model || "").toLowerCase() : "";
    const fromSel = String($("modelSelect").value || "").toLowerCase();
    const pool = [fromAnn, state.lastFlagPool, fromSel].find(p => FLAG_POOLS.includes(p)) || FLAG_POOLS[0];
    target = {kind: "frame", pool, frameIdx: state.frameIdx,
              label: `${state.videoName} · frame ${state.frameIdx + 1}`};
  } else {
    setStatus("Flagging works on RAW images (Dataset Fixer → Load raw) and on a loaded video frame.");
    return;
  }
  state.flagTarget = target;
  state.flagCategory = null;
  state.flagExisting = null;
  $("flagComment").value = "";
  $("flagPoolSelect").value = target.pool;
  showFlagError("");
  state.flagPanelOpen = true;
  $("flagPanel").classList.remove("hidden");
  renderFlagPanel();
  await loadFlagExisting();
}

function closeFlagPanel() {
  state.flagPanelOpen = false;
  state.flagTarget = null;
  state.flagExisting = null;
  state.flagCategory = null;
  const panel = document.getElementById("flagPanel");
  if (panel) panel.classList.add("hidden");
}

function selectFlagCategory(key) {
  if (!flagIsMutable(state.flagExisting)) return;
  state.flagCategory = key;
  renderFlagPanel();
}

async function saveFlag() {
  const t = state.flagTarget;
  if (!t || !flagIsMutable(state.flagExisting)) return;
  if (!state.flagCategory) { showFlagError("Pick a reason first (keys 1–4)."); return; }
  const comment = $("flagComment").value;
  try {
    let res;
    if (t.kind === "raw") {
      res = await api.rawFlagSet(t.imageKey, t.pool, state.flagCategory, comment);
      state.rawFlag = res.entry;
      updateFlagBadge();
    } else {
      res = await api.frameFlagSet(t.pool, t.frameIdx, state.flagCategory, comment);
    }
    state.lastFlagPool = t.pool;
    closeFlagPanel();
    setStatus(`Flagged (${res.entry.category}): ${t.label}`);
  } catch (e) {
    showFlagError(e.message || String(e));
  }
}

async function removeFlag() {
  const t = state.flagTarget;
  if (!t || !state.flagExisting || !flagIsMutable(state.flagExisting)) return;
  try {
    if (t.kind === "raw") {
      await api.rawFlagRemove(t.imageKey, t.pool);
      state.rawFlag = null;
      updateFlagBadge();
    } else {
      await api.frameFlagRemove(t.pool, t.frameIdx);
    }
    closeFlagPanel();
    setStatus(`Unflagged: ${t.label}`);
  } catch (e) {
    showFlagError(e.message || String(e));
  }
}

function installFlagPanel() {
  const cats = $("flagCats");
  cats.innerHTML = "";
  FLAG_CATEGORIES.forEach((c, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "flagCat";
    b.dataset.cat = c.key;
    const name = document.createElement("div");
    name.className = "flagCatName";
    name.textContent = `${i + 1} · ${c.label}`;
    const hint = document.createElement("div");
    hint.className = "flagCatHint";
    hint.textContent = c.hint;
    b.appendChild(name);
    b.appendChild(hint);
    b.onclick = () => selectFlagCategory(c.key);
    cats.appendChild(b);
  });
  const ps = $("flagPoolSelect");
  ps.innerHTML = "";
  for (const p of FLAG_POOLS) {
    const o = document.createElement("option");
    o.value = p;
    o.textContent = p;
    ps.appendChild(o);
  }
  ps.onchange = async () => {
    if (!state.flagTarget || state.flagTarget.kind !== "frame") return;
    state.flagTarget.pool = ps.value;
    state.flagCategory = null;
    $("flagComment").value = "";
    await loadFlagExisting();
  };
  $("flagSaveBtn").onclick = saveFlag;
  $("flagRemoveBtn").onclick = removeFlag;
  $("flagCancelBtn").onclick = closeFlagPanel;
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

    if (state.flagPanelOpen) {
      const typing = isTypingTarget(e.target);
      if (e.key === "Escape") { e.preventDefault(); closeFlagPanel(); }
      else if (e.key === "Enter" && (e.ctrlKey || e.metaKey || !typing)) { e.preventDefault(); await saveFlag(); }
      else if (!typing && FLAG_CATEGORIES.some((_c, i) => e.key === String(i + 1))) {
        e.preventDefault();
        selectFlagCategory(FLAG_CATEGORIES[parseInt(e.key, 10) - 1].key);
      }
      return;
    }

    // X = flag the current RAW image / video frame for review (Dataset Fixer raw-browse + Video Labeler).
    if ((e.key === "x" || e.key === "X") && !e.ctrlKey && !e.metaKey && !e.altKey && !e.repeat
        && !isTypingTarget(e.target) && (state.mode === "video" || state.mode === "dataset")) {
      e.preventDefault();
      await openFlagPanel();
      return;
    }

    if (state.mode === "dataset") {
      // RAW browsing has its own renderer (renderDatasetImage is a no-op for a RAW session) — same routing as Prev/Next.
      const render = state.rawLoaded ? renderRawImage : renderDatasetImage;
      if (e.key === "ArrowLeft") { e.preventDefault(); await render(state.datasetImageIdx - 1); }
      if (e.key === "ArrowRight") { e.preventDefault(); await render(state.datasetImageIdx + 1); }
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
    if (e.shiftKey && e.key === "ArrowLeft") { e.preventDefault(); await gotoFrame(state.frameIdx - Math.round((state.fps || 25) * 10)); }
    if (e.shiftKey && e.key === "ArrowRight") { e.preventDefault(); await gotoFrame(state.frameIdx + Math.round((state.fps || 25) * 10)); }
  });
}

async function init() {
  state.canvas = $("canvas");
  state.ctx = state.canvas.getContext("2d");

  setStatus("Loading config…");
  const [cfg, classVars, analyzerCfg] = await Promise.all([
    api.getConfig(),
    api.getClassVariations(),
    fetch("/api/analyzer/config").then(r => r.json()).catch(() => ({available: false, models: []})),
  ]);
  state.config = cfg;

  // Disable Analyzer tab on machines without GPU models (labelers' Windows PCs)
  if (!analyzerCfg.available) {
    const tabBtn = $("tabAnalyzer");
    tabBtn.disabled = true;
    tabBtn.title = "Analyzer requires GPU server — not available on this machine";
    tabBtn.style.opacity = "0.35";
    tabBtn.style.cursor = "not-allowed";
    tabBtn.onclick = (e) => e.preventDefault();
  }
  // Normalize model keys to lowercase so they match regardless of YAML filename casing
  const _rawVars = classVars || {};
  state.classVariations = Object.fromEntries(
    Object.entries(_rawVars).map(([m, classes]) => [m.toLowerCase(), classes])
  );
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
    // Rebuild speed map for this video's actual FPS, then refresh label.
    speedMap = buildSpeedMap();
    $("speedText").textContent = speedMap[state.speed].label;
    if ((res.loaded_existing || 0) > 0) {
      const lastFrame = (res.last_annotated_frame != null) ? res.last_annotated_frame : 0;
      await gotoFrame(lastFrame);
      setStatus(`Loaded ${res.loaded_existing} existing labels — jumped to last annotated frame (${lastFrame + 1})`);
    } else {
      await renderFrame(0);
    }
    if ((res.models_loaded || 0) === 0) {
      setStatus(`Video loaded. Now add model *.yaml in: ${state.config?.models_dir || ""}`);
      window.alert(`Video loaded.\n\nNo models found yet.\n\nPut YOLO *.yaml into:\n${state.config?.models_dir || ""}\n\n(You can still play/jump frames, but labeling needs models.)`);
    }
    // Load hint/info sidecars (UV ghost bbox + instruction banner)
    const [hintsRes, infoRes] = await Promise.all([
      api.getVideoHints(videoName),
      api.getVideoCaseInfo(videoName),
    ]);
    state.hintData = (hintsRes && hintsRes.detected_class) ? hintsRes : null;
    state.caseInfo = (infoRes && infoRes.instruction) ? infoRes : null;
    renderCaseInfoBanner();
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
  function disableTestMode() {
    if (!state.testMode) return;
    state.testMode = false;
    state.videoFrameTags = {};
    api.setTestMode(false).catch(() => {});
    const label = "🧪 Test Mode: OFF";
    const styleOff = (btn) => { btn.textContent = label; btn.style.backgroundColor = ""; btn.style.color = ""; };
    if ($("videoTestModeBtn")) styleOff($("videoTestModeBtn"));
    if ($("testModeBtn"))      styleOff($("testModeBtn"));
    if ($("frameTagsPanel"))   $("frameTagsPanel").classList.add("hidden");
  }

  setMode("video");
  $("tabVideo").onclick = () => {
    disableTestMode();
    if (state.mode !== "video") resetWorkspaceUI(`Video Labeler. Videos: ${state.config?.videos_dir || ""}`);
    setMode("video");
    setStatus(`Video Labeler. Videos: ${state.config?.videos_dir || ""}`);
  };
  $("tabDataset").onclick = async () => {
    disableTestMode();
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
    // Populate raw controls — disable entirely if raw_base_path not configured
    const rawConfigured = state.config?.raw_configured === true;
    const rawSection = [$("rawModelSelect"), $("rawClassSelect"), $("rawCameraSelect"), $("loadRawBtn")];
    rawSection.forEach(el => { if (el) el.disabled = !rawConfigured; });
    if (rawConfigured) {
      try {
        // Camera options from config
        const cameraSel = $("rawCameraSelect");
        cameraSel.innerHTML = "";
        const allOpt = document.createElement("option");
        allOpt.value = "ALL"; allOpt.textContent = "All cameras";
        cameraSel.appendChild(allOpt);
        for (const cam of (state.config.bar_counter_options || [])) {
          const opt = document.createElement("option");
          opt.value = cam; opt.textContent = cam;
          cameraSel.appendChild(opt);
        }
        // Model options
        const raw = await api.rawListModels();
        const modelSel = $("rawModelSelect");
        modelSel.innerHTML = "";
        for (const m of (raw.models || [])) {
          const opt = document.createElement("option");
          opt.value = m; opt.textContent = m;
          modelSel.appendChild(opt);
        }
        if ((raw.models || []).length > 0) {
          await _rawPopulateClasses(modelSel.value);
        }
      } catch (e) {
        rawSection.forEach(el => { if (el) el.disabled = true; });
      }
    }
  };

  async function _rawPopulateClasses(model) {
    const classSel = $("rawClassSelect");
    classSel.innerHTML = "";
    try {
      const res = await api.rawListClasses(model);
      for (const c of (res.classes || [])) {
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        classSel.appendChild(opt);
      }
    } catch (e) {
      setStatus(`Raw classes failed: ${e.message || e}`);
    }
  }

  $("rawModelSelect").onchange = () => _rawPopulateClasses($("rawModelSelect").value);

  $("loadRawBtn").onclick = async () => {
    const model = $("rawModelSelect").value;
    const className = $("rawClassSelect").value;
    const camera = $("rawCameraSelect").value || "ALL";
    if (!model || !className) {
      window.alert("Select a model and class first.");
      return;
    }
    try {
      resetWorkspaceUI(`Loading raw: ${model} / ${className}…`);
      setMode("dataset");
      const res = await api.rawLoad(model, className, camera);
      state.rawLoaded = true;
      state.datasetLoaded = false;
      state.datasetName = `${model}/${className}`;
      state.datasetModel = model;
      state.datasetImageCount = res.image_count || 0;
      state.datasetImageIdx = 0;
      const camLabel = (camera && camera !== "ALL") ? ` [${camera}]` : "";
      setStatus(`Raw loaded: ${model} / ${className}${camLabel} — ${state.datasetImageCount} images`);
      await renderRawImage(0);
    } catch (e) {
      setStatus(`Load raw failed: ${e.message || e}`);
      window.alert(`Load raw failed: ${e.message || e}`);
    }
  };

  // ---- Background Labeler ----
  $("tabBg").onclick = async () => {
    disableTestMode();
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
      existingLabel.textContent = cfgBg.existing_path || `${cfgBg.datasets_dir}/existing`;
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
      // Backend uses configured existing_datasets_dir from config, no need to send it
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
      if (res.zip_path) window.alert(`Background session finished.\n\nZip ready to send:\n${res.zip_path}`);
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
    if (state.rawLoaded) { await renderRawImage(state.datasetImageIdx - 1); return; }
    if (!state.datasetLoaded) return;
    await renderDatasetImage(state.datasetImageIdx - 1);
  };
  $("nextImgBtn").onclick = async () => {
    if (state.rawLoaded) { await renderRawImage(state.datasetImageIdx + 1); return; }
    if (!state.datasetLoaded) return;
    await renderDatasetImage(state.datasetImageIdx + 1);
  };
  $("testModeBtn").addEventListener("click", async () => {
    const newMode = !state.testMode;
    await api.setTestMode(newMode);
    state.testMode = newMode;
    const label = `🧪 Test Mode: ${newMode ? "ON" : "OFF"}`;
    $("testModeBtn").textContent = label;
    $("testModeBtn").style.backgroundColor = newMode ? "#7c3aed" : "";
    $("testModeBtn").style.color = newMode ? "#fff" : "";
    if ($("videoTestModeBtn")) {
      $("videoTestModeBtn").textContent = label;
      $("videoTestModeBtn").style.backgroundColor = newMode ? "#7c3aed" : "";
      $("videoTestModeBtn").style.color = newMode ? "#fff" : "";
    }
    await refreshLists();
  });

  $("videoTestModeBtn").addEventListener("click", async () => {
    const newMode = !state.testMode;
    await api.setTestMode(newMode);
    state.testMode = newMode;
    const label = `🧪 Test Mode: ${newMode ? "ON" : "OFF"}`;
    $("videoTestModeBtn").textContent = label;
    $("videoTestModeBtn").style.backgroundColor = newMode ? "#7c3aed" : "";
    $("videoTestModeBtn").style.color = newMode ? "#fff" : "";
    if ($("testModeBtn")) {
      $("testModeBtn").textContent = label;
      $("testModeBtn").style.backgroundColor = newMode ? "#7c3aed" : "";
      $("testModeBtn").style.color = newMode ? "#fff" : "";
    }
    await refreshLists();
  });
  $("saveOverwriteBtn").onclick = async () => {
    if (!state.datasetLoaded) return;
    try {
      // Check for deletions
      const status = await api.getDatasetStatus();
      if (status.deleted_count > 0) {
        const confirmMsg = `WARNING: ${status.deleted_count} image(s) marked for deletion.\n\nThese will be permanently removed from the dataset.\n\nContinue?`;
        if (!window.confirm(confirmMsg)) return;
      }
      const res = await api.saveDataset("overwrite");
      let msg = `Saved (overwrite).\n\nZip ready to send:\n${res.zip_path}`;
      if (res.deleted_count > 0) {
        msg += `\n\n${res.deleted_count} image(s) deleted.`;
      }
      window.alert(msg);
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
      let msg = `Saved as _fixed.\n\nZip ready to send:\n${res.zip_path}`;
      if (res.deleted_count > 0) {
        msg += `\n\nNote: ${res.deleted_count} deleted image(s) were excluded from _fixed dataset.`;
      }
      window.alert(msg);
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
            `This video already has exports on disk:\n\n${info.output_root}\n\nImages: ${info.image_files}   Labels: ${info.label_files}\n\nLoad existing annotations and continue from where you left off?`
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
    if ($("exportBtn").disabled) return; // prevent double-click
    $("exportBtn").disabled = true;
    $("exportBtn").textContent = "Exporting…";
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
      const zipList = (res.zip_paths || []).join("\n");
      setStatus(`Export complete: ${(res.zip_paths || []).length} zip(s) ready`);
      window.alert(`Export complete.\n\nZip(s) ready to send:\n${zipList}\n\nFrames: ${res.frames_labeled}\nLabel files: ${res.written_label_files}`);

      // After export, backend clears workspace; mirror that in the UI.
      if (res.cleared) {
        resetWorkspaceUI("Export complete. Video closed. Load a new video to continue.");
      }
    } catch (e) {
      setStatus(`Export error: ${e.message || e}`);
      window.alert(`Export failed: ${e.message || e}`);
    } finally {
      $("exportBtn").disabled = false;
      $("exportBtn").textContent = "Export ALL";
    }
  };

  $("playBtn").onclick = () => togglePlay();
  $("startBtn").onclick = async () => { stopPlayback(); await gotoFrame(0); };
  $("endBtn").onclick = async () => { stopPlayback(); await gotoFrame(state.totalFrames - 1); };
  function skip10s(dir) {
    const frames = Math.round((state.fps || 25) * 10);
    stopPlayback();
    return gotoFrame(state.frameIdx + dir * frames);
  }
  $("backBtn").onclick = () => skip10s(-1);
  $("fwdBtn").onclick = () => skip10s(+1);

  $("speedSlider").oninput = (e) => {
    state.speed = parseInt(e.target.value, 10);
    $("speedText").textContent = speedMap[state.speed].label;
    if (state.playing) {
      stopPlayback();
      startPlayback();
    }
  };
  $("speedText").textContent = speedMap[state.speed].label;

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
  installFlagPanel();

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

  // ── Analyzer Tab ──────────────────────────────────────────────────────────

  window.analyzerState = {};
  const analyzerState = window.analyzerState;
  Object.assign(analyzerState, {
    subMode: "inspector",    // "inspector" | "overlap"
    imageLoaded: false,
    imageW: 0,
    imageH: 0,
    bbox: null,              // {x1, y1, x2, y2} in image coords
    drawing: false,
    drawStart: null,
    overlapJobId: null,
    overlapPollTimer: null,
    analyzerModels: [],
  });

  function setAnalyzerSubMode(mode) {
    window.__setAnalyzerSubMode = setAnalyzerSubMode;
    analyzerState.subMode = mode;
    const isInspector = mode === "inspector";
    $("analyzerModeInspector").classList.toggle("primary", isInspector);
    $("analyzerModeOverlap").classList.toggle("primary", !isInspector);
    $("analyzerInspectorBar").style.display = isInspector ? "flex" : "none";
    $("analyzerOverlapBar").style.display = isInspector ? "none" : "flex";
    $("analyzerInspectorPanel").style.display = isInspector ? "flex" : "none";
    $("analyzerOverlapPanel").classList.toggle("hidden", isInspector);
  }

  $("analyzerModeInspector").onclick = () => setAnalyzerSubMode("inspector");
  $("analyzerModeOverlap").onclick = () => setAnalyzerSubMode("overlap");

  $("tabAnalyzer").onclick = async () => {
    disableTestMode();
    if (state.dirty) {
      const ok = window.confirm("You have un-exported video annotations. Switch and lose them?");
      if (!ok) return;
    }
    resetWorkspaceUI("Analyzer");
    setMode("analyzer");
    setStatus("Analyzer — Inspector: upload an image and draw a bbox to identify objects");
    try {
      const cfg = await fetch("/api/analyzer/config").then(r => r.json());
      analyzerState.analyzerModels = cfg.models || [];
      const sel = $("analyzerModelSelect");
      sel.innerHTML = "";
      for (const m of cfg.models) {
        const opt = document.createElement("option");
        opt.value = m; opt.textContent = m;
        sel.appendChild(opt);
      }
    } catch (e) {
      setStatus(`Analyzer config failed: ${e.message || e}`);
    }
  };

  // ── Inspector canvas ───────────────────────────────────────────────────────

  const aCanvas = $("analyzerCanvas");
  const aCtx = aCanvas.getContext("2d");
  let _analyzerImg = null;

  function analyzerCanvasToImg(cx, cy) {
    const rect = aCanvas.getBoundingClientRect();
    const scaleX = analyzerState.imageW / aCanvas.width;
    const scaleY = analyzerState.imageH / aCanvas.height;
    return {
      x: Math.round((cx - rect.left) * (aCanvas.width / rect.width) * scaleX),
      y: Math.round((cy - rect.top)  * (aCanvas.height / rect.height) * scaleY),
    };
  }

  function analyzerRedraw() {
    if (!_analyzerImg) return;
    aCanvas.width = _analyzerImg.naturalWidth;
    aCanvas.height = _analyzerImg.naturalHeight;
    aCtx.drawImage(_analyzerImg, 0, 0);
    if (analyzerState.bbox) {
      const {x1, y1, x2, y2} = analyzerState.bbox;
      aCtx.strokeStyle = "#2dd4bf";
      aCtx.lineWidth = 3;
      aCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      aCtx.fillStyle = "rgba(45,212,191,0.12)";
      aCtx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }
  }

  aCanvas.addEventListener("mousedown", (e) => {
    if (!analyzerState.imageLoaded) return;
    const p = analyzerCanvasToImg(e.clientX, e.clientY);
    analyzerState.drawing = true;
    analyzerState.drawStart = p;
    analyzerState.bbox = null;
    analyzerRedraw();
  });

  aCanvas.addEventListener("mousemove", (e) => {
    if (!analyzerState.drawing) return;
    const p = analyzerCanvasToImg(e.clientX, e.clientY);
    analyzerState.bbox = {
      x1: Math.min(analyzerState.drawStart.x, p.x),
      y1: Math.min(analyzerState.drawStart.y, p.y),
      x2: Math.max(analyzerState.drawStart.x, p.x),
      y2: Math.max(analyzerState.drawStart.y, p.y),
    };
    analyzerRedraw();
  });

  aCanvas.addEventListener("mouseup", () => {
    analyzerState.drawing = false;
    if (analyzerState.bbox) {
      const b = analyzerState.bbox;
      if ((b.x2 - b.x1) < 10 || (b.y2 - b.y1) < 10) {
        analyzerState.bbox = null;
      }
    }
    $("analyzerAnalyzeBtn").disabled = !analyzerState.bbox;
    analyzerRedraw();
  });

  $("analyzerUploadBtn").onclick = () => $("analyzerFileInput").click();

  $("analyzerFileInput").onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      _analyzerImg = img;
      analyzerState.imageLoaded = true;
      analyzerState.imageW = img.naturalWidth;
      analyzerState.imageH = img.naturalHeight;
      analyzerState.bbox = null;
      $("analyzerCanvas").style.display = "block";
      $("analyzerCanvasPlaceholder").style.display = "none";
      $("analyzerClearBtn").style.display = "";
      $("analyzerAnalyzeBtn").disabled = true;
      $("analyzerInspectorHint").textContent = "Draw a bounding box around the object to identify";
      analyzerRedraw();
      $("analyzerResults").classList.add("hidden");
      $("analyzerResultsPlaceholder").style.display = "";
    };
    img.src = url;
    analyzerState._fileRef = file;
    e.target.value = "";
  };

  $("analyzerClearBtn").onclick = () => {
    _analyzerImg = null;
    analyzerState.imageLoaded = false;
    analyzerState.bbox = null;
    aCanvas.style.display = "none";
    $("analyzerCanvasPlaceholder").style.display = "";
    $("analyzerClearBtn").style.display = "none";
    $("analyzerAnalyzeBtn").disabled = true;
    $("analyzerInspectorHint").textContent = "Upload an image, then draw a bounding box around the object";
    $("analyzerResults").classList.add("hidden");
    $("analyzerResultsPlaceholder").style.display = "";
  };

  $("analyzerAnalyzeBtn").onclick = async () => {
    if (!analyzerState.imageLoaded || !analyzerState.bbox || !analyzerState._fileRef) return;
    const model = $("analyzerModelSelect").value;
    if (!model) { window.alert("Select a model first."); return; }

    $("analyzerAnalyzeBtn").disabled = true;
    $("analyzerAnalyzeBtn").textContent = "Analyzing…";
    setStatus("Running inference…");

    try {
      // Convert file to base64
      const b64 = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(",")[1]);
        reader.onerror = reject;
        reader.readAsDataURL(analyzerState._fileRef);
      });

      const res = await fetch("/api/analyzer/classify", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          model,
          image_b64: b64,
          bbox: analyzerState.bbox,
        }),
      }).then(async r => {
        if (!r.ok) throw new Error((await r.json()).detail || "Classify failed");
        return r.json();
      });

      renderAnalyzerScores(res.scores || [], model);
      setStatus("Analysis complete");
    } catch (e) {
      setStatus(`Analysis failed: ${e.message || e}`);
      window.alert(`Analysis failed: ${e.message || e}`);
    } finally {
      $("analyzerAnalyzeBtn").disabled = false;
      $("analyzerAnalyzeBtn").textContent = "Analyze";
    }
  };

  function renderAnalyzerScores(scores, model) {
    $("analyzerResultsPlaceholder").style.display = "none";
    $("analyzerResults").classList.remove("hidden");

    const list = $("analyzerScoreList");
    list.innerHTML = "";

    if (!scores.length) {
      list.innerHTML = "<div style='color:var(--muted);font-size:12px;'>No detections in the selected region.</div>";
      $("analyzerExamplesSection").style.display = "none";
      return;
    }

    const maxConf = scores[0].confidence;
    scores.slice(0, 10).forEach((s, i) => {
      const row = document.createElement("div");
      row.className = "analyzerScoreRow" + (i === 0 ? " top" : "");
      const pct = maxConf > 0 ? Math.round((s.confidence / maxConf) * 100) : 0;
      const color = i === 0 ? "var(--accent)" : i < 3 ? "var(--accent2)" : "var(--muted)";
      row.innerHTML = `
        <div class="analyzerScoreLabel" title="${s.class}">${s.class}</div>
        <div class="analyzerScoreBar"><div class="analyzerScoreBarFill" style="width:${pct}%;background:${color}"></div></div>
        <div class="analyzerScoreVal">${(s.confidence * 100).toFixed(0)}%</div>
      `;
      row.onclick = () => loadAnalyzerExamples(model, s.class);
      list.appendChild(row);
    });

    // Auto-load examples for top match
    if (scores.length > 0) {
      loadAnalyzerExamples(model, scores[0].class);
    }
  }

  async function loadAnalyzerExamples(model, className) {
    $("analyzerTopClass").textContent = className;
    $("analyzerExamplesSection").style.display = "";
    const grid = $("analyzerExampleGrid");
    grid.innerHTML = "<span style='color:var(--muted);font-size:11px;'>Loading…</span>";
    try {
      const res = await fetch(`/api/analyzer/examples?model=${encodeURIComponent(model)}&class_name=${encodeURIComponent(className)}&n=4`).then(r => r.json());
      grid.innerHTML = "";
      if (!res.examples || !res.examples.length) {
        grid.innerHTML = "<span style='color:var(--muted);font-size:11px;'>No examples found</span>";
        return;
      }
      for (const src of res.examples) {
        const img = document.createElement("img");
        img.className = "analyzerExampleImg";
        img.src = src;
        img.title = className;
        grid.appendChild(img);
      }
    } catch (e) {
      grid.innerHTML = `<span style='color:var(--danger);font-size:11px;'>${e.message || e}</span>`;
    }
  }

  // ── Overlap Report ─────────────────────────────────────────────────────────

  $("analyzerRunReportBtn").onclick = async () => {
    const model = $("analyzerModelSelect").value;
    if (!model) { window.alert("Select a model first."); return; }
    const samples = parseInt($("analyzerSamplesInput").value) || 20;

    $("analyzerRunReportBtn").disabled = true;
    $("analyzerOverlapProgress").classList.remove("hidden");
    $("analyzerProgressFill").style.width = "0%";
    $("analyzerProgressText").textContent = "Starting…";
    $("analyzerOverlapResults").classList.add("hidden");
    $("analyzerOverlapPlaceholder").style.display = "none";
    setStatus(`Running overlap report for ${model}…`);

    try {
      const res = await fetch("/api/analyzer/overlap_report", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({model, samples}),
      }).then(async r => {
        if (!r.ok) throw new Error((await r.json()).detail || "Failed to start report");
        return r.json();
      });
      analyzerState.overlapJobId = res.job_id;
      if (analyzerState.overlapPollTimer) clearInterval(analyzerState.overlapPollTimer);
      analyzerState.overlapPollTimer = setInterval(pollOverlapJob, 1500);
    } catch (e) {
      setStatus(`Overlap report failed: ${e.message || e}`);
      $("analyzerRunReportBtn").disabled = false;
      $("analyzerOverlapProgress").classList.add("hidden");
    }
  };

  async function pollOverlapJob() {
    if (!analyzerState.overlapJobId) return;
    try {
      const status = await fetch(`/api/analyzer/overlap_status?job_id=${analyzerState.overlapJobId}`).then(r => r.json());
      const prog = status.progress || {};
      const done = prog.done || 0;
      const total = prog.total || 0;
      const pct = total > 0 ? Math.round((done / total) * 100) : 0;
      $("analyzerProgressFill").style.width = pct + "%";
      $("analyzerProgressText").textContent = total > 0
        ? `${done}/${total} images${prog.current_class ? " — " + prog.current_class : ""}`
        : "Loading model…";

      if (status.status === "done") {
        clearInterval(analyzerState.overlapPollTimer);
        analyzerState.overlapPollTimer = null;
        $("analyzerProgressFill").style.width = "100%";
        $("analyzerProgressText").textContent = "Done";
        $("analyzerRunReportBtn").disabled = false;
        renderOverlapResults(status.result);
        setStatus(`Overlap report complete — ${(status.result?.flagged_pairs || []).length} flagged pairs`);
      } else if (status.status === "error") {
        clearInterval(analyzerState.overlapPollTimer);
        analyzerState.overlapPollTimer = null;
        $("analyzerRunReportBtn").disabled = false;
        $("analyzerProgressText").textContent = "Error";
        setStatus(`Overlap report error: ${status.error || "unknown"}`);
        window.alert(`Overlap report failed:\n${status.error || "unknown error"}`);
      }
    } catch (e) {
      // transient network error — keep polling
    }
  }

  function renderOverlapResults(result) {
    if (!result) return;
    $("analyzerOverlapResults").classList.remove("hidden");

    // Flagged pairs
    const flaggedList = $("analyzerFlaggedList");
    flaggedList.innerHTML = "";
    const flagged = result.flagged_pairs || [];
    if (!flagged.length) {
      flaggedList.innerHTML = "<div style='color:var(--muted);font-size:12px;'>No flagged pairs above 15% confusion rate. Dataset looks clean.</div>";
    } else {
      for (const fp of flagged) {
        const row = document.createElement("div");
        row.className = "analyzerFlaggedRow";
        row.innerHTML = `
          <span style="font-weight:700;color:var(--text)">${fp.class_a}</span>
          <span style="color:var(--muted)">confused as</span>
          <span style="font-weight:700;color:var(--danger)">${fp.class_b}</span>
          <span style="color:var(--muted);font-size:11px;">(${fp.count} images)</span>
          <div class="analyzerFlaggedRate">${Math.round(fp.rate * 100)}%</div>
        `;
        flaggedList.appendChild(row);
      }
    }

    // Per-class accuracy
    const accList = $("analyzerClassAccList");
    accList.innerHTML = "";
    const perClass = result.per_class || {};
    const sorted = Object.entries(perClass).sort((a, b) => a[1].accuracy - b[1].accuracy);
    for (const [cls, info] of sorted) {
      const pct = Math.round(info.accuracy * 100);
      const color = pct >= 85 ? "var(--accent)" : pct >= 60 ? "#f59e0b" : "var(--danger)";
      const row = document.createElement("div");
      row.className = "analyzerAccRow";
      const confused = Object.entries(info.confused_as || {})
        .filter(([k]) => !k.startsWith("__"))
        .slice(0, 2)
        .map(([k, v]) => `${k}(${v})`)
        .join(", ");
      row.innerHTML = `
        <div style="min-width:160px;font-size:11px;font-weight:600;">${cls}</div>
        <div class="analyzerAccBar"><div class="analyzerAccFill" style="width:${pct}%;background:${color}"></div></div>
        <div style="width:36px;text-align:right;font-size:11px;color:${color}">${pct}%</div>
        ${confused ? `<div style="font-size:10px;color:var(--muted);margin-left:6px;">→ ${confused}</div>` : ""}
      `;
      accList.appendChild(row);
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  init().catch((e) => setStatus(`Init error: ${e.message || e}`));
});


