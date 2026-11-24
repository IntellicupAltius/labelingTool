"""Model cards: a read-only summary of every trained model, filled from files that already exist.

Sources, each value keeps the name of the one it came from:
  - the model file (.pt): only its description is read (date, training settings, classes, final
    results). The checkpoint is a zip; its data.pkl is read with an unpickler that builds empty
    stand-ins for every class and never loads the weights, so neither torch nor YOLO is imported;
  - the training run folder (args.yaml, results.csv) when the file lacks something;
  - <model>_models_versions.json (version number, sample counts) and the promotion log;
  - the training configuration (dataset balancing), which training does not copy per run, so it is
    today's files with a note when one changed after the model was trained;
  - the archived training set (_metadata.json), which has the real per-class balancing plan.

Building the cards runs outside the web process (`python -m web_labeler.catalog model-cards`); the
tool only reads the finished snapshot. Nothing here writes to the models folder.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import pickle
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

SNAPSHOT_NAME = "model_cards.json"
SNAPSHOT_VERSION = 1

SRC_FILE = "fajl modela"
SRC_RUN = "zapis treninga"
SRC_VERSIONS = "evidencija verzija"
SRC_PROMOTIONS = "evidencija promocija"
SRC_CONFIG = "trening konfiguracija (današnji fajlovi, trening ne čuva svoju kopiju)"
SRC_ARCHIVE = "arhiva trening seta"

TRAINING_KEYS = ("epochs", "patience", "imgsz", "batch", "optimizer", "lr0", "lrf", "cos_lr", "seed", "close_mosaic")
AUGMENT_KEYS = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud",
                "fliplr", "mosaic", "mixup", "cutmix", "copy_paste", "auto_augment", "erasing")
METRIC_KEYS = (("precision", "metrics/precision(B)"), ("recall", "metrics/recall(B)"),
               ("mAP50", "metrics/mAP50(B)"), ("mAP50-95", "metrics/mAP50-95(B)"))
_TS = re.compile(r"ts(\d{8})_(\d{6})")
_ARCH = re.compile(r"yolo(v?\d+)([nslmx])", re.IGNORECASE)


@dataclass(frozen=True)
class ModelCardSources:
    models_dir: Path
    promotions_path: Optional[Path] = None
    training_root: Optional[Path] = None      # training_pipeline/config
    archives_dir: Optional[Path] = None       # datasets/temp_archives
    tenant: Optional[str] = None


# ---------------- Reading a checkpoint without torch ----------------

class _Stub:
    """Stand-in for any class in the pickle. Keeps plain state so attributes like `names` stay readable."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            state = {**(state[0] or {}), **state[1]}
        if isinstance(state, dict):
            self.__dict__.update({k: v for k, v in state.items() if isinstance(k, str)})

    def __call__(self, *args: Any, **kwargs: Any) -> "_Stub":
        return _Stub()

    def __setitem__(self, key: Any, value: Any) -> None:
        pass

    def append(self, value: Any) -> None:
        pass

    def extend(self, values: Any) -> None:
        pass


def _reconstruct(*args: Any, **kwargs: Any) -> _Stub:
    return _Stub()


_SAFE = {("collections", "OrderedDict"), ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "slice"),
         ("builtins", "complex"), ("datetime", "datetime"), ("datetime", "date"), ("datetime", "timedelta")}


class _DescriptionUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _SAFE:
            return super().find_class(module, name)
        if (module, name) in {("copyreg", "_reconstructor"), ("copyreg", "__newobj__")}:
            return _reconstruct
        return type(name, (_Stub,), {"__module__": "stub." + module})

    def persistent_load(self, pid: Any) -> None:  # tensor storages: never read
        return None


def read_checkpoint(path: Path) -> Dict[str, Any]:
    """The description stored in a YOLO checkpoint: date, version, train_args, train_metrics, epochs run, classes."""
    with zipfile.ZipFile(path) as z:
        member = next((n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl"), None)
        if member is None:
            raise ValueError("no data.pkl")
        data = _DescriptionUnpickler(io.BytesIO(z.read(member))).load()
    if not isinstance(data, dict):
        raise ValueError("unexpected checkpoint layout")
    out: Dict[str, Any] = {}
    for key in ("date", "version"):
        if isinstance(data.get(key), str):
            out[key] = data[key]
    if isinstance(data.get("train_args"), dict):
        out["train_args"] = _plain(data["train_args"])
    if isinstance(data.get("train_metrics"), dict):
        out["train_metrics"] = _plain(data["train_metrics"])
    results = data.get("train_results")
    if isinstance(results, dict) and isinstance(results.get("epoch"), list):
        out["train_results"] = _plain(results)
    names = getattr(data.get("model"), "names", None)
    if isinstance(names, dict):
        out["names"] = [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
    elif isinstance(names, list):
        out["names"] = [str(n) for n in names]
    return out


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# ---------------- Small readers ----------------

def _yaml(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _json(path: Optional[Path]) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _results_csv(path: Path) -> Optional[Dict[str, List[float]]]:
    if not path.is_file():
        return None
    columns: Dict[str, List[float]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for key, value in row.items():
                    try:
                        columns.setdefault((key or "").strip(), []).append(float(value))
                    except (TypeError, ValueError):
                        pass
    except OSError:
        return None
    return columns if columns.get("epoch") else None


def _promotions(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("kind") == "PRODUKCIJA":
            rows.append(row)
    return rows


def _deep_merge(base: Dict[str, Any], over: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        out[key] = _deep_merge(out[key], value) if isinstance(value, Mapping) and isinstance(out.get(key), dict) else value
    return out


def _run_stamp(file_name: str) -> Optional[str]:
    m = _TS.search(file_name)
    return f"{m.group(1)}_{m.group(2)}" if m else None


def _architecture(*names: Optional[str]) -> Optional[str]:
    for name in names:
        m = _ARCH.search(Path(str(name or "")).name)
        if m:
            gen = m.group(1).lower()
            return f"YOLO{gen}{m.group(2).lower()}"
    return None


def _value(value: Any, source: str) -> Dict[str, Any]:
    return {"value": value, "source": source if value is not None else None}


# ---------------- Balancing (training configuration + archive) ----------------

def _config_files(root: Path, tenant: Optional[str], model: str, stamp: Optional[str]) -> List[Path]:
    files = [root / "defaults.yaml", root / "augmentations.yaml"]
    if tenant:
        files += [root / "tenants" / f"{tenant}.yaml", root / "models" / f"{model}.yaml",
                  root / "tenants" / tenant / f"{model}.yaml"]
    if stamp:
        # Assumption: a run override is named <model>_<anything>_<YYYYMMDD of the run>.yaml.
        files += sorted((root / "overrides").glob(f"{model}_*{stamp[:8]}.yaml"))
    return [p for p in files if p.is_file()]


def _balance_from_config(sources: ModelCardSources, model: str, stamp: Optional[str],
                         trained: Optional[str]) -> Optional[Dict[str, Any]]:
    if sources.training_root is None:
        return None
    files = _config_files(sources.training_root, sources.tenant, model, stamp)
    if not files:
        return None
    cfg: Dict[str, Any] = {}
    for p in files:
        cfg = _deep_merge(cfg, _yaml(p) or {})
    aug = cfg.get("augmentations") if isinstance(cfg.get("augmentations"), dict) else {}
    balance = aug.get("balance") if isinstance(aug.get("balance"), dict) else None
    roi = aug.get("roi") if isinstance(aug.get("roi"), dict) else {}
    glob = aug.get("global") if isinstance(aug.get("global"), dict) else {}
    changed = []
    if trained:
        for p in files:
            when = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")
            if when > trained[:10]:
                changed.append({"file": p.name, "changed": when})
    enabled = lambda d: sorted(k for k, v in d.items() if isinstance(v, dict) and v.get("enable", True))
    return {
        "source": SRC_CONFIG,
        "files": [p.name for p in files],
        "changed_after_training": changed,
        "min_samples_per_class": (balance or {}).get("min_samples_per_class"),
        "auto": (balance or {}).get("auto") if isinstance((balance or {}).get("auto"), dict) else None,
        "adaptive_strategy": aug.get("adaptive_strategy") if isinstance(aug.get("adaptive_strategy"), list) else None,
        "max_aug_per_class": roi.get("max_aug_per_class"),
        "roi_transforms": enabled(roi.get("transforms") or {}) if roi.get("enable", True) else [],
        "global_transforms": enabled({k: v for k, v in glob.items() if k != "enable"}) if glob.get("enable", True) else [],
        "roi_crop": cfg.get("roi_crop"),
    }


def _archive_plan(sources: ModelCardSources, snapshot: Optional[str], stamp: Optional[str]) -> Optional[Dict[str, Any]]:
    if sources.archives_dir is None:
        return None
    names = [n for n in (snapshot, f"tmp_training_{stamp}" if stamp else None) if n]
    for name in dict.fromkeys(names):
        path = sources.archives_dir / f"{name}.zip"
        if not path.is_file():
            continue
        try:
            with zipfile.ZipFile(path) as z:
                meta = json.loads(z.read("_metadata.json"))
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            continue
        plan = meta.get("augmentation_plan") if isinstance(meta.get("augmentation_plan"), dict) else {}
        split = meta.get("per_class_split") if isinstance(meta.get("per_class_split"), dict) else {}
        rows = []
        for cls in sorted(set(plan) | set(split)):
            counts = split.get(cls) if isinstance(split.get(cls), dict) else {}
            rows.append({"class": cls, "added": plan.get(cls), "train": counts.get("train"),
                         "val": counts.get("val"), "eval": counts.get("eval_holdout")})
        return {"source": SRC_ARCHIVE, "archive": path.name, "real_images": meta.get("raw_total_samples"),
                "added_images": meta.get("augmented_total_samples"), "classes": rows}
    return None


# ---------------- Cards ----------------

def _best_epoch(results: Mapping[str, List[float]], metrics: Optional[Mapping[str, Any]]) -> Optional[int]:
    m50, m5095 = results.get("metrics/mAP50(B)"), results.get("metrics/mAP50-95(B)")
    epochs = results.get("epoch")
    if not m50 or not m5095 or not epochs or len(m50) != len(m5095) or len(m50) != len(epochs):
        return None
    # Ultralytics' own fitness: older versions 0.1*mAP50 + 0.9*mAP50-95, newer mAP50-95 alone. The file's
    # final metrics say which one this run used.
    w50 = 0.1
    if metrics and isinstance(metrics.get("fitness"), (int, float)) and isinstance(metrics.get("metrics/mAP50-95(B)"), (int, float)):
        if abs(metrics["fitness"] - metrics["metrics/mAP50-95(B)"]) < 1e-4:
            w50 = 0.0
    scores = [w50 * a + (1 - w50) * b for a, b in zip(m50, m5095)]
    return int(epochs[scores.index(max(scores))])


def build_card(sources: ModelCardSources, model: str, pt: Optional[Path], version_label: Optional[str],
               version_entry: Optional[Mapping[str, Any]], promotions: List[Dict[str, Any]],
               live_file: Optional[str]) -> Dict[str, Any]:
    file_name = pt.name if pt else str((version_entry or {}).get("model_filename") or "")
    stamp = _run_stamp(file_name)
    entry = dict(version_entry or {})
    problems: List[str] = []
    ck: Dict[str, Any] = {}
    if pt is None or not pt.is_file():
        problems.append("Fajl modela nije na disku. Prikazano je samo ono što je ostalo u evidenciji verzija.")
    else:
        try:
            ck = read_checkpoint(pt)
        except Exception:  # a broken or foreign file must not break the other cards
            problems.append("Fajl modela nije mogao da se pročita.")
    run_dir = sources.models_dir / model / f"run_{stamp}" if stamp else None
    run_args = _yaml(run_dir / "args.yaml") if run_dir else None
    run_results = _results_csv(run_dir / "results.csv") if run_dir else None
    args, args_src = (ck["train_args"], SRC_FILE) if ck.get("train_args") else (run_args or {}, SRC_RUN)

    date_value = ck.get("date")
    date = _value(date_value[:10] if date_value else None, SRC_FILE)
    if date["value"] is None and entry.get("date"):
        date = _value(str(entry["date"]), SRC_VERSIONS)
    if date["value"] is None and stamp:
        date = _value(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}", "ime fajla modela")

    training = {k: _value(args.get(k), args_src) for k in TRAINING_KEYS}
    if training["epochs"]["value"] is None and entry.get("epochs") is not None:
        training["epochs"] = _value(entry["epochs"], SRC_VERSIONS)
    results_cols = ck.get("train_results") or run_results
    results_src = SRC_FILE if ck.get("train_results") else SRC_RUN
    epochs_run = len(results_cols["epoch"]) if results_cols else None
    training["epochs_run"] = _value(epochs_run, results_src)
    training["best_epoch"] = _value(_best_epoch(results_cols, ck.get("train_metrics")) if results_cols else None, results_src)

    names = ck.get("names")
    names_src = SRC_FILE
    if not names and isinstance(entry.get("classes"), list):
        names, names_src = [str(n) for n in entry["classes"]], SRC_VERSIONS
    metrics = ck.get("train_metrics") or {}
    result = {key: _value(metrics.get(src), SRC_FILE) for key, src in METRIC_KEYS}
    if result["mAP50"]["value"] is None and entry.get("map50") is not None:
        result["mAP50"] = _value(entry["map50"], SRC_VERSIONS)

    promoted = [{"date": p.get("date"), "before": p.get("before"), "why": p.get("why") or None, "source": SRC_PROMOTIONS}
                for p in promotions if p.get("after") == file_name]
    trained_on = date["value"]
    return {
        "file": file_name,
        "live": bool(live_file) and file_name == live_file,
        "version": _value(version_label, SRC_VERSIONS),
        "date": date,
        "ultralytics": _value(ck.get("version"), SRC_FILE),
        "architecture": _value(_architecture(args.get("model"), entry.get("base_model"), file_name),
                               args_src if _architecture(args.get("model")) else (SRC_VERSIONS if _architecture(entry.get("base_model")) else "ime fajla modela")),
        "base_model": _value(Path(str(args.get("model") or entry.get("base_model") or "")).name or None,
                             args_src if args.get("model") else SRC_VERSIONS),
        "training": training,
        "augmentations": {k: _value(args.get(k), args_src) for k in AUGMENT_KEYS},
        "classes": {"count": len(names) if names else None, "names": names or [], "source": names_src if names else None},
        "samples": {"train": _value(entry.get("train_samples"), SRC_VERSIONS),
                    "val": _value(entry.get("val_samples"), SRC_VERSIONS)},
        "results": result,
        "promotions": promoted,
        "balance_config": _balance_from_config(sources, model, stamp, trained_on),
        "balance_plan": _archive_plan(sources, entry.get("tmp_snapshot"), stamp),
        "problems": problems,
    }


def _model_dirs(models_dir: Path) -> List[Path]:
    if not models_dir.is_dir():
        return []
    return sorted(p for p in models_dir.iterdir() if p.is_dir() and not p.is_symlink() and (p / "current.pt").exists())


def build_cards(sources: ModelCardSources) -> Dict[str, Any]:
    promotions = _promotions(sources.promotions_path)
    models = []
    for folder in _model_dirs(sources.models_dir):
        model = folder.name
        live_target = Path(os.readlink(folder / "current.pt")) if (folder / "current.pt").is_symlink() else folder / "current.pt"
        live_file = live_target.name
        versions = _json(folder / f"{model}_models_versions.json")
        versions = versions if isinstance(versions, dict) else {}
        by_file = {str(v.get("model_filename")): (k, v) for k, v in versions.items() if isinstance(v, dict)}
        files = {p.name: p for p in folder.glob("*.pt") if p.is_file() and p.name != "current.pt"}
        cards = []
        for name in sorted(set(files) | set(by_file)):
            label, entry = by_file.get(name, (None, None))
            cards.append(build_card(sources, model, files.get(name), label, entry, promotions, live_file))
        cards.sort(key=lambda c: (c["date"]["value"] or "", c["file"]), reverse=True)
        warnings = []
        if live_file not in files:
            warnings.append(f"Živi model pokazuje na fajl koji nije u folderu modela: {live_file}.")
        model_promotions = [p for p in promotions if p.get("item") == model]
        if model_promotions and model_promotions[-1].get("after") != live_file:
            warnings.append(f"Evidencija promocija kaže da je živ {model_promotions[-1].get('after')}, a stvarno je živ {live_file}.")
        elif not model_promotions:
            warnings.append("Promocija ovog modela nije zabeležena u evidenciji promocija.")
        active = _yaml(folder / "active_classes.yaml")
        live_card = next((c for c in cards if c["live"]), None)
        if active and live_card and live_card["classes"]["names"] and isinstance(active.get("names"), list):
            if [str(n) for n in active["names"]] != live_card["classes"]["names"]:
                warnings.append("Spisak klasa živog modela razlikuje se od spiska aktivnih klasa za pipeline.")
        models.append({"model": model, "live_file": live_file, "warnings": warnings, "cards": cards})
    return {"snapshot_version": SNAPSHOT_VERSION, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fingerprint": fingerprint(sources), "models": models}


# ---------------- Snapshot ----------------

def _watched(sources: ModelCardSources) -> Iterable[Path]:
    for folder in _model_dirs(sources.models_dir):
        yield folder
        yield folder / "current.pt"
        yield folder / f"{folder.name}_models_versions.json"
        yield folder / "active_classes.yaml"
    yield sources.models_dir
    if sources.promotions_path:
        yield sources.promotions_path
    if sources.training_root:
        root = sources.training_root
        yield from (root / "defaults.yaml", root / "augmentations.yaml", root / "overrides")
        if sources.tenant:
            yield from (root / "tenants" / f"{sources.tenant}.yaml", root / "tenants" / sources.tenant)
            tenant_dir = root / "tenants" / sources.tenant
            if tenant_dir.is_dir():
                yield from sorted(tenant_dir.glob("*.yaml"))
    if sources.archives_dir:
        yield sources.archives_dir


def fingerprint(sources: ModelCardSources) -> str:
    """Cheap: names, sizes, times and link targets only, so the web process can tell a stale snapshot."""
    h = hashlib.sha256()
    for path in _watched(sources):
        try:
            st = path.lstat()
            link = os.readlink(path) if path.is_symlink() else ""
            h.update(f"{path}|{st.st_size}|{st.st_mtime_ns}|{link}\n".encode())
        except OSError:
            h.update(f"{path}|-\n".encode())
    return h.hexdigest()


def write_snapshot(sources: ModelCardSources, target: Path) -> Dict[str, Any]:
    if Path(target).resolve().is_relative_to(sources.models_dir.resolve()):
        raise ValueError("model cards are never written into the models folder")
    data = build_cards(sources)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, target)
    return {"models": len(data["models"]), "cards": sum(len(m["cards"]) for m in data["models"]), "snapshot": str(target)}


def read_snapshot(target: Path) -> Optional[Dict[str, Any]]:
    data = _json(target)
    return data if isinstance(data, dict) and data.get("snapshot_version") == SNAPSHOT_VERSION else None
