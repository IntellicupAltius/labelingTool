"""KATALOG-1 ticket 07: read-only model cards, on a small fake model folder.

    PYTHONPATH=web_labeler/tests /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest web_labeler.tests.test_model_cards -v
"""
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from web_labeler import server
from web_labeler.catalog import model_cards as mc
from web_labeler.catalog import router as catalog_router

REPO = Path(__file__).resolve().parents[2]


class DetectionModel:  # pickled under this test module's name, read back as a stand-in
    def __init__(self, names):
        self.names = names
        self.weights = FakeStorage()


class FakeStorage:
    pass


class Evil:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


class _Pickler(pickle.Pickler):
    def persistent_id(self, obj):
        return ("storage", "FloatStorage", "0", "cpu", 10) if isinstance(obj, FakeStorage) else None


def write_checkpoint(path, payload):
    buf = io.BytesIO()
    _Pickler(buf, protocol=2).dump(payload)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{path.stem}/data.pkl", buf.getvalue())
        z.writestr(f"{path.stem}/data/0", b"\0" * 40)  # stands for the weights; never read


def tree_state(root):
    return {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()}


class ModelCardTests(unittest.TestCase):
    LIVE = "acme_glasses_yolo11m_e300_c2_trval-100-20_ts20261122_210725.pt"
    OLD = "acme_glasses_yolov8m_e300_c3_trval-80-10_ts20260509_183927.pt"
    GONE = "acme_glasses_yolov8m_e300_c3_trval-50-10_ts20260101_100000.pt"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.models = self.tmp / "models"
        g = self.models / "glasses"
        g.mkdir(parents=True)
        write_checkpoint(g / self.LIVE, {
            "date": "2026-11-22T23:59:37", "version": "8.4.102", "epoch": -1, "model": DetectionModel({0: "CASA", 1: "NISKA"}),
            "ema": None, "optimizer": None,
            "train_args": {"model": "/some/where/yolo11m.pt", "epochs": 300, "patience": 100, "imgsz": 640, "batch": 16,
                           "optimizer": "SGD", "mosaic": 1.0, "fliplr": 0.5},
            "train_metrics": {"metrics/precision(B)": 0.8, "metrics/recall(B)": 0.7, "metrics/mAP50(B)": 0.75,
                              "metrics/mAP50-95(B)": 0.6, "fitness": 0.6},
            "train_results": {"epoch": [1, 2, 3], "metrics/mAP50(B)": [0.5, 0.9, 0.7], "metrics/mAP50-95(B)": [0.4, 0.6, 0.5]},
        })
        # An older model: the file knows nothing beyond its classes; the run folder has args and results.
        write_checkpoint(g / self.OLD, {"model": DetectionModel(["A", "B", "C"])})
        run = g / "run_20260509_183927"
        run.mkdir()
        (run / "args.yaml").write_text("model: yolov8m.yaml\nepochs: 300\npatience: 50\nimgsz: 640\n", encoding="utf-8")
        (run / "results.csv").write_text(
            "   epoch,   metrics/mAP50(B),   metrics/mAP50-95(B)\n   1,   0.5,   0.4\n   2,   0.6,   0.45\n", encoding="utf-8")
        os.symlink(g / self.LIVE, g / "current.pt")
        (g / "glasses_models_versions.json").write_text(json.dumps({
            "v1": {"date": "2026-01-01", "model_filename": self.GONE, "train_samples": 50, "val_samples": 10, "epochs": 300,
                   "base_model": "yolov8m.yaml", "class_count": 3, "classes": ["A", "B", "C"]},
            "v2": {"date": "2026-11-23", "model_filename": self.LIVE, "train_samples": 100, "val_samples": 20, "epochs": 300,
                   "base_model": "/some/where/yolo11m.pt", "class_count": 2, "tmp_snapshot": "tmp_training_20261122_210725"},
        }), encoding="utf-8")
        (g / "active_classes.yaml").write_text("names: [CASA, NISKA]\n", encoding="utf-8")
        (self.models / "deepfeatures").mkdir()  # not a model: no current.pt

        self.promotions = self.tmp / "promotions.jsonl"
        self.promotions.write_text(json.dumps({"date": "2026-11-23", "kind": "PRODUKCIJA", "item": "glasses",
                                               "before": self.OLD, "after": self.LIVE, "why": "bolji na probnom danu"}) + "\n",
                                   encoding="utf-8")
        cfg = self.tmp / "training"
        (cfg / "tenants" / "acme").mkdir(parents=True)
        (cfg / "overrides").mkdir()
        (cfg / "defaults.yaml").write_text("yolo:\n  epochs: 300\n", encoding="utf-8")
        (cfg / "augmentations.yaml").write_text(
            "augmentations:\n  balance:\n    min_samples_per_class: auto\n    auto: {percentile: 75, min_target: 120, max_target: 500}\n"
            "  global:\n    enable: true\n    gaussian_noise: {enable: true}\n    blur: {enable: false}\n", encoding="utf-8")
        (cfg / "tenants" / "acme" / "glasses.yaml").write_text(
            "roi_crop: true\naugmentations:\n  balance:\n    auto: {percentile: 70, min_target: 180, max_target: 550}\n", encoding="utf-8")
        (cfg / "overrides" / "glasses_yolo11m_20261122.yaml").write_text("yolo:\n  batch: 16\n", encoding="utf-8")
        old_time = time.mktime((2026, 11, 1, 12, 0, 0, 0, 0, -1))
        for p in cfg.rglob("*.yaml"):
            os.utime(p, (old_time, old_time))
        self.archives = self.tmp / "archives"
        self.archives.mkdir()
        with zipfile.ZipFile(self.archives / "tmp_training_20261122_210725.zip", "w") as z:
            z.writestr("_metadata.json", json.dumps({"raw_total_samples": 90, "augmented_total_samples": 10,
                                                     "augmentation_plan": {"CASA": 10, "NISKA": 0},
                                                     "per_class_split": {"CASA": {"train": 60, "val": 10, "eval_holdout": 2},
                                                                         "NISKA": {"train": 40, "val": 10, "eval_holdout": 0}}}))
        self.sources = mc.ModelCardSources(self.models, self.promotions, cfg, self.archives, "acme")

    def tearDown(self):
        self._tmp.cleanup()

    def cards(self):
        data = mc.build_cards(self.sources)
        self.assertEqual([m["model"] for m in data["models"]], ["glasses"])
        return data["models"][0]

    def test_reads_checkpoint_description_without_torch(self):
        ck = mc.read_checkpoint(self.models / "glasses" / self.LIVE)
        self.assertEqual(ck["date"], "2026-11-22T23:59:37")
        self.assertEqual(ck["names"], ["CASA", "NISKA"])
        self.assertEqual(ck["train_args"]["patience"], 100)
        self.assertEqual(ck["train_results"]["epoch"], [1, 2, 3])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("ultralytics", sys.modules)

    def test_pickle_cannot_run_code(self):
        marker = self.tmp / "ran"
        bad = self.tmp / "bad.pt"
        write_checkpoint(bad, {"date": "x", "model": Evil(str(marker))})
        mc.read_checkpoint(bad)
        self.assertFalse(marker.exists())

    def test_live_card_from_model_file_and_records(self):
        entry = self.cards()
        self.assertEqual(entry["live_file"], self.LIVE)
        self.assertEqual(entry["warnings"], [])
        live = entry["cards"][0]
        self.assertTrue(live["live"])
        self.assertEqual([c["live"] for c in entry["cards"]].count(True), 1)
        self.assertEqual(live["date"], {"value": "2026-11-22", "source": mc.SRC_FILE})
        self.assertEqual(live["version"]["value"], "v2")
        self.assertEqual(live["architecture"]["value"], "YOLO11m")
        self.assertEqual(live["training"]["patience"], {"value": 100, "source": mc.SRC_FILE})
        self.assertEqual(live["training"]["epochs_run"]["value"], 3)
        self.assertEqual(live["training"]["best_epoch"]["value"], 2)
        self.assertEqual(live["augmentations"]["mosaic"]["value"], 1.0)
        self.assertEqual(live["classes"]["names"], ["CASA", "NISKA"])
        self.assertEqual(live["results"]["mAP50"]["value"], 0.75)
        self.assertEqual(live["samples"]["train"], {"value": 100, "source": mc.SRC_VERSIONS})
        self.assertEqual(live["promotions"][0]["why"], "bolji na probnom danu")

    def test_balancing_from_training_config_with_source(self):
        live = self.cards()["cards"][0]
        cfg = live["balance_config"]
        self.assertEqual(cfg["source"], mc.SRC_CONFIG)
        self.assertEqual(cfg["auto"], {"percentile": 70, "min_target": 180, "max_target": 550})
        self.assertEqual(cfg["files"], ["defaults.yaml", "augmentations.yaml", "glasses.yaml", "glasses_yolo11m_20261122.yaml"])
        self.assertEqual(cfg["global_transforms"], ["gaussian_noise"])
        self.assertTrue(cfg["roi_crop"])
        self.assertEqual(cfg["changed_after_training"], [])
        plan = live["balance_plan"]
        self.assertEqual(plan["source"], mc.SRC_ARCHIVE)
        self.assertEqual(plan["classes"][0], {"class": "CASA", "added": 10, "train": 60, "val": 10, "eval": 2})
        # A config file edited after training is flagged.
        tenant_file = self.sources.training_root / "tenants" / "acme" / "glasses.yaml"
        later = time.mktime((2026, 11, 24, 12, 0, 0, 0, 0, -1))
        os.utime(tenant_file, (later, later))
        changed = self.cards()["cards"][0]["balance_config"]["changed_after_training"]
        self.assertEqual(changed, [{"file": "glasses.yaml", "changed": "2026-11-24"}])

    def test_missing_values_stay_empty(self):
        entry = self.cards()
        old = next(c for c in entry["cards"] if c["file"] == self.OLD)
        self.assertEqual(old["version"], {"value": None, "source": None})
        self.assertEqual(old["training"]["patience"], {"value": 50, "source": mc.SRC_RUN})
        self.assertEqual(old["training"]["batch"], {"value": None, "source": None})
        self.assertEqual(old["training"]["epochs_run"]["value"], 2)
        self.assertEqual(old["results"]["mAP50"], {"value": None, "source": None})
        self.assertEqual(old["date"]["source"], "ime fajla modela")
        self.assertIsNone(old["balance_plan"])
        self.assertEqual(old["promotions"], [])
        gone = next(c for c in entry["cards"] if c["file"] == self.GONE)
        self.assertEqual(gone["classes"]["names"], ["A", "B", "C"])
        self.assertEqual(gone["classes"]["source"], mc.SRC_VERSIONS)
        self.assertTrue(gone["problems"])

    def test_warns_when_records_disagree_with_live_file(self):
        g = self.models / "glasses"
        (g / "current.pt").unlink()
        os.symlink(g / self.OLD, g / "current.pt")
        entry = self.cards()
        self.assertEqual(entry["live_file"], self.OLD)
        self.assertTrue(next(c for c in entry["cards"] if c["file"] == self.OLD)["live"])
        self.assertTrue(any("Evidencija promocija" in w for w in entry["warnings"]))
        self.assertTrue(any("aktivnih klasa" in w for w in entry["warnings"]))

    def test_model_files_stay_unchanged_and_snapshot_never_goes_there(self):
        before = tree_state(self.models)
        link = os.readlink(self.models / "glasses" / "current.pt")
        out = self.tmp / "catalog" / mc.SNAPSHOT_NAME
        mc.write_snapshot(self.sources, out)
        self.assertEqual(tree_state(self.models), before)
        self.assertEqual(os.readlink(self.models / "glasses" / "current.pt"), link)
        with self.assertRaises(ValueError):
            mc.write_snapshot(self.sources, self.models / "glasses" / mc.SNAPSHOT_NAME)
        self.assertEqual(tree_state(self.models), before)

    def config(self):
        cfg = {"data_root": str(self.tmp / "data"), "catalog_venue_id": "acme",
               "catalog_db_path": str(self.tmp / "catalog" / "catalog.db"),
               "raw_review_dir": str(self.tmp / "raw_review"), "relabel_batches_dir": str(self.tmp / "batches"),
               "catalog_live_models_dir": str(self.models), "catalog_promotions_path": str(self.promotions),
               "catalog_training_root": str(self.sources.training_root),
               "catalog_training_archives_dir": str(self.archives)}
        path = self.tmp / "labeler_config.json"
        path.write_text(json.dumps(cfg))
        return path

    def routes(self):
        old = os.environ.get("LABELER_CONFIG_PATH")
        os.environ["LABELER_CONFIG_PATH"] = str(self.config())
        try:
            app = server.create_app()
        finally:
            if old is None:
                os.environ.pop("LABELER_CONFIG_PATH", None)
            else:
                os.environ["LABELER_CONFIG_PATH"] = old
        return {(r.path, m): r.endpoint for r in app.routes for m in getattr(r, "methods", ())}

    def test_built_in_separate_process_and_read_only_in_web(self):
        routes = self.routes()
        methods = {m for (path, m) in routes if path.startswith("/api/catalog/model-cards")}
        self.assertEqual(methods, {"GET"})
        started = []
        original = catalog_router.start_model_cards_refresh
        catalog_router.start_model_cards_refresh = lambda settings: started.append(settings) or True
        try:
            first = routes[("/api/catalog/model-cards", "GET")]()
            self.assertEqual(first, {"refreshing": True, "snapshot": None})
            self.assertEqual(len(started), 1)
            env = dict(os.environ, LABELER_CONFIG_PATH=str(self.config()))
            done = subprocess.run([sys.executable, "-m", "web_labeler.catalog", "model-cards"], cwd=str(REPO), env=env,
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stderr)
            second = routes[("/api/catalog/model-cards", "GET")]()
            self.assertFalse(second["refreshing"])
            self.assertEqual(second["snapshot"]["models"][0]["live_file"], self.LIVE)
            self.assertEqual(len(started), 1)
            # A promotion changes the live link: the snapshot is stale and a refresh starts again.
            g = self.models / "glasses"
            (g / "current.pt").unlink()
            os.symlink(g / self.OLD, g / "current.pt")
            self.assertTrue(routes[("/api/catalog/model-cards", "GET")]()["refreshing"])
            self.assertEqual(len(started), 2)
        finally:
            catalog_router.start_model_cards_refresh = original


if __name__ == "__main__":
    unittest.main()
