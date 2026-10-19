"""MDQ-16b tests — /api/raw/action + the RAW-browse badge status, on a throw-away app.

The app is built by ``create_app()`` against a tmp ``labeler_config.json`` (``LABELER_CONFIG_PATH``),
so RAW, sidecars and relabel batches are all tmp dirs. httpx is not in the labeler interpreter, so
the route functions are called directly instead of through a TestClient.

    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest discover -s web_labeler/tests -v
"""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from web_labeler import flag_store as fs
from web_labeler import server

POOL = "glasses"


class RawActionApiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.raw = tmp / "raw" / "blaznavac"
        self.review = tmp / "raw_review"
        self.batches = tmp / "relabel_batches"
        for d in ("CASA", "CASA2"):
            (self.raw / POOL / "images" / d).mkdir(parents=True)
            (self.raw / POOL / "labels" / d).mkdir(parents=True)
        self.put("CASA", "a.jpg")
        self.put("CASA", "b.jpg", label=None)
        self.put("CASA2", "a.jpg")          # same file name in another class folder
        cfg = tmp / "labeler_config.json"
        cfg.write_text(json.dumps({"data_root": str(tmp / "data"), "raw_base_path": str(self.raw),
                                   "raw_review_dir": str(self.review),
                                   "relabel_batches_dir": str(self.batches)}))
        self._env = os.environ.get("LABELER_CONFIG_PATH")
        os.environ["LABELER_CONFIG_PATH"] = str(cfg)
        app = server.create_app()
        self.routes = {(r.path, m): r.endpoint for r in app.routes for m in getattr(r, "methods", ())}
        self.store = fs.FlagStore(self.review)

    def tearDown(self):
        if self._env is None:
            os.environ.pop("LABELER_CONFIG_PATH", None)
        else:
            os.environ["LABELER_CONFIG_PATH"] = self._env
        self._tmp.cleanup()

    def put(self, cls, name, label="0 0.5 0.5 0.2 0.2\n"):
        # a real (tiny) JPEG: /api/raw/load reads the image size; a different color per name
        # keeps the checksums distinct
        Image.new("RGB", (8, 8), (len(name) * 20 % 255, ord(cls[-1]), 0)).save(self.raw / POOL / "images" / cls / name)
        if label is not None:
            (self.raw / POOL / "labels" / cls / (Path(name).stem + ".txt")).write_text(label)

    def act(self, key, action, comment=""):
        return self.routes[("/api/raw/action", "POST")](
            server.RawActionRequest(image_key=key, action=action, comment=comment, pool=POOL))

    def open_batch(self, key, status="open", bid="RELABEL_glasses_20261019_000000"):
        bdir = self.batches / POOL / bid
        bdir.mkdir(parents=True)
        (bdir / "manifest.json").write_text(json.dumps({
            "format_version": 1, "batch_id": bid, "pool": POOL, "status": status,
            "items": [{"item_name": key.split("/")[1], "image_key": key, "image_keys": [key]}]}))
        return bid

    # -- decisions --------------------------------------------------------------

    def test_delete_then_undo(self):
        r = self.act("CASA/a.jpg", "approve_delete")
        self.assertEqual(r["entry"]["source"], fs.SOURCE_OWNER)
        self.assertEqual(r["raw_status"]["state"], "delete")
        r = self.act("CASA/a.jpg", None)
        self.assertIsNone(r["entry"])
        self.assertEqual(r["raw_status"]["state"], "none")
        self.assertIsNone(self.store.get(POOL, "CASA/a.jpg"))

    def test_relabel_records_raw_checksums(self):
        r = self.act("CASA/a.jpg", "relabel", "oznaci sve case")
        snap = r["entry"]["relabel_snapshot"]
        img = (self.raw / POOL / "images" / "CASA" / "a.jpg").read_bytes()
        lab = (self.raw / POOL / "labels" / "CASA" / "a.txt").read_bytes()
        self.assertEqual(snap["image_sha256"], hashlib.sha256(img).hexdigest())
        self.assertEqual(snap["label_sha256"], hashlib.sha256(lab).hexdigest())
        self.assertEqual(r["raw_status"], {"state": "relabel", "batch_id": None, "locked": False,
                                           "comment": "oznaci sve case",
                                           "reason": fs.describe_reason(r["entry"])})

    def test_relabel_without_label_file(self):
        r = self.act("CASA/b.jpg", "relabel")
        self.assertIsNone(r["entry"]["relabel_snapshot"]["label_sha256"])

    def test_flag_is_pending(self):
        r = self.act("CASA/a.jpg", "flag", "nisam siguran")
        self.assertIsNone(r["entry"]["owner_decision"])
        self.assertEqual(r["raw_status"]["state"], "flag")

    def test_missing_raw_image_is_404(self):
        with self.assertRaises(server.HTTPException) as cm:
            self.act("CASA/nema.jpg", "approve_delete")
        self.assertEqual(cm.exception.status_code, 404)

    def test_bad_action(self):
        with self.assertRaises(fs.FlagValidationError):
            self.act("CASA/a.jpg", "keep")

    # -- lock by open relabel batch ---------------------------------------------

    def test_key_in_open_batch_is_locked(self):
        bid = self.open_batch("CASA/a.jpg")
        with self.assertRaises(fs.FlagConflict) as cm:
            self.act("CASA/a.jpg", "approve_delete")
        self.assertIn(bid, str(cm.exception))
        self.assertIsNone(self.store.get(POOL, "CASA/a.jpg"))

    def test_same_file_name_in_other_class_is_locked_too(self):
        self.open_batch("CASA/a.jpg")
        with self.assertRaises(fs.FlagConflict):
            self.act("CASA2/a.jpg", "relabel")

    def test_applied_batch_does_not_lock(self):
        self.open_batch("CASA/a.jpg", status="applied")
        self.assertEqual(self.act("CASA/a.jpg", "approve_delete")["raw_status"]["state"], "delete")

    def test_lock_index_refreshes_when_a_batch_appears(self):
        self.act("CASA/a.jpg", "flag")                    # index built and cached with no batches
        self.open_batch("CASA/a.jpg")
        with self.assertRaises(fs.FlagConflict):
            self.act("CASA/a.jpg", None)

    # -- badge states (served by /api/raw/annotations) ---------------------------

    def badge(self, idx):
        self.routes[("/api/raw/load", "POST")]({"model": POOL, "class_name": "CASA"})
        return self.routes[("/api/raw/annotations", "GET")](index=idx)["raw_status"]

    def test_annotations_badge_states(self):
        # CASA holds a.jpg (index 0) and b.jpg (index 1)
        self.assertEqual(self.badge(0)["state"], "none")
        self.store.add_auto_flag(POOL, "CASA/a.jpg", fs.SOURCE_ANALYZER_AUTO, {"metric": "m", "value": 1}, "model_pass")
        self.assertEqual(self.badge(0)["state"], "model")
        self.act("CASA/a.jpg", "relabel", "bolje")
        st = self.badge(0)
        self.assertEqual((st["state"], st["comment"], st["locked"]), ("relabel", "bolje", False))
        self.act("CASA/a.jpg", None)
        self.store.set_owner_decision(POOL, "CASA/a.jpg", "keep")
        self.assertEqual(self.badge(0)["state"], "decided")
        self.act("CASA/b.jpg", "approve_delete")
        self.assertEqual(self.badge(1)["state"], "delete")
        self.act("CASA/a.jpg", "relabel")
        self.store.set_relabel_result(POOL, "CASA/a.jpg", {"status": "applied", "batch_id": "B", "apply_run": "R"})
        self.assertEqual(self.badge(0)["state"], "none")      # already corrected in RAW

    def test_annotations_badge_in_batch(self):
        bid = self.open_batch("CASA2/a.jpg")              # same file name, other class -> CASA/a.jpg locked
        st = self.badge(0)
        self.assertEqual((st["state"], st["batch_id"], st["locked"]), ("in_batch", bid, True))

    def test_annotations_survive_a_broken_manifest(self):
        bdir = self.batches / POOL / "RELABEL_glasses_broken"
        bdir.mkdir(parents=True)
        (bdir / "manifest.json").write_text("{not json")
        self.assertIsNone(self.badge(0))


if __name__ == "__main__":
    unittest.main()
