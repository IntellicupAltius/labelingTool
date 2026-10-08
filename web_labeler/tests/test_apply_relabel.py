"""MDQ-15c-3 tests — throw-away copy of RAW in a temp dir; real RAW / sidecars are never touched.

    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest discover -s web_labeler/tests -v
"""
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import apply_relabel as ar   # noqa: E402
import archive_raw           # noqa: E402
import flag_store as fs      # noqa: E402
import relabel_batch as rb   # noqa: E402

sys.path.insert(0, str(ar.DEFAULT_INGEST_ROOT))

POOL = "cups"
W = H = 200
ORIG = "0 0.5 0.5 0.2 0.2\n"
ORIG_REQ = "0 0.500500 0.499500 0.201000 0.200500\n"      # ~1 px re-quantization drift of ORIG
MOVED = "0 0.7 0.5 0.2 0.2\n"                              # box moved 20 px
ADD1 = "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n"            # a second class added
ONLY0 = "0 0.5 0.5 0.2 0.2\n"


def img_bytes(color) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (W, H), color).save(b, "PNG")
    return b.getvalue()


def tree_hash(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()} if root.exists() else {}


class Env:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.raw = tmp / "raw" / "blaznavac"
        self.review = tmp / "raw_review"
        self.batches = tmp / "relabel_batches"
        self.archive = tmp / "raw_archive"
        (self.raw / POOL).mkdir(parents=True)
        (self.raw / POOL / "data.yaml").write_text("nc: 2\nnames: ['CAJ', 'CAFFE_LATTE']\n")
        self.store = fs.FlagStore(self.review)
        self.batch = None

    def put(self, cls, name, img, label):
        for kind in ("images", "labels"):
            (self.raw / POOL / kind / cls).mkdir(parents=True, exist_ok=True)
        (self.raw / POOL / "images" / cls / name).write_bytes(img)
        if label is not None:
            (self.raw / POOL / "labels" / cls / (Path(name).stem + ".txt")).write_text(label)

    def flag(self, cls, name):
        key = f"{cls}/{name}"
        ip = self.raw / POOL / "images" / cls / name
        lp = self.raw / POOL / "labels" / cls / (Path(name).stem + ".txt")
        self.store.add_manual_flag(POOL, key, "mislabeled_background", "fali boks")
        self.store.set_owner_decision(POOL, key, "relabel", relabel_snapshot={
            "image_sha256": rb.sha256_file(ip), "label_sha256": rb.sha256_file(lp) if lp.is_file() else None})
        return key

    def export(self) -> Path:
        self.batch = rb.write_batch(rb.plan_pool(POOL, self.raw, self.review, self.batches), self.raw, self.batches)
        return self.batch

    def correct(self, name, label):
        (self.batch / "labels" / (Path(name).stem + ".txt")).write_text(label)

    def run(self, apply=True, fail_after=None, stamp="20261005_100000_relabel", tol=2.0):
        out = io.StringIO()
        with redirect_stdout(out):
            res = ar.process_batch(POOL, self.batch, self.raw, self.store, self.archive, stamp, tol,
                                   ar.DEFAULT_INGEST_ROOT, apply, fail_after=fail_after)
        return res, out.getvalue()

    def where(self, name):
        return sorted(c["cls"] for c in ar._copies(self.raw, POOL, name))


class ApplyRelabelTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.e = Env(Path(self._td.name))
        import ingestion_pipeline.distributor as d
        self._raw_base = d.RAW_BASE

    def tearDown(self):
        import ingestion_pipeline.distributor as d
        d.RAW_BASE = self._raw_base
        self._td.cleanup()

    def lbl(self, cls, name):
        return (self.e.raw / POOL / "labels" / cls / (Path(name).stem + ".txt")).read_text()

    def test_single_folder_adds_class(self):
        e = self.e
        img = img_bytes("red")
        e.put("CAJ", "a.png", img, ORIG)
        key = e.flag("CAJ", "a.png")
        e.export()
        e.correct("a.png", ADD1)
        res, _ = e.run()
        self.assertEqual((res["applied"], res["failed"]), (1, 0))
        self.assertEqual(e.where("a.png"), ["CAFFE_LATTE", "CAJ"])      # new class folder gets it too
        for cls in ("CAJ", "CAFFE_LATTE"):                              # same label, same image, in every folder
            self.assertEqual(self.lbl(cls, "a.png"), ADD1)
            self.assertEqual((e.raw / POOL / "images" / cls / "a.png").read_bytes(), img)
        arch = e.archive / "20261005_100000_relabel"
        self.assertEqual(sorted(p.name for p in (arch / POOL / "images" / "CAJ").iterdir()), ["a.png"])
        self.assertEqual((arch / POOL / "labels" / "CAJ" / "a.txt").read_text(), ORIG)   # original kept, not deleted
        man = json.loads((arch / "apply_manifest.json").read_text())
        rec = man["batches"][0]["items"][0]
        self.assertEqual(rec["status"], "applied")
        self.assertEqual(len(rec["archived"]), 2)
        self.assertEqual(len(rec["inserted"]), 4)
        self.assertEqual(rb.read_manifest(e.batch)["status"], "applied")
        ent = e.store.get(POOL, key)
        self.assertEqual(ent["relabel_result"]["status"], "applied")
        self.assertEqual(ent["owner_decision"], "relabel")              # entry and decision kept
        self.assertIn("relabel_snapshot", ent)

    def test_image_in_two_folders(self):
        e = self.e
        img = img_bytes("green")
        both = "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n"
        e.put("CAJ", "b.png", img, both)
        e.put("CAFFE_LATTE", "b.png", img, both)
        e.flag("CAJ", "b.png")
        e.export()
        self.assertEqual(len(rb.read_manifest(e.batch)["items"][0]["raw_paths"]), 2)
        e.correct("b.png", "0 0.7 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n")     # box of class 0 moved
        res, _ = e.run()
        self.assertEqual(res["applied"], 1)
        self.assertEqual(e.where("b.png"), ["CAFFE_LATTE", "CAJ"])
        self.assertEqual(self.lbl("CAJ", "b.png"), self.lbl("CAFFE_LATTE", "b.png"))
        arch = e.archive / "20261005_100000_relabel" / POOL
        self.assertTrue((arch / "images" / "CAJ" / "b.png").is_file() and (arch / "images" / "CAFFE_LATTE" / "b.png").is_file())

    def test_class_removed_leaves_only_remaining_folder(self):
        e = self.e
        img = img_bytes("blue")
        both = "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n"
        e.put("CAJ", "c.png", img, both)
        e.put("CAFFE_LATTE", "c.png", img, both)
        e.flag("CAJ", "c.png")
        e.export()
        e.correct("c.png", ONLY0)
        res, out = e.run()
        self.assertEqual(res["applied"], 1)
        self.assertEqual(e.where("c.png"), ["CAJ"])
        self.assertIn("folders differ", out)

    def test_stale_original_changed(self):
        e = self.e
        e.put("CAJ", "d.png", img_bytes("red"), ORIG)
        e.flag("CAJ", "d.png")
        e.export()
        e.correct("d.png", ADD1)
        (e.raw / POOL / "labels" / "CAJ" / "d.txt").write_text(MOVED)   # RAW changed after the export
        before = tree_hash(e.raw)
        res, out = e.run()
        self.assertEqual((res["applied"], res["skipped"]), (0, 1))
        self.assertIn("stale", out)
        self.assertEqual(tree_hash(e.raw), before)
        self.assertNotIn("relabel_result", e.store.get(POOL, "CAJ/d.png"))
        self.assertEqual(rb.read_manifest(e.batch)["status"], "open")   # not resolved -> stays open

    def test_stale_new_copy_appeared(self):
        e = self.e
        img = img_bytes("red")
        e.put("CAJ", "d.png", img, ORIG)
        e.flag("CAJ", "d.png")
        e.export()
        e.correct("d.png", ADD1)
        e.put("CAFFE_LATTE", "d.png", img, ORIG)                        # a second copy showed up
        res, out = e.run()
        self.assertEqual((res["applied"], res["skipped"]), (0, 1))
        self.assertIn("stale", out)

    def test_corrected_image_edited_or_missing_or_bad_label(self):
        e = self.e
        for n in ("e1.png", "e2.png", "e3.png"):
            e.put("CAJ", n, img_bytes("red" if n == "e1.png" else "blue" if n == "e2.png" else "green"), ORIG)
            e.flag("CAJ", n)
        e.export()
        Image.new("RGB", (W, H), "white").save(e.batch / "images" / "e1.png")      # image edited
        (e.batch / "labels" / "e2.txt").unlink()                                    # label gone
        e.correct("e3.png", "5 0.5 0.5 0.2 0.2\n")                                  # class not in data.yaml
        before = tree_hash(e.raw)
        res, out = e.run()
        self.assertEqual((res["applied"], res["skipped"]), (0, 3))
        self.assertEqual(tree_hash(e.raw), before)

    def test_not_corrected_requantized_is_unchanged(self):
        e = self.e
        e.put("CAJ", "f.png", img_bytes("red"), ORIG)
        e.put("CAJ", "g.png", img_bytes("blue"), ORIG)
        kf, kg = e.flag("CAJ", "f.png"), e.flag("CAJ", "g.png")
        e.export()
        e.correct("f.png", ORIG_REQ)                                    # only Save drift
        e.correct("g.png", MOVED)                                       # real correction
        before = tree_hash(e.raw)
        res, out = e.run(apply=False)                                   # dry run: no change at all
        self.assertEqual((res["unchanged"], res["applied"]), (1, 1))
        self.assertEqual(tree_hash(e.raw), before)
        self.assertFalse(e.archive.exists())
        self.assertFalse((e.batch / "apply_state.json").exists())
        self.assertNotIn("relabel_result", e.store.get(POOL, kf))
        res, out = e.run()
        self.assertEqual((res["unchanged"], res["applied"]), (1, 1))
        self.assertIn("NIJE ISPRAVLJENO", out)
        self.assertEqual(self.lbl("CAJ", "f.png"), ORIG)                # untouched, no drift written to RAW
        self.assertEqual(self.lbl("CAJ", "g.png"), MOVED)
        self.assertEqual(e.store.get(POOL, kf)["relabel_result"]["status"], "unchanged")
        self.assertEqual(e.store.get(POOL, kg)["relabel_result"]["status"], "applied")
        self.assertEqual(rb.read_manifest(e.batch)["status"], "applied")

    def test_tolerance_is_a_parameter(self):
        e = self.e
        e.put("CAJ", "t.png", img_bytes("red"), ORIG)
        e.flag("CAJ", "t.png")
        e.export()
        e.correct("t.png", "0 0.52 0.5 0.2 0.2\n")                      # 4 px shift
        res, _ = e.run(apply=False, tol=2.0)
        self.assertEqual(res["applied"], 1)
        res, _ = e.run(apply=False, tol=5.0)
        self.assertEqual(res["unchanged"], 1)

    def test_failure_mid_apply_rolls_back(self):
        for stage in ("archive", "ingest", "verify"):
            with self.subTest(stage=stage):
                with tempfile.TemporaryDirectory() as td:
                    e = Env(Path(td))
                    img = img_bytes("red")
                    both = "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n"
                    e.put("CAJ", "h.png", img, both)
                    e.put("CAFFE_LATTE", "h.png", img, both)
                    e.put("CAJ", "ok.png", img_bytes("blue"), ORIG)
                    k = e.flag("CAJ", "h.png")
                    e.flag("CAJ", "ok.png")
                    e.export()
                    e.correct("h.png", "0 0.7 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n")
                    e.correct("ok.png", MOVED)
                    before = tree_hash(e.raw)
                    res, out = e.run(fail_after={"h.png": stage})
                    self.assertEqual((res["failed"], res["applied"]), (1, 1))   # other item still applied
                    after = tree_hash(e.raw)
                    for rel, h in before.items():
                        if "ok." not in rel:                                    # h.png: back as it was
                            self.assertEqual(after.get(rel), h, rel)
                    self.assertEqual(e.where("h.png"), ["CAFFE_LATTE", "CAJ"])
                    self.assertNotIn("relabel_result", e.store.get(POOL, k))
                    self.assertNotIn("h.png", ar._read_state(e.batch))
                    self.assertEqual(rb.read_manifest(e.batch)["status"], "open")
                    self.assertIn("rolled back", out)

    def test_idempotent_rerun(self):
        e = self.e
        e.put("CAJ", "i.png", img_bytes("red"), ORIG)
        k = e.flag("CAJ", "i.png")
        e.export()
        e.correct("i.png", MOVED)
        e.run()
        raw1, arch1 = tree_hash(e.raw), tree_hash(e.archive)
        res, out = e.run(stamp="20261005_110000_relabel")
        self.assertEqual((res["applied"], res["done"]), (0, 1))
        self.assertEqual(tree_hash(e.raw), raw1)
        self.assertEqual(tree_hash(e.archive), arch1)
        # sidecar mark lost (e.g. crash after RAW step) -> repaired on re-run, RAW still untouched
        e.store.clear_relabel_result(POOL, k)
        e.run(stamp="20261005_120000_relabel")
        self.assertEqual(e.store.get(POOL, k)["relabel_result"]["status"], "applied")
        self.assertEqual(tree_hash(e.raw), raw1)

    def test_rollback_command(self):
        e = self.e
        img = img_bytes("red")
        e.put("CAJ", "r.png", img, ORIG)
        k = e.flag("CAJ", "r.png")
        e.export()
        e.correct("r.png", ADD1)
        before = tree_hash(e.raw)
        e.run()
        self.assertNotEqual(tree_hash(e.raw), before)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(ar.rollback_run("20261005_100000_relabel", e.archive, e.batches, e.store, False), 0)
        self.assertNotEqual(tree_hash(e.raw), before)                   # rollback dry-run changes nothing
        with redirect_stdout(out):
            self.assertEqual(ar.rollback_run("20261005_100000_relabel", e.archive, e.batches, e.store, True), 0)
        self.assertEqual({k2: v for k2, v in tree_hash(e.raw).items() if "_ingest_index" not in k2},
                         {k2: v for k2, v in before.items() if "_ingest_index" not in k2})
        self.assertNotIn("relabel_result", e.store.get(POOL, k))
        self.assertEqual(rb.read_manifest(e.batch)["status"], "open")
        self.assertEqual(ar._read_state(e.batch), {})
        res, _ = e.run(stamp="20261005_130000_relabel")                 # can be applied again
        self.assertEqual(res["applied"], 1)

    def test_old_sidecar_read_without_migration(self):
        e = self.e
        e.put("CAJ", "o.png", img_bytes("red"), ORIG)
        e.flag("CAJ", "o.png")
        p = e.store.path_for(POOL)
        doc = json.loads(p.read_text())
        self.assertEqual(doc["schema_version"], 1)
        e.export()
        e.correct("o.png", MOVED)
        e.run()
        doc = json.loads(p.read_text())
        self.assertEqual(doc["schema_version"], 1)                      # schema untouched
        self.assertEqual(set(doc["entries"]["CAJ/o.png"]["relabel_result"]), {"status", "at", "batch_id", "apply_run"})


class ZeroBoxDecisionTests(unittest.TestCase):
    """Zero boxes in the corrected label: Background only when marked, otherwise Delete (archive every copy)."""
    setUp = ApplyRelabelTests.setUp
    tearDown = ApplyRelabelTests.tearDown
    lbl = ApplyRelabelTests.lbl

    def _two(self, name, color="red"):
        e = self.e
        both = "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n"
        e.put("CAJ", name, img_bytes(color), both)
        e.put("CAFFE_LATTE", name, img_bytes(color), both)
        return e.flag("CAJ", name)

    def test_no_choice_is_delete_and_archives_every_copy(self):
        e = self.e
        k = self._two("d.png")
        e.export()
        e.correct("d.png", "")
        res, out = e.run(apply=False)
        self.assertEqual((res["deleted"], res["applied"]), (1, 0))
        self.assertIn("Delete (default)", out)
        self.assertEqual(e.where("d.png"), ["CAFFE_LATTE", "CAJ"])     # dry run changes nothing
        res, _ = e.run()
        self.assertEqual((res["deleted"], res["failed"]), (1, 0))
        self.assertEqual(e.where("d.png"), [])
        arch = e.archive / "20261005_100000_relabel" / POOL
        for cls in ("CAJ", "CAFFE_LATTE"):
            self.assertTrue((arch / "images" / cls / "d.png").is_file())
            self.assertTrue((arch / "labels" / cls / "d.txt").is_file())
        self.assertEqual(e.store.get(POOL, k)["relabel_result"]["status"], "deleted")
        self.assertEqual(rb.read_manifest(e.batch)["status"], "applied")

    def test_marked_delete(self):
        e = self.e
        e.put("CAJ", "m.png", img_bytes("red"), ORIG)
        e.flag("CAJ", "m.png")
        e.export()
        e.correct("m.png", "")
        rb.write_fixer_decisions(e.batch, {"m.png": "delete"})
        res, out = e.run()
        self.assertEqual(res["deleted"], 1)
        self.assertIn("marked Delete", out)
        self.assertEqual(e.where("m.png"), [])

    def test_marked_background_goes_to_background(self):
        e = self.e
        e.put("CAJ", "b.png", img_bytes("red"), ORIG)
        k = e.flag("CAJ", "b.png")
        e.export()
        e.correct("b.png", "")
        rb.write_fixer_decisions(e.batch, {"b.png": "background"})
        res, _ = e.run()
        self.assertEqual((res["applied"], res["deleted"]), (1, 0))
        self.assertEqual(e.where("b.png"), ["background"])
        self.assertEqual(self.lbl("background", "b.png"), "")
        self.assertEqual(e.store.get(POOL, k)["relabel_result"]["status"], "applied")

    def test_mark_on_image_with_boxes_is_invalid(self):
        e = self.e
        e.put("CAJ", "i.png", img_bytes("red"), ORIG)
        e.flag("CAJ", "i.png")
        e.export()
        e.correct("i.png", MOVED)
        rb.write_fixer_decisions(e.batch, {"i.png": "delete"})
        res, out = e.run()
        self.assertEqual((res["skipped"], res["deleted"], res["applied"]), (1, 0, 0))
        self.assertEqual(e.where("i.png"), ["CAJ"])

    def test_delete_rollback_restores_every_copy(self):
        e = self.e
        k = self._two("r.png", "blue")
        e.export()
        e.correct("r.png", "")
        before = tree_hash(e.raw)
        e.run()
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(ar.rollback_run("20261005_100000_relabel", e.archive, e.batches, e.store, True), 0)
        self.assertEqual(tree_hash(e.raw), before)
        self.assertNotIn("relabel_result", e.store.get(POOL, k))
        self.assertEqual(rb.read_manifest(e.batch)["status"], "open")

    def test_failure_after_archive_rolls_back_delete(self):
        e = self.e
        self._two("f.png")
        e.export()
        e.correct("f.png", "")
        before = tree_hash(e.raw)
        res, _ = e.run(fail_after={"f.png": "archive"})
        self.assertEqual((res["failed"], res["deleted"]), (1, 0))
        self.assertEqual(tree_hash(e.raw), before)


class AppendedClassTests(unittest.TestCase):
    """A class appended to RAW data.yaml after the export (e.g. shots BLOW_JOB) keeps the batch usable."""
    setUp = ApplyRelabelTests.setUp
    tearDown = ApplyRelabelTests.tearDown
    lbl = ApplyRelabelTests.lbl

    def _yaml(self, names):
        (self.e.raw / POOL / "data.yaml").write_text(f"nc: {len(names)}\nnames: {names!r}\n")

    def test_compatible_helper(self):
        ok = rb.class_names_compatible
        self.assertTrue(ok(["A", "B"], ["A", "B"]))
        self.assertTrue(ok(["A", "B"], ["A", "B", "NEW"]))
        self.assertFalse(ok(["A", "B"], ["B", "A", "NEW"]))     # reorder
        self.assertFalse(ok(["A", "B"], ["A", "X"]))            # rename
        self.assertFalse(ok(["A", "B"], ["A"]))                 # removal
        self.assertFalse(ok([], ["A"]))
        self.assertFalse(ok(None, ["A"]))

    def test_new_class_appended_after_export_is_applied(self):
        e = self.e
        e.put("CAJ", "n.png", img_bytes("red"), ORIG)
        k = e.flag("CAJ", "n.png")
        e.export()                                               # batch has 2 classes
        self._yaml(["CAJ", "CAFFE_LATTE", "NEW_CLS"])           # class appended afterwards
        e.correct("n.png", "2 0.5 0.5 0.2 0.2\n")              # Goca uses the new class (id 2)
        res, _ = e.run()
        self.assertEqual((res["applied"], res["failed"], res["skipped"]), (1, 0, 0))
        self.assertEqual(e.where("n.png"), ["NEW_CLS"])
        self.assertEqual(self.lbl("NEW_CLS", "n.png"), "2 0.5 0.5 0.2 0.2\n")
        self.assertEqual(e.store.get(POOL, k)["relabel_result"]["status"], "applied")

    def test_new_class_id_before_raw_has_it_is_skipped(self):
        e = self.e
        e.put("CAJ", "s.png", img_bytes("red"), ORIG)
        e.flag("CAJ", "s.png")
        e.export()
        e.correct("s.png", "2 0.5 0.5 0.2 0.2\n")              # id 2 but RAW still has 2 classes
        res, out = e.run()
        self.assertEqual((res["applied"], res["skipped"]), (0, 1))
        self.assertIn("class id that is not in RAW data.yaml", out)
        self.assertEqual(e.where("s.png"), ["CAJ"])

    def test_reordered_raw_classes_refuse_the_batch(self):
        e = self.e
        e.put("CAJ", "r.png", img_bytes("red"), ORIG)
        e.flag("CAJ", "r.png")
        e.export()
        self._yaml(["CAFFE_LATTE", "CAJ", "NEW_CLS"])
        e.correct("r.png", MOVED)
        before = tree_hash(e.raw)
        with self.assertRaises(ar.ApplyError):
            e.run()
        self.assertEqual(tree_hash(e.raw), before)


class FixerSaveRelabelTests(unittest.TestCase):
    """Dataset Fixer Save on a relabel batch never erases files; marks round-trip through fixer_decisions.json."""

    def test_save_keeps_deleted_files_and_rejects_create_new(self):
        import dataset as ds
        with tempfile.TemporaryDirectory() as td:
            b = Path(td) / "B"
            (b / "images").mkdir(parents=True)
            (b / "labels").mkdir()
            for n in ("a.png", "b.png"):
                (b / "images" / n).write_bytes(img_bytes("red"))
                (b / "labels" / (Path(n).stem + ".txt")).write_text(ORIG)
            sess = ds.load_dataset_session(b, "cups", ["CAJ", "CAFFE_LATTE"])
            sess.relabel_items = {"a.png": {}, "b.png": {}}
            sess.ann_by_image[0] = []
            sess.deleted_images.add(0)
            with self.assertRaises(ValueError):
                ds.save_dataset_session(sess, "create_new")
            ds.save_dataset_session(sess, "overwrite")
            self.assertTrue((b / "images" / "a.png").is_file())
            self.assertEqual((b / "labels" / "a.txt").read_text(), "")
            self.assertTrue((b / "labels" / "b.txt").read_text().startswith("0 "))
            rb.write_fixer_decisions(b, {"a.png": "delete"})
            self.assertEqual(rb.read_fixer_decisions(b), {"a.png": "delete"})
            with self.assertRaises(rb.RelabelBatchError):
                rb.write_fixer_decisions(b, {"a.png": "nonsense"})


class ArchiveRawRegression(unittest.TestCase):
    """archive_raw.py is not modified by 15c-3; its approve_delete behaviour is pinned here."""

    def test_approve_delete_still_moves_pair_and_ignores_relabel(self):
        with tempfile.TemporaryDirectory() as td:
            e = Env(Path(td))
            e.put("CAJ", "x.png", img_bytes("red"), ORIG)
            e.put("CAJ", "y.png", img_bytes("blue"), ORIG)
            e.store.add_manual_flag(POOL, "CAJ/x.png", "gibberish", "")
            e.store.set_owner_decision(POOL, "CAJ/x.png", "approve_delete")
            e.flag("CAJ", "y.png")                                      # relabel: must NOT be archived
            plan = archive_raw.build_plan(e.store, e.raw, e.archive, "S", [POOL])
            self.assertEqual([p.image_key for p in plan.to_move], ["CAJ/x.png"])
            self.assertEqual(archive_raw.execute_plan(plan), [])
            self.assertFalse((e.raw / POOL / "images" / "CAJ" / "x.png").exists())
            self.assertTrue((e.archive / "S" / POOL / "images" / "CAJ" / "x.png").is_file())
            self.assertTrue((e.archive / "S" / POOL / "labels" / "CAJ" / "x.txt").is_file())
            self.assertTrue((e.raw / POOL / "images" / "CAJ" / "y.png").is_file())

    def test_execute_writes_audit_log_and_manifest_dry_run_does_not(self):
        with tempfile.TemporaryDirectory() as td:
            e = Env(Path(td))
            img = img_bytes("red")
            e.put("CAJ", "x.png", img, ORIG)
            e.store.add_manual_flag(POOL, "CAJ/x.png", "gibberish", "")
            e.store.set_owner_decision(POOL, "CAJ/x.png", "approve_delete")
            argv = ["--pool", POOL, "--raw-base-path", str(e.raw), "--review-dir", str(e.review),
                    "--archive-root", str(e.archive)]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(archive_raw.main(argv), 0)
            self.assertFalse(e.archive.exists())                         # dry run: nothing at all
            with redirect_stdout(io.StringIO()):
                self.assertEqual(archive_raw.main(argv + ["--execute"]), 0)
            [run] = list(e.archive.iterdir())
            man = json.loads((run / "archive_manifest.json").read_text())
            self.assertEqual(man["counts"], {"planned": 1, "moved": 1, "errors": 0, "skipped": 0})
            m = man["moved"][0]
            self.assertEqual(m["image_key"], "CAJ/x.png")
            self.assertEqual(m["image_sha256"], hashlib.sha256(img).hexdigest())
            self.assertEqual(m["label_sha256"], hashlib.sha256(ORIG.encode()).hexdigest())
            log = (run / "archive_log.txt").read_text()
            self.assertIn("CAJ/x.png", log)
            self.assertIn("Moved 1/1 pair(s).", log)


if __name__ == "__main__":
    unittest.main()
