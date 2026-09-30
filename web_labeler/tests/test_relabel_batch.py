"""MDQ-15c-2 tests — run on throw-away copies only (tmp dirs); never touches real RAW / sidecars.

    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest discover -s web_labeler/tests -v
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flag_store as fs      # noqa: E402
import relabel_batch as rb   # noqa: E402

POOL = "cups"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Env:
    def __init__(self, tmp: Path):
        self.raw = tmp / "raw" / "blaznavac"
        self.review = tmp / "raw_review"
        self.batches = tmp / "relabel_batches"
        (self.raw / POOL).mkdir(parents=True)
        (self.raw / POOL / "data.yaml").write_text("nc: 2\nnames: ['CAJ', 'CAFFE_LATTE']\n")
        self.store = fs.FlagStore(self.review)

    def put(self, cls, name, img=b"IMG", label="0 0.5 0.5 0.2 0.2\n"):
        (self.raw / POOL / "images" / cls).mkdir(parents=True, exist_ok=True)
        (self.raw / POOL / "labels" / cls).mkdir(parents=True, exist_ok=True)
        (self.raw / POOL / "images" / cls / name).write_bytes(img)
        if label is not None:
            (self.raw / POOL / "labels" / cls / (Path(name).stem + ".txt")).write_text(label)

    def flag_relabel(self, cls, name, comment="fali boks", snapshot=None):
        key = f"{cls}/{name}"
        self.store.add_manual_flag(POOL, key, "mislabeled_background", comment)
        ip = self.raw / POOL / "images" / cls / name
        lp = self.raw / POOL / "labels" / cls / (Path(name).stem + ".txt")
        snap = snapshot or {"image_sha256": rb.sha256_file(ip), "label_sha256": rb.sha256_file(lp) if lp.is_file() else None}
        self.store.set_owner_decision(POOL, key, "relabel", relabel_snapshot=snap)
        return key

    def plan(self):
        return rb.plan_pool(POOL, self.raw, self.review, self.batches)

    def tree(self, root: Path):
        return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


class RelabelBatchTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.env = Env(Path(self._td.name))

    def tearDown(self):
        self._td.cleanup()

    def test_export_copies_and_manifest(self):
        e = self.env
        e.put("CAJ", "a.jpg", b"A")
        e.put("CAJ", "b.jpg", b"B", label=None)              # no label file -> empty label in the batch
        e.flag_relabel("CAJ", "a.jpg", "fali SHOT boks levo")
        e.flag_relabel("CAJ", "b.jpg")
        raw_before = e.tree(e.raw)
        out = rb.write_batch(e.plan(), e.raw, e.batches)
        self.assertEqual(e.tree(e.raw), raw_before)           # RAW untouched
        self.assertTrue(out.is_relative_to(e.batches / POOL))
        man = json.loads((out / "manifest.json").read_text())
        self.assertEqual((man["format_version"], man["pool"], man["status"]), (1, POOL, "open"))
        self.assertEqual(man["class_names"], ["CAJ", "CAFFE_LATTE"])
        items = {i["item_name"]: i for i in man["items"]}
        a = items["a.jpg"]
        self.assertEqual(a["image_key"], "CAJ/a.jpg")
        self.assertEqual(a["comment"], "fali SHOT boks levo")
        self.assertEqual(a["flag_source"], "manual_goca")
        self.assertEqual(a["relabel_snapshot"], {"image_sha256": sha(b"A"), "label_sha256": sha(b"0 0.5 0.5 0.2 0.2\n")})
        self.assertEqual(a["raw_paths"], [{"image": "cups/images/CAJ/a.jpg", "label": "cups/labels/CAJ/a.txt"}])
        self.assertEqual((out / "images" / "a.jpg").read_bytes(), b"A")
        self.assertEqual((out / "labels" / "a.txt").read_text(), "0 0.5 0.5 0.2 0.2\n")
        self.assertEqual((out / "labels" / "b.txt").read_bytes(), b"")
        self.assertIsNone(items["b.jpg"]["relabel_snapshot"]["label_sha256"])
        self.assertIsNone(items["b.jpg"]["raw_paths"][0]["label"])
        # real copies, not links to RAW
        self.assertNotEqual((out / "images" / "a.jpg").stat().st_ino, (e.raw / POOL / "images" / "CAJ" / "a.jpg").stat().st_ino)
        self.assertEqual(rb.read_manifest(out)["batch_id"], out.name)
        self.assertEqual([p.name for p in (e.batches / POOL).iterdir()], [out.name])   # no .tmp_ leftovers

    def test_changing_the_batch_copy_does_not_touch_raw(self):
        e = self.env
        e.put("CAJ", "a.jpg", b"A")
        e.flag_relabel("CAJ", "a.jpg")
        out = rb.write_batch(e.plan(), e.raw, e.batches)
        (out / "labels" / "a.txt").write_text("1 0.1 0.1 0.1 0.1\n")   # what Dataset Fixer Save does
        self.assertEqual((e.raw / POOL / "labels" / "CAJ" / "a.txt").read_text(), "0 0.5 0.5 0.2 0.2\n")

    def test_stale_image_and_label(self):
        e = self.env
        e.put("CAJ", "a.jpg", b"A")
        e.put("CAJ", "b.jpg", b"B")
        e.put("CAJ", "c.jpg", b"C")
        e.flag_relabel("CAJ", "a.jpg")
        e.flag_relabel("CAJ", "b.jpg")
        e.flag_relabel("CAJ", "c.jpg")
        e.put("CAJ", "a.jpg", b"A2")                                    # image changed
        e.put("CAJ", "b.jpg", b"B", label="1 0.5 0.5 0.2 0.2\n")        # label changed
        (e.raw / POOL / "images" / "CAJ" / "c.jpg").unlink()            # image gone
        plan = e.plan()
        self.assertEqual(plan.items, [])
        self.assertEqual({s.image_key: s.reason for s in plan.skipped},
                         {"CAJ/a.jpg": "stale", "CAJ/b.jpg": "stale", "CAJ/c.jpg": "stale"})
        self.assertIsNone(rb.write_batch(plan, e.raw, e.batches))
        self.assertFalse(e.batches.exists())

    def test_same_name_in_two_class_folders_one_copy(self):
        e = self.env
        e.put("CAJ", "x.jpg", b"X")
        e.put("CAFFE_LATTE", "x.jpg", b"X")
        e.flag_relabel("CAJ", "x.jpg", "c1")
        e.flag_relabel("CAFFE_LATTE", "x.jpg", "c2")
        plan = e.plan()
        self.assertEqual(len(plan.items), 1)
        out = rb.write_batch(plan, e.raw, e.batches)
        self.assertEqual(sorted(p.name for p in (out / "images").iterdir()), ["x.jpg"])
        it = rb.read_manifest(out)["items"][0]
        self.assertEqual(sorted(p["image"] for p in it["raw_paths"]), ["cups/images/CAFFE_LATTE/x.jpg", "cups/images/CAJ/x.jpg"])
        self.assertEqual(sorted(it["image_keys"]), ["CAFFE_LATTE/x.jpg", "CAJ/x.jpg"])
        self.assertEqual(it["comment"], "c2 | c1")   # sorted key order: CAFFE_LATTE first
        self.assertEqual(len(it["flags"]), 2)

    def test_unflagged_extra_copy_is_listed(self):
        e = self.env
        e.put("CAJ", "x.jpg", b"X")
        e.put("CAFFE_LATTE", "x.jpg", b"X")
        e.flag_relabel("CAJ", "x.jpg")          # only one of the copies flagged
        it = e.plan().items[0]
        self.assertEqual(len(it["raw_paths"]), 2)
        self.assertEqual(it["image_keys"], ["CAJ/x.jpg"])

    def test_conflict_copies_with_different_labels(self):
        e = self.env
        e.put("CAJ", "x.jpg", b"X", label="0 0.5 0.5 0.2 0.2\n")
        e.put("CAFFE_LATTE", "x.jpg", b"X", label="1 0.5 0.5 0.2 0.2\n")
        e.flag_relabel("CAJ", "x.jpg")
        plan = e.plan()
        self.assertEqual(plan.items, [])
        self.assertEqual([(s.image_key, s.reason) for s in plan.skipped], [("CAJ/x.jpg", "conflict")])

    def test_conflict_class_vs_background(self):
        e = self.env
        e.put("CAJ", "x.jpg", b"X", label="0 0.5 0.5 0.2 0.2\n")
        e.put("background", "x.jpg", b"X", label=None)
        e.flag_relabel("CAJ", "x.jpg")
        self.assertEqual([s.reason for s in e.plan().skipped], ["conflict"])

    def test_conflict_copies_with_different_image_bytes(self):
        e = self.env
        e.put("CAJ", "x.jpg", b"X")
        e.put("CAFFE_LATTE", "x.jpg", b"Y")
        e.flag_relabel("CAJ", "x.jpg")
        self.assertEqual([s.reason for s in e.plan().skipped], ["conflict"])

    def test_missing_label_equals_empty_label_no_conflict(self):
        e = self.env
        e.put("CAJ", "x.jpg", b"X", label="")
        e.put("background", "x.jpg", b"X", label=None)
        e.flag_relabel("CAJ", "x.jpg")
        self.assertEqual(len(e.plan().items), 1)

    def test_idempotent_open_batch_not_exported_again(self):
        e = self.env
        e.put("CAJ", "a.jpg", b"A")
        e.put("CAJ", "b.jpg", b"B")
        e.flag_relabel("CAJ", "a.jpg")
        first = rb.write_batch(e.plan(), e.raw, e.batches)
        self.assertIsNotNone(first)
        again = e.plan()
        self.assertEqual(again.items, [])
        self.assertEqual([(s.image_key, s.reason) for s in again.skipped], [("CAJ/a.jpg", "in_open_batch")])
        self.assertIsNone(rb.write_batch(again, e.raw, e.batches))
        e.flag_relabel("CAJ", "b.jpg")            # a new one does get exported, the old one does not
        second = e.plan()
        self.assertEqual([i["item_name"] for i in second.items], ["b.jpg"])
        self.assertEqual(len(rb.list_batches(e.batches)), 1)

    def test_closed_batch_does_not_block(self):
        e = self.env
        e.put("CAJ", "a.jpg", b"A")
        e.flag_relabel("CAJ", "a.jpg")
        out = rb.write_batch(e.plan(), e.raw, e.batches)
        man = json.loads((out / "manifest.json").read_text())
        man["status"] = "applied"
        (out / "manifest.json").write_text(json.dumps(man))
        self.assertEqual(len(e.plan().items), 1)

    def test_only_relabel_decisions_and_forward_skipped(self):
        e = self.env
        e.put("CAJ", "a.jpg", b"A")
        e.put("CAJ", "k.jpg", b"K")
        e.put("CAJ", "p.jpg", b"P")
        e.flag_relabel("CAJ", "a.jpg")
        e.store.add_manual_flag(POOL, "CAJ/k.jpg", "gibberish", "x")
        e.store.set_owner_decision(POOL, "CAJ/k.jpg", "keep")
        e.store.add_manual_flag(POOL, "CAJ/p.jpg", "gibberish", "x")          # pending
        e.store.add_manual_flag(POOL, "_forward/v1.jpg", "gibberish", "x", flag_context=fs.CONTEXT_FORWARD)
        plan = e.plan()
        self.assertEqual([i["item_name"] for i in plan.items], ["a.jpg"])
        self.assertEqual(plan.skipped, [])

    def test_name_clash_same_stem(self):
        e = self.env
        e.put("CAJ", "a.jpg", b"A")
        e.put("CAJ", "a.png", b"A")
        e.flag_relabel("CAJ", "a.jpg")
        e.flag_relabel("CAJ", "a.png")
        plan = e.plan()
        self.assertEqual(len(plan.items), 1)
        self.assertEqual([s.reason for s in plan.skipped], ["name_clash"])

    def test_dry_run_writes_nothing(self):
        e = self.env
        e.put("CAJ", "a.jpg", b"A")
        e.flag_relabel("CAJ", "a.jpg")
        side_before = (e.review / "cups_flagged.json").read_bytes()
        argv = ["--raw-base-path", str(e.raw), "--review-dir", str(e.review), "--batches-dir", str(e.batches)]
        self.assertEqual(rb.main(argv + ["--dry-run"]), 0)
        self.assertEqual(rb.main(argv), 0)                    # dry-run is the default
        self.assertFalse(e.batches.exists())
        self.assertEqual((e.review / "cups_flagged.json").read_bytes(), side_before)
        self.assertEqual(rb.main(argv + ["--pool", "cups", "--execute"]), 0)
        self.assertEqual(len(rb.list_batches(e.batches)), 1)
        self.assertEqual((e.review / "cups_flagged.json").read_bytes(), side_before)   # sidecar never written

    def test_batches_dir_inside_raw_refused(self):
        e = self.env
        self.assertEqual(rb.main(["--raw-base-path", str(e.raw), "--review-dir", str(e.review),
                                  "--batches-dir", str(e.raw / "relabel")]), 2)

    def test_resolve_batch_guards(self):
        e = self.env
        e.put("CAJ", "a.jpg", b"A")
        e.flag_relabel("CAJ", "a.jpg")
        out = rb.write_batch(e.plan(), e.raw, e.batches)
        self.assertEqual(rb.resolve_batch(e.batches, POOL, out.name, e.raw), out.resolve())
        for bad in ("../cups/" + out.name, "..", ".tmp_x", "a/b"):
            with self.assertRaises(rb.RelabelBatchError):
                rb.resolve_batch(e.batches, POOL, bad, e.raw)

    def test_old_sidecar_without_relabel_fields_reads_fine(self):
        e = self.env
        e.review.mkdir(parents=True)
        (e.review / "cups_flagged.json").write_text(json.dumps({"schema_version": 1, "entries": {
            "CAJ/o.jpg": {"source": "manual_goca", "category": "gibberish", "comment": "", "signal": None,
                          "flagged_at": "2026-09-01T00:00:00+02:00", "flagged_by": "goca",
                          "owner_decision": "keep", "owner_decision_at": None, "flag_context": "retroactive_review"}}}))
        self.assertEqual(e.plan().items, [])


if __name__ == "__main__":
    unittest.main()
