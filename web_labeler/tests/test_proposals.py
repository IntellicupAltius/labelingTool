"""MDQ-17c tests — model proposals in a relabel batch (throw-away temp dirs; real RAW / batches never touched).

    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest discover -s web_labeler/tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as ds          # noqa: E402
import relabel_batch as rb    # noqa: E402
from test_apply_relabel import Env, ORIG, POOL, img_bytes, lbl   # noqa: E402

NAMES = ["CAJ", "CAFFE_LATTE"]


def write_props(bdir: Path, items: dict) -> None:
    (bdir / rb.PROPOSALS).write_text(json.dumps({"format_version": 1, "source": "test", "items": items}))


def prop(pid, cls="CAJ", box=(0.1, 0.1, 0.3, 0.3), **kw):
    return dict(pid=pid, class_name=cls, box=list(box), **kw)


def label_classes(p: Path):
    return sorted(int(l.split()[0]) for l in p.read_text().splitlines() if l.strip())


class ProposalFileTests(unittest.TestCase):
    def test_read_validates(self):
        with tempfile.TemporaryDirectory() as td:
            b = Path(td)
            self.assertEqual(rb.read_proposals(b), {})                       # no file = no proposals
            for bad in ([prop("p1", box=(0.3, 0.1, 0.1, 0.3))],             # x1 > x2
                        [prop("p1", box=(0.1, 0.1, 1.3, 0.3))],             # outside 0-1
                        [prop("p1"), prop("p1")],                           # duplicate pid
                        [{"pid": "p1", "box": [0.1, 0.1, 0.2, 0.2]}]):      # no class
                write_props(b, {"a.png": bad})
                with self.assertRaises(rb.RelabelBatchError):
                    rb.read_proposals(b)
            write_props(b, {"a.png": [prop("p1"), prop("p2")]})
            self.assertEqual(len(rb.read_proposals(b)["a.png"]), 2)

    def test_decisions_roundtrip_and_pending(self):
        with tempfile.TemporaryDirectory() as td:
            b = Path(td)
            write_props(b, {"a.png": [prop("p1"), prop("p2")], "b.png": [prop("p3")]})
            self.assertEqual(rb.pending_proposals(b), {"a.png": 2, "b.png": 1})
            rb.write_proposal_decisions(b, {"a.png": {"p1": "confirmed", "p2": "rejected"}})
            self.assertEqual(rb.pending_proposals(b), {"b.png": 1})
            with self.assertRaises(rb.RelabelBatchError):
                rb.write_proposal_decisions(b, {"a.png": {"p1": "maybe"}})


class SessionTests(unittest.TestCase):
    """The Dataset Fixer side: proposals load as proposal boxes, only decided ones reach labels/."""

    def _batch(self, td):
        b = Path(td) / "B"
        (b / "images").mkdir(parents=True)
        (b / "labels").mkdir()
        for n in ("a.png", "b.png"):
            (b / "images" / n).write_bytes(img_bytes("red"))
            (b / "labels" / (Path(n).stem + ".txt")).write_text(ORIG)
        write_props(b, {"a.png": [prop("p1"), prop("p2", box=(0.6, 0.6, 0.8, 0.8)),
                                  prop("p3", box=(0.05, 0.6, 0.2, 0.9)), prop("p4", box=(0.7, 0.05, 0.9, 0.2))],
                        "b.png": [prop("d1", decoy=True), prop("x1", cls="NOT_A_CLASS")]})
        return b

    def _load(self, b):
        sess = ds.load_dataset_session(b, "cups", NAMES)
        sess.relabel_items = {"a.png": {}, "b.png": {}}
        n = ds.add_proposals(sess, rb.read_proposals(b), rb.read_proposal_decisions(b), NAMES)
        return sess, n

    def test_full_cycle(self):
        with tempfile.TemporaryDirectory() as td:
            b = self._batch(td)
            sess, n = self._load(b)
            self.assertEqual(n, 5)                                   # unknown class skipped
            self.assertEqual(ds.pending_count(sess), 5)
            self.assertTrue(all("decoy" not in ds.ann_to_dict(a) for a in sess.ann_by_image[1]))
            a = {x.proposal_id: x for x in sess.ann_by_image[0] if x.proposal_id}
            a["p1"].proposal = False                                  # OK
            a["p2"].proposal = False; a["p2"].class_name, a["p2"].class_id = "CAFFE_LATTE", 1   # re-classed
            sess.ann_by_image[0].remove(a["p3"])                      # deleted; p4 left undecided
            sess.ann_by_image[1][-1].proposal = False                 # the decoy is OK'd (wrongly)

            ds.save_dataset_session(sess, "overwrite")
            dec = ds.proposal_decisions(sess)
            self.assertEqual(dec, {"a.png": {"p1": "confirmed", "p2": "reclassed", "p3": "rejected"},
                                   "b.png": {"d1": "confirmed"}})
            self.assertEqual(label_classes(b / "labels" / "a.txt"), [0, 0, 1])   # ORIG + p1 + p2, not p4
            self.assertEqual(label_classes(b / "labels" / "b.txt"), [0])         # decoy never written
            rb.write_proposal_decisions(b, dec)

            sess2, n2 = self._load(b)                                 # reopen: only p4 comes back
            self.assertEqual(n2, 1)
            self.assertEqual([x.proposal_id for x in sess2.ann_by_image[0] if x.proposal], ["p4"])
            self.assertEqual(len(sess2.ann_by_image[0]), 4)           # 3 saved boxes + p4, no duplicates


class ApplyPendingTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        import ingestion_pipeline.distributor as d
        self._raw_base = d.RAW_BASE
        self.e = Env(Path(self._td.name))

    def tearDown(self):
        import ingestion_pipeline.distributor as d
        d.RAW_BASE = self._raw_base
        self._td.cleanup()

    def test_pending_proposal_blocks_the_image_only(self):
        e = self.e
        e.put("CAJ", "a.png", img_bytes("red"), ORIG)
        e.put("CAJ", "b.png", img_bytes("blue"), ORIG)
        e.flag("CAJ", "a.png"); e.flag("CAJ", "b.png")
        e.export()
        write_props(e.batch, {"a.png": [prop("p1")]})
        added = ORIG + "0 0.200000 0.200000 0.200000 0.200000\n"
        e.correct("a.png", added)           # as if p1 were OK'd but the decision never saved
        e.correct("b.png", added)
        res, out = e.run()
        self.assertEqual((res["applied"], res["skipped"]), (1, 1))
        self.assertIn("pending_proposals", out)
        self.assertEqual(lbl(e, "a.png"), ORIG)                      # RAW untouched for the pending image
        self.assertEqual(lbl(e, "b.png"), added)
        rb.write_proposal_decisions(e.batch, {"a.png": {"p1": "confirmed"}})
        res, _ = e.run(stamp="20261005_110000_relabel")
        self.assertEqual(res["applied"], 1)
        self.assertEqual(lbl(e, "a.png"), added)

    def test_only_pending_and_no_boxes_is_not_deleted(self):
        e = self.e
        e.put("CAJ", "a.png", img_bytes("red"), ORIG)
        e.flag("CAJ", "a.png")
        e.export()
        write_props(e.batch, {"a.png": [prop("p1")]})
        e.correct("a.png", "")              # 0 boxes would be the default Delete — but a proposal is pending
        res, _ = e.run()
        self.assertEqual((res["deleted"], res["skipped"]), (0, 1))
        self.assertTrue((e.raw / POOL / "images" / "CAJ" / "a.png").is_file())


if __name__ == "__main__":
    unittest.main()
