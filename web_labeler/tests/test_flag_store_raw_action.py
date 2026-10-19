"""MDQ-16a tests — owner's direct D / R / flag from RAW browse (``FlagStore.set_raw_action``).

Throw-away tmp sidecars only; never touches the real raw_review dir.

    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest discover -s web_labeler/tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flag_store as fs  # noqa: E402

POOL = "glasses"
KEY = "CASA/a.jpg"
SNAP = {"image_sha256": "i" * 64, "label_sha256": "l" * 64}
AUTO_SIGNAL = {"metric": "missing_label", "value": 0.9}


class RawActionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = fs.FlagStore(Path(self._tmp.name) / "raw_review")

    def tearDown(self):
        self._tmp.cleanup()

    def get(self, key=KEY):
        return self.store.get(POOL, key)

    # -- new decisions on a clean image ---------------------------------------

    def test_delete_on_clean_image_is_decided_at_once(self):
        e = self.store.set_raw_action(POOL, KEY, "approve_delete")["entry"]
        self.assertEqual(e["source"], fs.SOURCE_OWNER)
        self.assertEqual(e["owner_decision"], "approve_delete")
        self.assertIsNotNone(e["owner_decision_at"])
        self.assertEqual(e["category"], fs.CATEGORY_OWNER_DIRECT)
        self.assertNotIn("replaced_flag", e)
        self.assertEqual(self.get()["owner_decision"], "approve_delete")
        self.assertIsNotNone(fs.describe_reason(e))   # review queue can still show a reason

    def test_relabel_needs_snapshot_and_stores_it(self):
        with self.assertRaises(fs.FlagValidationError):
            self.store.set_raw_action(POOL, KEY, "relabel", "oznaci sve case")
        e = self.store.set_raw_action(POOL, KEY, "relabel", "oznaci sve case", relabel_snapshot=SNAP)["entry"]
        self.assertEqual(e["owner_decision"], "relabel")
        self.assertEqual(e["comment"], "oznaci sve case")
        self.assertEqual(e["relabel_snapshot"]["image_sha256"], SNAP["image_sha256"])

    def test_relabel_with_empty_comment_is_allowed(self):
        e = self.store.set_raw_action(POOL, KEY, "relabel", "", relabel_snapshot=SNAP)["entry"]
        self.assertEqual(e["comment"], "")

    def test_flag_is_pending_for_the_review_queue(self):
        e = self.store.set_raw_action(POOL, KEY, "flag", "nisam siguran")["entry"]
        self.assertEqual(e["source"], fs.SOURCE_MANUAL)
        self.assertEqual(e["category"], fs.CATEGORY_FLAG)
        self.assertEqual(e["flagged_by"], "owner")
        self.assertIsNone(e["owner_decision"])

    def test_validation(self):
        with self.assertRaises(fs.FlagValidationError):
            self.store.set_raw_action(POOL, KEY, "keep")
        with self.assertRaises(fs.FlagValidationError):
            self.store.set_raw_action(POOL, "_forward/v.jpg", "approve_delete")
        with self.assertRaises(fs.FlagValidationError):
            self.store.set_raw_action(POOL, KEY, "flag", "x" * (fs.MAX_COMMENT_LEN + 1))

    # -- changing and undoing ---------------------------------------------------

    def test_switch_delete_to_relabel_and_back(self):
        self.store.set_raw_action(POOL, KEY, "approve_delete")
        e = self.store.set_raw_action(POOL, KEY, "relabel", "bolje boks", relabel_snapshot=SNAP)["entry"]
        self.assertEqual(e["owner_decision"], "relabel")
        e = self.store.set_raw_action(POOL, KEY, "approve_delete")["entry"]
        self.assertEqual(e["owner_decision"], "approve_delete")
        self.assertNotIn("relabel_snapshot", e)

    def test_undo_on_clean_image_removes_entry(self):
        self.store.set_raw_action(POOL, KEY, "approve_delete")
        self.assertEqual(self.store.set_raw_action(POOL, KEY, None), {"entry": None})
        self.assertIsNone(self.get())

    def test_undo_without_entry_is_noop(self):
        self.assertEqual(self.store.set_raw_action(POOL, KEY, None), {"entry": None})

    def test_undo_flag(self):
        self.store.set_raw_action(POOL, KEY, "flag")
        self.store.set_raw_action(POOL, KEY, None)
        self.assertIsNone(self.get())

    def test_undo_refuses_entries_that_are_not_the_owners_raw_action(self):
        self.store.add_manual_flag(POOL, KEY, "gibberish", "goca")
        with self.assertRaises(fs.FlagConflict):
            self.store.set_raw_action(POOL, KEY, None)
        self.assertEqual(self.get()["flagged_by"], "goca")

    # -- overriding earlier flags, evidence kept --------------------------------

    def test_override_pending_auto_flag_keeps_evidence_and_undo_restores_it(self):
        self.store.add_auto_flag(POOL, KEY, fs.SOURCE_ANALYZER_AUTO, AUTO_SIGNAL, "model_pass",
                                 "FALI LABEL: model predvidja CASA")
        e = self.store.set_raw_action(POOL, KEY, "approve_delete")["entry"]
        self.assertEqual(e["replaced_flag"]["source"], fs.SOURCE_ANALYZER_AUTO)
        self.assertEqual(e["replaced_flag"]["signal"], AUTO_SIGNAL)
        # switching the decision keeps the original evidence, not the owner's previous action
        e = self.store.set_raw_action(POOL, KEY, "relabel", relabel_snapshot=SNAP)["entry"]
        self.assertEqual(e["replaced_flag"]["source"], fs.SOURCE_ANALYZER_AUTO)
        restored = self.store.set_raw_action(POOL, KEY, None)["entry"]
        self.assertEqual(restored["source"], fs.SOURCE_ANALYZER_AUTO)
        self.assertIsNone(restored["owner_decision"])
        self.assertEqual(self.get(), restored)

    def test_relabel_with_empty_comment_inherits_model_comment(self):
        self.store.add_auto_flag(POOL, KEY, fs.SOURCE_ANALYZER_AUTO, AUTO_SIGNAL, "model_pass",
                                 "FALI LABEL: model predvidja CASA")
        e = self.store.set_raw_action(POOL, KEY, "relabel", "", relabel_snapshot=SNAP)["entry"]
        self.assertEqual(e["comment"], "FALI LABEL: model predvidja CASA")
        e = self.store.set_raw_action(POOL, KEY, "relabel", "moj komentar", relabel_snapshot=SNAP)["entry"]
        self.assertEqual(e["comment"], "moj komentar")

    def test_override_goca_flag_and_undo_restores_it(self):
        self.store.add_manual_flag(POOL, KEY, "gibberish", "goca komentar")
        e = self.store.set_raw_action(POOL, KEY, "approve_delete")["entry"]
        self.assertEqual(e["replaced_flag"]["flagged_by"], "goca")
        restored = self.store.set_raw_action(POOL, KEY, None)["entry"]
        self.assertEqual(restored["comment"], "goca komentar")
        self.assertEqual(restored["source"], fs.SOURCE_MANUAL)

    def test_override_queue_decision_and_undo_restores_it(self):
        self.store.add_manual_flag(POOL, KEY, "gibberish", "")
        self.store.set_owner_decision(POOL, KEY, "keep")
        e = self.store.set_raw_action(POOL, KEY, "approve_delete")["entry"]
        self.assertEqual(e["replaced_flag"]["owner_decision"], "keep")
        restored = self.store.set_raw_action(POOL, KEY, None)["entry"]
        self.assertEqual(restored["owner_decision"], "keep")

    def test_owner_flag_over_auto_then_unflag_manual_restores_auto(self):
        self.store.add_auto_flag(POOL, KEY, fs.SOURCE_DATA4_AUDIT, AUTO_SIGNAL, "data4_audit")
        self.store.set_raw_action(POOL, KEY, "flag", "pogledaj")
        res = self.store.unflag_manual(POOL, KEY)
        self.assertTrue(res["removed"])
        self.assertEqual(self.get()["source"], fs.SOURCE_DATA4_AUDIT)

    def test_owner_flag_decided_in_queue_is_no_longer_undoable_from_raw_but_can_be_overridden(self):
        self.store.set_raw_action(POOL, KEY, "flag")
        self.store.set_owner_decision(POOL, KEY, "keep")
        with self.assertRaises(fs.FlagConflict):
            self.store.set_raw_action(POOL, KEY, None)
        e = self.store.set_raw_action(POOL, KEY, "approve_delete")["entry"]
        self.assertEqual(e["replaced_flag"]["owner_decision"], "keep")

    def test_resolved_relabel_can_get_a_fresh_decision(self):
        self.store.set_raw_action(POOL, KEY, "relabel", relabel_snapshot=SNAP)
        self.store.set_relabel_result(POOL, KEY, {"status": "applied", "batch_id": "B", "apply_run": "R"})
        # an applied entry is not "own" any more in the sense of undo: a new action wraps it
        e = self.store.set_raw_action(POOL, KEY, "approve_delete")["entry"]
        self.assertEqual(e["owner_decision"], "approve_delete")
        self.assertNotIn("relabel_result", e)

    # -- lock -------------------------------------------------------------------

    def test_lock_check_blocks_every_change(self):
        self.store.set_raw_action(POOL, KEY, "relabel", relabel_snapshot=SNAP)
        before = self.get()
        seen = []

        def lock(key, existing):
            seen.append((key, existing["owner_decision"] if existing else None))
            return "already in open relabel batch RELABEL_glasses_X"

        for action in ("approve_delete", "flag", None):
            with self.assertRaises(fs.FlagConflict) as cm:
                self.store.set_raw_action(POOL, KEY, action, lock_check=lock,
                                          relabel_snapshot=SNAP)
            self.assertIn("RELABEL_glasses_X", str(cm.exception))
        self.assertEqual(self.get(), before)
        self.assertEqual(seen[0], (KEY, "relabel"))

    def test_lock_check_none_allows(self):
        e = self.store.set_raw_action(POOL, KEY, "approve_delete", lock_check=lambda k, e: None)["entry"]
        self.assertEqual(e["owner_decision"], "approve_delete")

    # -- categories -------------------------------------------------------------

    def test_new_flag_category_and_legacy_still_accepted(self):
        self.assertEqual(self.store.add_manual_flag(POOL, "CASA/b.jpg", "flag")["category"], "flag")
        self.assertEqual(self.store.add_manual_flag(POOL, "CASA/c.jpg", "near_duplicate")["category"],
                         "near_duplicate")
        with self.assertRaises(fs.FlagValidationError):
            self.store.add_manual_flag(POOL, "CASA/d.jpg", "owner_direct")

    def test_add_manual_flag_flagged_by(self):
        self.assertEqual(self.store.add_manual_flag(POOL, KEY, "flag")["flagged_by"], "goca")
        self.assertEqual(self.store.add_manual_flag(POOL, KEY, "flag", flagged_by="owner")["flagged_by"], "owner")


if __name__ == "__main__":
    unittest.main()
