"""LR-3 / LR-4 class picker helpers.

    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest discover -s web_labeler/tests -v
"""
import tempfile
import unittest
from pathlib import Path

from web_labeler import class_docs as cd


class ClassDocsTest(unittest.TestCase):
    def test_shots_hides_excluded_and_describes(self):
        names = ["B53", "DORUCAK-U-BLAZNAVCU", "SHOT_BLACK", "SHOT_BLURRY", "SHOT_AMBER"]
        out = cd.picker_classes("shots", names)
        got = [c["name"] for c in out]
        self.assertNotIn("DORUCAK-U-BLAZNAVCU", got)   # excluded in documented_classes
        self.assertIn("B53", got)                      # owner labels B53 although training config excludes it
        d = {c["name"]: c["description"] for c in out}
        self.assertIn("mlečna", d["SHOT_BLURRY"])
        self.assertIn("NE mutan snimak", d["SHOT_BLURRY"])
        self.assertTrue(d["SHOT_AMBER"])
        self.assertEqual(got, [n for n in names if n in got])   # model order preserved

    def test_other_models_hide_but_no_description(self):
        out = cd.picker_classes("pitchers", ["SPRICER_BELO", "BLUE_FAMILY", "LEGACY_BOKAL"])
        self.assertEqual([c["name"] for c in out], ["BLUE_FAMILY"])
        self.assertEqual(out[0]["description"], "")

    def test_missing_files_fail_soft(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(cd.hidden_classes("nope", Path(t)), set())
            self.assertEqual(cd.descriptions("shots", Path(t)), {})


if __name__ == "__main__":
    unittest.main()
