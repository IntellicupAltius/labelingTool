"""RAW-browse D / R / X keys and the X X quick flag (MDQ-16c): static wiring checks on app.js (the JS has
no module harness). Was the MDQ-3 "X X = quick gibberish" test; categories are retired since MDQ-16.

    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest discover -s web_labeler/tests -v
"""
import re
import shutil
import subprocess
import unittest
from pathlib import Path

APP_JS = Path(__file__).resolve().parents[1] / "static" / "app.js"
INDEX = Path(__file__).resolve().parents[1] / "static" / "index.html"


def _src():
    return APP_JS.read_text(encoding="utf-8")


def _panel_branch():
    s = _src()
    i = s.index("if (state.flagPanelOpen) {")
    return s[i:s.index("return;", i)]


class FlagPanelWiring(unittest.TestCase):
    def test_x_in_open_panel_saves_flag_without_comment(self):
        m = re.search(r'e\.key === "x" \|\| e\.key === "X"\) && (.*?)\{(.*?)\n      \}', _panel_branch(), re.S)
        self.assertTrue(m, "no X branch in the flag panel hotkeys")
        cond, body = m.groups()
        self.assertIn('state.flagMode === "flag"', cond)       # never in the R (relabel) panel
        self.assertIn('!$("flagComment").value', cond)        # a started comment keeps X as a character
        self.assertIn("!e.repeat", cond)
        self.assertIn('saveFlag({comment: ""})', body)

    def test_enter_saves_shift_enter_is_newline(self):
        self.assertIn('e.key === "Enter" && !e.shiftKey', _panel_branch())
        self.assertIn('e.key === "Escape"', _panel_branch())

    def test_categories_are_gone(self):
        s = _src()
        for gone in ("FLAG_CATEGORIES", "gibberish", "selectFlagCategory"):
            self.assertNotIn(gone, s)
        self.assertNotIn("flagCats", INDEX.read_text(encoding="utf-8"))

    def test_raw_keys_route_to_owner_actions(self):
        s = _src()
        i = s.index("MDQ-16c — RAW browse")
        branch = s[i:s.index("return;", i)]
        self.assertIn("state.rawLoaded", branch)
        self.assertIn("!isTypingTarget(e.target)", branch)
        self.assertIn('k === "d" ? "approve_delete" : k === "r" ? "relabel" : "flag"', branch)
        # RAW branch must come before the generic X (video) branch
        self.assertLess(i, s.index("// X = flag the current video frame"))

    def test_owner_key_undo_lock_and_advance(self):
        s = _src()
        i = s.index("async function ownerRawKey(kind)")
        body = s[i:s.index("\n}\n", i)]
        self.assertIn("st.locked", body)
        self.assertIn("current === kind", body)
        self.assertIn("rawAction(null)", body)
        self.assertIn("renderRawImage(state.datasetImageIdx + 1)", body)

    def test_api_route(self):
        self.assertIn('"/api/raw/action"', _src())

    def test_node_syntax(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        self.assertEqual(subprocess.run([node, "--check", str(APP_JS)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
