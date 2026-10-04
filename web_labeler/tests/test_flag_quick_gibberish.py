"""X-X quick gibberish flag: static wiring checks on app.js (the JS has no module harness).

The shortcut must go through the same saveFlag() path as the Gibberish button, with an empty comment,
and must not fire while typing in the comment field.

    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest discover -s web_labeler/tests -v
"""
import re
import shutil
import subprocess
import unittest
from pathlib import Path

APP_JS = Path(__file__).resolve().parents[1] / "static" / "app.js"


def _src():
    return APP_JS.read_text(encoding="utf-8")


def _panel_branch():
    s = _src()
    i = s.index("if (state.flagPanelOpen) {")
    return s[i:s.index("return;", i)]


class QuickGibberishWiring(unittest.TestCase):
    def test_x_in_open_panel_saves_gibberish_without_comment(self):
        m = re.search(r'e\.key === "x" \|\| e\.key === "X"\) && ([^{]*)\{(.*?)\n      \}', _panel_branch(), re.S)
        self.assertTrue(m, "no X branch in the flag panel hotkeys")
        cond, body = m.groups()
        self.assertIn("!typing", cond)
        self.assertIn("!e.repeat", cond)
        self.assertIn('saveFlag({category: "gibberish", comment: ""})', body)

    def test_gibberish_category_shows_shortcut(self):
        s = _src()
        self.assertRegex(s, r'key: "gibberish", label: "Gibberish", shortcut: "X"')
        self.assertIn('[${c.shortcut}]', s)

    def test_other_hotkeys_untouched(self):
        b = _panel_branch()
        for frag in ('e.key === "Escape"', 'e.key === "Enter"', "String(i + 1)"):
            self.assertIn(frag, b)

    def test_node_syntax(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        self.assertEqual(subprocess.run([node, "--check", str(APP_JS)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
