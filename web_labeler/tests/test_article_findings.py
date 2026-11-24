"""KATALOG-1 ticket 12: findings about articles from new POS rows (D1).

    PYTHONPATH=web_labeler/tests /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest web_labeler.tests.test_article_findings -v
"""
import unittest

from web_labeler.catalog import article_findings as af

REAL_RENAMES = [("DUPLI ESPRESO", "DUPLI ESPRESSO"), ("EVERVES BITER 0.25", "EVERVESS BITTER 0.25"),
                ("SCHEWEPPES TONIC 0.25", "SCHWEPPES TONIC 0.25"), ("GUSTO 0.20", "GUSTO POMORANDZA 0.20"),
                ("ASPALL", "ASPALL JABUKA"), ("ZARIC KRALJICA 0.03", "ZARIC SLJIVA KRALJICA 0.03"),
                ("ZARIC MAGIJA 0.03", "ZARIC KAJSIJA MAGIJA 0.03"), ("ZARIC NOSTALGIJA 0.03", "ZARIC DUNJA NOSTALGIJA 0.03"),
                ("ZARIC KOKETA 0.03", "ZARIC VISNJA KOKETA 0.03")]


class ArticleFindingTests(unittest.TestCase):
    def known(self):
        k = af.Known()
        k.add("1313", "ZARIC KRALJICA 0.03", 200)
        k.add("0712", "MIX SOKOVI", 85)
        k.add("0100", "CASA VINA", 300)
        return k

    def test_all_real_renames_are_renames(self):
        for old, new in REAL_RENAMES:
            self.assertTrue(af.is_rename(old, new), (old, new))

    def test_different_drinks_are_not_renames(self):
        for old, new in [("ZARIC KRALJICA 0.03", "COCA COLA 0.25"), ("ASPALL", "HEINEKEN 0.33"), ("GUSTO 0.20", "RED BULL")]:
            self.assertFalse(af.is_rename(old, new), (old, new))

    def test_the_five_cases(self):
        found = {f["kind"]: f for f in af.detect(self.known(), {
            ("1313", "ZARIC KRALJICA 0.03"): [200],               # known pair: nothing
            ("1313", "ZARIC SLJIVA KRALJICA 0.03"): [200, 200],   # rename
            ("0712", "PEPSI 0.25"): [180],                        # code reused
            ("9999", " casa vina "): [300],                       # second code, known name
            ("8888", "NOVO PICE"): [400]})}                       # new article
        self.assertEqual(sorted(found), ["code_reused", "new_article", "rename", "second_code"])
        self.assertEqual(found["rename"]["old_name"], "ZARIC KRALJICA 0.03")
        self.assertEqual(found["code_reused"]["old_names"], ["MIX SOKOVI"])
        self.assertEqual(found["second_code"]["other_codes"], ["0100"])

    def test_price_jump_but_not_an_exact_double_or_half(self):
        k = self.known()
        self.assertEqual(af.detect(k, {("0712", "MIX SOKOVI"): [42.5]}), [])     # half portion
        self.assertEqual(af.detect(k, {("1313", "ZARIC KRALJICA 0.03"): [400]}), [])  # exact double
        jump = af.detect(k, {("0100", "CASA VINA"): [520]})
        self.assertEqual([(f["kind"], f["old_price"], f["new_price"]) for f in jump], [("price_jump", 300, 520)])
        self.assertEqual(af.detect(k, {("0100", "CASA VINA"): [390]}), [])  # +30% is normal


if __name__ == "__main__":
    unittest.main()
