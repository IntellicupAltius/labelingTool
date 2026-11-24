"""KATALOG-1 ticket 19: new class flow (D9) on a temporary Catalog, class lists and live model.

    PYTHONPATH=web_labeler/tests /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest web_labeler.tests.test_new_class -v
"""
import unittest
from datetime import date, timedelta
from pathlib import Path

from web_labeler import catalog
from web_labeler.catalog import class_lists, store
from web_labeler.tests import test_releases
from web_labeler.yolo import load_yolo_names_from_yaml

GLASSES = """train: train/images
val: valid/images

nc: 4
names: [
  'CASA',
  'NISKA_BELO', # spaja se
  'NISKA_ZUTO',
  'LEGACY_OLD'
]
roboflow:
  workspace: x
"""




class NewClassTests(unittest.TestCase):
    setUp = test_releases.ReleaseTests.setUp
    tearDown = test_releases.ReleaseTests.tearDown
    write_live = test_releases.ReleaseTests.write_live
    write_map = test_releases.ReleaseTests.write_map
    make_settings = test_releases.ReleaseTests.make_settings
    production_state = test_releases.ReleaseTests.production_state
    identity = test_releases.ReleaseTests.identity

    def glasses(self):
        return load_yolo_names_from_yaml(self.tmp / "Models" / "Glasses.yaml")

    def ask_new_class(self, name="NEW DRINK"):
        v = self.identity(name)
        return catalog.record_decision(self.settings, "propose", "Goca", "labeler", identity_id=v["identity_id"],
                                       payload={"kind": "new_class", "change": {"note": "zelena čaša"}}, reason="nema klase")

    def open_class(self, cls="MATCHA", open_new=True, role="admin", proposal=None, name="NEW DRINK"):
        payload = {"model": "Glasses", "class": cls, "open_new": open_new}
        if proposal:
            payload["from_proposal"] = proposal["decision_id"]
        return catalog.record_decision(self.settings, "class_target", "Aleksa" if role == "admin" else "Goca", role,
                                       identity_id=self.identity(name)["identity_id"], payload=payload, reason="novo piće")

    def flow(self, name="NEW DRINK"):
        return self.identity(name)["flow"]

    def inbox_flow(self, name="NEW DRINK"):
        return next(i["flow"] for i in store.list_inbox(self.settings)["new_class"] if i["name"] == name)

    def add_raw_images(self, cls, count=2):
        folder = self.tmp / "raw" / "glasses" / "images" / cls
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (folder / f"img{i}.jpg").write_bytes(b"x")

    def promote(self, names):
        self.write_live("glasses", names + ["background"])
        catalog.import_catalog(self.settings)

    # ---- who may open a class, and where it goes ----

    def test_only_admin_opens_a_class_and_it_goes_to_the_end_of_the_tool_list(self):
        before = self.glasses()
        with self.assertRaises(catalog.NotAllowed):
            self.open_class(role="labeler")
        self.assertEqual(self.glasses(), before)
        decision = self.open_class(proposal=self.ask_new_class())
        self.assertEqual(self.glasses(), before + ["MATCHA"])
        classes = catalog.list_classes(self.settings, "Glasses")
        self.assertEqual([(c["id"], c["name"]) for c in classes], list(enumerate(before + ["MATCHA"])))
        self.assertEqual(decision["text"], "nova klasa glasses.MATCHA (broj 4 u modelu Glasses), mapira se kad je živi model nauči")
        catalog.import_catalog(self.settings)  # the next import agrees with the extended list; numbers stay
        self.assertEqual([c["name"] for c in catalog.list_classes(self.settings, "Glasses")], before + ["MATCHA"])
        backups = list((self.tmp / "db" / "class_list_backups").iterdir())
        self.assertEqual(len(backups), 1)
        self.assertEqual(load_yolo_names_from_yaml(backups[0]), before)

    def test_existing_numbers_never_change_and_comments_survive(self):
        (self.tmp / "Models" / "Glasses.yaml").write_text(GLASSES, encoding="utf-8")
        self.open_class()
        text = (self.tmp / "Models" / "Glasses.yaml").read_text(encoding="utf-8")
        self.assertIn("'NISKA_BELO', # spaja se", text)
        self.assertIn("nc: 5\n", text)
        self.assertIn("roboflow:\n  workspace: x", text)
        self.assertEqual(self.glasses(), ["CASA", "NISKA_BELO", "NISKA_ZUTO", "LEGACY_OLD", "MATCHA"])
        self.open_class("ZELENA_CASA", name="Casa Vina")
        self.assertEqual(self.glasses()[-2:], ["MATCHA", "ZELENA_CASA"])

    def test_refused_name_or_mismatch_changes_nothing(self):
        before = self.production_state()
        for bad, open_new in (("casa", True), ("NISKA_BELO_ZUTO", True), ("a", True), ("NE POSTOJI", False), ("LEGACY_OLD", False)):
            with self.assertRaises(catalog.CatalogError, msg=bad):
                self.open_class(bad, open_new=open_new)
        (self.tmp / "Models" / "Glasses.yaml").write_text("names: [CASA, NISKA_BELO]\n", encoding="utf-8")  # list differs
        with self.assertRaises(catalog.CatalogError):
            self.open_class()
        self.assertEqual(catalog.list_decisions(self.settings, "class_target")["items"], [])
        self.assertEqual(len(catalog.list_classes(self.settings, "Glasses")), 4)
        (self.tmp / "Models" / "Glasses.yaml").write_text("names: [CASA, NISKA_BELO, NISKA_ZUTO, LEGACY_OLD]\n", encoding="utf-8")
        self.assertEqual(self.production_state(), before)

    def test_file_is_put_back_when_the_catalog_cannot_save(self):
        path = self.tmp / "Models" / "Glasses.yaml"
        before = path.read_bytes()
        original = store._open_class

        def open_then_fail(settings, conn, venue, change):  # the file is written, then COMMIT fails
            written = original(settings, conn, venue, change)
            conn.execute("PRAGMA defer_foreign_keys = ON")
            conn.execute("INSERT INTO decisions (venue_id, kind, target_id, payload, actor, role, created_at, valid_from) "
                         "VALUES ('venue_a', 'revoke', 999999, '{}', 'x', 'admin', 'x', 'x')")
            return written
        store._open_class = open_then_fail
        try:
            with self.assertRaises(catalog.CatalogError):
                self.open_class()
        finally:
            store._open_class = original
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(catalog.list_classes(self.settings, "Glasses")), 4)

    # ---- the four steps ----

    def test_every_step_from_asking_to_tracked(self):
        proposal = self.ask_new_class()
        self.assertEqual(self.flow()["stage"], "waiting_class")
        self.assertEqual(self.inbox_flow()["label"], "Čeka klasu")
        self.assertEqual(store.list_inbox(self.settings)["items"], [])  # not in "waiting" any more

        self.open_class(proposal=proposal)
        self.assertEqual(catalog.get_decision(self.settings, proposal["decision_id"])["status"], "applied")
        v = self.identity("NEW DRINK")
        self.assertEqual((v["flow"]["stage"], v["flow"]["label"], v["state"]), ("labeling", "U označavanju", "pending"))
        self.assertEqual(v["mapping"], "čeka klasu glasses.MATCHA (živi model je još ne zna)")
        self.assertEqual(self.inbox_flow()["stage"], "labeling")

        self.add_raw_images("MATCHA")
        self.assertEqual((self.flow()["stage"], self.flow()["raw_images"]), ("training", 2))
        self.assertEqual(self.inbox_flow()["label"], "Čeka trening")

        self.promote(["CASA", "NISKA_BELO_ZUTO", "MATCHA"])
        v = self.identity("NEW DRINK")
        self.assertEqual((v["flow"]["stage"], v["state"], v["mapping"]), ("tracked", "tracked", "glasses.MATCHA"))
        self.assertNotIn("NEW DRINK", [i["name"] for i in store.list_inbox(self.settings)["new_class"]])
        self.assertIn("NEW DRINK", [i["name"] for i in catalog.list_articles(self.settings, state="new_class")["items"]])

    def test_existing_class_waits_for_training_and_united_names_map_to_the_model_name(self):
        self.open_class("NISKA_BELO", open_new=False)  # trained, united into NISKA_BELO_ZUTO
        self.assertEqual((self.flow()["stage"], self.identity("NEW DRINK")["mapping"]), ("tracked", "glasses.NISKA_BELO_ZUTO"))
        self.write_live("shots", ["RED"])  # AMBER is not in the live shots model
        catalog.import_catalog(self.settings)
        catalog.record_decision(self.settings, "class_target", "Aleksa", "admin", identity_id=self.identity("OMLET")["identity_id"],
                                payload={"model": "shots", "class": "AMBER"}, reason="x")
        self.assertEqual(self.flow("OMLET")["stage"], "labeling")

    def test_card_and_articles_show_the_step(self):
        self.ask_new_class()
        card = catalog.article_card(self.settings, self.identity("NEW DRINK")["article_id"])
        self.assertEqual(card["variations"][0]["flow"]["label"], "Čeka klasu")
        listed = catalog.list_articles(self.settings, state="new_class")
        self.assertEqual(([i["name"] for i in listed["items"]], listed["counts"]["new_class"]), (["NEW DRINK"], 1))
        self.open_class()
        self.assertEqual(catalog.list_articles(self.settings, state="missing")["counts"]["missing"],
                         catalog.list_articles(self.settings)["counts"]["missing"])
        self.assertNotIn("NEW DRINK", [i["name"] for i in catalog.list_articles(self.settings, state="missing")["items"]])

    # ---- releases ----

    def test_release_maps_the_article_only_when_the_live_model_knows_the_class(self):
        self.open_class()
        nxt = store.next_release(self.settings)
        self.assertTrue(all(c["ok"] for c in nxt["checks"]), nxt["checks"])  # a waiting class never blocks a release
        self.assertNotIn("NEW DRINK", " ".join(c["text"] for c in nxt["changes"]))
        self.assertTrue(nxt["can_publish"])
        self.promote(["CASA", "NISKA_BELO_ZUTO", "MATCHA"])
        nxt = store.next_release(self.settings)
        self.assertIn("NEW DRINK: počinje da se prati kao glasses.MATCHA.", [c["text"] for c in nxt["changes"]])
        self.assertTrue(all(c["ok"] for c in nxt["checks"]), nxt["checks"])
        self.write_live("glasses", ["CASA", "NISKA_BELO_ZUTO", "background"])  # model rolled back, import not run yet
        checks = {c["check"]: c for c in store.next_release(self.settings)["checks"]}
        self.assertFalse(checks["classes"]["ok"])  # the existing check still stops a map to a class the model lacks

    # ---- history and Miro/Word text ----

    def test_decision_has_miro_text_until_lock_and_can_be_revoked(self):
        decision = self.open_class()
        self.assertEqual(decision["kind_label"], "Klasa za novi artikal")
        self.assertIn("od ", decision["miro_text"])
        self.assertIn("ide u glasses.MATCHA čim je živi model nauči.", decision["miro_text"])
        self.assertIn("Nova klasa MATCHA dodata je na kraj liste modela Glasses (broj 4).", decision["miro_text"])
        card = catalog.article_card(self.settings, self.identity("NEW DRINK")["article_id"])
        self.assertEqual(card["history"][0]["decision_id"], decision["decision_id"])
        catalog.record_decision(self.settings, "revoke", "Aleksa", "admin", target_id=decision["decision_id"], reason="greška")
        self.assertIsNone(self.flow())
        self.assertEqual(self.glasses()[-1], "MATCHA")  # the class stays: classes are never removed

    def test_upgrade_from_v11_keeps_every_decision_and_number(self):
        import re
        import sqlite3
        self.ask_new_class()
        catalog.record_decision(self.settings, "ignore", "Goca", "labeler", article_id=self.identity("OMLET")["article_id"])
        db = self.tmp / "db" / "catalog.db"
        conn = sqlite3.connect(db)
        old_check = re.sub(r",\s*'class_target'\)", ")", store._SCHEMA_V12)  # the v11 table, same columns
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.executescript(f"BEGIN;DROP TABLE clip_events;{old_check}PRAGMA user_version = 11;COMMIT;")
        rows = conn.execute("SELECT * FROM decisions ORDER BY decision_id").fetchall()
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO decisions (venue_id, kind, payload, actor, role, created_at, valid_from) "
                         "VALUES ('venue_a', 'class_target', '{}', 'x', 'admin', 'x', 'x')")
        conn.close()
        with self.assertRaises(catalog.CatalogUnavailable):
            catalog.list_articles(self.settings)  # an old database is read only after a write upgraded it
        catalog.import_catalog(self.settings)
        conn = sqlite3.connect(db)
        self.assertEqual(conn.execute("SELECT * FROM decisions ORDER BY decision_id").fetchall(), rows)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], store.SCHEMA_VERSION)
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM decisions")  # still append-only
        conn.close()
        decision = self.open_class()
        self.assertEqual(decision["decision_id"], rows[-1][0] + 1)

    def test_x_ends_the_path(self):
        self.open_class()
        catalog.record_decision(self.settings, "ignore", "Goca", "labeler", article_id=self.identity("NEW DRINK")["article_id"])
        self.assertIsNone(self.flow())
        self.assertEqual(store.list_inbox(self.settings)["new_class"], [])


class AppendTextTests(unittest.TestCase):
    def test_forms(self):
        cases = {
            "names: [A, B]\n": "names: [A, B, 'NEW']\n",
            "names: ['A', 'B',\n  'C']\n": "names: ['A', 'B',\n  'C', 'NEW']\n",
            "names: [\n  'A',\n  'B' # x\n]\n": "names: [\n  'A',\n  'B', # x\n  'NEW'\n]\n",
            "names: [\n  'A',\n  'B',\n]\n": "names: [\n  'A',\n  'B',\n  'NEW',\n]\n",
            "names: []\n": "names: ['NEW']\n",
            "names: ['it''s]', B]\n": "names: ['it''s]', B, 'NEW']\n",
        }
        for old, new in cases.items():
            self.assertEqual(class_lists.appended_text(old, "NEW"), new, old)
        for bad in ("names:\n- A\n", "names: {0: A}\n"):
            with self.assertRaises(class_lists.ClassListError):
                class_lists.appended_text(bad, "NEW")


if __name__ == "__main__":
    unittest.main()
