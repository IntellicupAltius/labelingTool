"""KATALOG-1 ticket 01: class import and the Catalog API, on a temporary app and database.

    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest web_labeler.tests.test_catalog -v
"""
import csv
import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from web_labeler import catalog, server
from web_labeler.catalog import store
from web_labeler.catalog import miro
from web_labeler.yolo import load_yolo_names_from_yaml


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.models = self.tmp / "data" / "Models"
        self.models.mkdir(parents=True)
        (self.models / "Nested").mkdir()
        self.write("Glasses", ["CASA", "NISKA"])
        self.write("Shots", {0: "B53", 1: "SHOT_RED"})
        self.write("Nested/Ignored", ["NOT_A_MODEL"])
        self.db = self.tmp / "catalog_data" / "catalog.db"
        self.settings = self.make_settings()

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, names):
        (self.models / f"{name}.yaml").write_text(json.dumps({"names": names}), encoding="utf-8")

    def make_settings(self, venue="venue_a", db=None):
        return catalog.CatalogSettings.from_config(
            {"catalog_venue_id": venue, "catalog_venue_name": "Test", "catalog_db_path": str(db or self.db)},
            self.models, self.tmp / "raw")

    def source_hashes(self):
        return {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                for p in self.models.rglob("*.yaml")}

    def routes(self, extra=None):
        cfg = {"data_root": str(self.tmp / "data"), "catalog_venue_id": "venue_a", "catalog_db_path": str(self.db),
               "raw_review_dir": str(self.tmp / "raw_review"), "relabel_batches_dir": str(self.tmp / "batches")}
        cfg.update(extra or {})
        path = self.tmp / "labeler_config.json"
        path.write_text(json.dumps(cfg))
        old = os.environ.get("LABELER_CONFIG_PATH")
        os.environ["LABELER_CONFIG_PATH"] = str(path)
        try:
            app = server.create_app()
        finally:
            if old is None:
                os.environ.pop("LABELER_CONFIG_PATH", None)
            else:
                os.environ["LABELER_CONFIG_PATH"] = old
        return {(r.path, m): r.endpoint for r in app.routes for m in getattr(r, "methods", ())}

    def test_import_matches_tool_lists_through_api(self):
        catalog.import_class_lists(self.settings)
        routes = self.routes()
        models = routes[("/api/catalog/models", "GET")]()["models"]
        self.assertEqual(models, [{"name": "Glasses", "class_count": 2}, {"name": "Shots", "class_count": 2}])
        for model in ("Glasses", "Shots"):
            rows = routes[("/api/catalog/models/{model}/classes", "GET")](model)["classes"]
            self.assertEqual([r["id"] for r in rows], [0, 1])
            self.assertEqual([r["name"] for r in rows], load_yolo_names_from_yaml(self.models / f"{model}.yaml"))

    def test_repeat_import_adds_nothing(self):
        catalog.import_class_lists(self.settings)
        before = catalog.list_classes(self.settings, "Glasses")
        self.assertEqual(catalog.import_class_lists(self.settings)["added"], 0)
        self.assertEqual(catalog.list_classes(self.settings, "Glasses"), before)

    def test_new_class_is_appended(self):
        catalog.import_class_lists(self.settings)
        self.write("Glasses", ["CASA", "NISKA", "NOVA"])
        self.assertEqual(catalog.import_class_lists(self.settings)["added"], 1)
        self.assertEqual([(c["id"], c["name"]) for c in catalog.list_classes(self.settings, "Glasses")],
                         [(0, "CASA"), (1, "NISKA"), (2, "NOVA")])

    def test_changed_or_removed_number_refuses_whole_import(self):
        catalog.import_class_lists(self.settings)
        before = catalog.list_classes(self.settings, "Glasses")
        for changed in (["NISKA", "CASA"], ["CASA"]):
            self.write("Glasses", changed)
            self.write("Shots", ["B53", "SHOT_RED", "MUST_NOT_ENTER"])
            with self.assertRaises(catalog.CatalogError):
                catalog.import_class_lists(self.settings)
            self.assertEqual(catalog.list_classes(self.settings, "Glasses"), before)
            self.assertEqual(len(catalog.list_classes(self.settings, "Shots")), 2)

    def test_bad_source_writes_nothing(self):
        (self.models / "Broken.yaml").write_text("train: x\n", encoding="utf-8")
        with self.assertRaises(catalog.CatalogError):
            catalog.import_class_lists(self.settings)
        self.assertFalse(self.db.exists())

    def test_import_only_reads_sources(self):
        before = self.source_hashes()
        catalog.import_class_lists(self.settings)
        self.assertEqual(self.source_hashes(), before)
        self.assertEqual(sorted(p.name for p in self.models.iterdir()), ["Glasses.yaml", "Nested", "Shots.yaml"])

    def test_database_refused_inside_raw_or_sources(self):
        for bad in (self.tmp / "raw" / "catalog.db", self.models / "catalog.db"):
            with self.assertRaises(catalog.CatalogError):
                catalog.import_class_lists(self.make_settings(db=bad))
            self.assertFalse(bad.exists())

    def test_records_are_separated_by_venue(self):
        catalog.import_class_lists(self.settings)
        other = self.make_settings(venue="venue_b")
        self.assertEqual(catalog.list_models(other), [])
        self.write("Glasses", ["DRUGA", "CASA"])
        catalog.import_class_lists(other)
        self.assertEqual(catalog.list_classes(other, "Glasses")[0]["name"], "DRUGA")
        self.assertEqual(catalog.list_classes(self.settings, "Glasses")[0]["name"], "CASA")

    def test_missing_database_gives_clear_message_and_is_not_created(self):
        routes = self.routes()
        status = routes[("/api/catalog/status", "GET")]()
        self.assertFalse(status["available"])
        self.assertIn("nije napravljen", status["message"])
        with self.assertRaises(HTTPException) as e:
            routes[("/api/catalog/models", "GET")]()
        self.assertEqual(e.exception.status_code, 503)
        self.assertFalse(self.db.exists())
        self.assertIn(("/api/config", "GET"), routes)
        self.assertEqual(routes[("/api/config", "GET")]()["models"][0]["name"], "Glasses")

    def test_corrupt_database_gives_clear_message(self):
        self.db.parent.mkdir(parents=True)
        self.db.write_bytes(b"this is not a database" * 100)
        routes = self.routes()
        status = routes[("/api/catalog/status", "GET")]()
        self.assertFalse(status["available"])
        self.assertIn("ne može da se pročita", status["message"])
        self.assertNotIn(str(self.tmp), status["message"])
        with self.assertRaises(catalog.CatalogError):
            catalog.import_class_lists(self.settings)

    def test_unconfigured_venue_gives_clear_message(self):
        settings = catalog.CatalogSettings.from_config({"data_root": str(self.tmp)}, self.models)
        self.assertIn("nije podešen", catalog.catalog_status(settings)["message"])

    def test_unknown_model_is_404(self):
        catalog.import_class_lists(self.settings)
        with self.assertRaises(HTTPException) as e:
            self.routes()[("/api/catalog/models/{model}/classes", "GET")]("Nope")
        self.assertEqual(e.exception.status_code, 404)


def pos_row(code, name, when, qty=1, price=100, paid=True, table="1"):
    row = [""] * 42
    row[16], row[18], row[22], row[23], row[24] = when, table, "NENAPLACENO" if not paid else when, code, name
    row[25], row[26], row[27] = f"{qty:.2f}", f"{price:,.2f}", f"{qty * price:,.2f}"
    return row


class AssortmentFixture(unittest.TestCase):
    """Shared POS, map and class lists; holds no tests itself."""


    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.models = self.tmp / "Models"
        self.models.mkdir()
        (self.models / "Glasses.yaml").write_text(json.dumps({"names": ["CASA", "NISKA"]}), encoding="utf-8")
        (self.models / "Shots.yaml").write_text(json.dumps({"names": ["AMBER", "RED"]}), encoding="utf-8")
        self.pos = self.tmp / "pos"
        self.pos.mkdir()
        self.map_path = self.tmp / "map.yaml"
        self.blacklist = self.tmp / "blacklist.txt"
        self.blacklist.write_text("# food\nomlet\n", encoding="utf-8")
        self.write_map([
            {"pos_name": "DRINK", "status": "incl", "mapping_mode": "any", "provisional": "not sure",
             "mappings": [{"model": "glasses", "class": "CASA", "priority": 1},
                          {"model": "glasses", "class": "NISKA", "priority": 2}]},
            {"pos_name": "OLD NAME", "status": "incl", "mapping_mode": "fallback",
             "mappings": [{"model": "shots", "class": "AMBER", "priority": 1}]},
            {"pos_name": "NEW NAME", "status": "incl", "mapping_mode": "fallback",
             "mappings": [{"model": "shots", "class": "RED", "priority": 1}]},
            {"pos_name": "MAIN", "aliases": ["OTHER DRINK"], "status": "incl", "mapping_mode": "fallback",
             "mappings": [{"model": "glasses", "class": "CASA", "priority": 1}]},
            {"pos_name": "NEVER SOLD", "status": "incl", "mapping_mode": "fallback",
             "mappings": [{"model": "glasses", "class": "UNITED_CLASS", "priority": 1}]},
        ])
        self.write_pos("a.csv", [
            pos_row("0706", " drink ", "2026.11.20 10:00"),
            pos_row("1314", "OLD NAME", "2026.05.01 10:00"),
            pos_row("1314", "NEW NAME", "2026.11.20 11:00"),
            pos_row("23168", "omlet", "2026.11.20 09:00"),
            pos_row("23500", "omlet", "2026.11.20 09:30"),
            pos_row("0001", "MAIN", "2026.11.20 12:00"),
            pos_row("0002", "OTHER DRINK", "2026.11.20 12:30"),
            pos_row("5000", "NEW BURGER", "2026.11.19 12:00"),
            pos_row("5001", "OLD SNACK", "2026.05.01 12:00"),
            pos_row("6000", "ONLY FREE", "2026.11.20 13:00", paid=False),
            pos_row("7000", "CANCELLED", "2026.11.20 14:00"),
            pos_row("7000", "CANCELLED", "2026.11.20 14:05", qty=-1),
        ])
        self.settings = self.make_settings()

    def tearDown(self):
        self._tmp.cleanup()

    def write_map(self, articles):
        self.map_path.write_text(json.dumps({"articles": articles}), encoding="utf-8")

    def write_pos(self, name, rows):
        with open(self.pos / name, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)

    def make_settings(self, **extra):
        cfg = {"catalog_venue_id": "venue_a", "catalog_db_path": str(self.tmp / "db" / "catalog.db"),
               "catalog_pos_dir": str(self.pos), "catalog_pos_adapter": "blaznavac_csv",
               "article_map_path": str(self.map_path), "catalog_blacklist_path": str(self.blacklist)}
        cfg.update(extra)
        return catalog.CatalogSettings.from_config(cfg, self.models, self.tmp / "raw")

    def variation(self, model, cls, name):
        return next(v for v in catalog.list_variations(self.settings, model, cls) if v["name"] == name)

    def unclassified(self, state):
        groups = catalog.list_unclassified(self.settings)["groups"]
        return next(g["items"] for g in groups if g["state"] == state)


class AssortmentImportTests(AssortmentFixture):
    def test_leading_zero_stays(self):
        catalog.import_catalog(self.settings)
        self.assertEqual(self.variation("Glasses", "CASA", "drink")["code"], "0706")

    def test_two_codes_with_one_name_are_one_article(self):
        catalog.import_catalog(self.settings)
        items = [i for i in self.unclassified("ignored") if i["name"] == "omlet"]
        self.assertEqual(sorted(i["code"] for i in items), ["23168", "23500"])
        self.assertEqual(len({i["article_id"] for i in items}), 1)

    def test_rename_is_one_article_and_each_name_keeps_its_class(self):
        report = catalog.import_catalog(self.settings)["assortment"]
        old = self.variation("Shots", "AMBER", "OLD NAME")
        new = self.variation("Shots", "RED", "NEW NAME")
        self.assertEqual(old["article_id"], new["article_id"])
        self.assertEqual([x["code"] for x in report["same_code_different_classes"]], ["1314"])

    def test_map_alias_does_not_merge_different_pos_articles(self):
        catalog.import_catalog(self.settings)
        self.assertNotEqual(self.variation("Glasses", "CASA", "MAIN")["article_id"],
                            self.variation("Glasses", "CASA", "OTHER DRINK")["article_id"])

    def test_row_in_two_files_counts_once_but_real_repeat_counts_twice(self):
        row = pos_row("8000", "REPEAT", "2026.11.20 15:00")
        self.write_pos("b.csv", [row, row])
        self.write_pos("c.csv", [row])
        report = catalog.import_catalog(self.settings)["assortment"]
        self.assertEqual(next(i for i in self.unclassified("pending") if i["code"] == "8000")["sold"], 2)
        self.assertEqual(report["duplicate_rows_removed"], 1)

    def test_unpaid_rows_do_not_count(self):
        report = catalog.import_catalog(self.settings)["assortment"]
        self.assertEqual(report["unpaid_rows_skipped"], 1)
        self.assertFalse(any(i["code"] == "6000" for s in ("pending", "dormant", "ignored") for i in self.unclassified(s)))

    def test_cancelled_sale_nets_to_zero_and_is_marked(self):
        catalog.import_catalog(self.settings)
        item = next(i for i in self.unclassified("pending") if i["code"] == "7000")
        self.assertEqual(item["sold"], 0)
        self.assertEqual(item["last_sale"], "2026-11-20 14:00:00")
        self.assertIn("sve stornirano", item["tags"])

    def test_initial_states_flags_and_report(self):
        report = catalog.import_catalog(self.settings)["assortment"]
        self.assertEqual(report["counts"], {"tracked": 5, "ignored": 2, "pending": 2, "dormant": 1})
        self.assertEqual(sorted(i["code"] for i in report["unclassified"]), ["5000", "5001", "7000"])
        drink = self.variation("Glasses", "CASA", "drink")
        self.assertEqual(drink["mapping_mode"], "any")
        self.assertEqual(drink["classes"], ["glasses.CASA", "glasses.NISKA"])
        self.assertIn("nije provereno na snimku", drink["tags"])
        self.assertIn("privremeno", drink["tags"])
        self.assertNotIn("privremeno", self.variation("Glasses", "CASA", "MAIN")["tags"])

    def test_map_class_outside_tool_list_is_kept_and_shown_apart(self):
        catalog.import_catalog(self.settings)
        view = catalog.list_model_view(self.settings, "Glasses")
        self.assertEqual(view["map_only_classes"], [{"name": "UNITED_CLASS", "variation_count": 1}])
        self.assertEqual([c["name"] for c in view["classes"]], ["CASA", "NISKA"])
        self.assertIn("nije prodavano", self.variation("Glasses", "UNITED_CLASS", "NEVER SOLD")["tags"])

    def test_repeat_import_keeps_article_numbers(self):
        first = catalog.import_catalog(self.settings)["assortment"]
        before = self.variation("Shots", "RED", "NEW NAME")
        second = catalog.import_catalog(self.settings)["assortment"]
        self.assertEqual(self.variation("Shots", "RED", "NEW NAME"), before)
        self.assertEqual(first["counts"], second["counts"])

    def test_name_from_map_keeps_article_when_first_sold(self):
        catalog.import_catalog(self.settings)
        before = self.variation("Glasses", "UNITED_CLASS", "NEVER SOLD")["article_id"]
        self.write_pos("b.csv", [pos_row("9000", "NEVER SOLD", "2026.11.21 10:00")])
        catalog.import_catalog(self.settings)
        after = self.variation("Glasses", "UNITED_CLASS", "NEVER SOLD")
        self.assertEqual((after["article_id"], after["code"]), (before, "9000"))

    def test_newer_report_difference_is_explained(self):
        baseline = {"latest_sale": "2026-11-20", "counts": {"tracked": 5, "ignored": 2, "pending": 2, "dormant": 1}}
        self.write_pos("b.csv", [pos_row("9100", "BRAND NEW", "2026.11.22 10:00")])
        result = catalog.import_catalog(self.make_settings(catalog_import_baseline=baseline))["assortment"]["baseline"]
        self.assertEqual(result["unexplained"], {})
        self.assertIn({"code": "9100", "name": "BRAND NEW", "from": None, "to": "pending"},
                      result["explained_by_newer_reports"])

    def test_broken_report_changes_nothing(self):
        (self.pos / "broken.csv").write_bytes(b"\xff\xfe\x00broken")
        with self.assertRaises(catalog.CatalogError):
            catalog.import_catalog(self.settings)
        self.assertFalse((self.tmp / "db" / "catalog.db").exists())

    def test_import_only_reads_its_sources(self):
        paths = [*self.pos.iterdir(), self.map_path, self.blacklist, *self.models.iterdir()]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        catalog.import_catalog(self.settings)
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}, before)

    def test_first_version_database_is_upgraded(self):
        db = self.tmp / "db" / "catalog.db"
        db.parent.mkdir()
        conn = sqlite3.connect(str(db))
        conn.executescript(store._SCHEMA + "PRAGMA user_version = 1;")
        conn.close()
        catalog.import_catalog(self.settings)
        self.assertEqual(len(catalog.list_classes(self.settings, "Glasses")), 2)
        self.assertEqual(self.variation("Glasses", "CASA", "drink")["code"], "0706")


class ClassRulesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        for name in ("Models", "pos", "training", "documented", "cards", "live/glasses", "word/2026-09-25/text"):
            (self.tmp / name).mkdir(parents=True)
        (self.tmp / "Models" / "Glasses.yaml").write_text(json.dumps(
            {"names": ["NISKA_BELO", "NISKA_ZUTO", "LEGACY_OLD", "BAZA", "BEZ_OPISA", "NEMA_SLIKA"]}), encoding="utf-8")
        self.write_training(exclude=["BAZA"])
        self.write_live(["NISKA_BELO_ZUTO", "BEZ_OPISA", "background"])
        self.write_documented({"excluded": [{"class": "BAZA", "reason": "van menija"}],
                               "unite": [{"into": "NISKA_BELO_ZUTO", "classes": ["NISKA_BELO", "NISKA_ZUTO"]}]})
        (self.tmp / "cards" / "kartice_glasses_v1.md").write_text(
            "Status: NACRT\n\n### OSTALO\n\n### NISKA_ZUTO\n- **Prepoznaje se:** niska zuta (DATA-4 test)\n"
            "  casa.\n- **Meša se sa:** NISKA_BELO.\n- Slika: `img/x.jpg`\n\n### NISKA_BELO\n"
            "- **Prepoznaje se:** bela.\n\n### NEMA_SLIKA\n- **Prepoznaje se:** retka.\n", encoding="utf-8")
        (self.tmp / "word" / "2026-09-25" / "text" / "overview_Glasses.txt").write_text(
            "NISKA BELO NISKA ZUTO LEGACY OLD BAZA NEMA SLIKA Word kaze drugacije", encoding="utf-8")
        (self.tmp / "rules.md").write_text("| Klasa | Kada |\n|---|---|\n| NISKA_ZUTO | jarko zuto |\n", encoding="utf-8")
        row = pos_row("0100", "SOLD DRINK", "2026.11.20 10:00")
        with open(self.tmp / "pos" / "a.csv", "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)
        (self.tmp / "blacklist.txt").write_text("", encoding="utf-8")
        self.write_map("NISKA_BELO_ZUTO")
        self.settings = self.make_settings()

    def tearDown(self):
        self._tmp.cleanup()

    def write_training(self, exclude):
        (self.tmp / "training" / "glasses.yaml").write_text(json.dumps({"classes": {
            "exclude": exclude, "exclude_prefixes": ["LEGACY_"],
            "unite": [{"classes": ["NISKA_BELO", "NISKA_ZUTO"], "as": "NISKA_BELO_ZUTO"}]}}), encoding="utf-8")

    def write_live(self, names):
        (self.tmp / "live" / "glasses" / "active_classes.yaml").write_text(json.dumps({"names": names}), encoding="utf-8")

    def write_documented(self, data):
        (self.tmp / "documented" / "glasses.yaml").write_text(json.dumps(data), encoding="utf-8")

    def write_map(self, cls):
        (self.tmp / "map.yaml").write_text(json.dumps({"articles": [
            {"pos_name": "SOLD DRINK", "status": "incl", "mapping_mode": "fallback",
             "mappings": [{"model": "glasses", "class": cls, "priority": 1}]}]}), encoding="utf-8")

    def make_settings(self, **extra):
        t = self.tmp
        cfg = {"catalog_venue_id": "venue_a", "catalog_db_path": str(t / "db" / "catalog.db"),
               "catalog_pos_dir": str(t / "pos"), "catalog_pos_adapter": "blaznavac_csv",
               "article_map_path": str(t / "map.yaml"), "catalog_blacklist_path": str(t / "blacklist.txt"),
               "catalog_training_config_dir": str(t / "training"), "catalog_documented_classes_dir": str(t / "documented"),
               "catalog_cards_dir": str(t / "cards"), "catalog_word_exports_dir": str(t / "word"),
               "catalog_live_models_dir": str(t / "live"),
               "catalog_rule_files": {"glasses": {"label": "Pravila v2", "path": str(t / "rules.md")}}}
        cfg.update(extra)
        return catalog.CatalogSettings.from_config(cfg, t / "Models", t / "raw")

    def titles(self, settings=None):
        return [f["title"] for f in catalog.source_report(settings or self.settings)["findings"]]

    def test_united_class_shows_drawn_and_model_names_and_pos_articles(self):
        catalog.import_catalog(self.settings)
        detail = catalog.class_detail(self.settings, "Glasses", "NISKA_ZUTO")
        self.assertEqual(detail["trained_name"], "NISKA_BELO_ZUTO")
        self.assertEqual(detail["drawn_as"], ["NISKA_BELO", "NISKA_ZUTO"])
        self.assertEqual([v["name"] for v in detail["variations"]], ["SOLD DRINK"])
        self.assertEqual(detail["variations"][0]["via_class"], "NISKA_BELO_ZUTO")
        self.assertEqual(catalog.list_model_view(self.settings, "Glasses")["map_only_classes"], [])

    def test_excluded_classes_by_name_and_prefix(self):
        catalog.import_catalog(self.settings)
        by_name = {c["name"]: c for c in catalog.list_model_view(self.settings, "Glasses")["classes"]}
        for name in ("BAZA", "LEGACY_OLD"):
            self.assertEqual(by_name[name]["training_state"], "excluded")
            self.assertIsNone(by_name[name]["trained_name"])

    def test_class_not_excluded_but_missing_from_live_model(self):
        catalog.import_catalog(self.settings)
        by_name = {c["name"]: c for c in catalog.list_model_view(self.settings, "Glasses")["classes"]}
        self.assertEqual(by_name["NEMA_SLIKA"]["training_state"], "not_in_model")
        self.assertIn("NEMA_SLIKA: nije izbačena, a model je nema", self.titles())

    def test_cards_are_read_per_class_and_win_over_word(self):
        catalog.import_catalog(self.settings)
        detail = catalog.class_detail(self.settings, "Glasses", "NISKA_ZUTO")
        sections = {s["kind"]: s["items"] for s in detail["rules"]}
        self.assertEqual(sections["rule"][0]["text"], "jarko zuto")
        self.assertEqual(sections["recognize"][0]["text"], "niska zuta casa.")
        self.assertEqual(sections["recognize"][0]["source"], "Kartica Glasses (nacrt)")
        self.assertEqual(sections["confused"][0]["text"], "NISKA_BELO.")
        self.assertNotIn("Word kaze", json.dumps(detail, ensure_ascii=False))
        self.assertNotIn("img/", json.dumps(detail, ensure_ascii=False))

    def test_class_without_description_says_so(self):
        catalog.import_catalog(self.settings)
        self.assertFalse(catalog.class_detail(self.settings, "Glasses", "BEZ_OPISA")["has_description"])
        self.assertIn("BEZ_OPISA: nema kartice", self.titles())
        self.assertIn("BEZ_OPISA: Word je ne pominje", self.titles())

    def test_section_heading_is_not_reported_as_a_card(self):
        catalog.import_catalog(self.settings)
        self.assertFalse(any("OSTALO" in t for t in self.titles()))

    def test_sources_that_disagree_are_reported_with_each_source(self):
        self.write_training(exclude=[])
        self.write_documented({"excluded": [{"class": "BAZA", "reason": "van menija"}],
                               "unite": [{"into": "DRUGO_IME", "classes": ["NISKA_BELO", "NISKA_ZUTO"]}]})
        catalog.import_catalog(self.settings)
        findings = {f["title"]: f for f in catalog.source_report(self.settings)["findings"]}
        excluded = findings["BAZA: dokumentacija je izbacuje, trening ne"]
        self.assertEqual({s["says"] for s in excluded["sources"]}, {"van menija", "ulazi u trening"})
        unite = findings["Spajanje u NISKA_BELO_ZUTO nije isto zapisano"]
        self.assertTrue(any("DRUGO_IME" in s["says"] for s in unite["sources"]))

    def test_matching_sources_report_no_merge_difference(self):
        catalog.import_catalog(self.settings)
        self.assertFalse(any(t.startswith("Spajanje") for t in self.titles()))
        self.assertFalse(any(t.startswith("BAZA:") for t in self.titles()))

    def test_map_class_missing_from_model_is_important_only_when_sold(self):
        self.write_map("STARA_KLASA")
        catalog.import_catalog(self.settings)
        finding = next(f for f in catalog.source_report(self.settings)["findings"] if "STARA_KLASA" in f["title"])
        self.assertEqual(finding["severity"], "high")
        (self.tmp / "map.yaml").write_text(json.dumps({"articles": [
            {"pos_name": "NEVER SOLD", "status": "miss", "mappings": [{"model": "glasses", "class": "STARA_KLASA"}]}]}))
        catalog.import_catalog(self.settings)
        finding = next(f for f in catalog.source_report(self.settings)["findings"] if "STARA_KLASA" in f["title"])
        self.assertEqual(finding["severity"], "info")

    def test_known_conflict_is_reported_and_flags_a_changed_source(self):
        evidence = self.tmp / "rule.txt"
        evidence.write_text("flase u kibli se nikad ne oznacavaju", encoding="utf-8")
        known = self.tmp / "known.yaml"
        known.write_text(json.dumps({"conflicts": [{"model": "glasses", "class": "BAZA", "title": "Kibla", "text": "Protivrecno.",
            "sources": [{"label": "Odluka", "says": "da"},
                        {"label": "Kartica", "says": "ne", "file": str(evidence), "phrase": "nikad ne oznacavaju"}]}]}))
        settings = self.make_settings(catalog_known_conflicts_path=str(known))
        catalog.import_catalog(settings)
        finding = next(f for f in catalog.source_report(settings)["findings"] if f["title"] == "Kibla")
        self.assertEqual((finding["severity"], finding["text"]), ("high", "Protivrecno."))
        self.assertEqual([s["label"] for s in finding["sources"]], ["Odluka", "Kartica"])
        evidence.write_text("sada se oznacavaju", encoding="utf-8")
        catalog.import_catalog(settings)
        finding = next(f for f in catalog.source_report(settings)["findings"] if f["title"] == "Kibla")
        self.assertIn("promenio u izvoru: Kartica", finding["text"])

    def test_repeat_import_does_not_duplicate_rules_or_findings(self):
        catalog.import_catalog(self.settings)
        before = catalog.source_report(self.settings), catalog.class_detail(self.settings, "Glasses", "NISKA_ZUTO")
        catalog.import_catalog(self.settings)
        self.assertEqual((catalog.source_report(self.settings), catalog.class_detail(self.settings, "Glasses", "NISKA_ZUTO")), before)

    def test_import_only_reads_rule_sources(self):
        paths = [p for p in self.tmp.rglob("*") if p.is_file()]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        catalog.import_catalog(self.settings)
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}, before)

    def test_second_version_database_is_upgraded(self):
        db = self.tmp / "db" / "catalog.db"
        db.parent.mkdir()
        conn = sqlite3.connect(str(db))
        conn.executescript(store._SCHEMA + store._SCHEMA_V2 + "PRAGMA user_version = 2;")
        conn.close()
        catalog.import_catalog(self.settings)
        self.assertEqual(catalog.class_detail(self.settings, "Glasses", "NISKA_ZUTO")["trained_name"], "NISKA_BELO_ZUTO")


class MiroAndExamplesTests(AssortmentFixture):
    """Ticket 04: old Miro pages as whole images, admin confirmation, read-only dataset examples."""

    def setUp(self):
        super().setUp()
        from PIL import Image
        self.miro = self.tmp / "miro"
        self.miro.mkdir()
        self.page_sizes = [(300, 200), (320, 500), (280, 180)]
        pages = [Image.new("RGB", size, color) for size, color in zip(self.page_sizes, ("red", "green", "blue"))]
        pages[0].save(self.miro / "Glasses board.pdf", save_all=True, append_images=pages[1:])
        Image.new("RGB", (200, 100), "yellow").save(self.miro / "Shots.pdf")
        self.suggestions = self.tmp / "suggestions.yaml"
        self.write_suggestions({"glasses-01": {"heading": "CASA", "classes": ["CASA"]},
                                "glasses-02": {"heading": "OVERVIEW", "classes": []},
                                "shots-01": {"heading": "AMBER RED", "classes": ["AMBER", "RED"], "confidence": "low"}})
        self.raw = self.tmp / "raw"
        images, labels = self.raw / "glasses" / "images" / "CASA", self.raw / "glasses" / "labels" / "CASA"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (640, 360), "white").save(images / f"frame_{i}.jpg")
            (labels / f"frame_{i}.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.2 0.2 0.1 0.1\n", encoding="utf-8")
        Image.new("RGB", (640, 360), "white").save(images / "only_other_class.jpg")
        (labels / "only_other_class.txt").write_text("1 0.5 0.5 0.2 0.3\n", encoding="utf-8")
        self.settings = self.make_settings(
            catalog_miro_dir=str(self.miro), catalog_miro_suggestions_path=str(self.suggestions),
            catalog_miro_files={"Glasses": "Glasses board.pdf", "shots": "Shots.pdf"})

    def write_suggestions(self, pages):
        self.suggestions.write_text(json.dumps({"pages": pages}), encoding="utf-8")

    def raw_state(self):
        return {p: (p.is_dir() or p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(self.raw.rglob("*"))}

    def page(self, key):
        return next(p for p in catalog.list_miro_pages(self.settings)["pages"] if p["key"] == key)

    def test_every_page_is_imported_whole(self):
        from PIL import Image
        report = catalog.import_catalog(self.settings)["miro"]
        self.assertEqual(report["pages"], 4)
        data = catalog.list_miro_pages(self.settings)
        self.assertEqual([p["key"] for p in data["pages"]], ["glasses-01", "glasses-02", "glasses-03", "shots-01"])
        for p, size in zip(data["pages"], self.page_sizes):
            with Image.open(catalog.miro_page_image(self.settings, p["key"])) as im:
                self.assertEqual(im.size, size)
            self.assertEqual(p["source_label"], "Stara Miro strana")

    def test_admin_confirms_on_one_screen_and_decision_survives_reimport(self):
        catalog.import_catalog(self.settings)
        self.assertEqual(catalog.list_miro_pages(self.settings)["counts"], {"total": 4, "confirmed": 0, "without_class": 2, "retired": 0})
        self.assertEqual(self.page("shots-01")["classes"], ["AMBER", "RED"])
        self.assertTrue(self.page("shots-01")["low_confidence"])
        catalog.save_miro_page(self.settings, "glasses-01", ["NISKA", "CASA"])
        catalog.save_miro_page(self.settings, "glasses-03", [])
        catalog.import_catalog(self.settings)
        self.assertEqual(self.page("glasses-01")["classes"], ["CASA", "NISKA"])
        self.assertTrue(self.page("glasses-01")["confirmed"])
        counts = catalog.list_miro_pages(self.settings)["counts"]
        self.assertEqual(counts, {"total": 4, "confirmed": 2, "without_class": 2, "retired": 0})
        self.assertIn("glasses-03", [p["key"] for p in catalog.list_miro_pages(self.settings)["pages"] if not p["classes"]])
        catalog.reset_miro_page(self.settings, "glasses-01")
        self.assertEqual(self.page("glasses-01")["classes"], ["CASA"])
        self.assertFalse(self.page("glasses-01")["confirmed"])

    def test_new_export_keeps_confirmations_by_look_not_position(self):
        """Ticket 21: Miro re-encodes and reorders pages; a page is recognised by how it looks."""
        from PIL import Image, ImageDraw
        catalog.import_catalog(self.settings)
        catalog.save_miro_page(self.settings, "glasses-01", ["CASA"])   # red
        catalog.save_miro_page(self.settings, "glasses-02", ["NISKA"])  # green, will be gone
        catalog.save_miro_page(self.settings, "glasses-03", [])         # blue
        fresh = Image.new("RGB", (300, 200), "white")
        ImageDraw.Draw(fresh).rectangle((40, 40, 200, 160), fill="black")
        # new export: blue first and a little smaller (re-encoded), a new page, red last; green removed
        pages = [Image.new("RGB", (270, 168), "blue"), fresh, Image.new("RGB", (300, 200), (250, 2, 3))]
        pages[0].save(self.miro / "Glasses board.pdf", save_all=True, append_images=pages[1:])
        new_page = [p for p in miro.extract_pdf(self.miro / "Glasses board.pdf", "Glasses") if p.page_no == 2][0]
        new_key = miro.content_key("Glasses", new_page.sha256)
        self.write_suggestions({new_key: {"heading": "NOVA", "classes": ["NISKA"]},
                                "glasses-01": {"heading": "CASA", "classes": ["CASA"]},
                                "shots-01": {"heading": "AMBER RED", "classes": ["AMBER", "RED"]}})
        report = catalog.import_catalog(self.settings)["miro"]
        self.assertEqual(report["kept"], 3)  # red, blue, and the unchanged shots page
        self.assertEqual(report["new_pages"], ["Glasses 2"])
        self.assertEqual(report["confirmed_retired"], ["glasses-02 (OVERVIEW)"])
        self.assertEqual(self.page("glasses-01")["page_no"], 3)
        self.assertTrue(self.page("glasses-01")["confirmed"])
        self.assertEqual(self.page("glasses-01")["classes"], ["CASA"])
        self.assertTrue(self.page("glasses-03")["confirmed"])
        self.assertEqual(self.page("glasses-03")["page_no"], 1)
        self.assertEqual((self.page(new_key)["confirmed"], self.page(new_key)["classes"], self.page(new_key)["heading"]),
                         (False, ["NISKA"], "NOVA"))
        self.assertTrue(self.page("glasses-02")["retired"])
        self.assertEqual(catalog.list_miro_pages(self.settings)["counts"],
                         {"total": 4, "confirmed": 2, "without_class": 1, "retired": 1})
        with self.assertRaises(catalog.CatalogError):
            catalog.save_miro_page(self.settings, "glasses-02", ["NISKA"])
        self.assertEqual([p["key"] for p in catalog.class_detail(self.settings, "Glasses", "NISKA")["miro_pages"]], [new_key])
        # the same export again changes nothing
        again = catalog.import_catalog(self.settings)["miro"]
        self.assertEqual((again["kept"], again["new_pages"], again["retired"]), (4, [], []))

    def test_one_page_shows_on_every_class_it_belongs_to(self):
        catalog.import_catalog(self.settings)
        self.assertEqual(catalog.class_detail(self.settings, "Shots", "AMBER")["miro_pages"][0]["confirmed"], False)
        catalog.save_miro_page(self.settings, "glasses-01", ["CASA", "NISKA"])
        for cls in ("CASA", "NISKA"):
            pages = catalog.class_detail(self.settings, "Glasses", cls)["miro_pages"]
            self.assertEqual([(p["key"], p["confirmed"], p["source_label"]) for p in pages],
                             [("glasses-01", True, "Stara Miro strana")])

    def test_wrong_class_is_refused_through_api(self):
        catalog.import_catalog(self.settings)
        routes = self.routes_for(self.settings)
        save = routes[("/api/catalog/miro-pages/{page_key}", "POST")]
        with self.assertRaises(HTTPException) as e:
            save("glasses-01", {"classes": ["AMBER"]})
        self.assertEqual(e.exception.status_code, 400)
        with self.assertRaises(HTTPException) as e:
            save("glasses-99", {"classes": []})
        self.assertEqual(e.exception.status_code, 404)
        self.assertFalse(self.page("glasses-01")["confirmed"])
        self.assertTrue(save("glasses-01", {"classes": ["NISKA"]})["confirmed"])

    def routes_for(self, settings):
        from fastapi import FastAPI
        from web_labeler.catalog.router import build_catalog_router
        app = FastAPI()
        app.include_router(build_catalog_router(settings))
        return {(r.path, m): r.endpoint for r in app.routes for m in getattr(r, "methods", ())}

    def test_bad_suggestion_refuses_whole_import(self):
        self.write_suggestions({"glasses-01": {"classes": ["NOT_A_CLASS"]}})
        with self.assertRaises(catalog.CatalogError):
            catalog.import_catalog(self.settings)
        self.assertFalse((self.tmp / "db" / "catalog.db").exists())

    def test_examples_are_labeled_and_never_name_an_article(self):
        catalog.import_catalog(self.settings)
        data = catalog.class_examples(self.settings, "Glasses", "CASA", count=3, seed=1)
        self.assertEqual(data["source_label"], "Primer iz dataseta")
        self.assertEqual(len(data["examples"]), 3)
        for e in data["examples"]:
            self.assertEqual(set(e), {"url", "source_label"})
            self.assertNotIn("only_other_class", e["url"])
        self.assertEqual(catalog.class_examples(self.settings, "Glasses", "NISKA")["examples"], [])
        self.assertEqual(catalog.class_examples(self.settings, "Glasses", "UNITED_CLASS")["examples"], [])

    def test_example_crop_and_raw_stays_unchanged(self):
        from io import BytesIO
        from PIL import Image
        catalog.import_catalog(self.settings)
        before = self.raw_state()
        routes = self.routes_for(self.settings)
        listing = routes[("/api/catalog/models/{model}/classes/{class_name}/examples", "GET")]("Glasses", "CASA", 6, 7)
        image = routes[("/api/catalog/models/{model}/classes/{class_name}/examples/{file_name}", "GET")]
        for e in listing["examples"]:
            name, box = e["url"].rsplit("/", 1)[1].split("?box=")
            with Image.open(BytesIO(image("Glasses", "CASA", name, int(box)).body)) as im:
                self.assertEqual(im.format, "JPEG")
                self.assertLess(im.size[0], 640)
        for bad in ("../CASA/frame_0.jpg", "frame_0.txt", "missing.jpg"):
            with self.assertRaises(HTTPException):
                image("Glasses", "CASA", bad, 0)
        with self.assertRaises(HTTPException):
            image("Glasses", "CASA", "only_other_class.jpg", 0)
        catalog.class_examples(self.settings, "Glasses", "NISKA")
        self.assertEqual(self.raw_state(), before)
        self.assertFalse((self.raw / "glasses" / "labels" / "NISKA").exists())

    def test_article_without_evidence_has_no_image(self):
        catalog.import_catalog(self.settings)
        self.assertIsNone(self.variation("Glasses", "CASA", "drink")["image"])

    def test_images_refused_inside_raw(self):
        settings = self.make_settings(catalog_miro_dir=str(self.miro), catalog_media_dir=str(self.raw / "media"),
                                      catalog_miro_files={"Glasses": "Glasses board.pdf"})
        with self.assertRaises(catalog.CatalogError):
            catalog.import_catalog(settings)
        self.assertFalse((self.raw / "media").exists())

    def test_third_version_database_is_upgraded(self):
        db = self.tmp / "db" / "catalog.db"
        db.parent.mkdir()
        conn = sqlite3.connect(str(db))
        conn.executescript(store._SCHEMA + store._SCHEMA_V2 + store._SCHEMA_V3 + "PRAGMA user_version = 3;")
        conn.close()
        catalog.import_catalog(self.settings)
        self.assertEqual(catalog.list_miro_pages(self.settings)["counts"]["total"], 4)


class EvidenceTests(AssortmentFixture):
    """Ticket 05: a pasted screenshot waits for the admin, then proves one POS variation."""

    def setUp(self):
        super().setUp()
        catalog.import_catalog(self.settings)
        self.drink = self.variation("Glasses", "CASA", "drink")

    def png(self, color="red", size=(120, 80)):
        from io import BytesIO
        from PIL import Image
        out = BytesIO()
        Image.new("RGB", size, color).save(out, format="PNG")
        return out.getvalue()

    def add(self, data=None, identity=None, name="Goca"):
        return catalog.add_evidence(self.settings, identity or self.drink["identity_id"], data or self.png(), name,
                                    "labeler", "Glasses", "CASA")

    def test_added_image_waits_and_keeps_not_verified(self):
        item = self.add()
        self.assertEqual((item["status"], item["added_by"], item["name"], item["class"]), ("pending", "Goca", "drink", "CASA"))
        self.assertEqual([h["action"] for h in item["history"]], ["added"])
        drink = self.variation("Glasses", "CASA", "drink")
        self.assertIsNone(drink["image"])
        self.assertEqual(drink["pending_images"], 1)
        self.assertIn("nije provereno na snimku", drink["tags"])
        path, media_type = catalog.evidence_image(self.settings, item["evidence_id"])
        self.assertEqual(media_type, "image/png")
        self.assertTrue(store._is_within(path, self.tmp / "db" / "media"))
        self.assertFalse((self.tmp / "raw").exists())

    def test_confirmed_image_proves_the_variation_and_survives_reimport(self):
        item = self.add()
        done = catalog.decide_evidence(self.settings, item["evidence_id"], "confirmed", "Aleksa")
        self.assertEqual([h["action"] for h in done["history"]], ["added", "confirmed"])
        catalog.import_catalog(self.settings)
        drink = self.variation("Glasses", "CASA", "drink")
        self.assertEqual(drink["image"]["evidence_id"], item["evidence_id"])
        self.assertEqual(drink["image"]["source_label"], "Dokaz iz Kataloga")
        self.assertNotIn("nije provereno na snimku", drink["tags"])
        self.assertNotIn(drink["identity_id"], [v["identity_id"] for v in catalog.list_without_image(self.settings)["items"]])
        with self.assertRaises(catalog.CatalogError):
            catalog.decide_evidence(self.settings, item["evidence_id"], "rejected", "Aleksa")

    def test_rejected_image_stays_in_history_and_proves_nothing(self):
        item = self.add()
        catalog.decide_evidence(self.settings, item["evidence_id"], "rejected", "Aleksa", "  pogrešan   artikal ")
        rejected = catalog.list_evidence(self.settings, "rejected")["items"]
        self.assertEqual([(r["evidence_id"], r["reason"]) for r in rejected], [(item["evidence_id"], "pogrešan artikal")])
        self.assertEqual(catalog.list_evidence(self.settings)["items"], [])
        drink = self.variation("Glasses", "CASA", "drink")
        self.assertIsNone(drink["image"])
        self.assertEqual(drink["pending_images"], 0)
        self.assertIn("nije provereno na snimku", drink["tags"])
        with sqlite3.connect(str(self.tmp / "db" / "catalog.db")) as conn:
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("DELETE FROM evidence")
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("DELETE FROM evidence_events")

    def test_same_image_is_not_saved_twice(self):
        data = self.png()
        first = self.add(data)
        other = self.variation("Glasses", "CASA", "MAIN")["identity_id"]
        for identity in (self.drink["identity_id"], other):
            with self.assertRaises(catalog.CatalogError):
                self.add(data, identity)
        catalog.decide_evidence(self.settings, first["evidence_id"], "rejected", "Aleksa")
        with self.assertRaises(catalog.CatalogError):
            self.add(data)
        self.assertEqual(len(catalog.list_evidence(self.settings, None)["items"]), 1)
        self.assertEqual(len(list((self.tmp / "db" / "media" / "evidence").iterdir())), 1)

    def test_only_reasonable_images_from_a_named_person(self):
        from io import BytesIO
        from PIL import Image
        gif = BytesIO()
        Image.new("RGB", (100, 100)).save(gif, format="GIF")
        for data in (b"not an image", self.png(size=(20, 20)), gif.getvalue(), b"\x89PNG" + b"0" * (store.EVIDENCE_MAX_BYTES + 1)):
            with self.assertRaises(catalog.CatalogError):
                self.add(data)
        with self.assertRaises(catalog.CatalogError):
            self.add(name="   ")
        with self.assertRaises(catalog.UnknownEvidence):
            self.add(identity=999999)
        self.assertEqual(catalog.list_evidence(self.settings, None)["items"], [])

    def test_without_image_lists_tracked_sold_variations_best_first(self):
        items = catalog.list_without_image(self.settings, "Glasses")["items"]
        self.assertTrue(items)
        self.assertTrue(all(v["state"] == "tracked" and v["code"] for v in items))
        self.assertEqual([v["sold"] for v in items], sorted((v["sold"] for v in items), reverse=True))
        self.assertNotIn("NEVER SOLD", [v["name"] for v in items])
        self.assertNotIn("OLD NAME", [v["name"] for v in items])
        pending = self.add()
        self.assertIn(self.drink["identity_id"], [v["identity_id"] for v in catalog.list_without_image(self.settings, "Glasses")["items"]])
        catalog.decide_evidence(self.settings, pending["evidence_id"], "confirmed", "Aleksa")
        self.assertNotIn(self.drink["identity_id"], [v["identity_id"] for v in catalog.list_without_image(self.settings, "Glasses")["items"]])

    def test_paste_through_api(self):
        import base64
        from fastapi import FastAPI
        from web_labeler.catalog.router import build_catalog_router
        app = FastAPI()
        app.include_router(build_catalog_router(self.settings))
        routes = {(r.path, m): r.endpoint for r in app.routes for m in getattr(r, "methods", ())}
        post = routes[("/api/catalog/evidence", "POST")]
        url = "data:image/png;base64," + base64.b64encode(self.png()).decode()
        item = post({"identity_id": self.drink["identity_id"], "image_base64": url, "added_by": "Goca", "role": "labeler"})
        self.assertEqual(item["status"], "pending")
        for bad in ({"identity_id": self.drink["identity_id"], "image_base64": "%%%", "added_by": "Goca"},
                    {"identity_id": self.drink["identity_id"], "image_base64": url, "added_by": "Goca"},
                    {"image_base64": url, "added_by": "Goca"}):
            with self.assertRaises(HTTPException) as e:
                post(bad)
            self.assertEqual(e.exception.status_code, 400)
        decide = routes[("/api/catalog/evidence/{evidence_id}/decision", "POST")]
        with self.assertRaises(HTTPException):
            decide(item["evidence_id"], {"decision": "maybe", "decided_by": "Aleksa"})
        self.assertEqual(decide(item["evidence_id"], {"decision": "confirmed", "decided_by": "Aleksa"})["status"], "confirmed")

    def test_fourth_version_database_is_upgraded(self):
        db = self.tmp / "db2" / "catalog.db"
        db.parent.mkdir()
        conn = sqlite3.connect(str(db))
        conn.executescript(store._SCHEMA + store._SCHEMA_V2 + store._SCHEMA_V3 + store._SCHEMA_V4 + "PRAGMA user_version = 4;")
        conn.close()
        settings = self.make_settings(catalog_db_path=str(db))
        catalog.import_catalog(settings)
        self.assertEqual(catalog.list_evidence(settings)["counts"], {"pending": 0, "confirmed": 0, "rejected": 0})


class DecisionTests(AssortmentFixture):
    """Ticket 06: manual decisions are appended, survive imports and never touch production files."""

    def setUp(self):
        super().setUp()
        catalog.import_catalog(self.settings)
        self.burger = self.find("NEW BURGER")

    def find(self, name):
        return next(v for v in catalog.list_articles(self.settings)["items"] if v["name"] == name)

    def decide(self, kind, role="admin", **kw):
        kw.setdefault("reason", "test")
        return catalog.record_decision(self.settings, kind, "Aleksa" if role == "admin" else "Goca", role, **kw)

    def set_class(self, identity, cls, **kw):
        return self.decide("classes", identity_id=identity, payload={"links": [{"model": "glasses", "class": cls}]}, **kw)

    def test_whoever_is_at_the_tool_decides(self):
        """Ticket 20 (owner 24.11.2026): one pass, Goca or the owner; the decision counts at once and can be revoked."""
        decision = self.set_class(self.burger["identity_id"], "CASA", role="labeler")
        self.assertEqual((decision["role"], decision["actor"]), ("labeler", "Goca"))
        x = self.decide("ignore", role="labeler", article_id=self.burger["article_id"], reason="", valid_from="2030-01-01")
        self.assertEqual(x["valid_from"], store._today())  # only the admin may date a decision ahead
        undo = self.decide("revoke", role="labeler", target_id=x["decision_id"])
        self.assertEqual(undo["kind"], "revoke")
        proposal = self.decide("propose", role="labeler", identity_id=self.burger["identity_id"],
                               payload={"kind": "classes", "change": {"links": [{"model": "glasses", "class": "CASA"}]}})
        self.assertEqual(proposal["status"], "pending")

    def test_class_decision_changes_the_mapping_and_survives_reimport(self):
        decision = self.set_class(self.burger["identity_id"], "CASA")
        self.assertIn("glasses.CASA", decision["miro_text"])
        catalog.import_catalog(self.settings)
        burger = self.find("NEW BURGER")
        self.assertEqual((burger["state"], burger["mapping"]), ("tracked", "glasses.CASA"))
        self.assertIn("ručna odluka, čeka izdanje", burger["tags"])
        self.assertIn("nije provereno na snimku", burger["tags"])
        self.assertIn("NEW BURGER", [v["name"] for v in catalog.list_variations(self.settings, "Glasses", "CASA")])

    def test_changed_decision_keeps_the_old_one_and_revoke_steps_back(self):
        first = self.set_class(self.burger["identity_id"], "CASA")
        second = self.set_class(self.burger["identity_id"], "NISKA")
        self.assertEqual(self.find("NEW BURGER")["mapping"], "glasses.NISKA")
        history = catalog.article_card(self.settings, self.burger["article_id"])["history"]
        self.assertEqual([h["decision_id"] for h in history], [second["decision_id"], first["decision_id"]])
        self.decide("revoke", target_id=second["decision_id"])
        self.assertEqual(self.find("NEW BURGER")["mapping"], "glasses.CASA")
        with self.assertRaises(catalog.CatalogError):
            self.decide("revoke", target_id=second["decision_id"])
        self.decide("revoke", target_id=first["decision_id"])
        burger = self.find("NEW BURGER")
        self.assertEqual((burger["state"], burger["mapping"]), ("pending", "mapiranje nedostaje"))
        history = catalog.article_card(self.settings, self.burger["article_id"])["history"]
        self.assertEqual(len(history), 4)
        self.assertTrue(all(h["revoked"] for h in history if h["kind"] == "classes"))

    def test_decision_takes_effect_from_its_date(self):
        drink = self.find("drink")
        self.set_class(drink["identity_id"], "NISKA", valid_from="2030-05-01")
        self.assertEqual(self.find("drink")["mapping"], "glasses.CASA, glasses.NISKA")
        before = catalog.article_card(self.settings, drink["article_id"], as_of="2030-04-30")["variations"][0]
        after = catalog.article_card(self.settings, drink["article_id"], as_of="2030-05-01")["variations"][0]
        self.assertEqual((before["mapping"], after["mapping"]), ("glasses.CASA, glasses.NISKA", "glasses.NISKA"))
        self.assertTrue(catalog.list_decisions(self.settings)["items"][0]["future"])

    def test_x_ignores_the_whole_article_and_can_be_undone(self):
        drink = self.find("drink")
        x = self.decide("ignore", role="labeler", article_id=drink["article_id"], reason="")
        self.assertEqual(self.find("drink")["mapping"], "ne pratimo")
        ignored = next(g for g in catalog.list_unclassified(self.settings)["groups"] if g["state"] == "ignored")["items"]
        self.assertIn("drink", [v["name"] for v in ignored])
        self.assertEqual([d["decision_id"] for d in catalog.list_decisions(self.settings, "ignore")["items"]], [x["decision_id"]])
        self.decide("revoke", target_id=x["decision_id"])
        self.assertEqual(self.find("drink")["mapping"], "glasses.CASA, glasses.NISKA")

    def test_merge_puts_variations_on_one_card_and_can_be_undone(self):
        main, other = self.find("MAIN"), self.find("OTHER DRINK")
        self.assertNotEqual(main["article_id"], other["article_id"])
        merge = self.decide("merge", article_id=other["article_id"], payload={"into": main["article_id"]})
        card = catalog.article_card(self.settings, other["article_id"])
        self.assertEqual(card["merged_into"], main["article_id"])
        self.assertEqual(sorted(v["name"] for v in card["variations"]), ["MAIN", "OTHER DRINK"])
        self.assertEqual(self.find("OTHER DRINK")["article_id"], main["article_id"])
        with self.assertRaises(catalog.CatalogError):
            self.decide("merge", article_id=main["article_id"], payload={"into": other["article_id"]})
        self.decide("revoke", target_id=merge["decision_id"])
        self.assertEqual(self.find("OTHER DRINK")["article_id"], other["article_id"])

    def test_proposal_is_applied_with_a_star_or_rejected_with_a_reason(self):
        change = {"kind": "classes", "change": {"links": [{"model": "glasses", "class": "CASA"}]}}
        first = self.decide("propose", role="labeler", identity_id=self.burger["identity_id"], payload=change)
        applied = self.decide("classes", identity_id=self.burger["identity_id"],
                              payload={**first["payload"]["change"], "provisional": "nisam siguran", "from_proposal": first["decision_id"]})
        self.assertEqual(catalog.get_decision(self.settings, first["decision_id"])["status"], "applied")
        self.assertIn("privremeno", self.find("NEW BURGER")["tags"])
        with self.assertRaises(catalog.CatalogError):
            self.decide("classes", identity_id=self.burger["identity_id"],
                        payload={**first["payload"]["change"], "from_proposal": first["decision_id"]})
        second = self.decide("propose", role="labeler", identity_id=self.burger["identity_id"], payload=change)
        with self.assertRaises(catalog.CatalogError):
            self.decide("reject", target_id=second["decision_id"], reason="")
        self.decide("reject", target_id=second["decision_id"], reason="već rešeno")
        self.assertEqual(catalog.get_decision(self.settings, second["decision_id"])["status"], "rejected")
        with self.assertRaises(catalog.CatalogError):
            self.decide("reject", target_id=second["decision_id"])
        self.assertEqual(applied["payload"]["from_proposal"], first["decision_id"])

    def test_bad_decisions_are_refused(self):
        with self.assertRaises(catalog.CatalogError):
            self.set_class(self.burger["identity_id"], "NOT_A_CLASS")
        with self.assertRaises(catalog.CatalogError):
            self.set_class(self.burger["identity_id"], "CASA", reason="")
        with self.assertRaises(catalog.CatalogError):
            self.set_class(self.burger["identity_id"], "CASA", valid_from="01.12.2026")
        with self.assertRaises(catalog.CatalogError):
            catalog.record_decision(self.settings, "ignore", "  ", "labeler", article_id=self.burger["article_id"])
        with self.assertRaises(catalog.UnknownArticle):
            self.set_class(999999, "CASA")
        self.assertEqual(catalog.list_decisions(self.settings)["items"], [])

    def test_decisions_are_append_only_and_production_files_stay_the_same(self):
        files = [self.map_path, self.blacklist, *self.pos.iterdir(), *self.models.iterdir()]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
        d = self.set_class(self.burger["identity_id"], "CASA")
        self.decide("ignore", article_id=self.find("drink")["article_id"])
        self.decide("revoke", target_id=d["decision_id"])
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}, before)
        with sqlite3.connect(str(self.tmp / "db" / "catalog.db")) as conn:
            for sql in ("UPDATE decisions SET reason = 'x'", "DELETE FROM decisions"):
                with self.assertRaises(sqlite3.DatabaseError):
                    conn.execute(sql)

    def test_article_list_searches_and_says_when_mapping_is_missing(self):
        data = catalog.list_articles(self.settings)
        self.assertEqual(len(data["items"]), sum(data["counts"][s] for s in store.STATE_LABELS))
        self.assertEqual([v["name"] for v in catalog.list_articles(self.settings, "0706")["items"]], ["drink"])
        self.assertEqual([v["name"] for v in catalog.list_articles(self.settings, "burg")["items"]], ["NEW BURGER"])
        missing = catalog.list_articles(self.settings, state="missing")["items"]
        self.assertIn("NEW BURGER", [v["name"] for v in missing])
        self.assertTrue(all(v["mapping"] == "mapiranje nedostaje" for v in missing))
        self.assertTrue(all(v["mapping"] == "ne pratimo" for v in catalog.list_articles(self.settings, state="ignored")["items"]))

    def test_fifth_version_database_is_upgraded(self):
        db = self.tmp / "db2" / "catalog.db"
        db.parent.mkdir()
        conn = sqlite3.connect(str(db))
        conn.executescript(store._SCHEMA + store._SCHEMA_V2 + store._SCHEMA_V3 + store._SCHEMA_V4 + store._SCHEMA_V5
                           + "PRAGMA user_version = 5;")
        conn.close()
        settings = self.make_settings(catalog_db_path=str(db))
        catalog.import_catalog(settings)
        self.assertEqual(catalog.list_decisions(settings)["items"], [])


class BackupTests(AssortmentFixture):
    """Ticket 06: nightly copy of the database and images, kept 30 days, and a proven way back."""

    def setUp(self):
        super().setUp()
        catalog.import_catalog(self.settings)
        self.db = self.tmp / "db" / "catalog.db"
        self.media = self.tmp / "db" / "media"
        (self.media / "evidence").mkdir(parents=True)
        (self.media / "evidence" / "abc.png").write_bytes(b"image bytes")
        self.dest = self.tmp / "disk" / "backups"
        self.burger = next(v for v in catalog.list_articles(self.settings)["items"] if v["name"] == "NEW BURGER")

    def decide(self, cls):
        return catalog.record_decision(self.settings, "classes", "Aleksa", "admin", identity_id=self.burger["identity_id"],
                                       payload={"links": [{"model": "glasses", "class": cls}]}, reason="test")

    def test_copy_is_checked_and_kept_thirty_days(self):
        from datetime import datetime, timedelta
        from web_labeler.catalog import backup
        start = datetime(2026, 11, 1)
        for i in range(40):
            report = backup.backup(self.db, self.dest, "venue_a", self.media, today=start + timedelta(days=i))
        self.assertEqual(report["schema_version"], store.SCHEMA_VERSION)
        days = [b["day"] for b in backup.list_backups(self.dest, "venue_a")]
        self.assertEqual(len(days), 31)
        self.assertEqual(days[0], "2026-11-10")
        self.assertEqual((self.dest / "media" / "evidence" / "abc.png").read_bytes(), b"image bytes")
        self.assertFalse(list(self.dest.glob("*.part")))

    def test_restore_brings_back_the_copy_and_keeps_the_current_database(self):
        from web_labeler.catalog import backup
        first = self.decide("CASA")
        copy = Path(backup.backup(self.db, self.dest, "venue_a", self.media)["backup"])
        self.decide("NISKA")
        (self.media / "evidence" / "abc.png").unlink()
        report = backup.restore(copy, self.db, self.dest / "media", self.media)
        self.assertEqual([d["decision_id"] for d in catalog.list_decisions(self.settings)["items"]], [first["decision_id"]])
        self.assertEqual(next(v for v in catalog.list_articles(self.settings)["items"] if v["name"] == "NEW BURGER")["mapping"],
                         "glasses.CASA")
        self.assertTrue(Path(report["previous_kept_as"]).is_file())
        self.assertEqual((self.media / "evidence" / "abc.png").read_bytes(), b"image bytes")

    def test_broken_copy_is_refused_and_nothing_is_replaced(self):
        from web_labeler.catalog import backup
        bad = self.tmp / "catalog_venue_a_2026-11-24.db"
        bad.write_bytes(b"not a database" * 100)
        before = self.db.read_bytes()
        with self.assertRaises(backup.BackupError):
            backup.restore(bad, self.db)
        self.assertEqual(self.db.read_bytes(), before)
        with self.assertRaises(backup.BackupError):
            backup.backup(self.tmp / "missing.db", self.dest, "venue_a")


if __name__ == "__main__":
    unittest.main()
