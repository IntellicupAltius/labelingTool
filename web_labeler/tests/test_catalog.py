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


class AssortmentImportTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
