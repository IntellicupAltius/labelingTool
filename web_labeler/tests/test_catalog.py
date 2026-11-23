"""KATALOG-1 ticket 01: class import and the Catalog API, on a temporary app and database.

    /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest web_labeler.tests.test_catalog -v
"""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from web_labeler import catalog, server
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


if __name__ == "__main__":
    unittest.main()
