"""KATALOG-1 ticket 11: POS report intake on a fake drop folder (report only, never changes the folder or the map).

    PYTHONPATH=web_labeler/tests /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest web_labeler.tests.test_pos_intake -v
"""
import csv
import hashlib
import os
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from web_labeler import catalog
from web_labeler.catalog import store


def row(day, code, name, when, qty=1, paid=True):
    r = [""] * 42
    d = datetime.strptime(day, "%Y-%m-%d").strftime("%d/%m/%Y")
    r[3], r[5] = d, d
    r[16], r[22], r[23], r[24] = when, "NENAPLACENO" if not paid else when, code, name
    r[25], r[26], r[27] = f"{qty:.2f}", "100.00", f"{qty * 100:.2f}"
    return r


class PosIntakeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.drops = self.tmp / "drops"
        self.drops.mkdir()
        (self.tmp / "Models").mkdir()
        (self.tmp / "Models" / "Glasses.yaml").write_text("names: [CASA]\n")
        self.settings = catalog.CatalogSettings.from_config(
            {"catalog_venue_id": "venue_a", "catalog_db_path": str(self.tmp / "db" / "catalog.db"),
             "catalog_pos_dir": str(self.drops), "catalog_pos_adapter": "blaznavac_csv"}, self.tmp / "Models", self.tmp / "raw")
        self.now = datetime.now() + timedelta(minutes=10)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, rows, age_minutes=10):
        path = self.drops / name
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
        old = time.time() - age_minutes * 60
        os.utime(path, (old, old))
        return path

    def receive(self, now=None):
        return store.receive_pos_reports(self.settings, now or self.now)

    def findings(self, status="open"):
        return store.list_findings(self.settings, status)["items"]

    def drops_state(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.drops.iterdir()}

    def test_first_run_records_what_is_there_without_findings(self):
        self.write("01.11.csv", [row("2026-11-01", "0100", "Casa", "2026.11.01 10:00")])
        self.write("02.11..csv", [row("2026-11-03", "0100", "Casa", "2026.11.03 10:00")])  # wrong name, but old
        report = self.receive()
        self.assertTrue(report["baseline"])
        self.assertEqual(len(report["received"]), 2)
        self.assertEqual(self.findings(), [])

    def test_half_written_file_waits_and_nothing_is_read_twice(self):
        self.write("01.11.csv", [row("2026-11-01", "0100", "Casa", "2026.11.01 10:00")])
        self.receive()
        self.write("02.11.csv", [row("2026-11-02", "0100", "Casa", "2026.11.02 10:00")], age_minutes=0)
        report = store.receive_pos_reports(self.settings, datetime.now())
        self.assertEqual((report["received"], report["waiting"]), ([], ["02.11.csv"]))
        self.assertEqual(len(self.receive()["received"]), 1)
        self.assertEqual(self.receive()["received"], [])  # same file again: nothing
        (self.drops / "kopija.csv").write_bytes((self.drops / "02.11.csv").read_bytes())
        os.utime(self.drops / "kopija.csv", (0, 0))
        self.assertEqual(self.receive()["received"], [])  # same bytes under another name

    def test_date_comes_from_the_content_and_a_wrong_name_is_a_finding(self):
        self.write("01.11.csv", [row("2026-11-01", "0100", "Casa", "2026.11.01 10:00")])
        self.receive()
        self.write("24.04.csv", [row("2026-04-25", "0100", "Casa", "2026.04.24 17:43")])
        report = self.receive()
        self.assertEqual(report["received"][0]["day"], "2026-04-25")
        f = self.findings()
        self.assertEqual([x["kind"] for x in f], ["name_mismatch"])
        self.assertIn("24.04.2026.", f[0]["text"])
        self.assertIn("25.04.2026.", f[0]["text"])

    def test_changed_report_keeps_both_versions_and_the_admin_picks(self):
        self.write("01.11.csv", [row("2026-11-01", "0100", "Casa", "2026.11.01 10:00")])
        self.receive()
        self.write("05.11.csv", [row("2026-11-05", "0100", "Casa", "2026.11.05 10:00"),
                                 row("2026-11-05", "0200", "Niska", "2026.11.05 11:00")])
        self.receive()
        self.write("05.11 nova.csv", [row("2026-11-05", "0100", "Casa", "2026.11.05 10:00"),
                                      row("2026-11-05", "0300", "Pivo", "2026.11.05 12:00"),
                                      row("2026-11-05", "0300", "Pivo", "2026.11.05 12:00")])
        self.receive()
        changed = [x for x in self.findings() if x["kind"] == "changed_file"]
        self.assertEqual(len(changed), 1)
        self.assertEqual((changed[0]["data"]["only_new"], changed[0]["data"]["only_old"], changed[0]["data"]["both"]), (1, 1, 1))
        day = next(d for d in store.list_pos_days(self.settings)["days"] if d["day"] == "2026-11-05")
        old, new = [v["file_id"] for v in day["versions"]]
        self.assertEqual(day["active_file_id"], new)
        self.assertEqual([v["rows"] for v in day["versions"]], [2, 3])  # the repeated row inside one report counts twice
        store.decide_finding(self.settings, changed[0]["finding_id"], "denied", "Aleksa", "admin", "stara je tačna")
        self.assertEqual(next(d for d in store.list_pos_days(self.settings)["days"] if d["day"] == "2026-11-05")["active_file_id"], old)
        with self.assertRaises(catalog.CatalogError):
            store.decide_finding(self.settings, changed[0]["finding_id"], "applied", "Aleksa", "admin")
        store.decide_finding(self.settings, changed[0]["finding_id"], "reopened", "Aleksa", "admin", "ipak proveriti")
        item = store.decide_finding(self.settings, changed[0]["finding_id"], "applied", "Aleksa", "admin")
        self.assertEqual([h["action"] for h in item["history"]], ["denied", "reopened", "applied"])
        self.assertEqual(next(d for d in store.list_pos_days(self.settings)["days"] if d["day"] == "2026-11-05")["active_file_id"], new)
        self.assertEqual(len(list((self.tmp / "db" / "media" / "pos").iterdir())), 3)  # every received version kept

    def test_newest_version_is_the_last_to_arrive_not_the_last_by_name(self):
        self.write("01.11.csv", [row("2026-11-01", "0100", "Casa", "2026.11.01 10:00")])
        self.receive()
        self.write("b.csv", [row("2026-11-07", "0100", "Casa", "2026.11.07 10:00")], age_minutes=30)
        self.write("a.csv", [row("2026-11-07", "0200", "Niska", "2026.11.07 11:00")], age_minutes=20)
        self.receive()
        day = next(d for d in store.list_pos_days(self.settings)["days"] if d["day"] == "2026-11-07")
        self.assertEqual([v["file"] for v in day["versions"]], ["b.csv", "a.csv"])
        self.assertEqual(day["active_file_id"], day["versions"][-1]["file_id"])

    def test_bad_report_is_a_finding_and_is_not_used(self):
        self.write("01.11.csv", [row("2026-11-01", "0100", "Casa", "2026.11.01 10:00")])
        self.receive()
        self.write("06.11.csv", [row("2026-11-06", "0100", "Casa", "2026.11.06 10:00", paid=False)])
        (self.drops / "smece.csv").write_bytes(b"\xff\xfe\x00nije csv")
        os.utime(self.drops / "smece.csv", (0, 0))
        self.receive()
        bad = [x for x in self.findings() if x["kind"] == "bad_file"]
        self.assertEqual(len(bad), 2)
        days = store.list_pos_days(self.settings)["days"]
        self.assertIsNone(next(d for d in days if d["day"] == "2026-11-06")["active_file_id"])

    def test_only_admin_decides_and_nothing_is_erased(self):
        self.write("01.11.csv", [row("2026-11-01", "0100", "Casa", "2026.11.01 10:00")])
        self.receive()
        self.write("10.11.csv", [row("2026-11-11", "0100", "Casa", "2026.11.11 10:00")])
        self.receive()
        finding = self.findings()[0]
        with self.assertRaises(catalog.NotAllowed):
            store.decide_finding(self.settings, finding["finding_id"], "applied", "Goca", "labeler")
        conn = sqlite3.connect(str(self.tmp / "db" / "catalog.db"))
        for sql in ("DELETE FROM findings", "UPDATE pos_files SET status = 'bad'", "DELETE FROM pos_file_rows"):
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(sql)
        conn.close()

    def test_drop_folder_and_map_never_change(self):
        mapping = self.tmp / "map.yaml"
        mapping.write_text("articles: []\n")
        self.write("01.11.csv", [row("2026-11-01", "0100", "Casa", "2026.11.01 10:00")])
        self.receive()
        self.write("02.11.csv", [row("2026-11-02", "0100", "Casa", "2026.11.02 10:00")])
        before, map_before = self.drops_state(), mapping.read_bytes()
        self.receive()
        self.assertEqual(self.drops_state(), before)
        self.assertEqual(mapping.read_bytes(), map_before)



class DayTableTests(PosIntakeTests):
    """Ticket 13: one status per day, D+1 deadline, 'lokal nije radio', no late days before the module."""

    FUTURE = 8  # the table is read as if today were 8 days later; the module is switched on today

    def day(self, back):
        from datetime import date
        return (date.today() + timedelta(days=self.FUTURE - back)).isoformat()

    def report(self, back, n_rows, name=None, last_hour=22):
        d = self.day(back)
        compact = d.replace("-", ".")
        rows = [row(d, f"0{i:03d}", f"Pice {i}", f"{compact} {10 + i % (last_hour - 10):02d}:00") for i in range(n_rows - 1)]
        rows.append(row(d, "0999", "Kasno", f"{compact} {last_hour:02d}:00"))
        self.write(name or f"{d[8:10]}.{d[5:7]}.csv", rows)

    def status(self, back):
        from datetime import date
        future = (date.today() + timedelta(days=self.FUTURE)).isoformat()
        return next(x for x in store.day_table(self.settings, today=future)["days"] if x["day"] == self.day(back))

    def test_statuses(self):
        for weeks in (1, 2, 3):
            self.report(3 + 7 * weeks, 10)
        self.report(9, 10)  # before the module is switched on
        self.receive()       # baseline: the module starts today
        self.report(3, 2)                   # far fewer rows than the same weekday usually has
        self.report(4, 10, last_hour=15)    # last receipt too early
        self.report(5, 10, name="01.01.csv")  # name and date disagree
        self.report(6, 10)
        self.receive()
        self.assertEqual(self.status(6)["status"], "ok")
        self.assertEqual(self.status(9)["status"], "ok")
        self.assertIn("redova", " ".join(self.status(3)["notes"]))
        self.assertEqual(self.status(3)["status"], "suspicious")
        self.assertEqual(self.status(5)["status"], "suspicious")
        self.assertEqual(self.status(2)["status"], "late")
        self.assertEqual(self.status(1)["status"], "waiting")  # yesterday: due until the end of today
        self.assertEqual(self.status(12)["status"], "before")  # never late before the module

    def test_closed_day_by_labeler_with_reason_and_undo(self):
        self.report(20, 10)
        self.receive()
        with self.assertRaises(catalog.CatalogError):
            store.mark_day(self.settings, self.day(2), "closed", "Goca", "labeler", "")
        store.mark_day(self.settings, self.day(2), "closed", "Goca", "labeler", "praznik")
        st = self.status(2)
        self.assertEqual((st["status"], st["notes"]), ("closed", ["Goca: praznik"]))
        store.mark_day(self.settings, self.day(2), "reopened", "Aleksa", "admin", "ipak radio")
        self.assertEqual(self.status(2)["status"], "late")
        conn = sqlite3.connect(str(self.tmp / "db" / "catalog.db"))
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM pos_day_marks")
        conn.close()


if __name__ == "__main__":
    unittest.main()
