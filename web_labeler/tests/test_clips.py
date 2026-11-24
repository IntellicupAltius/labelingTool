"""KATALOG-1 ticket 16: footage of a new article, with a fake NVR downloader (no RTSP, no NVR).

    PYTHONPATH=web_labeler/tests /opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python -m unittest web_labeler.tests.test_clips -v
"""
import csv
import fcntl
import os
import sqlite3
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from web_labeler import catalog
from web_labeler.catalog import clips, store
from web_labeler.tests import test_releases
from web_labeler.tests.test_catalog import pos_row

FAKE = Path(__file__).resolve().parent / "fake_nvr_downloader.py"
NOW = datetime(2026, 11, 24, 12, 0)


class ClipBase(unittest.TestCase):
    tearDown_base = test_releases.ReleaseTests.tearDown
    write_live = test_releases.ReleaseTests.write_live
    write_map = test_releases.ReleaseTests.write_map
    make_settings = test_releases.ReleaseTests.make_settings
    identity = test_releases.ReleaseTests.identity

    def setUp(self):
        test_releases.ReleaseTests.setUp(self)
        self.write_pos("b.csv", [
            pos_row("0400", "NEW DRINK", "2026.11.21 10:00"), pos_row("0100", "Casa Vina", "2026.11.21 10:01"),
            pos_row("0100", "Casa Vina", "2026.11.21 10:02"), pos_row("0200", "Niska", "2026.11.21 10:03"),
            pos_row("0400", "NEW DRINK", "2026.11.21 15:00"),
            pos_row("0400", "NEW DRINK", "2026.11.22 12:00"),
            pos_row("0400", "NEW DRINK", "2026.11.23 09:00"),
            pos_row("0600", "OTHER NEW", "2026.11.23 20:00"),
            pos_row("0400", "NEW DRINK", "2026.11.24 11:55"),  # its 10 minutes are not over at NOW
        ])
        catalog.import_catalog(self.settings)
        self.clip_root = self.tmp / "clips"
        self.lock = self.tmp / "nvr.lock"
        self.log = self.tmp / "calls.log"
        self.proc = self.tmp / "proc"
        self.proc.mkdir()
        self.settings = self.clip_settings()
        os.environ.pop("FAKE_NVR_MISSING", None)
        os.environ["FAKE_NVR_LOG"] = str(self.log)

    def tearDown(self):
        for k in ("FAKE_NVR_MISSING", "FAKE_NVR_LOG", "FAKE_NVR_NEED_SECRET"):
            os.environ.pop(k, None)
        self.tearDown_base()

    def write_pos(self, name, rows):
        with open(self.tmp / "pos" / name, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)

    def clip_settings(self, **extra):
        cfg = {"catalog_clip_root": str(self.clip_root), "catalog_clip_downloader": [sys.executable, str(FAKE)],
               "catalog_clip_lock": str(self.lock), "catalog_clip_enabled": True, "catalog_clip_cap_gb": 1,
               "catalog_clip_est_mb_per_channel_minute": 0.02, "catalog_clip_busy_patterns": ["pipeline.run_day"],
               "catalog_clip_previews": False}
        cfg.update(extra)
        return self.make_settings(**cfg)

    def run_job(self, now=NOW, settings=None, **kw):
        kw.setdefault("clock", lambda: datetime(2026, 11, 24, 12, 0))
        return clips.run_job(settings or self.settings, now=now, proc=self.proc, **kw)

    def events(self, action=None):
        with store._open_for_read(self.settings) as conn:
            return [e for e in store._clip_events(conn, "venue_a") if action is None or e["action"] == action]

    def files(self):
        return sorted(str(p.relative_to(self.clip_root)) for p in self.clip_root.rglob("*.mkv")) if self.clip_root.exists() else []

    def new_drink(self):
        return self.identity("NEW DRINK")["identity_id"]


class ClipTests(ClipBase):
    # ---- choice of sales ----

    def test_sales_one_per_day_least_crowded_newest_first_only_finished_and_still_on_nvr(self):
        s = self.settings.clips
        t = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M")
        mine = [t("2026-11-21 10:00"), t("2026-11-21 15:00"), t("2026-11-22 12:00"), t("2026-11-24 11:55"), t("2026-10-01 12:00")]
        others = [t("2026-11-21 10:01"), t("2026-11-21 10:02"), t("2026-11-21 10:03")]
        ranked = clips.rank_sales(mine, mine + others, NOW, s)
        self.assertEqual([r.key for r in ranked], ["2026-11-22 12:00", "2026-11-21 15:00"])
        self.assertEqual(ranked[1].crowd, 0)

    def test_window_is_ring_to_plus_10_minutes_all_cameras_nothing_before(self):
        self.run_job(settings=self.clip_settings(catalog_clip_sales_per_article=1))
        calls = [l for l in self.log.read_text().splitlines() if "2026-11-23T09:00:00" in l]
        self.assertEqual(len(calls), 3)
        for ch, line in zip((1, 2, 3), calls):
            self.assertIn("--start 2026-11-23T09:00:00 --end 2026-11-23T09:10:00", line)
            self.assertIn(f"--channels {ch} ", line)

    # ---- downloads ----

    def test_downloads_up_to_three_sales_into_the_catalog_folder_never_raw(self):
        raw_before = sorted(p for p in (self.tmp / "raw").rglob("*")) if (self.tmp / "raw").exists() else []
        report = self.run_job()
        iid = self.new_drink()
        mine = [d for d in report["downloaded"] if d["identity_id"] == iid]
        self.assertEqual([d["sale"] for d in mine], ["2026-11-23 09:00", "2026-11-22 12:00", "2026-11-21 15:00"])
        self.assertTrue(all(d["channels"] == ["1", "2", "3"] for d in mine))
        self.assertIn(f"{iid}/20261123_0900/2026-11-23/channel_2/ch2_2026-11-23_09-00-00_10min.mkv", self.files())
        self.assertEqual(report["status"][iid], "ready")
        self.assertEqual(len(self.events("downloaded")), 4)  # 3 for NEW DRINK + 1 for OTHER NEW (one sale)
        raw_after = sorted(p for p in (self.tmp / "raw").rglob("*")) if (self.tmp / "raw").exists() else []
        self.assertEqual(raw_before, raw_after)
        self.assertEqual(self.run_job()["downloaded"], [])  # nothing twice
        item = next(i for i in store.list_inbox(self.settings)["items"] if i["identity_id"] == iid)
        self.assertEqual(item["clip"], {"status": "ready", "label": "Snimak spreman"})

    def test_breadth_first_and_stops_at_the_cap_with_a_finding(self):
        # one sale = 3 channels x 10 min x 0.02 MB = 0.6 MB estimate; real fake files 3 x 200 KB
        report = self.run_job(settings=self.clip_settings(catalog_clip_cap_gb=1.5 / 1024))
        got = [(d["identity_id"], d["sale"]) for d in report["downloaded"]]
        self.assertEqual(len({i for i, _ in got}), 2)  # both articles got their first sale
        self.assertLessEqual(len(got), 3)
        self.assertIn("granica", report["stopped"].lower())
        kinds = [f["kind"] for f in store.list_findings(self.settings)["items"]]
        self.assertEqual(kinds.count("clip_budget"), 1)
        self.run_job(settings=self.clip_settings(catalog_clip_cap_gb=1.5 / 1024))
        kinds = [f["kind"] for f in store.list_findings(self.settings)["items"]]
        self.assertEqual(kinds.count("clip_budget"), 1)  # once per day

    def test_skips_article_marked_x(self):
        v = self.identity("NEW DRINK")
        catalog.record_decision(self.settings, "ignore", "Aleksa", "admin", article_id=v["article_id"],
                                payload={}, reason="ne pratimo")
        report = self.run_job()
        self.assertNotIn(v["identity_id"], {d["identity_id"] for d in report["downloaded"]})
        self.assertNotIn(v["identity_id"], report["status"])

    def test_skips_article_with_a_pending_x_proposal(self):
        v = self.identity("OTHER NEW")
        catalog.record_decision(self.settings, "propose", "Claude (predlog)", "labeler", article_id=v["article_id"],
                                payload={"kind": "ignore", "change": {}}, reason="hrana")
        report = self.run_job()
        self.assertNotIn(v["identity_id"], report["status"])

    # ---- missing on the NVR ----

    def test_missing_sale_goes_to_next_and_with_none_left_says_nema_snimka(self):
        iid = self.new_drink()
        os.environ["FAKE_NVR_MISSING"] = "2026-11-23T09:00:00,2026-11-23T20:00:00"
        s1 = self.clip_settings(catalog_clip_sales_per_article=1)
        report = self.run_job(settings=s1)
        self.assertEqual([d["sale"] for d in report["downloaded"] if d["identity_id"] == iid], ["2026-11-22 12:00"])
        self.assertIn({"identity_id": iid, "sale": "2026-11-23 09:00"}, report["failed"])
        self.assertFalse((self.clip_root / str(iid) / "20261123_0900").exists())  # no stub left behind
        # everything gone from the NVR: OTHER NEW has one sale only; first try failed, second try = missing
        other = self.identity("OTHER NEW")["identity_id"]
        self.assertEqual(report["status"][other], "waiting")
        report = self.run_job(settings=s1)
        self.assertEqual(report["status"][other], "missing")
        self.assertEqual(clips.CLIP_STATUS_LABELS["missing"], "Nema snimka")
        # a new sale brings it back
        self.write_pos("c.csv", [pos_row("0600", "OTHER NEW", "2026.11.24 09:00")])
        catalog.import_catalog(self.settings)
        report = self.run_job(settings=s1)
        self.assertEqual(report["status"][other], "ready")

    def test_three_failures_in_a_row_stop_the_run(self):
        os.environ["FAKE_NVR_MISSING"] = ",".join(f"2026-11-{d}T{h}" for d, h in (
            ("23", "09:00:00"), ("22", "12:00:00"), ("21", "15:00:00"), ("20", "11:30:00"), ("23", "20:00:00")))
        report = self.run_job()
        self.assertEqual(len(report["failed"]), 3)
        self.assertIn("NVR", report["stopped"])

    def test_day_5_warning_once(self):
        os.environ["FAKE_NVR_MISSING"] = "2026-11-23T20:00:00"
        other = self.identity("OTHER NEW")["identity_id"]
        self.run_job()
        self.run_job(now=NOW + timedelta(days=3))
        self.assertEqual([f for f in store.list_findings(self.settings)["items"] if f["kind"] == "clip_late"], [])
        self.run_job(now=NOW + timedelta(days=5))
        self.run_job(now=NOW + timedelta(days=6))
        late = [f for f in store.list_findings(self.settings)["items"] if f["kind"] == "clip_late"]
        self.assertEqual(len(late), 1)
        self.assertEqual(late[0]["data"]["identity_id"], other)
        self.assertIn("OTHER NEW", late[0]["title"])

    # ---- guards ----

    def test_waits_for_the_nightly_sync_lock(self):
        with open(self.lock, "w") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX)
            report = self.run_job()
        self.assertFalse(report["ran"])
        self.assertIn("noćno", report["waiting"])
        self.assertEqual(self.files(), [])
        self.assertTrue(self.run_job()["ran"])

    def test_no_new_sale_after_22h(self):
        report = self.run_job(clock=lambda: datetime(2026, 11, 24, 22, 5))
        self.assertTrue(report["ran"])
        self.assertEqual(report["downloaded"], [])
        self.assertIn("23:55", report["stopped"])

    def test_waits_while_inference_or_training_runs(self):
        (self.proc / "4242").mkdir()
        (self.proc / "4242" / "cmdline").write_bytes(b"python\0-m\0pipeline.run_day\0--day\0x")
        report = self.run_job()
        self.assertFalse(report["ran"])
        self.assertIn("inference", report["waiting"])
        self.assertEqual(self.files(), [])

    def test_refuses_when_disk_not_mounted_or_folder_inside_raw(self):
        with self.assertRaises(clips.ClipError):
            self.run_job(settings=self.clip_settings(catalog_clip_mount=str(self.tmp / "not_a_mount")))
        (self.tmp / "raw").mkdir(exist_ok=True)
        with self.assertRaises(clips.ClipError):
            self.run_job(settings=self.clip_settings(catalog_clip_root=str(self.tmp / "raw" / "clips")))
        self.assertEqual(self.files(), [])

    def test_off_until_enabled_and_dry_run_changes_nothing(self):
        off = self.clip_settings(catalog_clip_enabled=False)
        with self.assertRaises(clips.ClipError):
            self.run_job(settings=off)
        report = self.run_job(settings=off, dry_run=True)
        self.assertTrue(report["downloaded"])
        self.assertEqual(self.files(), [])
        self.assertEqual(self.events(), [])
        self.assertFalse(self.log.exists())

    # ---- clean-up ----

    def test_deleted_30_days_after_decision(self):
        iid = self.new_drink()
        self.run_job(settings=self.clip_settings(catalog_clip_sales_per_article=1))
        self.assertTrue(any(f.startswith(f"{iid}/") for f in self.files()))
        v = self.identity("NEW DRINK")
        catalog.record_decision(self.settings, "ignore", "Aleksa", "admin", article_id=v["article_id"],
                                payload={}, reason="ne pratimo")
        real_now = datetime.now()
        self.run_job(now=real_now + timedelta(days=10))
        self.assertTrue(any(f.startswith(f"{iid}/") for f in self.files()))
        report = self.run_job(now=real_now + timedelta(days=31))
        self.assertEqual([d["identity_id"] for d in report["deleted"]], [iid])
        self.assertFalse(any(f.startswith(f"{iid}/") for f in self.files()))
        self.assertFalse((self.clip_root / str(iid)).exists())
        self.assertTrue(any(f.startswith(f"{self.identity('OTHER NEW')['identity_id']}/") for f in self.files()))

    def test_remove_folder_refuses_outside_root(self):
        with self.assertRaises(clips.ClipError):
            clips.remove_folder(self.settings.clips, self.tmp)

    # ---- secrets and history ----

    def test_secrets_are_sourced_for_the_downloader_and_never_on_its_command_line(self):
        env = self.tmp / "nvr.env"
        env.write_text("FAKE_NVR_SECRET=topsecret\n")
        os.environ["FAKE_NVR_NEED_SECRET"] = "1"
        report = self.run_job(settings=self.clip_settings(catalog_clip_secrets_env=str(env), catalog_clip_sales_per_article=1))
        self.assertTrue(report["downloaded"])
        text = self.log.read_text()
        self.assertIn("secret=True", text)
        self.assertNotIn("topsecret", text)

    def test_upgrade_from_v12_keeps_decisions(self):
        v = self.identity("OMLET")
        catalog.record_decision(self.settings, "ignore", "Goca", "labeler", article_id=v["article_id"])
        conn = sqlite3.connect(str(self.settings.db_path))
        rows = conn.execute("SELECT * FROM decisions ORDER BY decision_id").fetchall()
        conn.executescript("BEGIN;DROP TABLE clip_events;ALTER TABLE miro_pages DROP COLUMN retired_at;PRAGMA user_version = 12;COMMIT;")
        conn.close()
        with self.assertRaises(catalog.CatalogUnavailable):
            catalog.list_articles(self.settings)
        catalog.import_catalog(self.settings)
        conn = sqlite3.connect(str(self.settings.db_path))
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], store.SCHEMA_VERSION)
        self.assertEqual(conn.execute("SELECT * FROM decisions ORDER BY decision_id").fetchall(), rows)
        conn.close()
        self.assertTrue(self.run_job(settings=self.clip_settings(catalog_clip_sales_per_article=1))["downloaded"])

    def test_clip_history_is_append_only(self):
        self.run_job(settings=self.clip_settings(catalog_clip_sales_per_article=1))
        conn = sqlite3.connect(str(self.settings.db_path))
        with self.assertRaises(sqlite3.DatabaseError):
            conn.execute("UPDATE clip_events SET action = 'deleted'")
        with self.assertRaises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM clip_events")
        conn.close()



class CardBase(ClipBase):
    """Real tiny videos from the fake NVR, previews on."""

    def setUp(self):
        super().setUp()
        os.environ["FAKE_NVR_REAL"] = "1"
        self.settings = self.clip_settings(catalog_clip_previews=True, catalog_clip_window_minutes=1,
                                           catalog_clip_min_bytes=1000, catalog_clip_sales_per_article=1,
                                           catalog_clip_camera_names={"1": "Šank levo", "2": "Točilica", "3": "Šank desno"},
                                           videos_dir=str(self.tmp / "videos"))
        self.run_job()
        self.iid = self.new_drink()

    def tearDown(self):
        os.environ.pop("FAKE_NVR_REAL", None)
        super().tearDown()

    def run_job(self, now=datetime(2026, 11, 24, 11, 55, 30), settings=None, **kw):
        return super().run_job(now=now, settings=settings, **kw)

    def seen(self, card, channel=2, second=12.5, box=(0.25, 0.2, 0.3, 0.4)):
        return store.mark_seen(self.settings, self.iid, card["sales"][0]["slug"], channel, second, list(box), "Goca", "labeler")


class CardTests(CardBase):
    """Ticket 17: the card, "Vidim ga" and "Nije na snimku"."""

    def test_card_plays_three_cameras_from_the_ring_without_paths(self):
        card = store.clip_card(self.settings, self.iid)
        self.assertEqual(card["status_label"], "Snimak spreman")
        self.assertEqual(len(card["sales"]), 1)
        sale = card["sales"][0]
        self.assertEqual(sale["label"], "Kucano 23.11.2026. u 09:00")
        self.assertEqual([c["name"] for c in sale["cameras"]], ["Šank levo", "Točilica", "Šank desno"])
        self.assertTrue(all(c["ready"] for c in sale["cameras"]))
        text = str(card)
        for bad in (".mkv", ".mp4", str(self.clip_root), "channel_", "/media"):
            self.assertNotIn(bad, text)
        path = store.clip_video(self.settings, self.iid, sale["slug"], 2)
        self.assertEqual(path.suffix, ".mp4")
        self.assertIn(self.clip_root.resolve(), path.parents)
        with self.assertRaises(store.UnknownClip):
            store.clip_video(self.settings, self.iid, "../../etc", 2)

    def test_vidim_ga_is_catalog_evidence_waiting_for_admin_never_raw(self):
        raw = self.tmp / "raw"
        raw_before = sorted(raw.rglob("*")) if raw.exists() else []
        result = self.seen(store.clip_card(self.settings, self.iid))
        ev = result["evidence"]
        self.assertEqual(ev["status"], "pending")
        self.assertEqual((ev["width"], ev["height"]), (160, 90))  # full camera frame, box drawn on it
        with store._open_for_read(self.settings) as conn:
            self.assertEqual(conn.execute("SELECT source FROM evidence WHERE evidence_id = ?", (ev["evidence_id"],)).fetchone()[0], "clip")
        self.assertEqual(result["card"]["status_label"], "Viđeno na snimku")
        seen = self.events("seen")[0]["detail"]
        self.assertEqual((seen["channel"], seen["second"], seen["box"]), (2, 12.5, [0.25, 0.2, 0.3, 0.4]))
        self.assertEqual(sorted(raw.rglob("*")) if raw.exists() else [], raw_before)
        media = self.settings.media_dir.resolve()
        self.assertTrue(any(p.is_file() for p in media.rglob("*")))
        for root in self.settings.forbidden_roots:
            self.assertNotIn(Path(root).resolve(), media.parents)
        # the admin decides it like a pasted screenshot
        catalog.decide_evidence(self.settings, ev["evidence_id"], "confirmed", "Aleksa")
        self.assertEqual(catalog.get_evidence(self.settings, ev["evidence_id"])["status"], "confirmed")

    def test_box_must_be_one_real_rectangle(self):
        card = store.clip_card(self.settings, self.iid)
        for box in ((0, 0, 0, 0), (0.9, 0.9, 0.3, 0.3), (0.1, 0.1)):
            with self.assertRaises(catalog.CatalogError):
                self.seen(card, box=box)
        with self.assertRaises(catalog.CatalogError):
            self.seen(card, second=9999)
        self.assertEqual(self.events("seen"), [])

    def test_nije_na_snimku_goes_to_next_sale_then_nema_snimka(self):
        card = store.clip_card(self.settings, self.iid)
        card = store.mark_not_visible(self.settings, self.iid, card["sales"][0]["slug"], "Goca", "labeler")
        self.assertEqual(card["sales"], [])
        self.assertEqual(card["status_label"], "Nema snimka")
        report = self.run_job()  # the job fetches the next sale
        self.assertIn({"identity_id": self.iid, "sale": "2026-11-22 12:00", "channels": ["1", "2", "3"]},
                      [{k: d[k] for k in ("identity_id", "sale", "channels")} for d in report["downloaded"]])
        card = store.clip_card(self.settings, self.iid)
        self.assertEqual(card["sales"][0]["label"], "Kucano 22.11.2026. u 12:00")


class SendToLabelingTests(CardBase):
    """Ticket 18 on a fake clip: cut ±10 s around "Vidim ga", model and class preselected, nothing in RAW."""

    def give_class(self):
        catalog.record_decision(self.settings, "class_target", "Aleksa", "admin", identity_id=self.iid,
                                payload={"model": "Glasses", "class": "CASA", "open_new": False}, reason="isto piće")

    def test_cut_10s_around_each_seen_on_that_camera_with_model_and_class(self):
        import json
        import subprocess
        self.give_class()
        card = store.clip_card(self.settings, self.iid)
        self.assertEqual(card["target"], {"model": "Glasses", "class": "CASA"})
        self.seen(card, channel=2, second=25.0)
        self.assertEqual(store.clip_card(self.settings, self.iid)["to_send"], 1)
        raw = self.tmp / "raw"
        raw_before = sorted(raw.rglob("*")) if raw.exists() else []
        result = store.send_to_labeling(self.settings, self.iid, "Goca", "labeler")
        self.assertEqual((result["sent"], result["videos"], result["model"], result["class"]), (1, ["Točilica"], "Glasses", "CASA"))
        batch = self.tmp / "videos" / "handoff_katalog"
        videos = sorted(p.name for p in batch.glob("*.mp4"))
        self.assertEqual(videos, ["BLAZNAVAC_KATALOG_0400_SANK_TOCILICA_20261123090015.mp4"])  # ring 09:00 + 15 s
        name = videos[0].lower()
        self.assertIn("sank_tocilica", name)  # IntelliCup picks the camera crop by this token
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(batch / videos[0])], capture_output=True, text=True).stdout)
        self.assertAlmostEqual(dur, 20.0, delta=1.0)
        hint = json.loads((batch / videos[0].replace(".mp4", "_hints.json")).read_text())
        self.assertEqual((hint["preselect"], hint["at_second"], hint["tlwh_norm"]),
                         ({"model": "Glasses", "class": "CASA"}, 10.0, [0.25, 0.2, 0.3, 0.4]))
        info = json.loads((batch / videos[0].replace(".mp4", "_info.json")).read_text())
        self.assertEqual(info["case_type"], "KATALOG")
        self.assertTrue((batch / "_session.json").is_file())
        self.assertEqual(sorted(raw.rglob("*")) if raw.exists() else [], raw_before)
        self.assertEqual(store.clip_card(self.settings, self.iid)["to_send"], 0)
        with self.assertRaises(catalog.CatalogError):
            store.send_to_labeling(self.settings, self.iid, "Goca", "labeler")  # nothing new
        self.assertEqual(len(self.events("sent")), 1)

    def test_cut_starts_at_zero_near_the_ring(self):
        self.give_class()
        self.seen(store.clip_card(self.settings, self.iid), channel=1, second=4.0)
        store.send_to_labeling(self.settings, self.iid, "Goca", "labeler")
        batch = self.tmp / "videos" / "handoff_katalog"
        self.assertEqual([p.name for p in batch.glob("*.mp4")], ["BLAZNAVAC_KATALOG_0400_SANK_LEVO_20261123090000.mp4"])
        import json
        hint = json.loads(next(batch.glob("*_hints.json")).read_text())
        self.assertEqual(hint["at_second"], 4.0)

    def test_article_card_offers_the_footage(self):
        v = self.identity("NEW DRINK")
        card = catalog.article_card(self.settings, v["article_id"])
        self.assertTrue(next(x for x in card["variations"] if x["identity_id"] == self.iid)["has_clips"])

    def test_needs_a_class_first(self):
        self.seen(store.clip_card(self.settings, self.iid))
        with self.assertRaises(catalog.CatalogError):
            store.send_to_labeling(self.settings, self.iid, "Goca", "labeler")
        self.assertFalse((self.tmp / "videos" / "handoff_katalog").exists())

    def test_till_name_is_escaped_in_the_banner_text(self):
        import json
        self.write_pos("d.csv", [pos_row("0700", "<b>NOVO</b>", "2026.11.23 10:00")])
        catalog.import_catalog(self.settings)
        self.run_job()
        iid = self.identity("<b>NOVO</b>")["identity_id"]
        catalog.record_decision(self.settings, "class_target", "Aleksa", "admin", identity_id=iid,
                                payload={"model": "Glasses", "class": "CASA", "open_new": False}, reason="x")
        card = store.clip_card(self.settings, iid)
        store.mark_seen(self.settings, iid, card["sales"][0]["slug"], 3, 30, [0.1, 0.1, 0.2, 0.2], "Goca", "labeler")
        store.send_to_labeling(self.settings, iid, "Goca", "labeler")
        info = json.loads(next((self.tmp / "videos" / "handoff_katalog").glob("*_info.json")).read_text())
        self.assertNotIn("<b>", info["instruction"])
        self.assertIn("&lt;b&gt;", info["instruction"])


if __name__ == "__main__":
    unittest.main()
