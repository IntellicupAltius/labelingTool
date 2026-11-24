"""Footage of a new article (KATALOG-1 ticket 16, D5).

For every article that waits for a class the job prepares footage from the NVR: up to 3 sales, all cameras, from the
ring time (POS column 16) to +10 minutes, nothing before the ring (clock check 24.11.2026: POS and cameras agree to
about a minute, the drink appears 0-3 min after the ring).

Rules:
- download over RTSP with the same downloader the nightly sync uses, with the NVR secrets sourced the same way;
- never at the same time as the nightly sync: the job holds the sync's own lock (/tmp/blaznavac_nvr_sync.lock) and
  simply skips this run when it is taken; it also waits while inference or training runs;
- clips live in their own Catalog folder on the Transcend disk, never in RAW; the job refuses when the disk is not
  mounted; it stops before passing the size cap (30 GB to start);
- one sale per article first (newest, from different days, the minute with the fewest other rings), the second and
  third only when there is room and nobody saw the drink yet ("Vidim ga");
- a sale the NVR no longer has is "missing"; with no sale left the article shows "nema snimka" and waits for its next
  sale; a warning (finding) on day 5 if an article still has no footage;
- clips of an article are deleted 30 days after its decision (the cuts sent to labeling live elsewhere and stay).

The job runs outside the web process: python -m web_labeler.catalog clips [--dry-run].
"""
from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

GB = 1024 ** 3


class ClipError(Exception):
    """The job refused to run; nothing was downloaded or deleted."""


@dataclass(frozen=True)
class ClipSettings:
    root: Path
    mount: Optional[Path]
    downloader: Tuple[str, ...]
    secrets_env: Optional[Path] = None
    lock_path: Path = Path("/tmp/blaznavac_nvr_sync.lock")
    cap_bytes: int = 30 * GB
    nvr_days: int = 30
    sales_per_article: int = 3
    window_minutes: int = 10
    channels: Tuple[int, ...] = (1, 2, 3)
    busy_patterns: Tuple[str, ...] = ("pipeline.run_day", "training_pipeline", "run_training")
    warn_days: int = 5
    keep_days: int = 30
    crowd_minutes: int = 5
    min_bytes: int = 100 * 1024
    est_bytes_per_channel_minute: int = 45 * 1024 * 1024  # 2.7 GB per camera hour (Nov 2026 footage)
    attempts_before_missing: int = 2
    stop_hour: int = 22  # no new sale after this hour: the nightly sync (23:55) must find its lock free
    enabled: bool = False  # real NVR downloads only after the owner switches them on (catalog_clip_enabled)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Optional["ClipSettings"]:
        root = str(cfg.get("catalog_clip_root") or "").strip()
        downloader = cfg.get("catalog_clip_downloader")
        if not root or not downloader:
            return None
        mount = str(cfg.get("catalog_clip_mount") or "").strip()
        secrets = str(cfg.get("catalog_clip_secrets_env") or "").strip()
        base = cls(Path(root).expanduser(), Path(mount).expanduser() if mount else None,
                   tuple(str(x) for x in downloader), Path(secrets).expanduser() if secrets else None)
        extra: Dict[str, Any] = {}
        if cfg.get("catalog_clip_enabled") is not None:
            extra["enabled"] = bool(cfg["catalog_clip_enabled"])
        if cfg.get("catalog_clip_lock"):
            extra["lock_path"] = Path(str(cfg["catalog_clip_lock"])).expanduser()
        if cfg.get("catalog_clip_cap_gb") is not None:
            extra["cap_bytes"] = int(float(cfg["catalog_clip_cap_gb"]) * GB)
        for key, attr in (("catalog_clip_nvr_days", "nvr_days"), ("catalog_clip_sales_per_article", "sales_per_article"),
                          ("catalog_clip_window_minutes", "window_minutes"), ("catalog_clip_warn_days", "warn_days"),
                          ("catalog_clip_keep_days", "keep_days"), ("catalog_clip_min_bytes", "min_bytes"),
                          ("catalog_clip_stop_hour", "stop_hour")):
            if cfg.get(key) is not None:
                extra[attr] = int(cfg[key])
        if cfg.get("catalog_clip_est_mb_per_channel_minute") is not None:
            extra["est_bytes_per_channel_minute"] = int(float(cfg["catalog_clip_est_mb_per_channel_minute"]) * 1024 * 1024)
        if cfg.get("catalog_clip_channels"):
            extra["channels"] = tuple(int(c) for c in cfg["catalog_clip_channels"])
        if cfg.get("catalog_clip_busy_patterns") is not None:
            extra["busy_patterns"] = tuple(str(p) for p in cfg["catalog_clip_busy_patterns"])
        return cls(**{**base.__dict__, **extra})

    def sale_estimate(self) -> int:
        return self.est_bytes_per_channel_minute * self.window_minutes * len(self.channels)


# ---------------- sale choice ----------------

@dataclass(frozen=True)
class Sale:
    ring_time: datetime
    crowd: int  # other rings within crowd_minutes

    @property
    def key(self) -> str:
        return self.ring_time.strftime("%Y-%m-%d %H:%M")

    @property
    def day(self) -> str:
        return self.ring_time.strftime("%Y-%m-%d")


def rank_sales(ring_times: Sequence[datetime], all_rings: Sequence[datetime], now: datetime, settings: ClipSettings) -> List[Sale]:
    """Best sales first: one per day (the least crowded minute that day, newer on a tie), days newest first.

    Only sales the NVR should still have (nvr_days) and whose 10-minute window is over."""
    oldest = now - timedelta(days=settings.nvr_days)
    window = timedelta(minutes=settings.window_minutes)
    crowd_gap = timedelta(minutes=settings.crowd_minutes)
    rings = sorted(all_rings)
    import bisect
    best: Dict[str, Sale] = {}
    for t in sorted(set(ring_times)):
        if t < oldest or t + window > now:
            continue
        lo = bisect.bisect_left(rings, t - crowd_gap)
        hi = bisect.bisect_right(rings, t + crowd_gap)
        sale = Sale(t, max(0, hi - lo - ring_times.count(t)))
        current = best.get(sale.day)
        if current is None or (sale.crowd, -sale.ring_time.timestamp()) < (current.crowd, -current.ring_time.timestamp()):
            best[sale.day] = sale
    return sorted(best.values(), key=lambda s: s.ring_time, reverse=True)


# ---------------- guards ----------------

def check_disk(settings: ClipSettings) -> None:
    if settings.mount is not None:
        if not os.path.ismount(settings.mount):
            raise ClipError(f"Disk za snimke nije montiran ({settings.mount}). Ništa nije skinuto.")
        try:
            settings.root.resolve().relative_to(settings.mount.resolve())
        except ValueError:
            raise ClipError(f"Folder za snimke {settings.root} nije na disku {settings.mount}.")


def forbid_inside(settings: ClipSettings, roots: Sequence[Path]) -> None:
    root = settings.root.resolve()
    for r in roots:
        try:
            root.relative_to(Path(r).resolve())
        except ValueError:
            continue
        raise ClipError(f"Folder za snimke ne sme biti unutar {r}")


def busy_processes(patterns: Sequence[str], proc: Path = Path("/proc")) -> List[str]:
    """Command lines of running inference/training (any pattern in the command line); our own process is skipped."""
    found: List[str] = []
    if not patterns or not proc.is_dir():
        return found
    me = os.getpid()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        if cmd and any(p in cmd for p in patterns):
            found.append(cmd[:200])
    return found


@contextmanager
def nvr_lock(path: Path) -> Iterator[bool]:
    """The nightly sync's own lock, never waited for: True when we hold it, False when the sync (or another job) runs."""
    handle = open(path, "a")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


@contextmanager
def _no_lock() -> Iterator[bool]:
    yield True


def folder_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


# ---------------- download ----------------

def sale_folder(settings: ClipSettings, identity_id: int, sale: Sale) -> Path:
    return settings.root / str(identity_id) / sale.ring_time.strftime("%Y%m%d_%H%M")


def expected_file(folder: Path, channel: int, start: datetime, minutes: int) -> Path:
    """Where the nightly downloader writes a chunk: <out>/<date>/channel_N/chN_<date>_<HH-MM-SS>_<M>min.mkv"""
    day = start.strftime("%Y-%m-%d")
    return folder / day / f"channel_{channel}" / f"ch{channel}_{day}_{start.strftime('%H-%M-%S')}_{minutes}min.mkv"


Runner = Callable[[List[str], Optional[Path]], int]


def _run(argv: List[str], secrets_env: Optional[Path]) -> int:
    """Run the downloader like the nightly sync: secrets sourced into its environment, never printed or logged."""
    if secrets_env is not None:
        argv = ["bash", "-c", 'set -a; source "$0"; set +a; exec "$@"', str(secrets_env), *argv]
    return subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3 * 3600).returncode


def download_sale(settings: ClipSettings, identity_id: int, sale: Sale, runner: Runner = _run) -> Dict[str, Any]:
    """All cameras, one after the other (gentle on the NVR). A camera counts only with a real file."""
    folder = sale_folder(settings, identity_id, sale)
    end = sale.ring_time + timedelta(minutes=settings.window_minutes)
    channels: Dict[str, str] = {}
    size = 0
    for ch in settings.channels:
        target = expected_file(folder, ch, sale.ring_time, settings.window_minutes)
        argv = [*settings.downloader, "--start", sale.ring_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "--end", end.strftime("%Y-%m-%dT%H:%M:%S"), "--channels", str(ch),
                "--chunk-minutes", str(settings.window_minutes), "--output", str(folder)]
        try:
            runner(argv, settings.secrets_env)
        except (OSError, subprocess.SubprocessError):
            pass
        if target.is_file() and target.stat().st_size >= settings.min_bytes:
            channels[str(ch)] = str(target.relative_to(settings.root))
            size += target.stat().st_size
        elif target.exists():
            target.unlink()  # a stub the NVR sent back: not footage
    if not channels and folder.exists():
        remove_folder(settings, folder)
    return {"channels": channels, "bytes": size, "folder": str(folder.relative_to(settings.root))}


def remove_folder(settings: ClipSettings, folder: Path) -> None:
    root = settings.root.resolve()
    target = folder.resolve()
    if target == root or root not in target.parents:
        raise ClipError(f"Odbijeno brisanje van foldera za snimke: {folder}")
    shutil.rmtree(target)
    parent = target.parent
    while parent != root and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent


# ---------------- state per article ----------------

CLIP_STATUS_LABELS = {"ready": "Snimak spreman", "waiting": "Čeka snimak", "late": "Snimak kasni",
                      "missing": "Nema snimka", "seen": "Viđeno na snimku", "deleted": "Snimak obrisan"}


@dataclass
class SaleState:
    key: str
    status: str = "planned"  # planned | downloaded | missing | failed | deleted
    attempts: int = 0
    channels: Dict[str, str] = field(default_factory=dict)
    bytes: int = 0
    folder: Optional[str] = None
    not_visible: bool = False
    seen: List[Dict[str, Any]] = field(default_factory=list)


def fold_events(events: Sequence[Mapping[str, Any]], attempts_before_missing: int = 2) -> Dict[int, Dict[str, Any]]:
    """Events (oldest first) -> per identity: first_planned, sales {key: SaleState}, warned."""
    out: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"first_planned": None, "sales": {}, "warned": False})
    for e in events:
        st = out[int(e["identity_id"])]
        action = e["action"]
        if action == "planned":
            st["first_planned"] = st["first_planned"] or e["at"]
            continue
        if action == "warned":
            st["warned"] = True
            continue
        key = e.get("sale_time")
        if not key:
            continue
        sale = st["sales"].setdefault(key, SaleState(key))
        detail = e.get("detail") or {}
        if action == "downloaded":
            sale.status, sale.channels, sale.bytes, sale.folder = "downloaded", dict(detail.get("channels") or {}), int(e.get("bytes") or 0), e.get("folder")
        elif action == "failed":
            sale.attempts += 1
            sale.status = "missing" if sale.attempts >= attempts_before_missing else "failed"
        elif action == "missing":
            sale.status = "missing"
        elif action == "deleted":
            sale.status = "deleted"
        elif action == "not_visible":
            sale.not_visible = True
        elif action == "seen":
            sale.seen.append(dict(detail, event_id=e.get("event_id")))
    return dict(out)


def article_status(state: Optional[Mapping[str, Any]], ranked_keys: Sequence[str], today: str, warn_days: int) -> str:
    if not state:
        return "waiting"
    sales: Mapping[str, SaleState] = state["sales"]
    if any(s.seen for s in sales.values()):
        return "seen"
    if any(s.status == "downloaded" and not s.not_visible for s in sales.values()):
        return "ready"
    if sales and all(s.status == "deleted" for s in sales.values()):
        return "deleted"
    untried = [k for k in ranked_keys if k not in sales or sales[k].status in ("planned", "failed")]
    if not untried:
        return "missing"
    first = state.get("first_planned")
    if first and (datetime.fromisoformat(today) - datetime.fromisoformat(first[:10])).days >= warn_days:
        return "late"
    return "waiting"


def _usable(sales: Mapping[str, SaleState]) -> int:
    return sum(1 for s in sales.values() if s.status == "downloaded" and not s.not_visible)


# ---------------- the job ----------------

def run_job(settings: Any, now: Optional[datetime] = None, dry_run: bool = False, runner: Runner = _run,
            proc: Path = Path("/proc"), clock: Callable[[], datetime] = datetime.now) -> Dict[str, Any]:
    """One run of the footage job. `settings` is the CatalogSettings (with .clips)."""
    from web_labeler.catalog import store
    from web_labeler.catalog.assortment import normalize_name, read_pos_folder

    clips: Optional[ClipSettings] = settings.clips
    if clips is None:
        raise ClipError("Snimci nisu podešeni (catalog_clip_root, catalog_clip_downloader).")
    now = now or datetime.now()
    today = now.date().isoformat()
    report: Dict[str, Any] = {"ran": False, "dry_run": dry_run, "downloaded": [], "missing": [], "failed": [],
                              "deleted": [], "warned": [], "planned": [], "stopped": None, "used_bytes": 0}
    if not clips.enabled and not dry_run:
        raise ClipError("Skidanje snimaka još nije uključeno (catalog_clip_enabled). Proba: --dry-run.")
    check_disk(clips)
    forbid_inside(clips, settings.forbidden_roots)
    busy = busy_processes(clips.busy_patterns, proc)
    if busy and not dry_run:
        report["waiting"] = "Radi inference ili trening; snimci čekaju sledeće pokretanje."
        report["busy"] = busy
        return report
    report["busy"] = busy
    with (_no_lock() if dry_run else nvr_lock(clips.lock_path)) as held:
        if not held:
            report["waiting"] = "Radi noćno skidanje sa NVR-a (brava je zauzeta); snimci čekaju sledeće pokretanje."
            return report
        report["ran"] = True
        ctx = store.clip_context(settings, today)
        candidates: Dict[int, Mapping[str, Any]] = {c["identity_id"]: c for c in ctx["candidates"]}
        states = fold_events(ctx["events"], clips.attempts_before_missing)
        if settings.pos_dir is None or settings.pos_adapter is None:
            raise ClipError("POS folder nije podešen.")
        rows = read_pos_folder(settings.pos_dir, settings.pos_adapter).rows
        all_rings = [r.ring_time for r in rows if r.quantity > 0]
        by_identity: Dict[Tuple[str, str], List[datetime]] = defaultdict(list)
        for r in rows:
            if r.quantity > 0:
                by_identity[(r.code, normalize_name(r.name))].append(r.ring_time)
        ranked: Dict[int, List[Sale]] = {}
        for iid, c in candidates.items():
            ranked[iid] = rank_sales(by_identity.get((c["code"], normalize_name(c["name"])), []), all_rings, now, clips)
        events: List[Dict[str, Any]] = []
        findings: List[Dict[str, Any]] = []

        def log(event: Dict[str, Any]) -> None:
            event.setdefault("at", store._now())
            events.append(event)
            if not dry_run:
                store.add_clip_events(settings, [event], [])
                events.pop()

        # 1. new candidates are noted once (the day-5 warning counts from here)
        for iid in candidates:
            if not states.get(iid, {}).get("first_planned"):
                report["planned"].append(iid)
                log({"identity_id": iid, "action": "planned"})
                states.setdefault(iid, {"first_planned": None, "sales": {}, "warned": False})["first_planned"] = now.isoformat()

        # 2. delete clips 30 days after the decision on an article that no longer waits
        for iid, st in states.items():
            if iid in candidates:
                continue
            decided = ctx["decided_at"].get(iid)
            if not decided or (now - datetime.fromisoformat(decided[:19])).days < clips.keep_days:
                continue
            for sale in st["sales"].values():
                if sale.status == "downloaded" and sale.folder:
                    if not dry_run:
                        folder = clips.root / sale.folder
                        if folder.exists():
                            remove_folder(clips, folder)
                    report["deleted"].append({"identity_id": iid, "sale": sale.key})
                    log({"identity_id": iid, "action": "deleted", "sale_time": sale.key, "folder": sale.folder})
                    sale.status = "deleted"

        # 3. day-5 warning, once per article
        for iid, c in candidates.items():
            st = states.get(iid)
            keys = [s.key for s in ranked[iid]]
            status = article_status(st, keys, today, clips.warn_days)
            if st and not st["warned"] and (status == "late" or (status == "missing" and st["first_planned"]
                                                                  and (datetime.fromisoformat(today) - datetime.fromisoformat(st["first_planned"][:10])).days >= clips.warn_days)):
                text = (f"Za \"{c['name'].strip()}\" (šifra {c['code']}) posle {clips.warn_days} dana još nema snimka. "
                        + ("Na NVR-u više nema snimka njegovih prodaja; čeka sledeću prodaju." if status == "missing"
                           else "Skidanje je na redu čim NVR i disk budu slobodni."))
                findings.append({"kind": "clip_late", "day": today, "title": f"Snimak kasni: {c['name'].strip()}", "text": text,
                                 "data": {"identity_id": iid, "status": status}})
                report["warned"].append(iid)
                log({"identity_id": iid, "action": "warned"})
                st["warned"] = True

        # 4. downloads, breadth first: every article one sale before any gets a second.
        #    Each sale is tried at most once per run; after a failure the next sale is tried; 3 failures in a row stop
        #    the run (the NVR probably does not answer).
        budget = {"used": folder_size(clips.root), "failures_in_row": 0}
        report["used_bytes"] = budget["used"]
        tried: set = set()

        def one_sale(iid: int, sale: Sale, sales: Dict[str, SaleState]) -> None:
            if dry_run:
                report["downloaded"].append({"identity_id": iid, "sale": sale.key, "would": True})
                sales[sale.key] = SaleState(sale.key, status="downloaded")
                budget["used"] += clips.sale_estimate()
                return
            result = download_sale(clips, iid, sale, runner)
            if result["channels"]:
                budget["used"] += result["bytes"]
                budget["failures_in_row"] = 0
                sales[sale.key] = SaleState(sale.key, "downloaded", 0, result["channels"], result["bytes"], result["folder"])
                report["downloaded"].append({"identity_id": iid, "sale": sale.key, "channels": sorted(result["channels"]),
                                             "bytes": result["bytes"]})
                log({"identity_id": iid, "action": "downloaded", "sale_time": sale.key, "day": sale.day,
                     "bytes": result["bytes"], "folder": result["folder"], "detail": {"channels": result["channels"]}})
                return
            budget["failures_in_row"] += 1
            prev = sales.get(sale.key) or SaleState(sale.key)
            prev.attempts += 1
            prev.status = "missing" if prev.attempts >= clips.attempts_before_missing else "failed"
            sales[sale.key] = prev
            report["missing" if prev.status == "missing" else "failed"].append({"identity_id": iid, "sale": sale.key})
            log({"identity_id": iid, "action": "failed", "sale_time": sale.key, "day": sale.day})
            if budget["failures_in_row"] >= 3:
                report["stopped"] = "NVR tri puta zaredom nije dao snimak; skidanje je stalo do sledećeg pokretanja."

        order = sorted(candidates, key=lambda i: candidates[i].get("last_sale") or "", reverse=True)
        for round_no in range(1, clips.sales_per_article + 1):
            for iid in order:
                sales: Dict[str, SaleState] = states.setdefault(iid, {"first_planned": None, "sales": {}, "warned": False})["sales"]
                while not report["stopped"] and not any(s.seen for s in sales.values()) and _usable(sales) < round_no:
                    nxt = next((s for s in ranked[iid] if (iid, s.key) not in tried
                                and (s.key not in sales or sales[s.key].status in ("planned", "failed"))), None)
                    if nxt is None:
                        break
                    if not dry_run and clock().hour >= clips.stop_hour:
                        report["stopped"] = (f"Posle {clips.stop_hour}:00 se ne skida, da noćno skidanje (23:55) nađe NVR "
                                             "slobodan. Nastavlja se sledeći put.")
                        break
                    if budget["used"] + clips.sale_estimate() > clips.cap_bytes:
                        report["stopped"] = (f"Dostignuta granica za snimke ({clips.cap_bytes / GB:.0f} GB, zauzeto "
                                             f"{budget['used'] / GB:.1f} GB). Skidanje je stalo.")
                        findings.append({"kind": "clip_budget", "day": today, "title": "Granica za snimke je dostignuta",
                                         "text": report["stopped"] + " Obriši stare snimke ili povećaj granicu.",
                                         "data": {"used_bytes": budget["used"], "cap_bytes": clips.cap_bytes}})
                        break
                    tried.add((iid, nxt.key))
                    one_sale(iid, nxt, sales)
                if report["stopped"]:
                    break
            if report["stopped"]:
                break
        report["used_bytes"] = budget["used"]
        if findings and not dry_run:
            store.add_clip_events(settings, [], findings)
        report["findings"] = [f["kind"] for f in findings]
        report["status"] = {iid: article_status(states.get(iid), [s.key for s in ranked[iid]], today, clips.warn_days)
                            for iid in candidates}
        if not dry_run and settings.db_path is not None:
            write_status(settings.db_path.parent, report["status"], now)
    return report


STATUS_FILE = "clip_status.json"


def write_status(folder: Path, status: Mapping[int, str], now: datetime) -> None:
    """Derived view for the inbox (rebuilt by every run): article -> footage status."""
    import json
    tmp = folder / (STATUS_FILE + ".tmp")
    tmp.write_text(json.dumps({"at": now.isoformat(timespec="seconds"), "status": {str(k): v for k, v in status.items()}},
                              ensure_ascii=False), encoding="utf-8")
    tmp.replace(folder / STATUS_FILE)


def read_status(folder: Path) -> Dict[int, str]:
    import json
    try:
        data = json.loads((folder / STATUS_FILE).read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in data.get("status", {}).items()}
    except (OSError, ValueError):
        return {}
