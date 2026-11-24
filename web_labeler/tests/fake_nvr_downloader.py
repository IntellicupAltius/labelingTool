"""Stand-in for IntelliCup/utils/video_utils/test_download.py in tests: same arguments, same output layout, no NVR.

FAKE_NVR_MISSING   comma list of --start values (2026-11-23T09:00:00) the NVR "no longer has": nothing is written
FAKE_NVR_BYTES     size of each written file (default 200000)
FAKE_NVR_LOG       file that gets one line per call (arguments, and whether FAKE_NVR_SECRET was in the environment)
FAKE_NVR_NEED_SECRET  when set, nothing is written unless FAKE_NVR_SECRET is in the environment (sourced secrets)
"""
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True)
ap.add_argument("--end", required=True)
ap.add_argument("--channels", type=int, nargs="+", required=True)
ap.add_argument("--chunk-minutes", type=int, default=60)
ap.add_argument("--output", required=True)
a = ap.parse_args()
log = os.environ.get("FAKE_NVR_LOG")
if log:
    with open(log, "a") as f:
        f.write(" ".join(sys.argv[1:]) + f" secret={'FAKE_NVR_SECRET' in os.environ}\n")
if a.start in (os.environ.get("FAKE_NVR_MISSING") or "").split(","):
    sys.exit(1)
if os.environ.get("FAKE_NVR_NEED_SECRET") and "FAKE_NVR_SECRET" not in os.environ:
    sys.exit(2)
start = datetime.fromisoformat(a.start)
minutes = int((datetime.fromisoformat(a.end) - start).total_seconds() // 60)
day = start.strftime("%Y-%m-%d")
for ch in a.channels:
    out = Path(a.output) / day / f"channel_{ch}" / f"ch{ch}_{day}_{start.strftime('%H-%M-%S')}_{minutes}min.mkv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"\0" * int(os.environ.get("FAKE_NVR_BYTES", "200000")))
