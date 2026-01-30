#!/usr/bin/env python3
from __future__ import annotations

import os

import uvicorn
import threading
import time
import webbrowser


def main():
    host = os.getenv("LABELER_HOST", "127.0.0.1")
    port = int(os.getenv("LABELER_PORT", "8000"))
    open_browser = os.getenv("LABELER_OPEN_BROWSER", "0").strip() not in ("", "0", "false", "False")

    if open_browser:
        def _open():
            time.sleep(1.0)
            try:
                webbrowser.open(f"http://{host}:{port}", new=1)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()
    uvicorn.run(
        "web_labeler.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        workers=1,
    )


if __name__ == "__main__":
    main()


