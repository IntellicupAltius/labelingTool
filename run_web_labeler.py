#!/usr/bin/env python3
from __future__ import annotations

import os

import uvicorn


def main():
    host = os.getenv("LABELER_HOST", "127.0.0.1")
    port = int(os.getenv("LABELER_PORT", "8000"))
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


