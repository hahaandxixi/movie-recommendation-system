from __future__ import annotations

from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent

from bootstrap import ensure_source_package

ensure_source_package()

from myidea.web.app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
