import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parents[1]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from bootstrap import ensure_source_package


ensure_source_package()
