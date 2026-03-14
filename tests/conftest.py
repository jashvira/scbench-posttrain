from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VGB_ROOT = ROOT / "external" / "VisGeomBench"

for path in (ROOT, VGB_ROOT):
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
