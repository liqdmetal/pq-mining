"""Print the candidate profile's current attack-screen report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pqmine.profiles import candidate_128_profile


if __name__ == "__main__":
    print(json.dumps(candidate_128_profile().summary(), indent=2, sort_keys=True))
