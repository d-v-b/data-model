from __future__ import annotations

import json
from pathlib import Path


def _load_zcm_examples() -> dict[str, dict[str, object]]:
    """Load a ZCM multiscale example."""
    examples_dir = Path(__file__).parent / "zcm_multiscales_examples"
    return {
        path.name: json.loads(path.read_text()) for path in examples_dir.glob("*.json")
    }


ZCM_MULTISCALES_EXAMPLES = _load_zcm_examples()
