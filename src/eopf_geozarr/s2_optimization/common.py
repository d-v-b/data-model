from __future__ import annotations

from importlib.util import find_spec

DISTRIBUTED_AVAILABLE = find_spec("distributed") is not None
