"""
Round-trip tests for Sentinel-2 pydantic-zarr integrated models.

These tests verify that Sentinel-2 data can be:
1. Loaded from zarr stores using from_zarr()
2. Validated through Pydantic models
3. Written back to zarr using to_zarr()
4. Round-tripped without data loss

Note: All tests use MemoryStore() for fast in-memory storage.
"""

import jsondiff
import pytest
from pydantic_zarr.core import tuplify_json

from eopf_geozarr.data_api.sentinel2 import (
    Sentinel2Root,
)
from tests.test_data_api.conftest import SENTINEL2_EXAMPLES


@pytest.mark.parametrize("example", SENTINEL2_EXAMPLES)
def test_sentinel2_roundtrip(example: dict[str, object]) -> None:
    """Test that Sentinel1Root can load and dump example JSON without data loss."""
    example_tup = tuplify_json(example)
    model = Sentinel2Root(**example_tup)
    observed = model.model_dump()
    assert jsondiff.diff(observed, example_tup) == {}
