"""
Round-trip tests for Sentinel-1C pydantic-zarr integrated models.

These tests verify that Sentinel-1C data can be:
1. Loaded from example JSON data using direct instantiation
2. Validated through Pydantic models
3. Round-tripped without data loss

Note: Documentation code examples are tested separately via pytest-examples
from the markdown files in docs/models/sentinel2.md
"""

import jsondiff
import pytest
from pydantic_zarr.core import tuplify_json

from eopf_geozarr.data_api.s1c import Sentinel1CRoot
from tests.test_data_api.conftest import S1C_EXAMPLES


@pytest.mark.parametrize("example", S1C_EXAMPLES)
def test_sentinel1c_roundtrip(example: dict[str, object]) -> None:
    """Test that we can round-trip JSON data without loss"""
    example_tup = tuplify_json(example)
    model = Sentinel1CRoot(**example_tup)
    assert jsondiff.diff(model.model_dump(), example_tup) == {}
