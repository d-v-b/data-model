"""
Round-trip tests for Sentinel-2C pydantic-zarr integrated models.

These tests verify that Sentinel-2C data can be:
1. Loaded from example JSON data using direct instantiation
2. Validated through Pydantic models
3. Round-tripped without data loss

Note: Documentation code examples are tested separately via pytest-examples
from the markdown files in docs/models/sentinel2.md
"""

import jsondiff
import pytest
from pydantic_zarr.core import tuplify_json

from eopf_geozarr.data_api.s2b import Sentinel2BRoot
from tests.test_data_api.conftest import S2B_EXAMPLES


@pytest.mark.parametrize("example", S2B_EXAMPLES)
def test_sentinel2b_roundtrip(example: dict[str, object]) -> None:
    """Test that we can round-trip JSON data without loss"""
    example_tup = tuplify_json(example)
    model = Sentinel2BRoot(**example_tup)
    assert jsondiff.diff(model.model_dump(), example_tup) == {}
