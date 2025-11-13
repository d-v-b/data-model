"""
Round-trip tests for Sentinel-1 pydantic-zarr integrated models.

These tests verify that Sentinel-1 data can be:
1. Loaded from example JSON data using direct instantiation
2. Validated through Pydantic models
3. Round-tripped without data loss

Note: Documentation code examples are tested separately via pytest-examples
from the markdown files in docs/models/sentinel1.md
"""

import pytest

from eopf_geozarr.data_api.s1a import Sentinel1ARoot
from tests.test_data_api.conftest import S1A_EXAMPLES


@pytest.mark.parametrize("example", S1A_EXAMPLES)
def test_sentinel1_roundtrip(example: dict[str, object]) -> None:
    """Test that we can round-trip JSON data without loss"""
    model1 = Sentinel1ARoot(**example)
    dumped = model1.model_dump()
    model2 = Sentinel1ARoot(**dumped)
    assert model1.model_dump() == model2.model_dump()
