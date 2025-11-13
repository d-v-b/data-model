"""
Round-trip tests for Sentinel-2 pydantic-zarr integrated models.

These tests verify that Sentinel-2 data can be:
1. Loaded from example JSON data using direct instantiation
2. Validated through Pydantic models
3. Round-tripped without data loss

Note: Documentation code examples are tested separately via pytest-examples
from the markdown files in docs/models/sentinel2.md
"""

import pytest

from eopf_geozarr.data_api.s2a import Sentinel2ARoot
from tests.test_data_api.conftest import S2A_EXAMPLES


@pytest.mark.parametrize("example", S2A_EXAMPLES)
def test_sentinel2_roundtrip(example: dict[str, object]) -> None:
    """Test that we can round-trip JSON data without loss"""
    model1 = Sentinel2ARoot(**example)
    dumped = model1.model_dump()
    model2 = Sentinel2ARoot(**dumped)
    assert model1.model_dump() == model2.model_dump()
