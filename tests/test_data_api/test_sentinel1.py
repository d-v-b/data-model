import jsondiff
import pytest
from pydantic_zarr.core import tuplify_json

from eopf_geozarr.data_api.sentinel1 import Sentinel1Root
from tests.test_data_api.conftest import SENTINEL1_EXAMPLES


@pytest.mark.parametrize("example", SENTINEL1_EXAMPLES)
def test_sentinel1_roundtrip(example: dict[str, object]) -> None:
    """Test that Sentinel1Root can load and dump example JSON without data loss."""
    example_tup = tuplify_json(example)
    model = Sentinel1Root(**example_tup)
    assert jsondiff.diff(model.model_dump(), example_tup) == {}
