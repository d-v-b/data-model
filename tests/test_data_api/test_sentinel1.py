import pytest

from tests.test_data_api.conftest import SENTINEL1_EXAMPLES
from eopf_geozarr.data_api.sentinel1 import Sentinel1Root


@pytest.mark.parametrize('example', SENTINEL1_EXAMPLES)
def test_sentinel1_roundtrip(example: dict[str, object]) -> None:
    """Test that Sentinel1Root can load and dump example JSON without data loss."""
    model = Sentinel1Root(**example)
    assert model.to_dict() == example