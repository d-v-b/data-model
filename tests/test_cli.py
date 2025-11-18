import pytest
from structlog.testing import capture_logs

from eopf_geozarr.cli_2 import convert, resolve_product
from eopf_geozarr.data_api.s1 import Sentinel1Root
from eopf_geozarr.data_api.s2 import Sentinel2Root
from eopf_geozarr.pyz.v2 import GroupSpec

from .test_data_api.conftest import S1_EXAMPLES, S2_EXAMPLES


@pytest.mark.parametrize("example", S1_EXAMPLES)
def test_resolve_s1_product(example: dict[str, object]) -> None:
    model = GroupSpec(**example)
    assert isinstance(resolve_product(model), Sentinel1Root)


@pytest.mark.parametrize("example", S2_EXAMPLES)
def test_resolve_s2_product(example: dict[str, object]) -> None:
    model = GroupSpec(**example)
    assert isinstance(resolve_product(model), Sentinel2Root)


def test_convert() -> None:
    test_url: str = "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202509-s02msil2a/08/products/cpm_v256/S2A_MSIL2A_20250908T100041_N0511_R122_T32TQM_20250908T115116.zarr"
    with capture_logs() as cap_logs:
        assert convert(test_url) is None
    assert cap_logs == [{"event": "Found a Sentinel2 product!", "log_level": "info"}]
