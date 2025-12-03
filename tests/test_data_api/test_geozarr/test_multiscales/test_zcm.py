import jsondiff
import pytest
from pydantic import ValidationError
from pydantic_zarr.core import tuplify_json

from eopf_geozarr.data_api.geozarr.multiscales.zcm import MultiscalesAttrs, ScaleLevel

from .conftest import ZCM_MULTISCALES_EXAMPLES


def test_json_examples_exist() -> None:
    assert len(ZCM_MULTISCALES_EXAMPLES) > 0


@pytest.mark.parametrize(
    "json_example", ZCM_MULTISCALES_EXAMPLES.items(), ids=lambda v: v[0]
)
def test_multiscales_rt(json_example: tuple[str, dict[str, object]]) -> None:
    """
    Test that the multiscales metadata round-trips input JSON
    """
    _, value = json_example
    value_tup = tuplify_json(value)
    attrs_json = value_tup["attributes"]
    model = MultiscalesAttrs(**attrs_json)
    observed = model.model_dump()
    expected = attrs_json
    assert jsondiff.diff(observed, expected) == {}


def test_scale_level_from_group() -> None:
    """
    Test that the ScaleLevel metadata rejects a dict with
    from_group but no "scale" attribute
    """
    meta = {"group": "1", "from_group": "0"}
    with pytest.raises(ValidationError):
        ScaleLevel(**meta)
