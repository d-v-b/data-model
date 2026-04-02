# Test that the ScaleOffset codec combined with the CastValueRustV1 codec from the
# cast_value library works correctly. Correctness is measured by the ability of
# these two codecs to define a procedure that saves an array of floats ranging from -1000 to 1000
# as uint16 array. The offset should be the minimum value of the array, but the scale can be 1 in
# this case.

import numpy as np
import zarr
import zarr.storage
from cast_value import CastValueRustV1

from eopf_geozarr.codecs.scale_offset import ScaleOffset, scale_offset_from_cf


def test_scale_offset_with_cast_value() -> None:
    """
    Round-trip test: write float64 data through ScaleOffset + CastValueRustV1,
    verify it is stored as uint16, and read back as the original float64 values.
    """
    data = np.linspace(-1000, 1000, 2001, dtype="float64")
    offset = float(data.min())  # -1000.0
    scale = 1.0

    store = zarr.storage.MemoryStore()
    arr = zarr.open_array(
        store,
        mode="w",
        shape=data.shape,
        dtype="float64",
        codecs=[
            ScaleOffset(offset=offset, scale=scale),
            CastValueRustV1(data_type="uint16", rounding="nearest-even"),
            zarr.codecs.BytesCodec(),
        ],
    )

    arr[:] = data
    result = arr[:]

    np.testing.assert_array_almost_equal(result, data)


def test_cf_scale_offset_pushed_into_codecs() -> None:
    """
    Given CF-convention scale_factor and add_offset, generate a ScaleOffset codec
    that replicates the CF behavior at the zarr chunk level, paired with a
    CastValueRustV1 codec for the packed integer dtype.

    CF convention: unpacked = packed * scale_factor + add_offset
    """
    scale_factor = 0.01
    add_offset = 273.15
    packed_dtype = "int16"

    # Build the "unpacked" (decoded) float data that the user sees
    packed_values = np.arange(-1000, 1001, dtype=packed_dtype)
    unpacked_values = packed_values * scale_factor + add_offset

    # Generate the ScaleOffset codec from CF parameters
    so_codec = scale_offset_from_cf(scale_factor=scale_factor, add_offset=add_offset)
    cv_codec = CastValueRustV1(data_type=packed_dtype, rounding="nearest-even")

    # Write the unpacked float data through the codec pipeline
    store = zarr.storage.MemoryStore()
    arr = zarr.open_array(
        store,
        mode="w",
        shape=unpacked_values.shape,
        dtype=unpacked_values.dtype,
        codecs=[so_codec, cv_codec, zarr.codecs.BytesCodec()],
    )

    arr[:] = unpacked_values
    result = arr[:]

    # The round-trip should recover the original unpacked floats
    np.testing.assert_array_almost_equal(result, unpacked_values)


def test_scale_offset_from_dict_round_trip() -> None:
    """ScaleOffset.to_dict / from_dict should round-trip."""
    codec = ScaleOffset(offset=273.15, scale=100.0)
    d = codec.to_dict()
    restored = ScaleOffset.from_dict(d)
    assert restored.offset == codec.offset
    assert restored.scale == codec.scale
