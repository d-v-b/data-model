from zarr.registry import register_codec

from eopf_geozarr.codecs.scale_offset import ScaleOffset

register_codec("scale_offset", ScaleOffset)

try:
    from cast_value.zarr_compat.v1 import CastValueRust

    register_codec("cast_value", CastValueRust)
except ImportError:
    try:
        from cast_value.zarr_compat.v1 import CastValueNumpy

        register_codec("cast_value", CastValueNumpy)
    except ImportError:
        pass
