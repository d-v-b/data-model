from zarr.registry import register_codec

from eopf_geozarr.codecs.scale_offset import ScaleOffset

register_codec("scale_offset", ScaleOffset)
