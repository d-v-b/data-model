"""Zarr multiscales convention support."""

from .geozarr import MultiscaleGroupAttrs, MultiscaleMeta
from .zcm import MULTISCALE_CONVENTION_METADATA, Multiscales, ScaleLevel, ScaleLevelJSON

__all__ = [
    "MultiscaleGroupAttrs",
    "MultiscaleMeta",
    "MULTISCALE_CONVENTION_METADATA",
    "Multiscales",
    "ScaleLevel",
    "ScaleLevelJSON",
]
