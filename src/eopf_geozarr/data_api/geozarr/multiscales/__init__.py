"""Zarr multiscales convention support."""

from .geozarr import MultiscaleGroupAttrs, MultiscaleMeta
from .zcm import MULTISCALE_CONVENTION_METADATA, Multiscales, ScaleLevel, ScaleLevelJSON

__all__ = [
    "MULTISCALE_CONVENTION_METADATA",
    "MultiscaleGroupAttrs",
    "MultiscaleMeta",
    "Multiscales",
    "ScaleLevel",
    "ScaleLevelJSON",
]
