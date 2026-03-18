"""Zarr multiscales convention support."""

from .geozarr import MultiscaleGroupAttrs, MultiscaleMeta
from .zcm import Multiscales, ScaleLevel, ScaleLevelJSON

__all__ = [
    "MultiscaleGroupAttrs",
    "MultiscaleMeta",
    "Multiscales",
    "ScaleLevel",
    "ScaleLevelJSON",
]
