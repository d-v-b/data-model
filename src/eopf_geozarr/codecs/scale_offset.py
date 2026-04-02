"""
Zarr V3 array-to-array codec implementing the scale_offset extension.

Encode: out = (in - offset) * scale
Decode: out = (in / scale) + offset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from zarr.abc.codec import ArrayArrayCodec
from zarr.core.common import JSON, parse_named_configuration

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.dtype import ZDType
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar
    from zarr.core.ndbuffer import NDBuffer


@dataclass(frozen=True)
class ScaleOffset(ArrayArrayCodec):
    """Array-to-array codec that applies a linear scale and offset transformation."""

    is_fixed_size = True

    offset: float
    scale: float

    def __init__(self, *, offset: float = 0.0, scale: float = 1.0) -> None:
        object.__setattr__(self, "offset", float(offset))
        object.__setattr__(self, "scale", float(scale))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "scale_offset")
        return cls(**configuration_parsed)

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": "scale_offset",
            "configuration": {"offset": self.offset, "scale": self.scale},
        }

    def validate(
        self,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        pass

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return self

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return chunk_spec

    async def _decode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)

    def _decode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        data = chunk_array.as_numpy_array()
        decoded = (data / self.scale) + self.offset
        return chunk_spec.prototype.nd_buffer.from_numpy_array(decoded)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def _encode_sync(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        data = chunk_array.as_numpy_array()
        encoded = (data - self.offset) * self.scale
        return chunk_array.from_numpy_array(encoded)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length


def scale_offset_from_cf(*, scale_factor: float, add_offset: float) -> ScaleOffset:
    """
    Convert CF-convention scale_factor and add_offset to a ScaleOffset codec.

    CF convention: unpacked = packed * scale_factor + add_offset

    ScaleOffset convention:
        encode: out = (in - offset) * scale
        decode: out = (in / scale) + offset

    To match CF: offset = add_offset, scale = 1 / scale_factor.
    """
    return ScaleOffset(offset=add_offset, scale=1.0 / scale_factor)
