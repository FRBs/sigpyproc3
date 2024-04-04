from __future__ import annotations
import attrs
import numpy as np

from typing import ClassVar, Any

from sigpyproc.core import kernels

nbits_to_dtype = {1: "<u1", 2: "<u1", 4: "<u1", 8: "<u1", 16: "<u2", 32: "<f4"}


def unpack(array: np.ndarray, nbits: int, unpacked: np.ndarray = None) -> np.ndarray:
    """Unpack 1, 2 and 4 bit array. Only unpacks in big endian bit ordering.

    Parameters
    ----------
    array : numpy.ndarray
        Array to unpack.
    nbits : int
        Number of bits to unpack.

    Returns
    -------
    numpy.ndarray
        Unpacked array.

    Raises
    ------
    ValueError
        if nbits is not 1, 2, or 4
    """
    assert array.dtype == np.uint8, "Array must be uint8"
    bitfact = 8//nbits
    if unpacked is None:
        unpacked = np.zeros(shape=array.size * bitfact, dtype=np.uint8)
    else:
        if unpacked.size != array.size * bitfact:
            raise ValueError(f"Unpacking array must be {bitfact}x input size")
    if nbits == 1:
        unpacked = kernels.unpack1_8(array, unpacked)
    elif nbits == 2:
        unpacked = kernels.unpack2_8(array, unpacked)
    elif nbits == 4:
        unpacked = kernels.unpack4_8(array, unpacked)
    else:
        raise ValueError("nbits must be 1, 2, or 4")
    return unpacked


def pack(array, nbits):
    """Pack 1, 2 and 4 bit array. Only packs in big endian bit ordering.

    Parameters
    ----------
    array : numpy.ndarray
        Array to pack.
    nbits : int
        Number of bits to pack.

    Returns
    -------
    numpy.ndarray
        Packed array.

    Raises
    ------
    ValueError
        if nbits is not 1, 2, or 4
    """
    assert array.dtype == np.uint8, "Array must be uint8"
    if nbits == 1:
        packed = np.packbits(array, bitorder="big")
    elif nbits == 2:
        packed = kernels.pack2_8(array)
    elif nbits == 4:
        packed = kernels.pack4_8(array)
    else:
        raise ValueError("nbits must be 1, 2, or 4")
    return packed


@attrs.define(auto_attribs=True, frozen=True, slots=True)
class BitsInfo(object):
    """Class to handle bits info.

    Raises
    ------
    ValueError
        if input `nbits` not in [1, 2, 4, 8, 16, 32]
    """

    nbits: int = attrs.field(validator=attrs.validators.in_(nbits_to_dtype.keys()))
    digi_sigma: float = attrs.field()

    default_sigma: ClassVar[dict[int, float]] = {
        1: 0.5,
        2: 1.5,
        4: 6,
        8: 6,
        16: 6,
        32: 6,
    }

    @property
    def dtype(self) -> np.dtype:
        """Type of the data (`np.dtype`, read-only)."""
        return np.dtype(nbits_to_dtype[self.nbits])

    @property
    def itemsize(self) -> int:
        """Element size of this data-type object (`int`, read-only)."""
        return self.dtype.itemsize

    @property
    def unpack(self) -> bool:
        """Whether to unpack bits (`bool`, read-only)."""
        return bool(self.nbits in {1, 2, 4})

    @property
    def bitfact(self) -> int:
        """Bit factor to unpack/pack bits (`int`, read-only)."""
        return 8 // self.nbits if self.unpack else 1

    @property
    def digi_min(self) -> int:
        """Minimum value used to quantize data (`int`, read-only)."""
        return 0

    @property
    def digi_max(self) -> int:
        """Maximum value used to quantize data (`int`, read-only)."""
        return (1 << self.nbits) - 1

    @property
    def digi_mean(self) -> float:
        """Mean used to quantize data (`float`, read-only)."""
        return (1 << (self.nbits - 1)) - 0.5

    @property
    def digi_scale(self) -> float:
        """Scale used to quantize data (`float`, read-only)."""
        return self.digi_mean / self.digi_sigma

    def to_dict(self) -> dict[str, Any]:
        """Get a dict of all property attributes.

        Returns
        -------
        dict
            property attributes
        """
        attributes = attrs.asdict(self)
        prop = {
            key: getattr(self, key)
            for key, value in vars(type(self)).items()  # noqa: WPS421
            if isinstance(value, property)
        }
        attributes.update(prop)
        return attributes

    @digi_sigma.default
    def _set_digi_sigma(self):
        return self.default_sigma[self.nbits]
