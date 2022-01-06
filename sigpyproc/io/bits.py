import attr
import numpy as np

from typing import Optional, ClassVar, Dict, Any

from sigpyproc.core import kernels

nbits_to_dtype = {1: "<u1", 2: "<u1", 4: "<u1", 8: "<u1", 16: "<u2", 32: "<f4"}


def unpack(array, nbits):
    """Unpack 1, 2 and 4 bit array. Only unpacks in big endian bit ordering.

    Parameters
    ----------
    array : ndarray
        Array to unpack.
    nbits : int
        Number of bits to unpack.

    Returns
    -------
    ndarray
        Unpacked array.

    Raises
    ------
    ValueError
        if nbits is not 1, 2, or 4
    """
    assert array.dtype == np.uint8, "Array must be uint8"
    if nbits == 1:
        unpacked = np.unpackbits(array, bitorder="big")
    elif nbits == 2:
        unpacked = kernels.unpack2_8(array)
    elif nbits == 4:
        unpacked = kernels.unpack4_8(array)
    else:
        raise ValueError("nbits must be 1, 2, or 4")
    return unpacked


def pack(array, nbits):
    """Pack 1, 2 and 4 bit array. Only packs in big endian bit ordering.

    Parameters
    ----------
    array : ndarray
        Array to pack.
    nbits : int
        Number of bits to pack.

    Returns
    -------
    ndarray
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


@attr.s(auto_attribs=True)
class BitsInfo(object):
    """Class to handle bits info.

    Raises
    ------
    ValueError
        if input `nbits` not in [1, 2, 4, 8, 16, 32]
    """

    nbits: int = attr.ib(validator=attr.validators.in_(nbits_to_dtype.keys()))
    digi_sigma: float = attr.ib()

    float_bits: ClassVar[int] = 32
    default_sigma: ClassVar[Dict[int, float]] = {
        1: 0.5,
        2: 1.5,
        4: 6,
        8: 6,
        16: 6,
        32: 6,
    }

    def __attrs_post_init__(self) -> None:
        self._digi_min = 0
        self._digi_max = (1 << self.nbits) - 1
        self._digi_mean = (1 << (self.nbits - 1)) - 0.5
        self._digi_scale = self._digi_mean / self.digi_sigma

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
    def digi_min(self) -> Optional[int]:
        """Minimum value used to quantize data (`int` or None, read-only)."""
        return None if self.nbits == self.float_bits else self._digi_min

    @property
    def digi_max(self) -> Optional[int]:
        """Maximum value used to quantize data (`int` or None, read-only)."""
        return None if self.nbits == self.float_bits else self._digi_max

    @property
    def digi_mean(self) -> Optional[float]:
        """Mean used to quantize data (`float` or None, read-only)."""
        return None if self.nbits == self.float_bits else self._digi_mean

    @property
    def digi_scale(self) -> Optional[float]:
        """Scale used to quantize data (`float` or None, read-only)."""
        return None if self.nbits == self.float_bits else self._digi_scale

    def to_dict(self) -> Dict[str, Any]:
        """Get a dict of all property attributes.

        Returns
        -------
        dict
            property attributes
        """
        return {
            key: getattr(self, key)
            for key, value in vars(type(self)).items()  # noqa: WPS421
            if isinstance(value, property)
        }

    @digi_sigma.default
    def _set_digi_sigma(self):
        return self.default_sigma[self.nbits]
