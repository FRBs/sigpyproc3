from __future__ import annotations

from typing import ClassVar

import attrs
import numpy as np

from sigpyproc.core import kernels

nbits_to_dtype = {1: "<u1", 2: "<u1", 4: "<u1", 8: "<u1", 16: "<u2", 32: "<f4"}


def unpack(
    array: np.ndarray,
    nbits: int,
    unpacked: np.ndarray | None = None,
    *,
    bitorder: str = "big",
    parallel: bool = False,
) -> np.ndarray:
    """Unpack 1, 2 and 4-bit data packed as 8-bit numpy array.

    Parameters
    ----------
    array : numpy.ndarray
        Array to unpack.
    nbits : int
        Number of bits of the packed data.
    unpacked : numpy.ndarray, optional
        Array to unpack into.
    bitorder : str, optional
        Bit order of the packed data.
    parallel : bool, optional
        Whether to use parallel unpacking.

    Returns
    -------
    numpy.ndarray
        Unpacked array.

    Raises
    ------
    ValueError
        if input array is not uint8 type
        if nbits is not 1, 2, or 4
        if bitorder is not 'big' or 'little'
        if unpacked array is not of the correct size
    """
    if array.dtype != np.uint8:
        msg = f"Input array must be uint8, got {array.dtype}"
        raise ValueError(msg)
    if nbits not in {1, 2, 4}:
        msg = f"nbits must be 1, 2, or 4, got {nbits}"
        raise ValueError(msg)
    if (not bitorder) or (bitorder[0] not in {"b", "l"}):
        msg = f"bitorder must be 'big' or 'little', got {bitorder}"
        raise ValueError(msg)
    bitorder_str = "big" if bitorder[0] == "b" else "little"
    parallel_str = "" if parallel else "_serial"
    bitfact = 8 // nbits
    if unpacked is None:
        unpacked = np.zeros(shape=array.size * bitfact, dtype=np.uint8)
    elif unpacked.size != array.size * bitfact:
        msg = f"Unpacking array must be {bitfact} x input size, got {unpacked.size}"
        raise ValueError(msg)
    unpack_func = getattr(kernels, f"unpack{nbits:d}_8_{bitorder_str}{parallel_str}")
    unpack_func(array, unpacked)
    return unpacked


def pack(
    array: np.ndarray,
    nbits: int,
    packed: np.ndarray | None = None,
    *,
    bitorder: str = "big",
    parallel: bool = False,
) -> np.ndarray:
    """Pack 1, 2 and 4-bit data into 8-bit numpy array.

    Parameters
    ----------
    array : numpy.ndarray
        Array to pack.
    nbits : int
        Number of bits of the unpacked data.
    packed : numpy.ndarray, optional
        Array to pack into.
    bitorder : str, optional
        Bit order in which to pack the data.
    parallel : bool, optional
        Whether to use parallel packing.

    Returns
    -------
    numpy.ndarray
        Packed array.

    Raises
    ------
    ValueError
        if input array is not uint8 type
        if nbits is not 1, 2, or 4
        if bitorder is not 'big' or 'little'
        if unpacked array is not of the correct size
    """
    if array.dtype != np.uint8:
        msg = f"Input array must be uint8, got {array.dtype}"
        raise ValueError(msg)
    if nbits not in {1, 2, 4}:
        msg = f"nbits must be 1, 2, or 4, got {nbits}"
        raise ValueError(msg)
    if (not bitorder) or (bitorder[0] not in {"b", "l"}):
        msg = f"bitorder must be 'big' or 'little', got {bitorder}"
        raise ValueError(msg)
    bitorder_str = "big" if bitorder[0] == "b" else "little"
    parallel_str = "" if parallel else "_serial"
    bitfact = 8 // nbits
    if packed is None:
        packed = np.zeros(shape=array.size // bitfact, dtype=np.uint8)
    elif packed.size != array.size // bitfact:
        msg = f"packing array must be input size // {bitfact}, got {packed.size}"
        raise ValueError(msg)
    pack_func = getattr(kernels, f"pack{nbits:d}_8_{bitorder_str}{parallel_str}")
    pack_func(array, packed)
    return packed


@attrs.define(auto_attribs=True, frozen=True, slots=True)
class BitsInfo:
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
    default_bitorder: ClassVar[dict[int, str]] = {
        1: "little",
        2: "big",
        4: "big",
        8: "invalid",
        16: "invalid",
        32: "invalid",
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
    def bitorder(self) -> str:
        """Bit order of the packed data (`str`, read-only)."""
        return self.default_bitorder[self.nbits]

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

    def to_dict(self) -> dict[str, int | float | np.dtype]:
        """Get a dict of all property attributes.

        Returns
        -------
        dict
            property attributes
        """
        attributes = attrs.asdict(self)
        prop = {
            key: getattr(self, key)
            for key, value in vars(type(self)).items()
            if isinstance(value, property)
        }
        attributes.update(prop)
        return attributes

    def quantize(self, arr_norm: np.ndarray) -> np.ndarray:
        """Quantize normalized data to given nbit-dependent mean and sigma.

        Parameters
        ----------
        arr : :py:obj:`~numpy.ndarray`
            a 1-D numpy array containing the data to quantize.

        Returns
        -------
        :py:obj:`~numpy.ndarray`
            Quantized data array to nbits.

        Notes
        -----
        Values outside the dynamic range of the nbits will be clipped.
        """
        arr = (arr_norm * self.digi_scale) + self.digi_mean + 0.5
        arr = arr.astype(np.int32)
        np.clip(arr, self.digi_min, self.digi_max, out=arr)
        return arr.astype(self.dtype, copy=False)

    @digi_sigma.default
    def _set_digi_sigma(self) -> float:
        return self.default_sigma[self.nbits]
