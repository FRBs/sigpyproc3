import attr
import numpy as np

from typing import Optional, ClassVar, Dict, Any

nbits_to_dtype = {1: "<u1", 2: "<u1", 4: "<u1", 8: "<u1", 16: "<u2", 32: "<f4"}


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
