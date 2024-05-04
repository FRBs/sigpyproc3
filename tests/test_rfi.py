import numpy as np

from sigpyproc.core import rfi
from sigpyproc.header import Header


class TestRFI:
    def test_double_mad_mask(self) -> None:
        input_arr = np.array([1, 2, 3, 4, 5, 20], dtype=np.uint8)
        desired = np.array([0, 0, 0, 0, 0, 1], dtype=bool)
        np.testing.assert_array_equal(desired, rfi.double_mad_mask(input_arr))


class TestRFIMask:
    def test_from_file(self, maskfile: str) -> None:
        mask = rfi.RFIMask.from_file(maskfile)
        assert isinstance(mask.header, Header)
        np.testing.assert_equal(mask.num_masked, 83)
        np.testing.assert_equal(mask.chan_mask.size, mask.header.nchans)
        np.testing.assert_almost_equal(mask.masked_fraction, 9.97, decimal=1)
