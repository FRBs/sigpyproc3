from pathlib import Path

import numpy as np
import pytest

from sigpyproc.core import rfi
from sigpyproc.header import Header


class TestRFI:
    def test_double_mad_mask(self) -> None:
        input_arr = np.array([1, 2, 3, 4, 5, 20], dtype=np.uint8)
        desired = np.array([0, 0, 0, 0, 0, 1], dtype=bool)
        np.testing.assert_array_equal(rfi.double_mad_mask(input_arr, 3), desired)
        with pytest.raises(ValueError):
            rfi.double_mad_mask(input_arr, -1)

    def test_iqrm_mask(self) -> None:
        input_arr = np.array([1, 2, 3, 4, 5, 20], dtype=np.uint8)
        desired = np.array([0, 0, 0, 0, 0, 1], dtype=bool)
        np.testing.assert_array_equal(rfi.iqrm_mask(input_arr, 3, 1), desired)
        with pytest.raises(ValueError):
            rfi.iqrm_mask(input_arr, -1)


class TestRFIMask:
    def test_from_file(self, maskfile: str) -> None:
        mask = rfi.RFIMask.from_file(maskfile)
        assert isinstance(mask.header, Header)
        np.testing.assert_equal(mask.num_masked, 83)
        np.testing.assert_equal(mask.chan_mask.size, mask.header.nchans)
        np.testing.assert_almost_equal(mask.masked_fraction, 9.97, decimal=1)
        test_mask = np.ones(mask.header.nchans, dtype=bool)
        mask.chan_mask = test_mask
        np.testing.assert_equal(mask.chan_mask, test_mask)

    def test_to_file(self, maskfile: str) -> None:
        mask = rfi.RFIMask.from_file(maskfile)
        outfile = mask.to_file()
        outpath = Path(outfile)
        mask2 = rfi.RFIMask.from_file(outfile)
        np.testing.assert_equal(mask.chan_mask, mask2.chan_mask)
        outpath.unlink()

    def test_apply_mask(self, maskfile: str) -> None:
        mask = rfi.RFIMask.from_file(maskfile)
        test_mask = np.ones(mask.header.nchans, dtype=bool)
        mask.apply_mask(test_mask)
        np.testing.assert_equal(mask.chan_mask, test_mask)
        with pytest.raises(ValueError):
            mask.apply_mask(np.zeros(mask.header.nchans - 1, dtype=bool))

    def test_apply_method(self, maskfile: str) -> None:
        mask = rfi.RFIMask.from_file(maskfile)
        mask.apply_method("mad")
        np.testing.assert_equal(len(mask.chan_mask), mask.header.nchans)
        mask.apply_method("iqrm")
        np.testing.assert_equal(len(mask.chan_mask), mask.header.nchans)
        with pytest.raises(ValueError):
            mask.apply_method("invalid")

    def test_apply_function(self, maskfile: str) -> None:
        mask = rfi.RFIMask.from_file(maskfile)
        mask.apply_funcn(rfi.double_mad_mask)
        np.testing.assert_equal(len(mask.chan_mask), mask.header.nchans)
        with pytest.raises(TypeError):
            mask.apply_funcn("invalid")  # type: ignore[arg-type]
