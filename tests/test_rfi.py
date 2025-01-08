from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

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
    def test_init(self, filfile_8bit_1: str) -> None:
        hdr = Header.from_sigproc(filfile_8bit_1)
        chan_stats = np.arange(0, hdr.nchans)
        threshold = 3
        rfimask = rfi.RFIMask(
            threshold,
            hdr,
            chan_stats,
            chan_stats,
            chan_stats,
            chan_stats,
            chan_stats,
            chan_stats,
        )
        np.testing.assert_equal(rfimask.chan_mask.size, hdr.nchans)
        np.testing.assert_equal(rfimask.user_mask.size, hdr.nchans)
        np.testing.assert_equal(rfimask.stats_mask.size, hdr.nchans)
        np.testing.assert_equal(rfimask.custom_mask.size, hdr.nchans)

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
        freq_mask = [(1400.0, 1800.0), (3200.0, 3400.0)]
        mask.apply_mask(freq_mask)
        np.testing.assert_equal(mask.user_mask.sum(), 150)
        mask1 = rfi.RFIMask.from_file(maskfile)
        freq_mask = []
        mask1.apply_mask(freq_mask)
        np.testing.assert_equal(mask1.user_mask.sum(), 0)

    def test_apply_method(self, maskfile: str) -> None:
        mask = rfi.RFIMask.from_file(maskfile)
        mask.apply_method("mad")
        np.testing.assert_equal(len(mask.chan_mask), mask.header.nchans)
        mask.apply_method("iqrm")
        np.testing.assert_equal(len(mask.chan_mask), mask.header.nchans)
        with pytest.raises(ValueError):
            mask.apply_method("invalid")  # type: ignore[arg-type]

    def test_apply_function(self, maskfile: str) -> None:
        mask = rfi.RFIMask.from_file(maskfile)
        mask.apply_funcn(rfi.double_mad_mask)
        np.testing.assert_equal(len(mask.chan_mask), mask.header.nchans)
        with pytest.raises(TypeError):
            mask.apply_funcn("invalid")  # type: ignore[arg-type]

    def test_plot(self, maskfile: str) -> None:
        mask = rfi.RFIMask.from_file(maskfile)
        fig = mask.plot()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
