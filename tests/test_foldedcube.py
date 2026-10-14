from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from sigpyproc.core.filters import MatchedFilter
from sigpyproc.foldedcube import FoldedData, FoldSlice, Profile
from sigpyproc.header import Header
from sigpyproc.readers import FilReader


@pytest.fixture
def sample_data() -> NDArray[np.float32]:
    rng = np.random.default_rng(42)
    return rng.normal(size=128).astype(np.float32)


@pytest.fixture
def sample_foldslice_data() -> NDArray[np.float32]:
    rng = np.random.default_rng(42)
    return rng.normal(size=(64, 128)).astype(np.float32)


@pytest.fixture
def sample_foldeddata_data() -> NDArray[np.float32]:
    rng = np.random.default_rng(42)
    return rng.normal(size=(32, 64, 128)).astype(np.float32)


@pytest.fixture
def sample_header(filfile_8bit_1: str) -> Header:
    return Header.from_sigproc(filfile_8bit_1)


class TestProfile:
    def test_init(self, sample_data: NDArray[np.float32]) -> None:
        profile = Profile(sample_data, tsamp=0.001)
        assert isinstance(profile, Profile)
        assert isinstance(profile.data, np.ndarray)
        assert profile.data.dtype == np.float32
        assert profile.data.shape == (128,)
        assert profile.tsamp == 0.001
        mf = profile.compute_mf()
        assert isinstance(mf, MatchedFilter)


class TestFoldSlice:
    def test_init(self, sample_foldslice_data: NDArray[np.float32]) -> None:
        foldslice = FoldSlice(sample_foldslice_data, tsamp=0.001)
        assert isinstance(foldslice, FoldSlice)
        assert isinstance(foldslice.data, np.ndarray)
        assert foldslice.data.dtype == np.float32
        assert foldslice.data.shape == (64, 128)
        assert foldslice.tsamp == 0.001

    def test_normalize(self, sample_foldslice_data: NDArray[np.float32]) -> None:
        foldslice = FoldSlice(sample_foldslice_data, tsamp=0.001)
        normalized = foldslice.normalize()
        assert isinstance(normalized, FoldSlice)
        assert np.allclose(np.mean(normalized.data, axis=1), 1.0)

    def test_get_profile(self, sample_foldslice_data: NDArray[np.float32]) -> None:
        foldslice = FoldSlice(sample_foldslice_data, tsamp=0.001)
        profile = foldslice.get_profile()
        assert isinstance(profile, Profile)
        assert np.array_equal(profile.data, np.sum(sample_foldslice_data, axis=0))
        assert profile.tsamp == 0.001


class TestFoldedData:
    def test_init_fails(
        self,
        sample_data: NDArray[np.float32],
        filfile_4bit: str,
    ) -> None:
        hdr = Header.from_sigproc(filfile_4bit)
        with pytest.raises(TypeError):
            FoldedData(sample_data, "not a header", period=1.0, dm=10.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            FoldedData(sample_data, hdr, period=1.0, dm=10.0)

    def test_init(self, filfile_4bit: str) -> None:
        fil = FilReader(filfile_4bit)
        cube = fil.fold(period=1, dm=10, nints=16, nbins=50)
        assert isinstance(cube, FoldedData)
        assert isinstance(cube.data, np.ndarray)
        assert cube.data.dtype == np.float32
        assert cube.data.shape == (16, 32, 50)
        assert cube.period == 1.0
        assert cube.dm == 10.0

    def test_gets(self, filfile_4bit: str) -> None:
        fil = FilReader(filfile_4bit)
        cube = fil.fold(period=1, dm=10, nints=16, nbins=50)
        subint = cube.get_subint(0)
        assert isinstance(subint, FoldSlice)
        subband = cube.get_subband(0)
        assert isinstance(subband, FoldSlice)
        profile = cube.get_profile()
        assert isinstance(profile, Profile)
        time_phase = cube.get_time_phase()
        assert isinstance(time_phase, FoldSlice)
        freq_phase = cube.get_freq_phase()
        assert isinstance(freq_phase, FoldSlice)
        centred = cube.centre()
        assert isinstance(centred, FoldedData)
        assert centred.data.shape == cube.data.shape

    def test_methods(self, filfile_4bit: str) -> None:
        fil = FilReader(filfile_4bit)
        cube = fil.fold(period=1, dm=10, nints=16, nbins=50)
        cube.update_dm(20.0)
        np.testing.assert_equal(cube.dm, 20.0)
        cube.update_period(2.0)
        np.testing.assert_equal(cube.period, 2.0)
        assert cube.get_profile().tsamp == pytest.approx(cube.period / cube.nbins)

    def test_repeated_updates_are_absolute(self, filfile_4bit: str) -> None:
        header = Header.from_sigproc(filfile_4bit)
        data = np.zeros((4, 8, 32), dtype=np.float32)
        data[0, :, 0] = np.arange(8)
        data[:, 0, 1] = 3
        original = data.copy()
        cube = FoldedData(data, header, period=0.5, dm=0.0)
        cube.update_dm(20.0)
        once = cube.data.copy()
        cube.update_dm(20.0)
        np.testing.assert_array_equal(cube.data, once)
        other = FoldedData(original.copy(), header, period=0.5, dm=0.0)
        other.update_dm(50.0)
        cube.update_dm(50.0)
        np.testing.assert_array_equal(cube.data, other.data)
        cube.update_dm(0.0)
        np.testing.assert_array_equal(cube.data, original)

        cube.update_period(0.48)
        shifted = cube.data.copy()
        cube.update_period(0.48)
        np.testing.assert_array_equal(cube.data, shifted)
        cube.update_period(0.5)
        np.testing.assert_array_equal(cube.data, original)

    def test_to_ascii_npz(self, filfile_4bit: str, tmpdir: str) -> None:
        fil = FilReader(filfile_4bit)
        cube = fil.fold(period=1, dm=10, nints=16, nbins=50)
        txt = cube.to_ascii(str(Path(tmpdir) / "prof.txt"))
        npz = cube.to_npz(str(Path(tmpdir) / "cube.npz"))
        loaded_txt = np.loadtxt(txt)
        np.testing.assert_allclose(loaded_txt, cube.get_profile().data)
        loaded = np.load(npz)
        np.testing.assert_array_equal(loaded["data"], cube.data)
        np.testing.assert_equal(float(loaded["period"]), cube.period)
        restored = FoldedData.from_npz(npz)
        np.testing.assert_array_equal(restored.data, cube.data)
        assert restored.period == pytest.approx(cube.period)
        assert restored.dm == pytest.approx(cube.dm)
        assert restored.accel == pytest.approx(cube.accel)
        assert restored.header.foff == pytest.approx(cube.header.foff)
        assert restored.header.tsamp == pytest.approx(cube.header.tsamp)

    def test_centre_puts_the_peak_in_the_middle(self, filfile_4bit: str) -> None:
        header = Header.from_sigproc(filfile_4bit)
        data = np.zeros((2, 4, 8), dtype=np.float32)
        data[:, :, 1] = 5
        data[:, :, 0] = 1
        cube = FoldedData(data, header, period=0.5, dm=10.0, accel=2.0)
        centred = cube.centre()
        assert int(np.argmax(centred.get_profile().data)) == centred.nbins // 2
        assert centred.accel == pytest.approx(2.0)
        assert centred.dm == pytest.approx(10.0)

    def test_flat_profile_is_not_rolled(self, filfile_4bit: str) -> None:
        header = Header.from_sigproc(filfile_4bit)
        data = np.ones((2, 2, 8), dtype=np.float32)
        cube = FoldedData(data, header, period=0.5, dm=0.0)
        np.testing.assert_array_equal(cube.centre().data, data)

    def test_zero_mean_slice_stays_finite(self) -> None:
        data = np.array([[1.0, -1.0, 1.0, -1.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        normalized = FoldSlice(data, tsamp=0.001).normalize()
        assert np.isfinite(normalized.data).all()
        np.testing.assert_array_equal(normalized.data[0], 0)
        assert normalized.data[1].mean() == pytest.approx(1)

    def test_invalid_period_is_rejected(self, filfile_4bit: str) -> None:
        header = Header.from_sigproc(filfile_4bit)
        data = np.ones((2, 2, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="period"):
            FoldedData(data, header, period=0.0, dm=0.0)
