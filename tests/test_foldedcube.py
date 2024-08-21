import numpy as np
import pytest
from numpy import typing as npt

from sigpyproc.core.filters import MatchedFilter
from sigpyproc.foldedcube import FoldedData, FoldSlice, Profile
from sigpyproc.header import Header
from sigpyproc.readers import FilReader


@pytest.fixture
def sample_data() -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(42)
    return rng.normal(size=128).astype(np.float32)


@pytest.fixture
def sample_foldslice_data() -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(42)
    return rng.normal(size=(64, 128)).astype(np.float32)


@pytest.fixture
def sample_foldeddata_data() -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(42)
    return rng.normal(size=(32, 64, 128)).astype(np.float32)


@pytest.fixture
def sample_header(filfile_8bit_1: str) -> Header:
    return Header.from_sigproc(filfile_8bit_1)


class TestProfile:
    def test_init(self, sample_data: npt.NDArray[np.float32]) -> None:
        profile = Profile(sample_data, tsamp=0.001)
        assert isinstance(profile, Profile)
        assert isinstance(profile.data, np.ndarray)
        assert profile.data.dtype == np.float32
        assert profile.data.shape == (128,)
        assert profile.tsamp == 0.001
        mf = profile.compute_mf()
        assert isinstance(mf, MatchedFilter)


class TestFoldSlice:
    def test_init(self, sample_foldslice_data: npt.NDArray[np.float32]) -> None:
        foldslice = FoldSlice(sample_foldslice_data, tsamp=0.001)
        assert isinstance(foldslice, FoldSlice)
        assert isinstance(foldslice.data, np.ndarray)
        assert foldslice.data.dtype == np.float32
        assert foldslice.data.shape == (64, 128)
        assert foldslice.tsamp == 0.001

    def test_normalize(self, sample_foldslice_data: npt.NDArray[np.float32]) -> None:
        foldslice = FoldSlice(sample_foldslice_data, tsamp=0.001)
        normalized = foldslice.normalize()
        assert isinstance(normalized, FoldSlice)
        assert np.allclose(np.mean(normalized.data, axis=1), 1.0)

    def test_get_profile(self, sample_foldslice_data: npt.NDArray[np.float32]) -> None:
        foldslice = FoldSlice(sample_foldslice_data, tsamp=0.001)
        profile = foldslice.get_profile()
        assert isinstance(profile, Profile)
        assert np.array_equal(profile.data, np.sum(sample_foldslice_data, axis=0))
        assert profile.tsamp == 0.001


class TestFoldedData:
    def test_init_fails(
        self,
        sample_data: npt.NDArray[np.float32],
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
        cube.dedisperse(20.0)
        np.testing.assert_equal(cube.dm, 20.0)
        cube.update_period(2.0)
        np.testing.assert_equal(cube.period, 2.0)
