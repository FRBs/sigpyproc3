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
        data = np.array(
            [[1.0, -1.0, 1.0, -1.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32
        )
        normalized = FoldSlice(data, tsamp=0.001).normalize()
        assert np.isfinite(normalized.data).all()
        np.testing.assert_array_equal(normalized.data[0], 0)
        assert normalized.data[1].mean() == pytest.approx(1)

    def test_invalid_period_is_rejected(self, filfile_4bit: str) -> None:
        header = Header.from_sigproc(filfile_4bit)
        data = np.ones((2, 2, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="period"):
            FoldedData(data, header, period=0.0, dm=0.0)


def _synthetic_series_header(nsamples: int = 128) -> Header:
    return Header(
        filename="synthetic.tim",
        data_type="time series",
        nchans=1,
        foff=1.0,
        fch1=1400.0,
        nbits=32,
        tsamp=0.001,
        tstart=50000.0,
        nsamples=nsamples,
    )


def _correction_cube(nsamples: int = 128) -> FoldedData:
    return FoldedData(
        np.arange(64, dtype=np.float32).reshape(4, 2, 8),
        _synthetic_series_header(nsamples).new_header({"nchans": 2}),
        0.5,
        10.0,
        3.0,
    )


@pytest.mark.parametrize("target", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_period_rejection_preserves_correction_state(target: float) -> None:
    cube = _correction_cube()
    cube.update_period(0.48)
    before = cube.data.copy()
    history = cube._tph_shifts.copy()
    with pytest.raises(ValueError, match="period"):
        cube.update_period(target)
    assert cube.period == 0.48
    np.testing.assert_array_equal(cube.data, before)
    np.testing.assert_array_equal(cube._tph_shifts, history)


@pytest.mark.parametrize("target", [np.nan, np.inf, -np.inf, 1e30])
def test_dm_rejection_preserves_correction_state(target: float) -> None:
    cube = _correction_cube()
    before = cube.data.copy()
    with pytest.raises(ValueError, match=r"finite|32-bit"):
        cube.update_dm(target)
    assert cube.dm == 10.0
    np.testing.assert_array_equal(cube.data, before)
    np.testing.assert_array_equal(cube._fph_shifts, 0)


@pytest.mark.parametrize("nsamples", [2**31 - 1, 2**31, 3_000_000_000])
def test_long_fold_archives_keep_duration_and_future_corrections(
    tmp_path: Path,
    nsamples: int,
) -> None:
    cube = _correction_cube(nsamples)
    cube.update_period(0.49999)
    cube.update_dm(11.0)
    cube = cube.centre()
    path = cube.to_npz(str(tmp_path / "fold"))
    assert path.endswith("fold.npz")
    restored = FoldedData.from_npz(path)
    assert restored.header.nsamples == nsamples
    assert restored.header.tobs == cube.header.tobs
    assert restored.accel == 3.0
    for item in (cube, restored):
        item.update_period(0.5)
        item.update_dm(10.0)
    np.testing.assert_array_equal(restored.data, cube.data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("period0", np.array(0.0), "original period"),
        ("period0", np.array(np.nan), "original period"),
        ("dm0", np.array(np.inf), "original period"),
        ("tph_shifts", np.zeros(3, dtype=np.int32), "shift vectors"),
        ("fph_shifts", np.zeros(2, dtype=np.float32), "shift vectors"),
        ("fph_shifts", np.zeros((2, 1), dtype=np.int32), "shift vectors"),
        ("version", np.array(99), "version"),
    ],
)
def test_fold_archive_rejects_malformed_state(
    tmp_path: Path,
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    path = _correction_cube().to_npz(str(tmp_path / "original"))
    with np.load(path, allow_pickle=False) as source:
        contents = dict(source)
    contents[field] = value
    changed = tmp_path / "invalid.npz"
    np.savez(changed, **contents)
    with pytest.raises(ValueError, match=message):
        FoldedData.from_npz(str(changed))


def test_old_integer_archive_and_missing_fields(tmp_path: Path) -> None:
    path = _correction_cube().to_npz(str(tmp_path / "original"))
    with np.load(path, allow_pickle=False) as source:
        contents = dict(source)
    contents["nsamples"] = np.int32(128)
    contents["tph_shifts"] = contents["tph_shifts"].astype(np.int32)
    np.savez(path, **contents)
    assert FoldedData.from_npz(path).header.nsamples == 128
    del contents["period0"]
    np.savez(path, **contents)
    with pytest.raises(ValueError, match="missing fields"):
        FoldedData.from_npz(path)
    np.savez(path, data=np.ones(8))
    with pytest.raises(ValueError, match="version"):
        FoldedData.from_npz(path)


@pytest.mark.parametrize("all_missing", [False, True])
def test_fold_nan_replacement_uses_finite_median(all_missing: bool) -> None:
    data = np.full((2, 2, 8), np.nan, dtype=np.float32)
    if not all_missing:
        data[0, 0, :4] = [1, 3, 5, 7]
    expected = np.zeros_like(data) if all_missing else np.where(np.isnan(data), 4, data)
    cube = FoldedData(
        data, _synthetic_series_header().new_header({"nchans": 2}), 0.5, 0
    )
    cube.replace_nan()
    np.testing.assert_allclose(cube.data, expected)


@pytest.mark.parametrize("changes", [{"dm": np.nan}, {"accel": np.inf}])
def test_fold_metadata_must_be_finite(changes: dict) -> None:
    values = {"period": 0.5, "dm": 0.0, "accel": 0.0, **changes}
    with pytest.raises(ValueError, match="finite"):
        FoldedData(np.ones((2, 2, 8)), _synthetic_series_header(), **values)
