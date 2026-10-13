from pathlib import Path

import numpy as np
import pytest

from sigpyproc.foldedcube import FoldedData
from sigpyproc.fourierseries import FourierSeries
from sigpyproc.header import Header
from sigpyproc.io import sigproc
from sigpyproc.timeseries import TimeSeries


class TestTimeSeries:
    def test_init_fail(self, tim_data: np.ndarray, tim_header: dict) -> None:
        with pytest.raises(TypeError):
            TimeSeries(tim_data, tim_data)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            TimeSeries(tim_data[0], Header(**tim_header))
        with pytest.raises(ValueError):
            TimeSeries(tim_data[:-1], Header(**tim_header))
        with pytest.raises(ValueError):
            TimeSeries([], Header(**tim_header))

    def test_timeseries(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        np.testing.assert_equal(tim.header.nbits, 32)
        np.testing.assert_almost_equal(tim.data.mean(), 128, decimal=0)

    def test_fold(self, tim_data: np.ndarray, tim_header: dict) -> None:
        period = 1
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_folded = tim.fold(period=period, nints=16)
        assert isinstance(tim_folded, FoldedData)

    def test_rfft(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_rfft = tim.rfft()
        assert isinstance(tim_rfft, FourierSeries)

    def test_deredden(self, tim_data: np.ndarray, tim_header: dict) -> None:
        window = 101
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_deredden = tim.deredden(window=window)
        np.testing.assert_equal(tim_deredden.data.size, tim.data.size)
        np.testing.assert_allclose(tim_deredden.data.mean(), 0, atol=0.1)
        with pytest.raises(ValueError):
            tim.deredden(window=-5)

    def test_apply_boxcar(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_boxcar = tim.apply_boxcar(width=5)
        assert isinstance(tim_boxcar, TimeSeries)
        np.testing.assert_equal(tim_boxcar.data.size, tim.data.size)
        with pytest.raises(ValueError):
            tim.apply_boxcar(width=0)

    def test_downsample(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tfactor = 16
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_decimated = tim.downsample(factor=tfactor)
        assert isinstance(tim_decimated, TimeSeries)
        np.testing.assert_equal(tim_decimated.data.size, tim.data.size // tfactor)
        np.testing.assert_allclose(
            tim.data[:tfactor].mean(),
            tim_decimated.data[0],
            atol=0.01,
        )
        tim_decimated_1 = tim.downsample(factor=1)
        np.testing.assert_equal(tim_decimated_1.data, tim.data)
        with pytest.raises(ValueError):
            tim.downsample(factor=0)
        with pytest.raises(ValueError):
            tim.downsample(factor=1.5)  # type: ignore[arg-type]

    def test_pad(self, tim_data: np.ndarray, tim_header: dict) -> None:
        npad = 100
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_padded = tim.pad(npad=npad)
        assert isinstance(tim_padded, TimeSeries)
        np.testing.assert_equal(tim_padded.data.size, tim.data.size + npad)

    def test_resample(self, tim_data: np.ndarray, tim_header: dict) -> None:
        accel = 1
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_resampled = tim.resample(accel=accel)
        assert isinstance(tim_resampled, TimeSeries)
        np.testing.assert_equal(tim_resampled.header.accel, accel)

    def test_correlate(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_corr = tim.correlate(tim_data)
        assert isinstance(tim_corr, TimeSeries)

    def test_to_dat(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        datfile = tim.to_dat()
        datfile_path = Path(datfile)
        inffile_path = datfile_path.with_suffix(".inf")
        assert datfile_path.is_file()
        assert inffile_path.is_file()
        assert datfile_path.stat().st_size == tim.nsamples * 4
        np.testing.assert_array_equal(np.fromfile(datfile_path, dtype="<f4"), tim.data)
        np.testing.assert_array_equal(TimeSeries.from_dat(datfile_path).data, tim.data)
        datfile_path.unlink()
        inffile_path.unlink()

    def test_to_tim(
        self,
        tim_data: np.ndarray,
        tim_header: dict,
        tmpfile: str,
    ) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        outfile = tim.to_tim(tmpfile)
        assert Path(outfile).is_file()
        outfile = tim.to_tim()
        outpath = Path(outfile)
        assert outpath.is_file()
        outpath.unlink()

    def test_from_dat(
        self,
        datfile: str,
        datfile_mean: float,
        datfile_std: float,
    ) -> None:
        tim = TimeSeries.from_dat(datfile)
        np.testing.assert_allclose(tim.data.mean(), datfile_mean, atol=1)
        np.testing.assert_allclose(tim.data.std(), datfile_std, atol=1)
        with pytest.raises(FileNotFoundError):
            TimeSeries.from_dat("nonexistent.dat")

    def test_from_tim(
        self,
        timfile: str,
        timfile_mean: float,
        timfile_std: float,
    ) -> None:
        tim = TimeSeries.from_tim(timfile)
        np.testing.assert_allclose(tim.data.mean(), timfile_mean, atol=1)
        np.testing.assert_allclose(tim.data.std(), timfile_std, atol=1)


@pytest.mark.parametrize("signed", [False, True])
def test_from_tim_eight_bit_sign(tmp_path: Path, signed: bool) -> None:
    path = tmp_path / "eight.tim"
    data = np.array(
        [-128, -1, 0, 127] if signed else [0, 127, 128, 255],
        dtype=np.int8 if signed else np.uint8,
    )
    header = Header(
        filename=str(path),
        data_type="time series",
        nchans=1,
        foff=1.0,
        fch1=1400.0,
        nbits=8,
        tsamp=0.001,
        tstart=50000.0,
        nsamples=4,
        signed=signed,
    )
    with path.open("wb") as handle:
        handle.write(sigproc.encode_header(header.to_sigproc()))
        handle.write(data.tobytes())
    series = TimeSeries.from_tim(path)
    np.testing.assert_array_equal(series.data, data.astype(np.float32))


@pytest.mark.parametrize("terminal_padding", [True, False])
def test_from_dat_terminal_padding(tmp_path: Path, terminal_padding: bool) -> None:
    path = tmp_path / "padding.dat"
    data = np.arange(128, dtype=np.float32)
    data[100:127] = 0
    data.tofile(path)
    header = Header(
        filename=str(path),
        data_type="time series",
        nchans=1,
        foff=1.0,
        fch1=1400.0,
        nbits=32,
        tsamp=0.001,
        tstart=50000.0,
        nsamples=data.size,
    )
    text = header.make_inf()
    last = "127, 127" if terminal_padding else "110, 127"
    text += (
        "\n Any breaks in the data? (1=yes, 0=no) = 1"
        "\n On/Off bin pair # 1 = 0, 99\n On/Off bin pair # 2 = " + last + "\n"
    )
    path.with_suffix(".inf").write_text(text)
    trimmed = TimeSeries.from_dat(path)
    count = 100 if terminal_padding else 128
    assert trimmed.nsamples == count
    assert trimmed.header.nsamples == count
    np.testing.assert_array_equal(trimmed.data, data[:count])
    raw = TimeSeries.from_dat(path, trim_padding=False)
    np.testing.assert_array_equal(raw.data, data)


def test_pulse_search_ignores_terminal_padding(tmp_path: Path) -> None:
    path = tmp_path / "observed.dat"
    nvalid, ntotal = 20500, 32000
    data = np.zeros(ntotal, dtype=np.float32)
    data[:nvalid] = 1000 + np.random.default_rng(99).normal(size=nvalid)
    start, width = 20460, 14
    data[start : start + width] += 18 / np.sqrt(width)
    data.tofile(path)
    header = Header(
        filename=str(path),
        data_type="time series",
        nchans=1,
        foff=1.0,
        fch1=1400.0,
        nbits=32,
        tsamp=0.001,
        tstart=50000.0,
        nsamples=ntotal,
    )
    text = header.make_inf() + (
        "\n Any breaks in the data? (1 yes, 0 no) = 1"
        f"\n On/Off bin pair # 1 = 0, {nvalid - 1}"
        f"\n On/Off bin pair # 2 = {ntotal - 1}, {ntotal - 1}\n"
    )
    path.with_suffix(".inf").write_text(text)
    series = TimeSeries.from_dat(path)
    cands = series.find_pulses(snr_min=8, max_width=150, widths=[width])
    bins = cands.values("bin")
    assert len(cands) >= 1
    assert np.min(np.abs(bins - (start + width // 2))) <= 1
    # Proximity pruning can retain shoulders of a bright event, but there
    # must be no unrelated high-SNR detections induced by the padding edge.
    assert np.all(np.abs(bins - (start + width // 2)) <= width)
    assert series.nsamples == nvalid


def test_from_dat_uses_actual_file_length(tmp_path: Path) -> None:
    path = tmp_path / "length.dat"
    data = np.arange(128, dtype=np.float32)
    data.tofile(path)
    header = Header(
        filename=str(path),
        data_type="time series",
        nchans=1,
        foff=1.0,
        fch1=1400.0,
        nbits=32,
        tsamp=0.001,
        tstart=50000.0,
        nsamples=16,
    )
    path.with_suffix(".inf").write_text(header.make_inf())
    series = TimeSeries.from_dat(path)
    assert series.nsamples == data.size
    assert series.header.nsamples == data.size
    np.testing.assert_array_equal(series.data, data)
