from pathlib import Path

import numpy as np
import pytest

from sigpyproc.header import Header
from sigpyproc.readers import FilReader
from sigpyproc.simulation.pulsar import PulsarParams, fake_filterbank, make_header


def _tiny_header(
    path: str,
    *,
    nbits: int = 32,
    nchans: int = 16,
    tobs: float = 1.0,
    fch1: float = 1400.0,
    foff: float = -1.0,
) -> Header:
    return make_header(
        filename=path,
        nchans=nchans,
        nbits=nbits,
        tsamp=1e-3,
        tobs=tobs,
        tstart=50000.0,
        fch1=fch1,
        foff=foff,
    )


class TestPulsarParams:
    def test_period_and_pulse_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            PulsarParams(period=0.1, pulse_time=0.5)

    def test_default_width(self) -> None:
        assert PulsarParams(period=1.0).resolved_width() == pytest.approx(0.04)
        assert PulsarParams().resolved_width() == 0.0
        with pytest.raises(ValueError, match="width is required"):
            PulsarParams(pulse_time=0.5).resolved_width()

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            PulsarParams(period=0.0)


class TestFakeFilterbank:
    def test_noise_stats(self, tmpdir: str) -> None:
        path = str(Path(tmpdir) / "noise.fil")
        header = _tiny_header(path, tobs=2.0)
        fake_filterbank(header, seed=1, gulp=128)
        block = FilReader(path).read_block(0, header.nsamples)
        assert block.data.shape == (header.nchans, header.nsamples)
        assert block.data.mean() == pytest.approx(0.0, abs=0.05)
        assert block.data.std() == pytest.approx(1.0, abs=0.05)

    def test_seed_reproducible(self, tmpdir: str) -> None:
        a = str(Path(tmpdir) / "a.fil")
        b = str(Path(tmpdir) / "b.fil")
        header = _tiny_header(a)
        fake_filterbank(header, seed=7, gulp=64)
        fake_filterbank(header.new_header({"filename": b}), seed=7, gulp=64)
        da = FilReader(a).read_block(0, header.nsamples).data
        db = FilReader(b).read_block(0, header.nsamples).data
        np.testing.assert_array_equal(da, db)

    def test_pulsar_on_phase(self, tmpdir: str) -> None:
        path = str(Path(tmpdir) / "psr.fil")
        period = 0.1
        header = _tiny_header(path, tobs=2.0)
        fake_filterbank(
            header,
            PulsarParams(period=period, width=0.02, snr=40.0, smear=False),
            seed=2,
            gulp=256,
        )
        ts = FilReader(path).read_block(0, header.nsamples).data.mean(axis=0)
        phase = np.mod(np.arange(header.nsamples) * header.tsamp / period, 1.0)
        on = ts[(phase > 0.4) & (phase < 0.6)].mean()
        off = ts[(phase < 0.2) | (phase > 0.8)].mean()
        assert on > off + 2.0

    def test_pulse_amplitude_not_rescaled(self, tmpdir: str) -> None:
        path = str(Path(tmpdir) / "amp.fil")
        header = _tiny_header(path, nchans=16, tobs=1.0)
        snr = 50.0
        fake_filterbank(
            header,
            PulsarParams(pulse_time=0.5, width=0.04, snr=snr, smear=False),
            seed=3,
            gulp=128,
        )
        ts = FilReader(path).read_block(0, header.nsamples).data.mean(axis=0)
        expected = snr / np.sqrt(header.nchans)
        peak = float(ts.max())
        assert peak == pytest.approx(expected, rel=0.25)

    def test_single_pulse_time(self, tmpdir: str) -> None:
        path = str(Path(tmpdir) / "sp.fil")
        pulse_time = 0.4
        header = _tiny_header(path)
        fake_filterbank(
            header,
            PulsarParams(pulse_time=pulse_time, width=0.01, snr=40.0, smear=False),
            seed=4,
        )
        ts = FilReader(path).read_block(0, header.nsamples).data.mean(axis=0)
        peak_t = int(ts.argmax()) * header.tsamp
        assert peak_t == pytest.approx(pulse_time, abs=0.01)

    def test_dm_delays_low_freq(self, tmpdir: str) -> None:
        path = str(Path(tmpdir) / "dm.fil")
        header = _tiny_header(path, nchans=32, tobs=1.0, fch1=1400.0, foff=-1.0)
        fake_filterbank(
            header,
            PulsarParams(pulse_time=0.3, width=0.004, dm=80.0, snr=80.0, smear=False),
            seed=5,
        )
        data = FilReader(path).read_block(0, header.nsamples).data
        first = int(data[0].argmax())
        last = int(data[-1].argmax())
        assert last > first

    def test_4bit_readable(self, tmpdir: str) -> None:
        path = str(Path(tmpdir) / "q.fil")
        header = _tiny_header(path, nbits=4, nchans=16, tobs=0.5)
        out = fake_filterbank(header, seed=6, gulp=64)
        fil = FilReader(out)
        assert fil.header.nbits == 4
        assert fil.header.nsamples == header.nsamples
        assert fil.header.nchans == header.nchans
        block = fil.read_block(0, fil.header.nsamples)
        assert block.data.shape == (header.nchans, header.nsamples)

    def test_bad_packed_nchans(self, tmpdir: str) -> None:
        path = str(Path(tmpdir) / "bad.fil")
        header = _tiny_header(path, nbits=4, nchans=3)
        with pytest.raises(ValueError, match="multiple of 8"):
            fake_filterbank(header, seed=1)


@pytest.mark.parametrize(
    "changes",
    [
        {"pdot": 1.0, "accel": 1.0},
        {"dm": np.nan},
        {"dm": np.inf},
        {"dm": -1.0},
        {"snr": np.nan},
        {"snr": 0.0},
        {"snr": -1.0},
        {"pdot": np.inf},
        {"accel": -np.inf},
    ],
)
def test_pulsar_parameters_reject_unphysical_values(changes: dict) -> None:
    with pytest.raises(ValueError):
        PulsarParams(**changes)


@pytest.mark.parametrize(
    "changes",
    [{"tsamp": 0.0}, {"tobs": 0.0}, {"nchans": 0}, {"tobs": 1e-8, "tsamp": 1.0}],
)
def test_simulation_geometry_cannot_create_empty_files(changes: dict) -> None:
    with pytest.raises(ValueError):
        make_header(**changes)


@pytest.mark.parametrize(
    "field",
    ["tsamp", "tobs", "tstart", "fch1", "foff", "dm", "period"],
)
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_fake_header_rejects_nonfinite_metadata(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_header(**{field: value})


def test_fake_header_accepts_finite_metadata() -> None:
    header = make_header(dm=12.5, period=0.25, tsamp=1e-3, tobs=0.1)
    assert header.dm == pytest.approx(12.5)
    assert header.period == pytest.approx(0.25)
    assert header.nsamples == 100


@pytest.mark.parametrize(
    "params",
    [PulsarParams(period=0.1, pdot=-1.0), PulsarParams(period=0.1, accel=-1e12)],
)
def test_period_evolution_cannot_cross_zero(
    tmp_path: Path, params: PulsarParams
) -> None:
    path = tmp_path / "invalid.fil"
    header = _tiny_header(str(path))
    with pytest.raises(ValueError, match="period through zero"):
        fake_filterbank(header, params)
    assert not path.exists()


@pytest.mark.parametrize("field", ["period", "pulse_time", "width"])
def test_optional_pulsar_parameters_reject_infinite_values(field: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        PulsarParams(**{field: np.inf})
