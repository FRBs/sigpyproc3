from pathlib import Path

import numpy as np
import pytest

from sigpyproc.header import Header
from sigpyproc.readers import FilReader
from sigpyproc.simulation.furby import FurbyGenerator, PulseParams
from sigpyproc.simulation.inject import Injector, inject_file, read_pulse_catalogue
from sigpyproc.simulation.pulsar import PulsarParams, fake_filterbank


def _header() -> Header:
    return Header(
        filename="tiny.fil",
        data_type="filterbank",
        nchans=8,
        foff=-1.0,
        fch1=1400.0,
        nbits=32,
        tsamp=1e-3,
        tstart=50000.0,
        nsamples=2000,
    )


class TestInjector:
    def test_channel_zero_peak_lands_on_arrival(self) -> None:
        header = _header()
        params = PulseParams(dm=0.0, snr=30.0, width=4e-3, dmsmear=False)
        pulse = FurbyGenerator(header, params, seed=1).generate()
        arrival = 1000
        injector = Injector.from_furbies([pulse], [arrival])
        block = np.zeros((1500, header.nchans), dtype=np.float32)
        injector.add_into(block, 0)
        profile_peak = int(np.argmax(pulse.block.data[0]))
        expected = arrival + profile_peak + int(pulse.sample0) + int(pulse.delays[0])
        found = int(np.argmax(block[:, 0]))
        assert abs(found - expected) <= 1
        assert abs(profile_peak + int(pulse.sample0)) <= 1

    def test_catalogue_skips_comments(self, tmp_path: Path) -> None:
        path = tmp_path / "pulses.txt"
        path.write_text("# comment\n\n0.1 10 0.004 20\n0.2 12 0.005 30  # tail\n")
        assert read_pulse_catalogue(path) == [
            (0.1, 10.0, 0.004, 20.0),
            (0.2, 12.0, 0.005, 30.0),
        ]

    def test_catalogue_rejects_a_short_row(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.txt"
        path.write_text("0.1 10 0.004\n")
        with pytest.raises(ValueError, match="time dm width snr"):
            read_pulse_catalogue(path)

    def test_catalogue_rejects_an_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("# nothing\n\n")
        with pytest.raises(ValueError, match="no pulses"):
            read_pulse_catalogue(path)

    def test_catalogue_with_optional_columns_and_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "pulses.csv"
        path.write_text(
            "# time, dm, width, snr, spec_idx, tau0, spec_kind\n"
            "0.1, 10.0, 0.002, 25.0\n"
            "0.2, 20.0, 0.003, 30.0, -1.5\n"
            "0.3, 30.0, 0.004, 35.0, -2.0, 0.001\n"
            "0.4, 40.0, 0.005, 40.0, -1.0, 0.002, scintillation\n",
        )
        rows = read_pulse_catalogue(path)
        assert len(rows) == 4
        assert rows[0] == (0.1, 10.0, 0.002, 25.0)
        assert rows[1] == (0.2, 20.0, 0.003, 30.0, -1.5)
        assert rows[2] == (0.3, 30.0, 0.004, 35.0, -2.0, 0.001)
        assert rows[3] == (0.4, 40.0, 0.005, 40.0, -1.0, 0.002, "scintillation")

    def test_inject_file_preserve_bandpass(self, tmp_path: Path) -> None:
        header = _header()
        infile = str(tmp_path / "base.fil")
        outfile = str(tmp_path / "injected.fil")
        fake_filterbank(header, PulsarParams(), infile, seed=42)

        params = PulseParams(dm=0.0, snr=50.0, width=4e-3, dmsmear=False)
        pulse = FurbyGenerator(header, params, seed=1).generate()
        injector = Injector.from_furbies([pulse], [500])

        out = inject_file(infile, outfile, injector, preserve_bandpass=True)
        assert Path(out).is_file()

        reader_in = FilReader(infile)
        reader_out = FilReader(outfile)
        assert reader_out.header.nsamples == reader_in.header.nsamples

        data_in = reader_in.read_block(0, reader_in.header.nsamples).data
        data_out = reader_out.read_block(0, reader_out.header.nsamples).data

        # Away from pulse (e.g. samples 1000..1500), values should closely match
        np.testing.assert_allclose(
            data_in[:, 1000:1500],
            data_out[:, 1000:1500],
            atol=1e-5,
        )
        # Around the pulse arrival (e.g. sample 500), power must be significantly higher
        assert data_out[:, 495:505].sum() > data_in[:, 495:505].sum() + 10.0

    def test_inject_file_preserve_bandpass_8bit_unsigned(self, tmp_path: Path) -> None:
        header = Header(
            filename=str(tmp_path / "base_u8.fil"),
            data_type="filterbank",
            nchans=8,
            foff=-1.0,
            fch1=1400.0,
            nbits=8,
            tsamp=1e-3,
            tstart=50000.0,
            nsamples=2000,
        )
        infile = str(tmp_path / "base_u8.fil")
        outfile = str(tmp_path / "injected_u8.fil")
        fake_filterbank(header, PulsarParams(), infile, seed=42)

        params = PulseParams(dm=0.0, snr=50.0, width=4e-3, dmsmear=False)
        pulse = FurbyGenerator(header, params, seed=1).generate()
        injector = Injector.from_furbies([pulse], [500])

        out = inject_file(infile, outfile, injector, preserve_bandpass=True)
        assert Path(out).is_file()

        reader_in = FilReader(infile)
        reader_out = FilReader(outfile)
        assert reader_out.header.nbits == 8
        assert reader_out.header.signed is False

        data_in = reader_in.read_block(0, 2000).data
        data_out = reader_out.read_block(0, 2000).data

        # Off-pulse samples match exactly
        np.testing.assert_allclose(
            data_in[:, 1000:1500],
            data_out[:, 1000:1500],
            atol=0.0,
        )
        assert data_out[:, 495:505].sum() > data_in[:, 495:505].sum() + 10.0

    def test_inject_file_preserve_bandpass_8bit_signed(self, tmp_path: Path) -> None:
        infile = str(tmp_path / "base_s8.fil")
        outfile = str(tmp_path / "injected_s8.fil")
        header = Header(
            filename=infile,
            data_type="filterbank",
            nchans=8,
            foff=-1.0,
            fch1=1400.0,
            nbits=8,
            signed=True,
            tsamp=1e-3,
            tstart=50000.0,
            nsamples=2000,
        )
        rng = np.random.default_rng(42)
        raw_data = rng.integers(-50, 50, size=(2000, 8), dtype=np.int8)
        with header.prep_outfile(infile) as writer:
            writer.write(raw_data.tobytes())

        reader_in = FilReader(infile)
        assert reader_in.header.signed is True
        data_in = reader_in.read_block(0, 2000).data

        params = PulseParams(dm=0.0, snr=50.0, width=4e-3, dmsmear=False)
        pulse = FurbyGenerator(header, params, seed=1).generate()
        injector = Injector.from_furbies([pulse], [500])

        out = inject_file(infile, outfile, injector, preserve_bandpass=True)
        assert Path(out).is_file()

        reader_out = FilReader(outfile)
        assert reader_out.header.nbits == 8
        assert reader_out.header.signed is True
        data_out = reader_out.read_block(0, 2000).data

        # Off-pulse signed samples match exactly
        np.testing.assert_allclose(
            data_in[:, 1000:1500],
            data_out[:, 1000:1500],
            atol=0.0,
        )
        assert data_out[:, 495:505].sum() > data_in[:, 495:505].sum() + 10.0

    def test_inject_file_preserve_bandpass_4bit(self, tmp_path: Path) -> None:
        header = Header(
            filename=str(tmp_path / "base_u4.fil"),
            data_type="filterbank",
            nchans=8,
            foff=-1.0,
            fch1=1400.0,
            nbits=4,
            tsamp=1e-3,
            tstart=50000.0,
            nsamples=2000,
        )
        infile = str(tmp_path / "base_u4.fil")
        outfile = str(tmp_path / "injected_u4.fil")
        fake_filterbank(header, PulsarParams(), infile, seed=42)

        params = PulseParams(dm=0.0, snr=50.0, width=4e-3, dmsmear=False)
        pulse = FurbyGenerator(header, params, seed=1).generate()
        injector = Injector.from_furbies([pulse], [500])

        out = inject_file(infile, outfile, injector, preserve_bandpass=True)
        assert Path(out).is_file()

        reader_in = FilReader(infile)
        reader_out = FilReader(outfile)
        assert reader_out.header.nbits == 4

        data_in = reader_in.read_block(0, 2000).data
        data_out = reader_out.read_block(0, 2000).data

        # Off-pulse 4-bit samples match exactly
        np.testing.assert_allclose(
            data_in[:, 1000:1500],
            data_out[:, 1000:1500],
            atol=0.0,
        )
        assert data_out[:, 495:505].sum() > data_in[:, 495:505].sum() + 10.0
