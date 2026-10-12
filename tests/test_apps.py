import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sigpyproc.apps import (
    spp_bandpass,
    spp_clean,
    spp_decimate,
    spp_dedisperse,
    spp_extract,
    spp_fake,
    spp_fold,
    spp_header,
    spp_merge,
    spp_search,
    spp_splice,
    spp_stats,
    spp_view,
    spp_zerodm,
)
from sigpyproc.header import Header
from sigpyproc.io import sigproc
from sigpyproc.readers import FilReader
from sigpyproc.timeseries import TimeSeries


class TestSppHeader:
    def test_print(self, filfile_8bit_1: str) -> None:
        runner = CliRunner()
        result = runner.invoke(spp_header.main, ["print", filfile_8bit_1])
        assert result.exit_code == 0
        assert "Header size (bytes)" in result.output
        assert "Signedness of 8-bit numbers" in result.output

    def test_print_fail(self) -> None:
        runner = CliRunner()
        filename = "invalid"
        result = runner.invoke(spp_header.main, ["print", filename])
        assert result.exit_code == 2
        assert f"Path '{filename}' does not exist." in result.output

    def test_get(self, filfile_8bit_1: str) -> None:
        runner = CliRunner()
        result = runner.invoke(
            spp_header.main,
            ["get", filfile_8bit_1, "-k", "source"],
        )
        assert result.exit_code == 0
        assert "source = " in result.output
        result = runner.invoke(
            spp_header.main,
            ["get", filfile_8bit_1, "-k", "source_name"],
        )
        assert result.exit_code == 0
        assert "source_name = " in result.output

    def test_get_keys_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(spp_header.main, ["get", "--keys-help"])
        assert result.exit_code == 0
        assert "source_name" in result.output
        assert "fch1" in result.output

    def test_update(self, filfile_8bit_1: str, tmpfile: str) -> None:
        shutil.copy(filfile_8bit_1, tmpfile)
        runner = CliRunner()
        result = runner.invoke(
            spp_header.main,
            ["update", tmpfile, "-i", "source_name", "TEST"],
        )
        assert result.exit_code == 0
        assert result.output == ""
        result = runner.invoke(spp_header.main, ["get", tmpfile, "-k", "source_name"])
        assert result.exit_code == 0
        assert "source_name = TEST" in result.output

    def test_update_dry_run(self, filfile_8bit_1: str, tmpfile: str) -> None:
        shutil.copy(filfile_8bit_1, tmpfile)
        before = Header.from_sigproc(tmpfile).source
        runner = CliRunner()
        result = runner.invoke(
            spp_header.main,
            ["update", tmpfile, "-i", "source_name", "DRYRUN", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "Before" in result.output
        assert "After" in result.output
        assert "DRYRUN" in result.output
        after = Header.from_sigproc(tmpfile).source
        assert after == before

    def test_update_numeric_values_from_cli(
        self, filfile_8bit_1: str, tmpfile: str
    ) -> None:
        shutil.copy(filfile_8bit_1, tmpfile)
        runner = CliRunner()
        result = runner.invoke(
            spp_header.main,
            ["update", tmpfile, "-i", "fch1", "1400.5", "-i", "ibeam", "3"],
        )
        assert result.exit_code == 0, result.output
        header = Header.from_sigproc(tmpfile)
        assert header.fch1 == 1400.5
        assert header.ibeam == 3

    @pytest.mark.parametrize(
        ("args", "exit_code", "message"),
        [
            (["-i", "nosuchkey", "1"], 2, "not a SIGPROC header key"),
            (["-i", "nbits", "3"], 1, "cannot be changed in place"),
            (["-i", "ibeam", "abc"], 1, "needs a whole number"),
            (["-i", "src_raj", "250000"], 1, "right ascension"),
            (["-i", "source_name", "X" * 200], 1, "has room for"),
            (["-i", "rawdatafile", ""], 1, "cannot be empty"),
            (["-i", "ibeam", "1", "-i", "ibeam", "2"], 2, "more than once"),
        ],
    )
    def test_update_rejects_bad_input(
        self,
        filfile_8bit_1: str,
        tmpfile: str,
        args: list[str],
        exit_code: int,
        message: str,
    ) -> None:
        shutil.copy(filfile_8bit_1, tmpfile)
        before = Path(tmpfile).read_bytes()
        runner = CliRunner()
        result = runner.invoke(spp_header.main, ["update", tmpfile, *args])
        assert result.exit_code == exit_code
        assert message in result.output
        assert "Traceback" not in result.output
        assert Path(tmpfile).read_bytes() == before

    def test_update_multiple_is_all_or_nothing(
        self, filfile_8bit_1: str, tmpfile: str
    ) -> None:
        shutil.copy(filfile_8bit_1, tmpfile)
        before = Path(tmpfile).read_bytes()
        runner = CliRunner()
        result = runner.invoke(
            spp_header.main,
            ["update", tmpfile, "-i", "ibeam", "5", "-i", "src_dej", "900100"],
        )
        assert result.exit_code == 1
        assert Path(tmpfile).read_bytes() == before

    def test_update_refuses_key_missing_from_file(self, tmpdir: str) -> None:
        path = Path(tmpdir) / "min.fil"
        payload = sigproc.encode_header(
            {
                "data_type": 1,
                "source_name": "T",
                "tstart": 60000.0,
                "tsamp": 1e-3,
                "nbits": 8,
                "fch1": 1400.0,
                "foff": -1.0,
                "nchans": 4,
                "nifs": 1,
            }
        )
        path.write_bytes(payload + bytes(16))
        runner = CliRunner()
        result = runner.invoke(
            spp_header.main, ["update", str(path), "-i", "nbeams", "2"]
        )
        assert result.exit_code == 1
        assert "cannot add keys" in result.output

    def test_get_unknown_key_is_clean_error(self, filfile_8bit_1: str) -> None:
        runner = CliRunner()
        result = runner.invoke(
            spp_header.main, ["get", filfile_8bit_1, "-k", "to_string"]
        )
        assert result.exit_code == 2
        assert "Unknown header key 'to_string'" in result.output
        assert "Traceback" not in result.output

    def test_print_non_sigproc_is_clean_error(self, tmpdir: str) -> None:
        path = Path(tmpdir) / "bad.fil"
        path.write_bytes(b"not a sigproc file at all")
        runner = CliRunner()
        result = runner.invoke(spp_header.main, ["print", str(path)])
        assert result.exit_code == 1
        assert "Traceback" not in result.output

    def test_samples(self, filfile_8bit_1: str, tmpfile: str) -> None:
        nsamples = 10
        runner = CliRunner()
        result = runner.invoke(
            spp_extract.main,
            ["samples", filfile_8bit_1, "-s", "0", "-n", "10", "-o", tmpfile],
        )
        assert result.exit_code == 0
        assert tmpfile in result.output
        result = runner.invoke(spp_header.main, ["get", tmpfile, "-k", "nsamples"])
        assert result.exit_code == 0
        assert f"nsamples = {nsamples}" in result.output

    def test_samples_duration(self, filfile_8bit_1: str, tmpfile: str) -> None:
        header = Header.from_sigproc(filfile_8bit_1)
        duration = 10 * header.tsamp
        runner = CliRunner()
        result = runner.invoke(
            spp_extract.main,
            [
                "samples",
                filfile_8bit_1,
                "--start-sec",
                "0",
                "--duration",
                str(duration),
                "-o",
                tmpfile,
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(spp_header.main, ["get", tmpfile, "-k", "nsamples"])
        assert "nsamples = 10" in result.output

    def test_channels(self, filfile_8bit_1: str, tmpfile: str) -> None:
        runner = CliRunner()
        result = runner.invoke(
            spp_extract.main,
            ["channels", filfile_8bit_1, "-c", "0", "-c", "1", "-o", tmpfile],
        )
        assert result.exit_code == 0
        outfiles = list(Path(tmpfile).parent.glob(f"{Path(tmpfile).name}_*"))
        assert len(outfiles) == 2

    def test_channels_range(self, filfile_8bit_1: str, tmpdir: str) -> None:
        outbase = str(Path(tmpdir) / "range")
        runner = CliRunner()
        result = runner.invoke(
            spp_extract.main,
            [
                "channels",
                filfile_8bit_1,
                "--chan-start",
                "0",
                "--nchans",
                "3",
                "-o",
                outbase,
            ],
        )
        assert result.exit_code == 0
        written = list(Path(tmpdir).glob("range_chan*.tim"))
        assert len(written) == 3

    def test_bands(self, filfile_8bit_1: str, tmpfile: str) -> None:
        runner = CliRunner()
        result = runner.invoke(
            spp_extract.main,
            ["bands", filfile_8bit_1, "-s", "0", "-n", "10", "-c", "2", "-o", tmpfile],
        )
        assert result.exit_code == 0
        outfiles = list(Path(tmpfile).parent.glob(f"{Path(tmpfile).name}_*"))
        assert outfiles


class TestSppStats:
    def test_quiet_flag(self, filfile_8bit_1: str) -> None:
        runner = CliRunner()
        result = runner.invoke(spp_stats.main, ["-q", filfile_8bit_1])
        assert result.exit_code == 0

    def test_print(self, filfile_8bit_1: str) -> None:
        runner = CliRunner()
        result = runner.invoke(spp_stats.main, [filfile_8bit_1])
        assert result.exit_code == 0
        assert "freq_mhz" in result.output
        header = Header.from_sigproc(filfile_8bit_1)
        assert result.output.count("\n") >= header.nchans

    def test_csv(self, filfile_8bit_1: str, tmpfile: str) -> None:
        runner = CliRunner()
        result = runner.invoke(spp_stats.main, [filfile_8bit_1, "-o", tmpfile])
        assert result.exit_code == 0
        text = Path(tmpfile).read_text()
        assert "freq_mhz" in text
        assert len(text.splitlines()) > 1


class TestSppDecimate:
    def test_main(self, filfile_8bit_1: str, tmpfile: str) -> None:
        runner = CliRunner()
        result = runner.invoke(
            spp_decimate.main,
            [filfile_8bit_1, "-t", "2", "-c", "2", "-o", tmpfile],
        )
        assert result.exit_code == 0
        out = FilReader(tmpfile)
        src = FilReader(filfile_8bit_1)
        assert out.header.tsamp == src.header.tsamp * 2
        assert out.header.nchans == src.header.nchans // 2


class TestSppBandpass:
    def test_print(self, filfile_8bit_1: str) -> None:
        runner = CliRunner()
        result = runner.invoke(spp_bandpass.main, [filfile_8bit_1])
        assert result.exit_code == 0
        assert "freq_mhz" in result.output


class TestSppZerodm:
    def test_main(self, filfile_8bit_1: str, tmpfile: str) -> None:
        runner = CliRunner()
        result = runner.invoke(spp_zerodm.main, [filfile_8bit_1, "-o", tmpfile])
        assert result.exit_code == 0
        assert Path(tmpfile).exists()


class TestSppMergeSplice:
    def test_merge_time(self, filfile_8bit_1: str, tmpdir: str) -> None:
        a = str(Path(tmpdir) / "a.fil")
        b = str(Path(tmpdir) / "b.fil")
        out = str(Path(tmpdir) / "merged.fil")
        src = FilReader(filfile_8bit_1)
        n = 32
        src.extract_samps(0, n, outfile_name=a)
        src.extract_samps(n, n, outfile_name=b)
        runner = CliRunner()
        result = runner.invoke(
            spp_merge.main,
            [a, b, "-a", "time", "-o", out],
        )
        assert result.exit_code == 0
        assert FilReader(out).header.nsamples == 2 * n

    def test_splice_freq(self, filfile_8bit_1: str, tmpdir: str) -> None:
        src = FilReader(filfile_8bit_1)
        nchan = src.header.nchans
        half = nchan // 2
        files = src.extract_bands(
            chanstart=0,
            nchans=nchan,
            chanpersub=half,
            outfile_base=str(Path(tmpdir) / "band"),
        )
        out = str(Path(tmpdir) / "spliced.fil")
        runner = CliRunner()
        result = runner.invoke(spp_splice.main, [*files, "-o", out])
        assert result.exit_code == 0
        merged = FilReader(out)
        assert merged.header.nchans == half * len(files)
        assert merged.header.nsamples == src.header.nsamples


class TestSppDedisperse:
    def test_tim(self, filfile_8bit_1: str, tmpfile: str) -> None:
        runner = CliRunner()
        result = runner.invoke(
            spp_dedisperse.main,
            ["tim", filfile_8bit_1, "-d", "10", "-o", tmpfile],
        )
        assert result.exit_code == 0
        tim = FilReader(tmpfile)
        assert tim.header.nchans == 1

    def test_subband(self, filfile_8bit_1: str, tmpfile: str) -> None:
        runner = CliRunner()
        result = runner.invoke(
            spp_dedisperse.main,
            ["subband", filfile_8bit_1, "-d", "10", "-b", "16", "-o", tmpfile],
        )
        assert result.exit_code == 0
        assert FilReader(tmpfile).header.nchans == 16

    def test_tree(self, filfile_8bit_1: str, tmpdir: str) -> None:
        band = str(Path(tmpdir) / "band.fil")
        src = FilReader(filfile_8bit_1)
        src.extract_bands(
            chanstart=0,
            nchans=32,
            chanpersub=32,
            outfile_base=str(Path(tmpdir) / "b"),
        )
        band = next(Path(tmpdir).glob("b_sub*.fil")).as_posix()
        runner = CliRunner()
        result = runner.invoke(
            spp_dedisperse.main,
            [
                "tree",
                band,
                "-l",
                "0",
                "-u",
                "2",
                "--chunk",
                "256",
                "-o",
                str(Path(tmpdir) / "t"),
            ],
        )
        assert result.exit_code == 0
        assert len(list(Path(tmpdir).glob("t.DM*.tim"))) == 3

    def test_plan(self, filfile_8bit_1: str, tmpdir: str) -> None:
        src = FilReader(filfile_8bit_1)
        src.extract_bands(
            chanstart=0,
            nchans=32,
            chanpersub=32,
            outfile_base=str(Path(tmpdir) / "p"),
        )
        band = next(Path(tmpdir).glob("p_sub*.fil")).as_posix()
        runner = CliRunner()
        result = runner.invoke(
            spp_dedisperse.main,
            ["plan", band, "--dm-max", "1", "-o", str(Path(tmpdir) / "plan")],
        )
        assert result.exit_code == 0
        assert list(Path(tmpdir).glob("plan_DM*.tim"))


class TestSppFold:
    def test_main(self, filfile_8bit_1: str, tmpdir: str) -> None:
        out = str(Path(tmpdir) / "fold")
        runner = CliRunner()
        result = runner.invoke(
            spp_fold.main,
            [
                filfile_8bit_1,
                "-p",
                "0.5",
                "-d",
                "0",
                "-n",
                "32",
                "--nints",
                "8",
                "--nbands",
                "8",
                "-o",
                out,
            ],
        )
        assert result.exit_code == 0
        assert Path(out + ".npz").exists()
        assert Path(out + ".txt").exists()


class TestSppClean:
    def test_rfi(self, filfile_8bit_1: str, tmpfile: str) -> None:
        runner = CliRunner()
        result = runner.invoke(spp_clean.main, ["rfi", filfile_8bit_1, "-o", tmpfile])
        assert result.exit_code == 0
        assert Path(tmpfile).exists()
        maskfile = tmpfile + ".mask"
        result = runner.invoke(
            spp_clean.main,
            ["rfi", filfile_8bit_1, "-o", tmpfile, "-s", maskfile],
        )
        assert result.exit_code == 0
        assert Path(tmpfile).exists()
        assert Path(maskfile).exists()

    def test_zap_channels(self, filfile_8bit_1: str, tmpfile: str) -> None:
        runner = CliRunner()
        result = runner.invoke(
            spp_clean.main,
            ["zap-channels", filfile_8bit_1, "-c", "0", "-c", "1", "-o", tmpfile],
        )
        assert result.exit_code == 0
        assert Path(tmpfile).exists()
        out = FilReader(tmpfile)
        src = FilReader(filfile_8bit_1)
        assert out.header.nchans == src.header.nchans
        zapped = out.read_block(0, out.header.nsamples).data
        original = src.read_block(0, src.header.nsamples).data
        np.testing.assert_array_equal(zapped[:2], 0)
        np.testing.assert_array_equal(zapped[2:], original[2:])

    def test_zap_times(self, filfile_8bit_1: str, tmpfile: str) -> None:
        runner = CliRunner()
        result = runner.invoke(
            spp_clean.main,
            ["zap-times", filfile_8bit_1, "-s", "0", "-e", "8", "-o", tmpfile],
        )
        assert result.exit_code == 0
        assert Path(tmpfile).exists()
        assert (
            FilReader(tmpfile).header.nsamples
            == FilReader(filfile_8bit_1).header.nsamples
        )


class TestSppFake:
    def test_noise(self, tmpdir: str) -> None:
        out = str(Path(tmpdir) / "noise.fil")
        runner = CliRunner()
        result = runner.invoke(
            spp_fake.main,
            [
                "noise",
                "-o",
                out,
                "--nchans",
                "16",
                "--nbits",
                "32",
                "--tsamp",
                "0.001",
                "--tobs",
                "0.2",
                "--seed",
                "1",
            ],
        )
        assert result.exit_code == 0
        assert out in result.output
        fil = FilReader(out)
        assert fil.header.nchans == 16
        assert fil.header.nbits == 32
        assert fil.header.nsamples == 200

    def test_pulsar(self, tmpdir: str) -> None:
        out = str(Path(tmpdir) / "psr.fil")
        runner = CliRunner()
        result = runner.invoke(
            spp_fake.main,
            [
                "pulsar",
                "-p",
                "0.05",
                "-d",
                "10",
                "-o",
                out,
                "--nchans",
                "16",
                "--nbits",
                "8",
                "--tsamp",
                "0.001",
                "--tobs",
                "0.3",
                "--seed",
                "2",
            ],
        )
        assert result.exit_code == 0
        fil = FilReader(out)
        assert fil.header.period == pytest.approx(0.05)
        assert fil.header.dm == pytest.approx(10.0)

    def test_pulse(self, tmpdir: str) -> None:
        out = str(Path(tmpdir) / "sp.fil")
        runner = CliRunner()
        result = runner.invoke(
            spp_fake.main,
            [
                "pulse",
                "-t",
                "0.1",
                "-w",
                "0.005",
                "-o",
                out,
                "--nchans",
                "16",
                "--nbits",
                "8",
                "--tsamp",
                "0.001",
                "--tobs",
                "0.3",
                "--seed",
                "3",
            ],
        )
        assert result.exit_code == 0
        assert FilReader(out).header.nsamples == 300

    def test_pulse_requires_width(self) -> None:
        runner = CliRunner()
        result = runner.invoke(spp_fake.main, ["pulse", "-t", "0.1"])
        assert result.exit_code != 0


def _write_search_tim(path: Path, nsamp: int = 4096, tsamp: float = 0.001) -> Path:
    rng = np.random.default_rng(7)
    data = rng.normal(0.0, 1.0, nsamp).astype(np.float32)
    t = np.arange(nsamp) * tsamp
    data += 4.0 * np.sin(2 * np.pi * t / 0.05).astype(np.float32)
    data[200:208] += 8.0
    header = Header(
        filename=path.name,
        rawdatafile=path.name,
        data_type="time series",
        nchans=1,
        foff=1.0,
        fch1=1400.0,
        nbits=32,
        tsamp=tsamp,
        tstart=50000.0,
        nsamples=nsamp,
    )
    TimeSeries(data, header).to_tim(str(path))
    return path


class TestSppSearch:
    def test_fft(self, tmpdir: str) -> None:
        tim = _write_search_tim(Path(tmpdir) / "inj.tim")
        out = str(Path(tmpdir) / "fft.csv")
        runner = CliRunner()
        result = runner.invoke(
            spp_search.main,
            ["fft", str(tim), "--nfolds", "1", "--pmax", "1", "-o", out],
        )
        assert result.exit_code == 0
        assert Path(out).is_file()
        text = Path(out).read_text()
        assert "period_s" in text
        assert "fft" in text

    def test_pulses(self, tmpdir: str) -> None:
        tim = _write_search_tim(Path(tmpdir) / "inj.tim")
        out = str(Path(tmpdir) / "pulses.csv")
        runner = CliRunner()
        result = runner.invoke(spp_search.main, ["pulses", str(tim), "-o", out])
        assert result.exit_code == 0
        assert Path(out).is_file()

    def test_cluster(self, tmpdir: str) -> None:
        src = Path(tmpdir) / "cands.csv"
        src.write_text(
            "period_s,freq_hz,snr,dm,accel,nharm,bin,width_s,kind\n"
            "1.0,1.0,20.0,0.0,0.0,1,10,nan,fft\n"
            "0.5,2.0,12.0,0.0,0.0,2,20,nan,fft\n",
        )
        out = str(Path(tmpdir) / "clustered.csv")
        runner = CliRunner()
        result = runner.invoke(spp_search.main, ["cluster", str(src), "-o", out])
        assert result.exit_code == 0
        lines = Path(out).read_text().strip().splitlines()
        assert len(lines) == 2  # header + loudest only

    def test_ffa(self, tmpdir: str) -> None:
        nsamp = 2048
        tsamp = 0.001
        path = Path(tmpdir) / "ffa.tim"
        rng = np.random.default_rng(8)
        data = rng.normal(0.0, 1.0, nsamp).astype(np.float32)
        header = Header(
            filename=path.name,
            rawdatafile=path.name,
            data_type="time series",
            nchans=1,
            foff=1.0,
            fch1=1400.0,
            nbits=32,
            tsamp=tsamp,
            tstart=50000.0,
            nsamples=nsamp,
        )
        TimeSeries(data, header).to_tim(str(path))
        out = str(Path(tmpdir) / "ffa.csv")
        runner = CliRunner()
        result = runner.invoke(
            spp_search.main,
            [
                "ffa",
                str(path),
                "--pmin-ms",
                "20",
                "--pmax-ms",
                "40",
                "-o",
                out,
            ],
        )
        assert result.exit_code == 0
        assert Path(out).is_file()

    def test_fft_stdout(self, tmpdir: str) -> None:
        tim = _write_search_tim(Path(tmpdir) / "inj.tim")
        runner = CliRunner()
        result = runner.invoke(
            spp_search.main,
            ["fft", str(tim), "--nfolds", "0", "--pmax", "1", "-o", "-"],
        )
        assert result.exit_code == 0
        assert "period_s" in result.output
        assert result.output.splitlines()[-1] != "-"

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(spp_search.main, ["fft", "--help"])
        assert result.exit_code == 0
        assert "harmonic folds" in result.output


class TestSppView:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(spp_view.main, ["--help"])
        assert result.exit_code == 0
        assert "Visualize pulsar/FRB filterbanks" in result.output

    def test_batch_png(self, filfile_8bit_1: str, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            spp_view.main,
            [
                filfile_8bit_1,
                "--batch",
                "--format",
                "png",
                "-o",
                str(tmp_path),
                "--stop",
                "2048",
                "-g",
                "1024",
                "-j",
                "1",
            ],
        )
        assert result.exit_code == 0
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) == 2
