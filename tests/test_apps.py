import shutil
from pathlib import Path

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
    spp_splice,
    spp_stats,
    spp_zerodm,
)
from sigpyproc.header import Header
from sigpyproc.readers import FilReader


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


class TestSppExtract:
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
