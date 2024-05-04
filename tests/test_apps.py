import shutil
from pathlib import Path

from click.testing import CliRunner

from sigpyproc.apps import (
    spp_clean,
    spp_extract,
    spp_header,
)


class TestSppHeader:
    def test_print(self, filfile_8bit_1: str) -> None:
        runner = CliRunner()
        result = runner.invoke(spp_header.main, ["print", filfile_8bit_1])
        assert result.exit_code == 0
        assert "Header size (bytes)" in result.output

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


class TestSppExtract:
    def test_samples(self, filfile_8bit_1: str, tmpfile: str) -> None:
        nsamples = 10
        runner = CliRunner()
        result = runner.invoke(
            spp_extract.main,
            ["samples", filfile_8bit_1, "-s", "0", "-n", "10", "-o", tmpfile],
        )
        assert result.exit_code == 0
        result = runner.invoke(spp_header.main, ["get", tmpfile, "-k", "nsamples"])
        assert result.exit_code == 0
        assert f"nsamples = {nsamples}" in result.output

    def test_channels(self, filfile_8bit_1: str, tmpfile: str) -> None:
        runner = CliRunner()
        result = runner.invoke(
            spp_extract.main,
            ["channels", filfile_8bit_1, "-c", "0", "-c", "1", "-o", tmpfile],
        )
        assert result.exit_code == 0
        outfiles = Path(tmpfile).parent.glob(f"{Path(tmpfile).name}_*")
        assert all(outfile.exists() for outfile in outfiles)

    def test_bands(self, filfile_8bit_1: str, tmpfile: str) -> None:
        runner = CliRunner()
        result = runner.invoke(
            spp_extract.main,
            ["bands", filfile_8bit_1, "-s", "0", "-n", "10", "-c", "2", "-o", tmpfile],
        )
        assert result.exit_code == 0
        outfiles = Path(tmpfile).parent.glob(f"{Path(tmpfile).name}_*")
        assert all(outfile.exists() for outfile in outfiles)


class TestSppClean:
    def test_main(self, filfile_8bit_1: str, tmpfile: str) -> None:
        runner = CliRunner()
        result = runner.invoke(spp_clean.main, [filfile_8bit_1, "-o", tmpfile])
        assert result.exit_code == 0
        assert Path(tmpfile).exists()
        maskfile = tmpfile + ".mask"
        result = runner.invoke(
            spp_clean.main,
            [filfile_8bit_1, "-o", tmpfile, "-s", maskfile],
        )
        assert result.exit_code == 0
        assert Path(tmpfile).exists()
        assert Path(maskfile).exists()
