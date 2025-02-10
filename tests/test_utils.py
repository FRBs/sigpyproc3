import logging
from pathlib import Path

import numpy as np
import pytest
from rich.logging import RichHandler

from sigpyproc import utils


class TestUtils:
    def test_detect_file_type(self, filfile_8bit_1: str, tmpfile: str) -> None:
        assert utils.detect_file_type(filfile_8bit_1) == "sigproc"
        filfile_tmp = utils.validate_path(filfile_8bit_1)
        assert utils.detect_file_type(filfile_tmp.with_suffix(".fits")) == "pfits"
        assert utils.detect_file_type(filfile_tmp.with_suffix(".h5")) == "fbh5"
        with pytest.raises(ValueError):
            utils.detect_file_type(tmpfile)

    def test_nearest_factor(self) -> None:
        np.testing.assert_equal(utils.nearest_factor(10, 0), 1)
        np.testing.assert_equal(utils.nearest_factor(10, 3), 2)
        np.testing.assert_equal(utils.nearest_factor(10, 7), 5)
        np.testing.assert_equal(utils.nearest_factor(10, 11), 10)

    @pytest.mark.parametrize(
        ("inp", "exp"),
        [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (1023, 1024), (1025, 2048)],
    )
    def test_next2_to_n(self, inp: int, exp: int) -> None:
        np.testing.assert_equal(utils.next2_to_n(inp), exp)

    def test_next2_to_n_fail(self) -> None:
        with pytest.raises(ValueError):
            utils.next2_to_n(0)
        with pytest.raises(ValueError):
            utils.next2_to_n(-1)

    def test_duration_string(self) -> None:
        np.testing.assert_equal(utils.duration_string(0), "0.0 seconds")
        np.testing.assert_equal(utils.duration_string(60), "1.0 minutes")
        np.testing.assert_equal(utils.duration_string(3600), "1.0 hours")
        np.testing.assert_equal(utils.duration_string(86400), "1.0 days")

    def test_gaussian(self) -> None:
        x = np.linspace(-5, 5, 101)
        result = utils.gaussian(x, mu=0, fwhm=1, amp=1)
        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)
        max_idx = np.argmax(result)
        np.testing.assert_equal(x[max_idx], 0)
        np.testing.assert_equal(result.shape, x.shape)

    def test_pad_centre(self) -> None:
        array = np.array([1, 2, 3])
        padded_array = utils.pad_centre(array, 7)
        np.testing.assert_equal(padded_array, np.array([0, 0, 1, 2, 3, 0, 0]))
        padded_array = utils.pad_centre(array, 3)
        np.testing.assert_equal(padded_array, array)

        with pytest.raises(ValueError):
            utils.pad_centre(array, 2)

    def test_pad_centre_cases(self) -> None:
        array = np.array([1, 2])
        padded_array = utils.pad_centre(array, 4)
        np.testing.assert_equal(padded_array, np.array([0, 1, 2, 0]))
        padded_array = utils.pad_centre(array, 5)
        np.testing.assert_equal(padded_array, np.array([0, 0, 1, 2, 0]))


class TestPaths:
    def test_basic_file_validation(self, tmpfile: str) -> None:
        with pytest.raises(ValueError):
            utils.validate_path(tmpfile, file_okay=False, dir_okay=False)
        with pytest.raises(NotADirectoryError):
            utils.validate_path(tmpfile, file_okay=False, dir_okay=True)
        assert utils.validate_path(tmpfile) == Path(tmpfile).resolve()

    def test_basic_dir_validation(self, tmpdir: str) -> None:
        with pytest.raises(IsADirectoryError):
            utils.validate_path(tmpdir)
        assert utils.validate_path(tmpdir, dir_okay=True) == Path(tmpdir).resolve()

    def test_permission_validation(self, read_only_file: str) -> None:
        assert utils.validate_path(read_only_file) == Path(read_only_file).resolve()
        with pytest.raises(PermissionError) as exc_info:
            utils.validate_path(read_only_file, writable=True)
        assert "write permission" in str(exc_info.value)

    def test_read_permission(self, tmpfile: str) -> None:
        path = Path(tmpfile)
        # Remove read permissions
        path.chmod(0o222)  # write-only permissions
        with pytest.raises(PermissionError) as exc_info:
            utils.validate_path(path, readable=True, writable=False)
        assert "read permission" in str(exc_info.value)
        path.chmod(0o666)

    def test_non_existent_path(self) -> None:
        with pytest.raises(FileNotFoundError):
            utils.validate_path("non_existent.txt")

    def test_resolve_path_option(self, tmpfile: str) -> None:
        assert utils.validate_path(tmpfile, resolve_path=False) == Path(tmpfile)
        assert (
            utils.validate_path(tmpfile, resolve_path=True) == Path(tmpfile).resolve()
        )


class TestLogger:
    def test_get_logger_default(self) -> None:
        name = "test_logger"
        logger_def = logging.getLogger(name)
        logger_def.propagate = False
        for handler in logger_def.handlers[:]:
            logger_def.removeHandler(handler)
        logger = utils.get_logger(name)
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], RichHandler)

    def test_get_logger_log_file(self, tmpfile: str) -> None:
        name = "test_logger"
        logger_def = logging.getLogger(name)
        logger_def.propagate = False
        for handler in logger_def.handlers[:]:
            logger_def.removeHandler(handler)
        logger = utils.get_logger(name, log_file=tmpfile)
        assert len(logger.handlers) == 2
        assert isinstance(logger.handlers[1], logging.FileHandler)
        assert logger.handlers[1].baseFilename == tmpfile


class TestFrequencyChannels:
    def test_from_sig(self) -> None:
        fch1 = 1500
        foff = -1
        nchans = 1024
        freqs = utils.FrequencyChannels.from_sig(fch1, foff, nchans)
        np.testing.assert_equal(freqs.fch1.value, fch1)
        np.testing.assert_equal(freqs.foff.value, foff)
        np.testing.assert_equal(freqs.nchans, nchans)
        np.testing.assert_equal(freqs.fbottom.value, fch1 + foff * (nchans - 0.5))
        np.testing.assert_equal(len(freqs.array), nchans)

    def test_from_pfits(self) -> None:
        fcenter = 1500
        bandwidth = -1024
        nchans = 1024
        freqs = utils.FrequencyChannels.from_pfits(fcenter, bandwidth, nchans)
        np.testing.assert_equal(freqs.fcenter.value, fcenter)
        np.testing.assert_equal(freqs.bandwidth.value, -bandwidth)
        np.testing.assert_equal(freqs.nchans, nchans)

    def test_fail(self) -> None:
        with np.testing.assert_raises(ValueError):
            utils.FrequencyChannels([])  # type: ignore[arg-type]
        arr = [1, 2, 4, 7]
        with np.testing.assert_raises(ValueError):
            utils.FrequencyChannels(arr)  # type: ignore[arg-type]
