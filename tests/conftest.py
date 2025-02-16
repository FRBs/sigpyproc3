from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

_testdir = Path(__file__).resolve().parent
_datadir = _testdir / "data"


@pytest.fixture
def tmpfile(
    tmp_path_factory: pytest.TempPathFactory,
    content: str = "",
) -> Generator[str, None, None]:
    temp_dir = tmp_path_factory.mktemp("pytest_data")
    test_file = temp_dir / "test.tmpfile"
    test_file.write_text(content)
    yield test_file.as_posix()
    if test_file.exists():
        test_file.unlink()


@pytest.fixture
def tmpdir(tmp_path_factory: pytest.TempPathFactory) -> Generator[str, None, None]:
    temp_dir = tmp_path_factory.mktemp("pytest_data")
    yield temp_dir.as_posix()
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def read_only_file(tmpfile: str) -> Generator[str, None, None]:
    tmppath = Path(tmpfile)
    tmppath.chmod(0o444)
    yield tmppath.as_posix()
    tmppath.chmod(0o666)


@pytest.fixture(scope="session", autouse=True)
def filfile_1bit() -> str:
    return Path(_datadir / "parkes_1bit.fil").as_posix()


@pytest.fixture(scope="session", autouse=True)
def filfile_2bit() -> str:
    return Path(_datadir / "parkes_2bit.fil").as_posix()


@pytest.fixture(scope="session", autouse=True)
def filfile_4bit() -> str:
    return Path(_datadir / "parkes_4bit.fil").as_posix()


@pytest.fixture(scope="session", autouse=True)
def filfile_8bit_1() -> str:
    return Path(_datadir / "parkes_8bit_1.fil").as_posix()


@pytest.fixture(scope="session", autouse=True)
def filfile_8bit_2() -> str:
    return Path(_datadir / "parkes_8bit_2.fil").as_posix()


@pytest.fixture(scope="session", autouse=True)
def filfiles() -> list:
    return [
        Path(_datadir / "parkes_1bit.fil").as_posix(),
        Path(_datadir / "parkes_2bit.fil").as_posix(),
        Path(_datadir / "parkes_4bit.fil").as_posix(),
        [
            Path(_datadir / "parkes_8bit_1.fil").as_posix(),
            Path(_datadir / "parkes_8bit_2.fil").as_posix(),
        ],
        Path(_datadir / "tutorial.fil").as_posix(),
        Path(_datadir / "tutorial_2bit.fil").as_posix(),
    ]


@pytest.fixture(scope="session", autouse=True)
def fitsfile_4bit() -> str:
    return Path(_datadir / "parkes_4bit.sf").as_posix()


@pytest.fixture(scope="session", autouse=True)
def maskfile() -> str:
    return Path(_datadir / "parkes_8bit_1_mask.h5").as_posix()


@pytest.fixture(scope="session", autouse=True)
def timfile() -> str:
    return Path(_datadir / "GBT_J1807-0847.tim").as_posix()


@pytest.fixture(scope="session", autouse=True)
def timfile_mean() -> int:
    return 100


@pytest.fixture(scope="session", autouse=True)
def timfile_std() -> int:
    return 691


@pytest.fixture(scope="session", autouse=True)
def datfile() -> str:
    return Path(_datadir / "GBT_J1807-0847.dat").as_posix()


@pytest.fixture(scope="session", autouse=True)
def datfile_mean() -> int:
    return 445404


@pytest.fixture(scope="session", autouse=True)
def datfile_std() -> int:
    return 3753


@pytest.fixture(scope="session", autouse=True)
def fftfile() -> str:
    return Path(_datadir / "GBT_J1807-0847.fft").as_posix()


@pytest.fixture(scope="session", autouse=True)
def inffile() -> str:
    return Path(_datadir / "GBT_J1807-0847.inf").as_posix()


@pytest.fixture(scope="class", autouse=True)
def tim_data() -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.normal(128, 20, 10000)


@pytest.fixture(scope="class", autouse=True)
def tim_header() -> dict[str, str | float]:
    header: dict[str, str | float] = {}
    header["rawdatafile"] = "tmp_test.tim"
    header["filename"] = "tmp_test.tim"
    header["data_type"] = "time series"
    header["nchans"] = 1
    header["foff"] = 1
    header["fch1"] = 2000
    header["nbits"] = 32
    header["tsamp"] = 0.000064
    header["tstart"] = 50000.0
    header["nsamples"] = 10000
    return header


@pytest.fixture(scope="class", autouse=True)
def fourier_data(tim_data: np.ndarray) -> np.ndarray:
    fft = np.fft.rfft(tim_data)
    return fft.view(np.float64).astype(np.float32).view(np.complex64)


@pytest.fixture(scope="class", autouse=True)
def inf_header() -> dict[str, str | float]:
    header: dict[str, str | float] = {}
    header["basename"] = "GBT_J1807-0847"
    header["telescope"] = "GBT"
    header["backend"] = "VEGAS"
    header["source"] = "J1807-0847"
    header["ra"] = "18:07:37.9999"
    header["dec"] = "-08:47:43.7463"
    header["tstart"] = 59313.309837974340741
    header["nsamples"] = 131072
    header["tsamp"] = 0.00016384
    # / header["freq_low"] = 720.78125
    header["bandwidth"] = 0.78125
    header["nchans"] = 1
    header["foff"] = 0.78125
    return header


@pytest.fixture(scope="class", autouse=True)
def filfile_8bit_1_header() -> dict[str, str | float]:
    header: dict[str, str | float] = {}
    header["telescope_id"] = 4
    header["machine_id"] = 0
    header["source_name"] = "J0534+2200"
    header["src_raj"] = 53431.9
    header["src_dej"] = 220052.0
    header["tstart"] = 58543.330387241345
    header["tsamp"] = 0.000512
    header["data_type"] = 1
    header["nchans"] = 832
    header["nbits"] = 8
    header["fch1"] = 4030.0
    header["foff"] = -4.0
    header["nifs"] = 1
    header["nsamples"] = 4096
    return header


@pytest.fixture(scope="session", autouse=True)
def random_normal_1d() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=5, scale=2, size=1000).astype(np.float32)


@pytest.fixture(scope="session", autouse=True)
def skewed_normal_1d() -> np.ndarray:
    rng = np.random.default_rng(42)
    skewed = np.concatenate([rng.normal(0, 1, 900), rng.normal(10, 1, 100)])
    return skewed.astype(np.float32)


@pytest.fixture(scope="session", autouse=True)
def random_normal_2d() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=5, scale=2, size=(10, 1000)).astype(np.float32)


@pytest.fixture(scope="session", autouse=True)
def random_normal_1d_complex() -> np.ndarray:
    rng = np.random.default_rng(42)
    re = rng.normal(loc=5, scale=2, size=1000).astype(np.float32)
    im = rng.normal(loc=5, scale=2, size=1000).astype(np.float32)
    return re + 1j * im
