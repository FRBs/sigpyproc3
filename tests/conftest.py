import pytest
import numpy as np
from pathlib import Path

_testdir = Path(__file__).resolve().parent
_datadir = _testdir / "data"


@pytest.fixture(scope="session")
def tmpfile(tmp_path_factory, content=""):
    fn = tmp_path_factory.mktemp("pytest_data") / "test.tmpfile"
    fn.write_text(content)
    return fn.as_posix()


@pytest.fixture(scope="session", autouse=True)
def filfile_1bit():
    return Path(_datadir / "parkes_1bit.fil").as_posix()


@pytest.fixture(scope="session", autouse=True)
def filfile_2bit():
    return Path(_datadir / "parkes_2bit.fil").as_posix()


@pytest.fixture(scope="session", autouse=True)
def filfile_4bit():
    return Path(_datadir / "parkes_4bit.fil").as_posix()


@pytest.fixture(scope="session", autouse=True)
def filfile_8bit_1():
    return Path(_datadir / "parkes_8bit_1.fil").as_posix()


@pytest.fixture(scope="session", autouse=True)
def filfile_8bit_2():
    return Path(_datadir / "parkes_8bit_2.fil").as_posix()


@pytest.fixture(scope="session", autouse=True)
def timfile():
    return Path(_datadir / "GBT_J1807-0847.tim").as_posix()


@pytest.fixture(scope="session", autouse=True)
def timfile_mean():
    return 100


@pytest.fixture(scope="session", autouse=True)
def timfile_std():
    return 691


@pytest.fixture(scope="session", autouse=True)
def datfile():
    return Path(_datadir / "GBT_J1807-0847.dat").as_posix()


@pytest.fixture(scope="session", autouse=True)
def datfile_mean():
    return 445404


@pytest.fixture(scope="session", autouse=True)
def datfile_std():
    return 3753


@pytest.fixture(scope="session", autouse=True)
def fftfile():
    return Path(_datadir / "GBT_J1807-0847.fft").as_posix()


@pytest.fixture(scope="session", autouse=True)
def fftfile_mean():
    return 443190


@pytest.fixture(scope="session", autouse=True)
def inffile():
    return Path(_datadir / "GBT_J1807-0847.inf").as_posix()


@pytest.fixture(scope="class", autouse=True)
def tim_data():
    np.random.seed(5)
    return np.random.normal(128, 20, 10000)


@pytest.fixture(scope="class", autouse=True)
def tim_header():
    header = {}
    header["rawdatafile"] = "tmp_test.tim"
    header["filename"] = "tmp_test.tim"
    header["data_type"] = 2
    header["nchans"] = 1
    header["foff"] = 1
    header["fch1"] = 2000
    header["nbits"] = 32
    header["tsamp"] = 0.000064
    header["tstart"] = 50000.0
    header["nsamples"] = 10000
    return header


@pytest.fixture(scope="class", autouse=True)
def fourier_data(tim_data):  # noqa: WPS442
    fft = np.fft.rfft(tim_data)
    return fft.view(np.float64).astype(np.float32)


@pytest.fixture(scope="class", autouse=True)
def inf_header():
    header = {}
    header["basename"] = "Lband_DM0.00"
    header["telescope"] = "GBT"
    header["machine"] = "Unknown"
    header["source_name"] = "Mystery_PSR"
    header["ra"] = "16:43:38.1000"
    header["dec"] = "-12:24:58.7000"
    header["tstart"] = 53010.484826388892543
    header["barycentric"] = 0
    header["nsamples"] = 66250
    header["tsamp"] = 0.000072
    header["freq_low"] = 1352.5
    header["bandwidth"] = 96
    header["nchans"] = 96
    header["foff"] = 1
    return header


@pytest.fixture(scope="class", autouse=True)
def filfile_8bit_1_header():
    header = {}
    header["telescope_id"] = 4
    header["machine_id"] = 0
    header["source_name"] = "J0534+2200"
    header["src_raj"] = 53431.9
    header["src_dej"] = 220052.0
    header["tstart"] = 58543.3303690369
    header["tsamp"] = 0.000256
    header["data_type"] = 1
    header["nchans"] = 1664
    header["nbits"] = 8
    header["fch1"] = 4031.0
    header["foff"] = -2.0
    header["nifs"] = 1
    header["nsamples"] = 2048
    return header


@pytest.fixture(scope="class", autouse=True)
def gaus_data():
    return [5, 10, 25, 50, 25, 10, 5]


@pytest.fixture(scope="class", autouse=True)
def gaus_data_snr(gaus_data):  # noqa: WPS442
    return np.array(gaus_data) / np.sqrt(len(gaus_data))


@pytest.fixture(scope="class", autouse=True)
def profile_data(gaus_data):  # noqa: WPS442
    np.random.seed(5)
    data = np.random.normal(0, 1, 1024)
    data[497:504] = gaus_data  # noqa: WPS362
    return data
