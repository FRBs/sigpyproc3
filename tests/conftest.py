import os
import pytest
import numpy as np

_topdir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
_testdir = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(scope="session", autouse=True)
def filfile():
    return os.path.join(_topdir, "examples/tutorial_2bit.fil")


@pytest.fixture(scope="session", autouse=True)
def filfile_8bit_1():
    return os.path.join(_testdir, "data/parkes_8bit_1.fil")


@pytest.fixture(scope="session", autouse=True)
def filfile_8bit_2():
    return os.path.join(_testdir, "data/parkes_8bit_2.fil")


@pytest.fixture(scope="session", autouse=True)
def timfile():
    return os.path.join(_testdir, "data/Lband_DM0.00.tim")


@pytest.fixture(scope="session", autouse=True)
def datfile():
    return os.path.join(_testdir, "data/Lband_DM0.00.dat")


@pytest.fixture(scope="session", autouse=True)
def fftfile():
    return os.path.join(_testdir, "data/Lband_DM0.00.fft")


@pytest.fixture(scope="session", autouse=True)
def inffile():
    return os.path.join(_testdir, "data/Lband_DM0.00.inf")


@pytest.fixture(scope="session")
def tmpfile(tmp_path_factory, content=""):
    fn = tmp_path_factory.mktemp("pytest_data") / "test.tmpfile"
    fn.write_text(content)
    return fn.as_posix()


@pytest.fixture(scope="class", autouse=True)
def tim_data():
    np.random.seed(5)
    return np.random.normal(128, 20, 10000)


@pytest.fixture(scope="class", autouse=True)
def fourier_data():
    np.random.seed(5)
    data = np.random.normal(128, 20, 10000)
    fft = np.fft.rfft(data)
    return fft.view(np.float64).astype(np.float32)


@pytest.fixture(scope="class", autouse=True)
def tim_header():
    header = {}
    header["telescope_id"] = 10
    header["machine_id"] = 9
    header["source_name"] = "test"
    header["basename"] = "tmp_test"
    header["src_raj"] = 0
    header["src_dej"] = 0
    header["tstart"] = 50000.0
    header["tsamp"] = 1
    header["data_type"] = 2
    header["nchans"] = 1
    header["nbits"] = 32
    header["hdrlen"] = 0
    header["nsamples"] = 0
    return header


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
