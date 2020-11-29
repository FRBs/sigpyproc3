import os
import pytest
import numpy as np

_topdir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


@pytest.fixture(scope="session", autouse=True)
def filfile():
    return os.path.join(_topdir, "examples/tutorial_2bit.fil")


@pytest.fixture(scope="session")
def tmpfile(tmp_path_factory, content):
    fn = tmp_path_factory.mktemp("data").join("test.tmpfile")
    with open(fn, 'w') as tmp:
        # do stuff with temp file
        tmp.write(content)
    return fn


@pytest.fixture(scope="class", autouse=True)
def tim_data():
    np.random.seed(5)
    return np.random.normal(128, 20, 10000)


@pytest.fixture(scope="class", autouse=True)
def tim_header():
    header = {}
    header["telescope_id"] = 10
    header["machine_id"]   = 9
    header["source_name"]  = "test"
    header["basename"]     = "tmp_test"
    header["src_raj"]      = 0
    header["src_dej"]      = 0
    header["tstart"]       = 50000.0
    header["tsamp"]        = 1
    header["data_type"]    = 2
    header["nchans"]       = 1
    header["nbits"]        = 32
    header["hdrlen"]       = 0
    header["nsamples"]     = 0
    return header
