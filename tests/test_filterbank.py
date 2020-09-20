from sigpyproc.Readers import FilReader
import os

_topdir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


def test_filterbank_headers():
    filfile = os.path.join(_topdir, "examples/tutorial.fil")
    myFil = FilReader(filfile)
    header = myFil.header
    assert header.nchans == 64
