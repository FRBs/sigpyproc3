import pytest
from sigpyproc.Readers import FilReader


class TestReaders:
    def test_filterbank_single(self, filfile):
        myFil  = FilReader(filfile)
        header = myFil.header
        assert header.nchans == 64

    def test_filterbank_empty(self):
        with pytest.raises(TypeError):
            FilReader()

    def test_filterbank_otherfile(tmpfile):
        with pytest.raises(TypeError):
            FilReader(tmpfile)
