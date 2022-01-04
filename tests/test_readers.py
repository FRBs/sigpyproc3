import pytest
from sigpyproc.readers import FilReader


class TestReaders(object):
    def test_filterbank_single(self, filfile_4bit):
        fil = FilReader(filfile_4bit)
        assert fil.header.nchans == 832

    def test_filterbank_empty(self):
        with pytest.raises(TypeError):
            FilReader()

    def test_filterbank_otherfile(self, inffile):
        with pytest.raises(OSError):
            FilReader(inffile)
