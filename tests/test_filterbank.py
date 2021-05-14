import numpy as np
from sigpyproc.readers import FilReader


class TestFilterbank(object):
    def test_collapse(self, filfile_4bit):
        fil = FilReader(filfile_4bit)
        tim = fil.collapse()
        np.testing.assert_equal(tim.dtype, np.float32)
        np.testing.assert_equal(tim.size, fil.header.nsamples)
        np.testing.assert_equal(tim.header.nchans, 1)

    def test_bandpass(self, filfile_4bit):
        fil = FilReader(filfile_4bit)
        tim = fil.bandpass()
        np.testing.assert_equal(tim.dtype, np.float32)
        np.testing.assert_equal(tim.size, fil.header.nchans)
        np.testing.assert_equal(tim.header.nchans, 1)

    def test_dedisperse(self, filfile_4bit):
        dm = 100
        fil = FilReader(filfile_4bit)
        tim = fil.dedisperse(100)
        np.testing.assert_equal(tim.dtype, np.float32)
        np.testing.assert_equal(tim.header.nchans, 1)
        np.testing.assert_equal(tim.header.dm, dm)

    def test_get_chan(self, filfile_4bit):
        fil = FilReader(filfile_4bit)
        tim = fil.get_chan(5)
        np.testing.assert_equal(tim.dtype, np.float32)
        np.testing.assert_equal(tim.size, fil.header.nsamples)
        np.testing.assert_equal(tim.header.nchans, 1)

    def test_invert_freq(self, filfile_4bit, tmpfile):
        fil = FilReader(filfile_4bit)
        data = fil.read_block(0, 100)
        outfile = fil.invert_freq(filename=tmpfile)
        new_fil = FilReader(outfile)
        newdata = new_fil.read_block(0, 100)
        np.testing.assert_equal(new_fil.header.dtype, fil.header.dtype)
        np.testing.assert_equal(new_fil.header.nsamples, fil.header.nsamples)
        np.testing.assert_equal(new_fil.header.foff, -1 * fil.header.foff)
        np.testing.assert_array_equal(data, np.flip(newdata, axis=0))
