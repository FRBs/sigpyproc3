import os
import numpy as np
from sigpyproc.Readers import FilReader
import sigpyproc.libSigPyProc as lib


class TestFilterbank:
    def test_threads(self, filfile):
        myFil = FilReader(filfile)
        myFil.setNthreads()
        assert lib.getNthreads == 4
        myFil.setNthreads(nthreads=2)
        assert lib.getNthreads == 2

    def test_collapse(self, filfile):
        myFil = FilReader(filfile)
        myTim = myFil.collapse()
        np.testing.assert_equal(myTim.dtype, np.float32)
        np.testing.assert_equal(myTim.size, myFil.header.nsamples)
        np.testing.assert_allclose(myTim.mean(), 104.7, atol=0.1)

    def test_invertFreq(self, filfile):
        myFil = FilReader(filfile)
        data = myFil.readBlock(0, 100)
        outfile = myFil.invertFreq()
        mynewFil = FilReader(outfile)
        newdata = mynewFil.readBlock(0, 100)
        np.testing.assert_equal(mynewFil.header.dtype, myFil.header.dtype)
        np.testing.assert_equal(mynewFil.header.nsamples, myFil.header.nsamples)
        np.testing.assert_equal(mynewFil.header.foff, -1 * myFil.header.foff)
        np.testing.assert_array_equal(data, np.flip(newdata, axis=0))
        os.remove(outfile)

    def test_bandpass(self, filfile):
        myFil = FilReader(filfile)
        myTim = myFil.bandpass()
        np.testing.assert_equal(myTim.dtype, np.float32)
        np.testing.assert_equal(myTim.size, myFil.header.nchans)
        np.testing.assert_allclose(myTim.mean(), 1.64, atol=0.1)
