import numpy as np
from sigpyproc.Header import Header


class TestHeader(object):
    def test_getDMdelays(self, timfile):
        header = Header.parseSigprocHeader(filename=timfile)
        delays_time = header.getDMdelays(dm=100, in_samples=False)
        delays_samp = header.getDMdelays(dm=100, in_samples=True)
        np.testing.assert_equal(delays_samp.mean(), 193.3125)
        np.testing.assert_allclose(delays_time.mean(), 0.013913)

    def test_prepOutfile(self, filfile, tmpfile):
        myheader = Header.parseSigprocHeader(filename=filfile)
        outfile = myheader.prepOutfile(tmpfile)
        out_header = Header.parseSigprocHeader(filename=outfile.name)
        np.testing.assert_equal(out_header, myheader)

    def test_makeInf(self, inffile, tmpfile):
        myheader = Header.parseInfHeader(filename=inffile)
        myheader.makeInf(outfile=tmpfile)
        with open(inffile, 'r') as file:
            infdata = file.read()
        with open(tmpfile, 'r') as file:
            tmpdata = file.read()
        np.testing.assert_string_equal(tmpdata, infdata)

    def test_parseInfHeader(self, inffile):
        header = Header.parseInfHeader(filename=inffile)
        assert header.nbits == 32
        assert header.source_name == "Mystery_PSR"
        assert header.telescope_id == 6
        assert header.machine_id == 9
        assert header.src_raj == 164338.1
        assert header.src_dej == -122458.7
        assert header.nsamples == 66250
        assert header.nchans == 1

    def test_parseSigprocHeader(self, timfile):
        header = Header.parseSigprocHeader(filename=timfile)
        assert header.nbits == 32
        assert header.source_name == "Mystery_PSR"
        assert header.telescope_id == 6
        assert header.machine_id == 9
        assert header.ra == "16:43:38.1000"
        assert header.dec == "-12:24:58.7000"
        assert header.nsamples == 66250
        assert header.nchans == 1
