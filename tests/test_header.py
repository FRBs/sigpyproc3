import pytest
import numpy as np
from sigpyproc.header import Header


class TestHeader(object):
    @pytest.mark.parametrize(
        "key, newval", [("nchans", 2000), ("fch1", 4000.5), ("source", "new_source")]
    )
    def test_new_header_pass(self, filfile_4bit, key, newval):
        header = Header.from_sigproc(filfile_4bit)
        newhdr = header.new_header(update_dict={key: newval})
        assert getattr(newhdr, key) == newval

    def test_new_header_fail(self, filfile_4bit):
        header = Header.from_sigproc(filfile_4bit)
        newhdr = header.new_header(update_dict={"random_key": 0})
        with pytest.raises(AttributeError):
            assert newhdr.random_key == 0

    def test_dedispersed_header(self, filfile_4bit):
        header = Header.from_sigproc(filfile_4bit)
        newhdr = header.dedispersed_header(dm=10)
        assert newhdr.dm == 10
        assert newhdr.nchans == 1

    def test_to_sigproc(self, filfile_4bit):
        header = Header.from_sigproc(filfile_4bit)
        spphdr = header.to_sigproc()
        assert len(spphdr) >= header.hdrlens[0]
        assert isinstance(spphdr, bytes)

    @pytest.mark.parametrize("dm, maxdelay", [(0, 0), (100, 0.8), (5000, 40.4)])
    def test_get_dmdelays(self, filfile_4bit, dm, maxdelay):
        header = Header.from_sigproc(filfile_4bit)
        delays_time = header.get_dmdelays(dm=dm, in_samples=False)
        np.testing.assert_allclose(delays_time.max(), maxdelay, atol=0.1)

    def test_prep_outfile(self, filfile_4bit, tmpfile):
        header = Header.from_sigproc(filfile_4bit)
        with header.prep_outfile(tmpfile) as outfile:
            outfilename = outfile.name
        out_header = Header.from_sigproc(outfilename)
        np.testing.assert_equal(out_header.to_sigproc(), header.to_sigproc())

    def test_from_inffile(self, inffile, inf_header):
        infheader = Header.from_inffile(inffile)
        for key, expected_value in inf_header.items():
            assert getattr(infheader, key) == expected_value

    def test_make_inf(self, inffile, inf_header, tmpfile):
        infheader = Header.from_inffile(inffile)
        infheader.make_inf(outfile=tmpfile)
        self.test_from_inffile(tmpfile, inf_header)
