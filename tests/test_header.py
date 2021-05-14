import pytest
import numpy as np
from sigpyproc.header import Header


class TestHeader(object):
    @pytest.mark.parametrize(
        "key, newval", [("nchans", 2000), ("fch1", 4000.5), ("source_name", "new_source")]
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
        assert newhdr.refdm == 10
        assert newhdr.nchans == 1

    def test_spp_header(self, filfile_4bit):
        header = Header.from_sigproc(filfile_4bit)
        spphdr = header.spp_header()
        assert len(spphdr) == header.hdrlen
        assert isinstance(spphdr, bytes)

    @pytest.mark.parametrize("dm, maxdelay", [(0, 2000), (100, 4000.5), (5000, 10)])
    def test_get_dmdelays(self, filfile_4bit, dm, maxdelay):
        header = Header.from_sigproc(filfile_4bit)
        delays_time = header.get_dmdelays(dm=dm, in_samples=False)
        delays_samp = header.get_dmdelays(dm=dm, in_samples=True)
        np.testing.assert_allclose(delays_time.max(), maxdelay, atol=0.01)
        np.testing.assert_equal(delays_samp.max(), round(maxdelay / header.tsamp))

    def test_prep_outfile(self, filfile_4bit, tmpfile):
        header = Header.from_sigproc(filfile_4bit)
        with header.prep_outfile(tmpfile) as outfile:
            outfilename = outfile.name
        out_header = Header.from_sigproc(outfilename)
        np.testing.assert_equal(out_header.spp_header(), header.spp_header())

    def test_from_inffile(self, inffile, inf_header):
        infheader = Header.from_inffile(inffile)
        for key, expected_value in inf_header.items():
            assert getattr(infheader, key) == expected_value

    def test_make_inf(self, inffile, inf_header, tmpfile):
        infheader = Header.from_inffile(inffile)
        infheader.make_inf(outfile=tmpfile)
        self.test_from_inffile(tmpfile, inf_header)
