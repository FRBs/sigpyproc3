from __future__ import annotations

import numpy as np
import pytest

from sigpyproc.header import Header
from sigpyproc.io import sigproc


class TestHeader:
    @pytest.mark.parametrize(
        ("key", "newval"),
        [("nchans", 2000), ("fch1", 4000.5), ("source", "new_source")],
    )
    def test_new_header_pass(
        self,
        filfile_4bit: str,
        key: str,
        newval: float | str,
    ) -> None:
        header = Header.from_sigproc(filfile_4bit)
        newhdr = header.new_header(update_dict={key: newval})
        np.testing.assert_equal(getattr(newhdr, key), newval)

    def test_new_header_fail(self, filfile_4bit: str) -> None:
        header = Header.from_sigproc(filfile_4bit)
        newhdr = header.new_header(update_dict={"random_key": 0})
        with np.testing.assert_raises(AttributeError):
            assert newhdr.random_key == 0  # type: ignore [attr-defined]

    def test_mjd_after_nsamps(self, filfile_4bit: str) -> None:
        header = Header.from_sigproc(filfile_4bit)
        np.testing.assert_equal(header.mjd_after_nsamps(0), header.tstart)

    @pytest.mark.parametrize(("dm", "maxdelay"), [(0, 0), (100, 0.8), (5000, 40.4)])
    def test_get_dmdelays(self, filfile_4bit: str, dm: float, maxdelay: float) -> None:
        header = Header.from_sigproc(filfile_4bit)
        delays_time = header.get_dmdelays(dm=dm, in_samples=False)
        np.testing.assert_allclose(delays_time.max(), maxdelay, atol=0.1)
        delays_sample = header.get_dmdelays(dm=dm, in_samples=True)
        np.testing.assert_equal(delays_sample.dtype, np.int32)

    def test_get_dmdelays_wrong_ref(self, filfile_4bit: str) -> None:
        dm = 100
        header = Header.from_sigproc(filfile_4bit)
        with np.testing.assert_raises(ValueError):
            header.get_dmdelays(dm=dm, in_samples=False, ref_freq="top")

    def test_new_header(self, filfile_4bit: str) -> None:
        nchans = 15
        header = Header.from_sigproc(filfile_4bit)
        newhdr = header.new_header({"nchans": nchans})
        np.testing.assert_equal(newhdr.nchans, nchans)

    def test_dedispersed_header(self, filfile_4bit: str) -> None:
        dm = 10
        header = Header.from_sigproc(filfile_4bit)
        newhdr = header.dedispersed_header(dm=dm)
        np.testing.assert_equal(newhdr.dm, dm)
        np.testing.assert_equal(newhdr.nchans, 1)

    def test_to_dict(self, filfile_4bit: str) -> None:
        header = Header.from_sigproc(filfile_4bit)
        header_dict = header.to_dict()
        np.testing.assert_equal(header_dict["nchans"], header.nchans)

    def test_to_sigproc(self, filfile_4bit: str) -> None:
        header = Header.from_sigproc(filfile_4bit)
        spphdr = header.to_sigproc()
        np.testing.assert_equal(spphdr["nchans"], header.nchans)
        for key in sigproc.SIGPROC_SCHEMA:
            assert key in spphdr

    def test_to_string(self, filfile_4bit: str) -> None:
        header = Header.from_sigproc(filfile_4bit)
        header_string = header.to_string()
        assert isinstance(header_string, str)

    def test_prep_outfile(self, filfile_4bit: str, tmpfile: str) -> None:
        header = Header.from_sigproc(filfile_4bit)
        with header.prep_outfile(tmpfile) as outfile:
            outfilename = outfile.file_cur
        assert outfilename is not None
        out_header = Header.from_sigproc(outfilename)
        expected_dict = header.to_sigproc()
        actual_dict = out_header.to_sigproc()
        for key, value in expected_dict.items():
            if key == "nsamples":
                assert value != actual_dict[key]
            else:
                assert value == actual_dict[key]

    def test_from_inffile(self, inffile: str, inf_header: dict) -> None:
        infheader = Header.from_inffile(inffile)
        for key, expected_value in inf_header.items():
            assert getattr(infheader, key) == expected_value

    def test_make_inf(self, inffile: str, inf_header: dict, tmpfile: str) -> None:
        infheader = Header.from_inffile(inffile)
        infheader.make_inf(outfile=tmpfile)
        self.test_from_inffile(tmpfile, inf_header)

    def test_from_pfits(self, fitsfile_4bit: str) -> None:
        header = Header.from_pfits(fitsfile_4bit)
        np.testing.assert_equal(header.nchans, 416)
        np.testing.assert_equal(header.nbits, 4)
        np.testing.assert_equal(header.nifs, 1)

    def test_from_pfits_otherfile(self, inffile: str) -> None:
        with np.testing.assert_raises(OSError):
            Header.from_pfits(inffile)
