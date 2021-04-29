import pytest
import struct
import shutil
import numpy as np
from astropy.coordinates import SkyCoord

from sigpyproc import Header


class TestHeader(object):
    @pytest.mark.parametrize(
        "key, newval", [("nchans", 2000), ("fch1", 4000.5), ("source_name", "new_source")]
    )
    def test_new_header_pass(self, filfile_4bit, key, newval):
        header = Header.Header.from_sigproc(filfile_4bit)
        newhdr = header.new_header(update_dict={key: newval})
        assert getattr(newhdr, key) == newval

    def test_new_header_fail(self, filfile_4bit):
        header = Header.Header.from_sigproc(filfile_4bit)
        newhdr = header.new_header(update_dict={"random_key": 0})
        with pytest.raises(AttributeError):
            assert newhdr.random_key == 0

    def test_dedispersed_header(self, filfile_4bit):
        header = Header.Header.from_sigproc(filfile_4bit)
        newhdr = header.dedispersed_header(dm=10)
        assert newhdr.refdm == 10
        assert newhdr.nchans == 1

    def test_spp_header(self, filfile_4bit):
        header = Header.Header.from_sigproc(filfile_4bit)
        spphdr = header.spp_header()
        assert len(spphdr) == header.hdrlen
        assert isinstance(spphdr, bytes)

    @pytest.mark.parametrize("dm, maxdelay", [(0, 2000), (100, 4000.5), (5000, 10)])
    def test_get_dmdelays(self, filfile_4bit, dm, maxdelay):
        header = Header.Header.from_sigproc(filfile_4bit)
        delays_time = header.get_dmdelays(dm=dm, in_samples=False)
        delays_samp = header.get_dmdelays(dm=dm, in_samples=True)
        np.testing.assert_allclose(delays_time.max(), maxdelay, atol=0.01)
        np.testing.assert_equal(delays_samp.max(), round(maxdelay / header.tsamp))

    def test_prep_outfile(self, filfile_4bit, tmpfile):
        header = Header.Header.from_sigproc(filfile_4bit)
        with header.prep_outfile(tmpfile) as outfile:
            outfilename = outfile.name
        out_header = Header.Header.from_sigproc(outfilename)
        np.testing.assert_equal(out_header.spp_header(), header.spp_header())

    def test_from_inffile(self, inffile, inf_header):
        infheader = Header.Header.from_inffile(inffile)
        for key, expected_value in inf_header.items():
            assert getattr(infheader, key) == expected_value

    def test_make_inf(self, inffile, inf_header, tmpfile):
        infheader = Header.Header.from_inffile(inffile)
        infheader.make_inf(outfile=tmpfile)
        self.test_from_inffile(tmpfile, inf_header)


class TestHeaderHelpers(object):
    @pytest.mark.parametrize(
        "ra, dec",
        [
            ("15h42m30.1234s", "+00d12m00.1234s"),
            ("00h42m30.1234s", "+41d12m00.1234s"),
            ("00h42m30.1234s", "-41d12m00.1234s"),
        ],
    )
    def test_parse_radec(self, ra, dec):
        src_raj = ra.translate({ord(ch): None for ch in "hdms"})
        src_dej = dec.translate({ord(ch): None for ch in "hdms"})
        assert Header.parse_radec(float(src_raj), float(src_dej)) == SkyCoord(
            ra, dec, frame="icrs"
        )

    def test_encode_header_str(self):
        key = "testkey"
        keylen_enc = struct.pack("I", len(key)).decode()
        assert Header.encode_header("testkey") == f"{keylen_enc}{key}".encode()
        value = "value"
        valuelen_enc = struct.pack("I", len(key)).decode()
        assert (
            Header.encode_header("testkey", value=value, value_type="str")
            == f"{keylen_enc}{key}{valuelen_enc}{value}".encode()
        )

    @pytest.mark.parametrize("value, val_type", [(123, "I"), (123.456, "d")])
    def test_encode_header_other(self, value, val_type):
        key = "testkey"
        keylen_enc = struct.pack("I", len(key)).decode()
        value_enc = struct.pack(val_type, value).decode()
        assert (
            Header.encode_header(key, value=value, value_type=val_type)
            == f"{keylen_enc}{key}{value_enc}".encode()
        )

    def test_read_string(self, tmpfile):
        header = Header.encode_header("HEADER_START")
        with open(tmpfile, mode="w") as wf:
            wf.write(header)
        with open(tmpfile, mode="rb") as fp:
            key = Header._read_string(fp)
        assert key == "HEADER_START"

    def test_parse_sigproc_header(self, filfile_8bit_1, filfile_8bit_1_header):
        header = Header.parse_sigproc_header(filfile_8bit_1)
        for key, expected_value in filfile_8bit_1_header.items():
            assert getattr(header, key) == expected_value

    @pytest.mark.parametrize(
        "key, newval", [("nchans", 2000), ("fch1", 4000.5), ("source_name", "new_source")]
    )
    def test_edit_header(self, filfile_8bit_1, tmpfile, key, newval):
        shutil.copyfile(filfile_8bit_1, tmpfile)
        Header.edit_header(tmpfile, key, newval)
        header = Header.parse_sigproc_header(tmpfile)
        assert getattr(header, key) == newval

    def test_match_header_pass(self, filfile_8bit_1, filfile_8bit_2):
        header1 = Header.parse_sigproc_header(filfile_8bit_1)
        header2 = Header.parse_sigproc_header(filfile_8bit_2)
        try:
            Header.match_header(header1, header2)
        except ValueError as msg:
            pytest.fail(msg)

    def test_match_header_fail(self, filfile_8bit_1, filfile_8bit_2):
        header1 = Header.parse_sigproc_header(filfile_8bit_1)
        header2 = Header.parse_sigproc_header(filfile_8bit_2)
        header2.fch1 += 1
        with pytest.raises(ValueError):
            Header.match_header(header1, header2)

    def test_ensure_contiguity_pass(self, filfile_8bit_1, filfile_8bit_2):
        try:
            Header.Header.from_sigproc(
                [filfile_8bit_1, filfile_8bit_2], check_contiguity=True
            )
        except ValueError as msg:
            pytest.fail(msg)

    def test_ensure_contiguity_fail(self, filfile_8bit_1, filfile_4bit):
        with pytest.raises(ValueError):
            Header.Header.from_sigproc(
                [filfile_8bit_1, filfile_4bit], check_contiguity=True
            )
