import pytest
import struct
import shutil
from astropy.coordinates import SkyCoord

from sigpyproc.io import sigproc


class TestSigprocHeader(object):
    def test_read_string(self, tmpfile):
        header = sigproc.encode_header("HEADER_START")
        with open(tmpfile, mode="w") as wf:
            wf.write(header)
        with open(tmpfile, mode="rb") as fp:
            key = sigproc._read_string(fp)  # noqa: WPS437
        assert key == "HEADER_START"

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
        assert sigproc.parse_radec(float(src_raj), float(src_dej)) == SkyCoord(
            ra, dec, frame="icrs"
        )

    def test_encode_key_str(self):
        key = "testkey"
        keylen_enc = struct.pack("I", len(key)).decode()
        assert sigproc.encode_key("testkey") == f"{keylen_enc}{key}".encode()
        value = "value"
        valuelen_enc = struct.pack("I", len(key)).decode()
        assert (
            sigproc.encode_key("testkey", value=value, value_type="str")
            == f"{keylen_enc}{key}{valuelen_enc}{value}".encode()
        )

    @pytest.mark.parametrize("value, val_type", [(123, "I"), (123.456, "d")])
    def test_encode_key_other(self, value, val_type):
        key = "testkey"
        keylen_enc = struct.pack("I", len(key)).decode()
        value_enc = struct.pack(val_type, value).decode()
        assert (
            sigproc.encode_key(key, value=value, value_type=val_type)
            == f"{keylen_enc}{key}{value_enc}".encode()
        )

    def test_parse_header(self, filfile_8bit_1, filfile_8bit_1_header):
        header = sigproc.parse_header(filfile_8bit_1)
        for key, expected_value in filfile_8bit_1_header.items():
            assert getattr(header, key) == expected_value

    @pytest.mark.parametrize(
        "key, newval", [("nchans", 2000), ("fch1", 4000.5), ("source_name", "new_source")]
    )
    def test_edit_header(self, filfile_8bit_1, tmpfile, key, newval):
        shutil.copyfile(filfile_8bit_1, tmpfile)
        sigproc.edit_header(tmpfile, key, newval)
        header = sigproc.parse_header(tmpfile)
        assert getattr(header, key) == newval

    def test_match_header_pass(self, filfile_8bit_1, filfile_8bit_2):
        header1 = sigproc.parse_header(filfile_8bit_1)
        header2 = sigproc.parse_header(filfile_8bit_2)
        try:
            sigproc.match_header(header1, header2)
        except ValueError as msg:
            pytest.fail(msg)

    def test_match_header_fail(self, filfile_8bit_1, filfile_8bit_2):
        header1 = sigproc.parse_header(filfile_8bit_1)
        header2 = sigproc.parse_header(filfile_8bit_2)
        header2.fch1 += 1
        with pytest.raises(ValueError):
            sigproc.match_header(header1, header2)

    def test_ensure_contiguity_pass(self, filfile_8bit_1, filfile_8bit_2):
        try:
            sigproc.parse_header_multi(
                [filfile_8bit_1, filfile_8bit_2], check_contiguity=True
            )
        except ValueError as msg:
            pytest.fail(msg)

    def test_ensure_contiguity_fail(self, filfile_8bit_1, filfile_4bit):
        with pytest.raises(ValueError):
            sigproc.parse_header_multi(
                [filfile_8bit_1, filfile_4bit], check_contiguity=True
            )
