import shutil
import struct

import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from sigpyproc.io import sigproc


class TestSigproc(object):
    def test_read_string(self, tmpfile):
        header = sigproc.encode_key("HEADER_START")
        with open(tmpfile, mode="wb") as wf:
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
        sig_coord = sigproc.parse_radec(float(src_raj), float(src_dej))
        coord = SkyCoord(ra, dec, frame="icrs")
        np.testing.assert_allclose(sig_coord.ra.deg, coord.ra.deg)
        np.testing.assert_allclose(sig_coord.dec.deg, coord.dec.deg)

    def test_encode_key_str(self):
        key = "testkey"
        keylen_enc = struct.pack("I", len(key)).decode()
        assert sigproc.encode_key("testkey") == f"{keylen_enc}{key}".encode()
        value = "value"
        valuelen_enc = struct.pack("I", len(value)).decode()
        assert (
            sigproc.encode_key("testkey", value=value, value_type="str")
            == f"{keylen_enc}{key}{valuelen_enc}{value}".encode()
        )

    @pytest.mark.parametrize("value, val_type", [(123, "I"), (123.456, "d")])
    def test_encode_key_other(self, value, val_type):
        key = "testkey"
        keylen_enc = struct.pack("I", len(key))
        value_enc = struct.pack(val_type, value)
        assert (
            sigproc.encode_key(key, value=value, value_type=val_type)
            == keylen_enc + key.encode() + value_enc
        )

    def test_parse_header(self, filfile_8bit_1, filfile_8bit_1_header):
        header = sigproc.parse_header(filfile_8bit_1)
        for key, expected_value in filfile_8bit_1_header.items():
            assert header[key] == expected_value

    @pytest.mark.parametrize(
        "key, newval", [("nchans", 2000), ("fch1", 4000.5), ("source_name", "new_source")]
    )
    def test_edit_header(self, filfile_8bit_1, tmpfile, key, newval):
        shutil.copyfile(filfile_8bit_1, tmpfile)
        sigproc.edit_header(tmpfile, key, newval)
        header = sigproc.parse_header(tmpfile)
        assert header[key] == newval

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
        header2["fch1"] += 1
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


class TestStreamInfo(object):
    def test_file_info_from_dict(self, filfile_8bit_1):
        header = sigproc.parse_header(filfile_8bit_1)
        file_info = sigproc.FileInfo.from_dict(header)
        assert file_info.hdrlen + file_info.datalen == header["filelen"]

    def test_stream_info_add_entry(self):
        stream_info = sigproc.StreamInfo()
        file_info = sigproc.FileInfo(
            filename="file.txt", hdrlen=100, datalen=1000, nsamples=500, tstart=10.0
        )

        stream_info.add_entry(file_info)
        assert len(stream_info.entries) == 1
        assert stream_info.entries[0] == file_info

    def test_stream_info_add_entry_invalid(self) -> None:
        stream_info = sigproc.StreamInfo()
        with pytest.raises(TypeError):
            stream_info.add_entry("invalid") # type: ignore [arg-type]

    def test_stream_info_check_contiguity_valid(self):
        file_info1 = sigproc.FileInfo(
            filename="file1.txt", hdrlen=100, datalen=1000, nsamples=500, tstart=50000.0
        )
        tsamp = 0.001
        tstart_valid = file_info1.tstart + file_info1.nsamples * tsamp / 86400
        file_info2 = sigproc.FileInfo(
            filename="file2.txt",
            hdrlen=200,
            datalen=2000,
            nsamples=1000,
            tstart=tstart_valid,
        )
        stream_info = sigproc.StreamInfo(entries=[file_info1, file_info2])
        assert stream_info.check_contiguity(tsamp) is True

    def test_stream_info_check_contiguity_invalid(self):
        file_info1 = sigproc.FileInfo(
            filename="file1.txt", hdrlen=100, datalen=1000, nsamples=500, tstart=50000.0
        )
        tsamp = 0.001
        tstart_invalid = file_info1.tstart + file_info1.nsamples * tsamp + 0.1
        file_info2 = sigproc.FileInfo(
            filename="file2.txt",
            hdrlen=200,
            datalen=2000,
            nsamples=1000,
            tstart=tstart_invalid,
        )
        stream_info = sigproc.StreamInfo(entries=[file_info1, file_info2])
        assert stream_info.check_contiguity(tsamp) is False
