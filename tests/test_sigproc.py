from __future__ import annotations

import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from sigpyproc.io import sigproc


class TestSigproc:
    def test_read_string(self, tmpfile: str) -> None:
        header = sigproc.encode_key("HEADER_START")
        with Path(tmpfile).open(mode="wb") as wf:
            wf.write(header)
        with Path(tmpfile).open(mode="rb") as fp:
            key = sigproc._read_string(fp)
        assert key == "HEADER_START"

    @pytest.mark.parametrize(
        ("ra", "dec"),
        [
            ("15h42m30.1234s", "+00d12m00.1234s"),
            ("00h42m30.1234s", "+41d12m00.1234s"),
            ("00h42m30.1234s", "-41d12m00.1234s"),
        ],
    )
    def test_parse_radec(self, ra: str, dec: str) -> None:
        coord = SkyCoord(ra, dec, frame="icrs")
        src_raj = ra.translate({ord(ch): None for ch in "hdms"})
        src_dej = dec.translate({ord(ch): None for ch in "hdms"})
        sig_coord = sigproc.parse_radec(float(src_raj), float(src_dej))
        np.testing.assert_allclose(sig_coord.ra.deg, coord.ra.deg)
        np.testing.assert_allclose(sig_coord.dec.deg, coord.dec.deg)

    def test_encode_key_str(self) -> None:
        key = "testkey"
        keylen_enc = struct.pack("I", len(key)).decode()
        assert sigproc.encode_key("testkey") == f"{keylen_enc}{key}".encode()
        value = "value"
        valuelen_enc = struct.pack("I", len(value)).decode()
        assert (
            sigproc.encode_key("testkey", value=value, value_type="str")
            == f"{keylen_enc}{key}{valuelen_enc}{value}".encode()
        )

    @pytest.mark.parametrize(("value", "val_type"), [(123, "I"), (123.456, "d")])
    def test_encode_key_other(self, value: float, val_type: str) -> None:
        key = "testkey"
        keylen_enc = struct.pack("I", len(key))
        value_enc = struct.pack(val_type, value)
        assert (
            sigproc.encode_key(key, value=value, value_type=val_type)
            == keylen_enc + key.encode() + value_enc
        )

    def test_encode_header(self) -> None:
        pass

    def test_parse_header(
        self,
        filfile_8bit_1: str,
        filfile_8bit_1_header: dict,
    ) -> None:
        header = sigproc.parse_header(filfile_8bit_1)
        for key, expected_value in filfile_8bit_1_header.items():
            assert header[key] == expected_value

    def test_parse_header_invalid(self, tmpfile: str) -> None:
        with Path(tmpfile).open(mode="wb") as wf:
            wf.write(b"HEADER_STARTNOT")
        with pytest.raises(OSError):
            sigproc.parse_header(tmpfile)
        with Path(tmpfile).open(mode="wb") as wf:
            wf.write(b"5")
        with pytest.raises(OSError):
            sigproc.parse_header(tmpfile)

    @pytest.mark.parametrize(
        ("key", "newval"),
        [("nchans", 2000), ("fch1", 4000.5), ("source_name", "new_source")],
    )
    def test_edit_header(
        self,
        filfile_8bit_1: str,
        tmpfile: str,
        key: str,
        newval: float | str,
    ) -> None:
        shutil.copyfile(filfile_8bit_1, tmpfile)
        sigproc.edit_header(tmpfile, key, newval)
        header = sigproc.parse_header(tmpfile)
        assert header[key] == newval

    def test_edit_header_invalid(
        self,
        filfile_8bit_1: str,
        tmpfile: str,
    ) -> None:
        shutil.copyfile(filfile_8bit_1, tmpfile)
        with pytest.raises(ValueError):
            sigproc.edit_header(tmpfile, "invalid", 0)
        with pytest.raises(ValueError):
            sigproc.edit_header(tmpfile, "nchans", None)  # type: ignore [arg-type]

    def test_match_header_pass(self, filfile_8bit_1: str, filfile_8bit_2: str) -> None:
        header1 = sigproc.parse_header(filfile_8bit_1)
        header2 = sigproc.parse_header(filfile_8bit_2)
        try:
            sigproc.match_header(header1, header2)
        except ValueError as msg:
            pytest.fail(str(msg))

    def test_match_header_fail(self, filfile_8bit_1: str, filfile_8bit_2: str) -> None:
        header1 = sigproc.parse_header(filfile_8bit_1)
        header2 = sigproc.parse_header(filfile_8bit_2)
        header2["fch1"] += 1
        with pytest.raises(ValueError):
            sigproc.match_header(header1, header2)

    def test_parse_header_multi(
        self,
        filfile_8bit_1: str,
        filfile_8bit_1_header: dict,
    ) -> None:
        header = sigproc.parse_header_multi(filfile_8bit_1)
        for key, expected_value in filfile_8bit_1_header.items():
            assert header[key] == expected_value

    def test_parse_header_multi_contiguity_pass(
        self,
        filfile_8bit_1: str,
        filfile_8bit_2: str,
    ) -> None:
        try:
            sigproc.parse_header_multi(
                [filfile_8bit_1, filfile_8bit_2],
                check_contiguity=True,
            )
        except ValueError as msg:
            pytest.fail(str(msg))

    def test_parse_header_multi_contiguity_fail(
        self,
        filfile_8bit_1: str,
        filfile_8bit_2: str,
        tmpfile: str,
    ) -> None:
        shutil.copyfile(filfile_8bit_2, tmpfile)
        sigproc.edit_header(tmpfile, "tstart", 60000)
        with pytest.raises(ValueError):
            sigproc.parse_header_multi(
                [filfile_8bit_1, tmpfile],
                check_contiguity=True,
            )


class TestStreamInfo:
    def test_file_info_from_dict(self, filfile_8bit_1: str) -> None:
        header = sigproc.parse_header(filfile_8bit_1)
        file_info = sigproc.FileInfo.from_dict(header)
        assert file_info.hdrlen + file_info.datalen == header["filelen"]

    def test_stream_info_add_entry(self) -> None:
        stream_info = sigproc.StreamInfo()
        file_info = sigproc.FileInfo(
            filename="file.txt",
            hdrlen=100,
            datalen=1000,
            nsamples=500,
            tstart=10.0,
            tsamp=0.001,
        )

        stream_info.add_entry(file_info)
        assert len(stream_info.entries) == 1
        assert stream_info.entries[0] == file_info
        assert stream_info.time_gaps == [0.0]

    def test_stream_info_add_entry_invalid(self) -> None:
        stream_info = sigproc.StreamInfo()
        with pytest.raises(TypeError):
            stream_info.add_entry("invalid")  # type: ignore [arg-type]

    def test_stream_info_check_contiguity_valid(self) -> None:
        tsamp = 0.001
        file_info1 = sigproc.FileInfo(
            filename="file1.txt",
            hdrlen=100,
            datalen=1000,
            nsamples=500,
            tstart=50000.0,
            tsamp=tsamp,
        )

        tstart_valid = file_info1.tstart + (file_info1.nsamples * tsamp) / 86400
        file_info2 = sigproc.FileInfo(
            filename="file2.txt",
            hdrlen=200,
            datalen=2000,
            nsamples=1000,
            tstart=tstart_valid,
            tsamp=tsamp,
        )
        stream_info = sigproc.StreamInfo(entries=[file_info1, file_info2])
        assert stream_info.check_contiguity() is True
        np.testing.assert_array_equal(stream_info.cumsum_datalens, [1000, 3000])

    def test_stream_info_check_contiguity_invalid(self) -> None:
        tsamp = 0.001
        file_info1 = sigproc.FileInfo(
            filename="file1.txt",
            hdrlen=100,
            datalen=1000,
            nsamples=500,
            tstart=50000.0,
            tsamp=0.001,
        )
        tstart_invalid = file_info1.tstart + file_info1.nsamples * tsamp + 0.1
        file_info2 = sigproc.FileInfo(
            filename="file2.txt",
            hdrlen=200,
            datalen=2000,
            nsamples=1000,
            tstart=tstart_invalid,
            tsamp=tsamp,
        )
        stream_info = sigproc.StreamInfo(entries=[file_info1, file_info2])
        assert stream_info.check_contiguity() is False
