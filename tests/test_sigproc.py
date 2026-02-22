from __future__ import annotations

import logging
import shutil
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from sigpyproc.io import sigproc


class SigprocBuilder:
    """sigproc-format byte streams for testing."""

    _UINT = struct.Struct("<I")
    _DOUBLE = struct.Struct("<d")
    _BYTE = struct.Struct("b")

    def __init__(self) -> None:
        self._buf = bytearray()

    def _write_string(self, s: str) -> SigprocBuilder:
        encoded = s.encode("utf-8")
        self._buf += self._UINT.pack(len(encoded)) + encoded
        return self

    def _write_uint(self, v: int) -> SigprocBuilder:
        self._buf += self._UINT.pack(v)
        return self

    def _write_double(self, v: float) -> SigprocBuilder:
        self._buf += self._DOUBLE.pack(v)
        return self

    def _write_byte(self, v: int) -> SigprocBuilder:
        self._buf += self._BYTE.pack(v)
        return self

    def header_start(self) -> SigprocBuilder:
        return self._write_string("HEADER_START")

    def header_end(self) -> SigprocBuilder:
        return self._write_string("HEADER_END")

    def uint_field(self, key: str, value: int) -> SigprocBuilder:
        return self._write_string(key)._write_uint(value)

    def double_field(self, key: str, value: float) -> SigprocBuilder:
        return self._write_string(key)._write_double(value)

    def byte_field(self, key: str, value: int) -> SigprocBuilder:
        return self._write_string(key)._write_byte(value)

    def string_field(self, key: str, value: str) -> SigprocBuilder:
        return self._write_string(key)._write_string(value)

    def raw(self, data: bytes) -> SigprocBuilder:
        """Append raw bytes (for data block or deliberate corruption)."""
        self._buf += data
        return self

    def build(self) -> bytes:
        return bytes(self._buf)


@pytest.fixture
def builder() -> SigprocBuilder:
    return SigprocBuilder()


def _minimal_payload(
    b: SigprocBuilder,
    *,
    nchans: int = 4,
    nbits: int = 8,
    nsamples: int | None = None,
    data: bytes | None = None,
) -> SigprocBuilder:
    """Write a minimal payload for reuse across tests."""
    base = (
        b.header_start()
        .string_field("source_name", "test_src")
        .uint_field("nchans", nchans)
        .uint_field("nbits", nbits)
    )
    if nsamples is not None:
        base.uint_field("nsamples", nsamples)
    base = (
        base.double_field("tsamp", 6.4e-5)
        .double_field("fch1", 1500.0)
        .double_field("foff", -0.1)
        .double_field("tstart", 60000.0)
        .header_end()
    )
    if data is not None:
        return base.raw(data)
    return base


class TestReadString:
    def test_roundtrip(self, tmpfile: str) -> None:
        f = Path(tmpfile)
        f.write_bytes(sigproc._encode_string("HEADER_START"))
        with f.open("rb") as fp:
            assert sigproc._read_string(fp) == "HEADER_START"

    def test_empty_file_raises(self, tmpfile: str) -> None:
        f = Path(tmpfile)
        f.write_bytes(b"")
        with (
            f.open("rb") as fp,
            pytest.raises(struct.error, match="EOF while reading string length"),
        ):
            sigproc._read_string(fp)

    def test_truncated_body_raises(self, tmpfile: str) -> None:
        # Write length=10 but only 3 bytes of body
        f = Path(tmpfile)
        f.write_bytes(struct.pack("<I", 10) + b"abc")
        with (
            f.open("rb") as fp,
            pytest.raises(struct.error, match="EOF while reading string body"),
        ):
            sigproc._read_string(fp)

    def test_implausible_length_raises(self, tmpfile: str) -> None:
        f = Path(tmpfile)
        f.write_bytes(struct.pack("<I", 99_999))
        with (
            f.open("rb") as fp,
            pytest.raises(ValueError, match="Implausible string length"),
        ):
            sigproc._read_string(fp)

    def test_zero_length_raises(self, tmpfile: str) -> None:
        # strlen=0 is outside the valid 1..4096 range
        f = Path(tmpfile)
        f.write_bytes(struct.pack("<I", 0))
        with (
            f.open("rb") as fp,
            pytest.raises(ValueError, match="Implausible string length"),
        ):
            sigproc._read_string(fp)


class TestEncodeKey:
    def test_known_string_field(self) -> None:
        result = sigproc.encode_key("source_name", value="CrabPulsar")
        # Reconstruct expected manually
        key_enc = struct.pack("<I", len("source_name")) + b"source_name"
        val_enc = struct.pack("<I", len("CrabPulsar")) + b"CrabPulsar"
        assert result == key_enc + val_enc

    def test_known_uint_field(self) -> None:
        result = sigproc.encode_key("nchans", value=1024)
        key_enc = struct.pack("<I", len("nchans")) + b"nchans"
        val_enc = struct.pack("<I", 1024)
        assert result == key_enc + val_enc

    def test_known_double_field(self) -> None:
        result = sigproc.encode_key("tsamp", value=6.4e-5)
        key_enc = struct.pack("<I", len("tsamp")) + b"tsamp"
        val_enc = struct.pack("<d", 6.4e-5)
        assert result == key_enc + val_enc

    def test_known_byte_field(self) -> None:
        result = sigproc.encode_key("signed", value=1)
        key_enc = struct.pack("<I", len("signed")) + b"signed"
        val_enc = struct.pack("b", 1)
        assert result == key_enc + val_enc

    def test_key_only_no_value(self) -> None:
        # encode_key with no value should only encode the key string
        result = sigproc.encode_key("HEADER_START")
        expected = struct.pack("<I", len("HEADER_START")) + b"HEADER_START"
        assert result == expected

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown SIGPROC header key"):
            sigproc.encode_key("bogus_key", value=42)

    def test_wrong_type_for_string_field_raises(self) -> None:
        with pytest.raises(TypeError, match="expects a string value"):
            sigproc.encode_key("source_name", value=42)


class TestEncodeHeader:
    def test_roundtrip(self, filfile_8bit_1_header: dict, tmpfile: str) -> None:
        """Encode → write → parse_header should recover the same values."""
        bytes_data = (
            filfile_8bit_1_header["nsamples"]
            * filfile_8bit_1_header["nchans"]
            * (filfile_8bit_1_header["nbits"] // 8)
        )
        payload = sigproc.encode_header(filfile_8bit_1_header) + bytes(bytes_data)
        f = Path(tmpfile)
        f.write_bytes(payload)
        parsed = sigproc.parse_header(f)
        for key, val in filfile_8bit_1_header.items():
            if isinstance(val, float):
                assert parsed[key] == pytest.approx(val), (
                    f"Float mismatch on key {key!r}"
                )
            else:
                assert parsed[key] == val, f"Value mismatch on key {key!r}"

    def test_deterministic_ordering(self, filfile_8bit_1_header: dict) -> None:
        """Encoding the same header twice must produce identical bytes."""
        assert sigproc.encode_header(filfile_8bit_1_header) == sigproc.encode_header(
            filfile_8bit_1_header
        )

    def test_nsamples_excluded_by_default(self, filfile_8bit_1_header: dict) -> None:
        h = {**filfile_8bit_1_header, "nsamples": 1000}
        encoded = sigproc.encode_header(h)
        # nsamples should not appear in the output unless explicitly allowed
        assert b"nsamples" not in encoded

    def test_nsamples_included_when_allowed(self, filfile_8bit_1_header: dict) -> None:
        h = {**filfile_8bit_1_header, "nsamples": 1000}
        encoded = sigproc.encode_header(h, allow_nsamples_overwrite=True)
        assert b"nsamples" in encoded

    def test_unknown_keys_in_dict_are_silently_ignored(
        self, filfile_8bit_1_header: dict
    ) -> None:
        h = {**filfile_8bit_1_header, "my_custom_key": 42}
        # Should not raise — unknown keys are simply skipped
        encoded = sigproc.encode_header(h)
        assert b"my_custom_key" not in encoded

    def test_starts_with_header_start(self, filfile_8bit_1_header: dict) -> None:
        encoded = sigproc.encode_header(filfile_8bit_1_header)
        expected_start = struct.pack("<I", len("HEADER_START")) + b"HEADER_START"
        assert encoded.startswith(expected_start)

    def test_ends_with_header_end(self, filfile_8bit_1_header: dict) -> None:
        encoded = sigproc.encode_header(filfile_8bit_1_header)
        expected_end = struct.pack("<I", len("HEADER_END")) + b"HEADER_END"
        assert encoded.endswith(expected_end)


class TestParseHeaderValid:
    def test_computed_fields_present(
        self, filfile_8bit_1: str, filfile_8bit_1_header: dict
    ) -> None:
        h = sigproc.parse_header(filfile_8bit_1)
        assert "hdrlen" in h
        assert "filelen" in h
        assert "datalen" in h
        assert "nsamples" in h
        assert "filename" in h
        for key, expected_value in filfile_8bit_1_header.items():
            assert h[key] == expected_value

    def test_filename_is_absolute_posix(self, filfile_8bit_1: str) -> None:
        h = sigproc.parse_header(filfile_8bit_1)
        assert h["filename"] == Path(filfile_8bit_1).resolve().as_posix()

    def test_accepts_path_and_str(self, filfile_8bit_1: Path) -> None:
        h_path = sigproc.parse_header(filfile_8bit_1)
        h_str = sigproc.parse_header(str(filfile_8bit_1))
        assert h_path == h_str

    def test_nsamples_in_header_agrees(
        self, tmpfile: str, builder: SigprocBuilder
    ) -> None:
        """When header nsamples matches file, no warning, value preserved."""
        nchans, nbits, nsamples = 4, 8, 50
        data = bytes(nchans * nsamples)
        payload = (
            builder.header_start()
            .string_field("source_name", "src")
            .uint_field("nchans", nchans)
            .uint_field("nbits", nbits)
            .uint_field("nsamples", nsamples)  # declared == actual
            .double_field("tsamp", 6.4e-5)
            .double_field("fch1", 1500.0)
            .double_field("foff", -0.1)
            .double_field("tstart", 60000.0)
            .header_end()
            .raw(data)
            .build()
        )
        f = Path(tmpfile)
        f.write_bytes(payload)
        h = sigproc.parse_header(f)
        assert h["nsamples"] == nsamples

    def test_nsamples_in_header_too_large_clamped(
        self, tmpfile: str, builder: SigprocBuilder
    ) -> None:
        """Header declares more samples than exist on disk → clamp to file size."""
        nchans, nbits, actual = 4, 8, 50
        data = bytes(nchans * actual)
        payload = _minimal_payload(
            builder,
            nchans=nchans,
            nbits=nbits,
            nsamples=actual + 1000,  # declared > actual
            data=data,
        ).build()
        f = Path(tmpfile)
        f.write_bytes(payload)
        h = sigproc.parse_header(f)
        assert h["nsamples"] == actual  # clamped to what's actually there

    def test_nsamples_in_header_too_small_used(
        self, tmpfile: str, builder: SigprocBuilder
    ) -> None:
        """Header declares fewer samples than disk — trust the header (min wins)."""
        nchans, nbits, declared = 4, 8, 30
        data = bytes(nchans * 50)  # 50 samples on disk
        payload = _minimal_payload(
            builder,
            nchans=nchans,
            nbits=nbits,
            nsamples=declared,
            data=data,
        ).build()
        f = Path(tmpfile)
        f.write_bytes(payload)
        h = sigproc.parse_header(f)
        assert h["nsamples"] == declared

    def test_unknown_key_skipped(self, tmpfile: str, builder: SigprocBuilder) -> None:
        """Unknown keys with uint-sized values should be skipped gracefully."""
        nchans, nbits, nsamples = 4, 8, 20
        data = bytes(nchans * nsamples)
        # Inject an unknown uint key between known keys
        unknown_key = struct.pack("<I", len("mystery")) + b"mystery"
        unknown_val = struct.pack("<I", 999)

        base = _minimal_payload(builder, nchans=nchans, nbits=nbits).build()
        # Splice unknown key+value right before HEADER_END
        header_end_marker = struct.pack("<I", len("HEADER_END")) + b"HEADER_END"
        split = base.rfind(header_end_marker)
        crafted = base[:split] + unknown_key + unknown_val + base[split:]

        f = Path(tmpfile)
        f.write_bytes(crafted + data)
        h = sigproc.parse_header(f)
        # Must still parse correctly and not contain the unknown key
        assert "mystery" not in h
        assert h["nchans"] == nchans

    def test_empty_data_block_warns(
        self, tmpfile: str, builder: SigprocBuilder, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Header-only file with no data block should log a warning, not crash."""
        payload = _minimal_payload(builder).build()  # no data appended
        f = Path(tmpfile)
        f.write_bytes(payload)
        with caplog.at_level(logging.WARNING):
            h = sigproc.parse_header(f)
        assert h["nsamples"] == 0
        assert any("nsamples is zero" in r.message for r in caplog.records)


class TestParseHeaderInvalid:
    def test_empty_file(self, tmpfile: str) -> None:
        f = Path(tmpfile)
        f.write_bytes(b"")
        with pytest.raises(OSError, match="sigproc format"):
            sigproc.parse_header(f)

    def test_wrong_sentinel(self, tmpfile: str) -> None:
        f = Path(tmpfile)
        f.write_bytes(struct.pack("<I", 12) + b"HEADER_NOPE!")
        with pytest.raises(OSError, match="not in sigproc format"):
            sigproc.parse_header(f)

    def test_file_not_found(self, tmpfile: str) -> None:
        f = Path(tmpfile)
        with pytest.raises(OSError):
            sigproc.parse_header(f)

    def test_truncated_mid_key(self, tmpfile: str, builder: SigprocBuilder) -> None:
        """File ends abruptly inside a key string."""
        payload = builder.header_start().build()
        # Append a partial key: length says 6 bytes but only 3 follow
        payload += struct.pack("<I", 6) + b"abc"
        f = Path(tmpfile)
        f.write_bytes(payload)
        with pytest.raises(OSError, match="truncated"):
            sigproc.parse_header(f)

    def test_truncated_mid_value(self, tmpfile: str, builder: SigprocBuilder) -> None:
        """File ends abruptly inside a known key's value."""
        payload = builder.header_start().build()
        payload += struct.pack("<I", len("nchans")) + b"nchans"
        payload += b"\x00\x00"  # only 2 bytes of a 4-byte uint
        f = Path(tmpfile)
        f.write_bytes(payload)
        with pytest.raises(OSError, match="truncated"):
            sigproc.parse_header(f)

    def test_missing_required_keys_raises(
        self, tmpfile: str, builder: SigprocBuilder
    ) -> None:
        """Header missing nbits/nchans should raise ValueError."""
        f = Path(tmpfile)
        payload = (
            builder.header_start()
            .string_field("source_name", "src")
            # nbits and nchans deliberately omitted
            .double_field("tsamp", 6.4e-5)
            .header_end()
            .build()
        )
        f.write_bytes(payload)
        with pytest.raises(ValueError, match="missing required header keys"):
            sigproc.parse_header(f)

    def test_zero_nbits_raises(self, tmpfile: str, builder: SigprocBuilder) -> None:
        payload = (
            builder.header_start()
            .uint_field("nchans", 4)
            .uint_field("nbits", 0)  # invalid
            .double_field("tsamp", 6.4e-5)
            .header_end()
            .build()
        )
        f = Path(tmpfile)
        f.write_bytes(payload)
        with pytest.raises(ValueError, match="nbits=0"):
            sigproc.parse_header(f)

    def test_raises_on_completely_unresolvable_key(
        self, tmpfile: str, builder: SigprocBuilder
    ) -> None:
        """No candidate size produces a valid next key — must raise OSError."""
        nchans, nbits = 4, 8
        base = _minimal_payload(builder, nchans=nchans, nbits=nbits).build()
        # Inject: unknown key name + pure garbage bytes that cannot be
        # interpreted as any valid next key at any candidate offset.
        unknown_key = struct.pack("<I", len("mystery")) + b"mystery"
        garbage = b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"

        header_end = struct.pack("<I", len("HEADER_END")) + b"HEADER_END"
        split = base.rfind(header_end)
        crafted = base[:split] + unknown_key + garbage
        # Deliberately omit HEADER_END so every probe fails to find a valid next key

        f = Path(tmpfile)
        f.write_bytes(crafted)
        with pytest.raises(OSError, match="Cannot determine size"):
            sigproc.parse_header(f)

    def test_eof_during_skip_tries_smaller(
        self, tmpfile: str, builder: SigprocBuilder
    ) -> None:
        """EOF mid-skip causes the loop to try the next candidate size."""
        # Inject unknown key with only 2 bytes of value (less than any candidate),
        # followed by HEADER_END — the skip loop will exhaust all candidates and raise.
        base = _minimal_payload(builder, nchans=4, nbits=8).build()
        unknown_key = struct.pack("<I", len("mystery")) + b"mystery"
        # 2 bytes — smaller than _BYTE (1B candidate will succeed reading 1 byte,
        # but then the remaining 1 byte won't parse as a valid key string)
        partial_value = b"\x00\x00"
        header_end = struct.pack("<I", len("HEADER_END")) + b"HEADER_END"
        split = base.rfind(header_end)
        crafted = base[:split] + unknown_key + partial_value
        # No HEADER_END after garbage — ensures failure

        f = Path(tmpfile)
        f.write_bytes(crafted)
        with pytest.raises(OSError):
            sigproc.parse_header(f)

    def test_negative_datalen_raises(
        self, tmpfile: str, monkeypatch: pytest.MonkeyPatch, builder: SigprocBuilder
    ) -> None:
        """Make datalen < 0 branch: mock filelen to be smaller than hdrlen."""
        import sigpyproc.io.sigproc as _mod  # noqa: PLC0415

        nchans, nbits = 4, 8
        payload = _minimal_payload(builder, nchans=nchans, nbits=nbits).build()
        f = Path(tmpfile)
        f.write_bytes(payload)  # header only, no data — datalen == 0, not negative

        # Force filelen to appear smaller than hdrlen by patching fp.seek/tell.
        # Simplest alternative: patch _read_sigproc_header to return a hdrlen
        # larger than the actual file size.
        original_read = _mod._read_sigproc_header

        def _patched_read(fp: BinaryIO, filepath: Path) -> dict:
            result = original_read(fp, filepath)
            result["hdrlen"] = result["hdrlen"] + 999  # larger than file
            return result

        monkeypatch.setattr(_mod, "_read_sigproc_header", _patched_read)
        with pytest.raises(OSError, match="smaller than header"):
            sigproc.parse_header(f)


class TestNsamplesInBinaryHeader:
    def test_returns_true_when_present(
        self, tmpfile: str, builder: SigprocBuilder
    ) -> None:
        nchans, nbits, nsamples = 4, 8, 50
        payload = _minimal_payload(
            builder,
            nchans=nchans,
            nbits=nbits,
            nsamples=nsamples,
            data=bytes(nchans * nsamples),
        ).build()
        f = Path(tmpfile)
        f.write_bytes(payload)
        assert sigproc._nsamples_in_binary_header(f) is True

    def test_returns_false_when_absent(
        self, tmpfile: str, builder: SigprocBuilder
    ) -> None:
        nchans, nbits, nsamples = 4, 8, 50
        payload = _minimal_payload(
            builder,
            nchans=nchans,
            nbits=nbits,
            data=bytes(nchans * nsamples),
        ).build()
        f = Path(tmpfile)
        f.write_bytes(payload)
        assert sigproc._nsamples_in_binary_header(f) is False

    def test_skips_unknown_key_before_nsamples(
        self, tmpfile: str, builder: SigprocBuilder
    ) -> None:
        """Exercises the _skip_unknown_key branch inside _nsamples_in_binary_header."""
        nchans, nbits, nsamples = 4, 8, 50
        data = bytes(nchans * nsamples)
        base = _minimal_payload(builder, nchans=nchans, nbits=nbits).build()
        # Inject unknown uint key + nsamples before HEADER_END
        unknown = struct.pack("<I", len("mystery")) + b"mystery" + struct.pack("<I", 42)
        nsamples_field = (
            struct.pack("<I", len("nsamples"))
            + b"nsamples"
            + struct.pack("<I", nsamples)
        )
        header_end = struct.pack("<I", len("HEADER_END")) + b"HEADER_END"

        split = base.rfind(header_end)
        crafted = base[:split] + unknown + nsamples_field + base[split:]

        f = Path(tmpfile)
        f.write_bytes(crafted + data)
        assert sigproc._nsamples_in_binary_header(f) is True


class TestSigproc:
    @pytest.mark.parametrize(
        ("src_raj", "src_dej", "ra_str", "dec_str"),
        [
            (154230.1234, 1200.1234, "15h42m30.1234s", "+00d12m00.1234s"),
            (4230.1234, 411200.1234, "00h42m30.1234s", "+41d12m00.1234s"),
            (4230.1234, -411200.1234, "00h42m30.1234s", "-41d12m00.1234s"),
        ],
    )
    def test_parse_radec(
        self, src_raj: float, src_dej: float, ra_str: str, dec_str: str
    ) -> None:
        expected = SkyCoord(ra_str, dec_str, frame="icrs")
        result = sigproc.parse_radec(src_raj, src_dej)
        np.testing.assert_allclose(result.ra.deg, expected.ra.deg, atol=1e-6)
        np.testing.assert_allclose(result.dec.deg, expected.dec.deg, atol=1e-6)

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

    def test_edit_header_source_name_truncated_to_original_length(
        self, filfile_8bit_1: str, tmpfile: str
    ) -> None:
        """source_name edits are padded/truncated to preserve header size."""
        shutil.copyfile(filfile_8bit_1, tmpfile)
        original = sigproc.parse_header(tmpfile)
        original_len = len(original["source_name"])
        long_name = "X" * (original_len + 20)
        sigproc.edit_header(tmpfile, "source_name", long_name)
        header = sigproc.parse_header(tmpfile)
        # Must be same byte length as before — header size is invariant
        assert len(header["source_name"]) == original_len
        assert header["hdrlen"] == original["hdrlen"]

    def test_edit_header_invalid(
        self,
        filfile_8bit_1: str,
        tmpfile: str,
    ) -> None:
        shutil.copyfile(filfile_8bit_1, tmpfile)
        with pytest.raises(ValueError):
            sigproc.edit_header(tmpfile, "invalid", 0)
        with pytest.raises((TypeError, struct.error)):
            sigproc.edit_header(tmpfile, "nchans", "not_a_number")
        with pytest.raises(ValueError):
            sigproc.edit_header(tmpfile, "nchans", None)  # type: ignore [arg-type]
        with pytest.raises(ValueError, match="non-ASCII characters"):
            sigproc.edit_header(tmpfile, "source_name", "Ångström")

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
        header2["fch1"] += 1  # type: ignore [unsupported-operator]
        with pytest.raises(ValueError):
            sigproc.match_header(header1, header2)

    def test_parse_header_multi_single_file(
        self,
        filfile_8bit_1: str,
        filfile_8bit_1_header: dict,
    ) -> None:
        header, sinfo = sigproc.parse_header_multi(filfile_8bit_1)
        for key, expected_value in filfile_8bit_1_header.items():
            assert header[key] == expected_value
        assert len(sinfo.entries) == 1

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

        # exactly one sample after last
        tstart_valid = file_info1.tend + tsamp / 86400
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
