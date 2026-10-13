from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import Angle, SkyCoord

from sigpyproc import utils
from sigpyproc.header import Header, _infer_horizon
from sigpyproc.io import sigproc
from sigpyproc.simulation.pulsar import PulsarParams, fake_filterbank

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

CORE_SIGPROC_KEYS = {
    "data_type",
    "telescope_id",
    "machine_id",
    "tstart",
    "tsamp",
    "nbits",
    "nsamples",
    "fch1",
    "foff",
    "nchans",
    "nifs",
    "unknown_keys",
}
# Written only when the header knows the source name or position.
COORD_SOURCE_KEYS = {"source_name", "src_raj", "src_dej"}
OPTIONAL_SIGPROC_KEYS = {
    "barycentric",
    "pulsarcentric",
    "signed",
    "refdm",
    "period",
    "az_start",
    "za_start",
    "ibeam",
    "nbeams",
    "rawdatafile",
}


def _core_fields(**extra: object) -> dict:
    fields: dict = {
        "data_type": 1,
        "telescope_id": 0,
        "machine_id": 0,
        "source_name": "test_src",
        "src_raj": 0.0,
        "src_dej": 0.0,
        "tstart": 60000.0,
        "tsamp": 6.4e-5,
        "nbits": 8,
        "fch1": 1500.0,
        "foff": -0.1,
        "nchans": 4,
        "nifs": 1,
    }
    fields.update(extra)
    return fields


def _write_fil(path: Path, nsamples: int = 16, **extra: object) -> str:
    """Write a minimal sigproc filterbank file with only core keys by default."""
    data = bytes(4 * nsamples)
    path.write_bytes(sigproc.encode_header(_core_fields(**extra)) + data)
    return str(path)


def _minimal_header(**extra: object) -> Header:
    return Header(
        filename="minimal.fil",
        data_type="filterbank",
        nchans=4,
        foff=-0.1,
        fch1=1500.0,
        nbits=8,
        tsamp=6.4e-5,
        tstart=60000.0,
        nsamples=16,
        **extra,
    )


def _schema_keys(path: str) -> set[str]:
    return {key for key in sigproc.parse_header(path) if key in sigproc.SIGPROC_SCHEMA}


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
        assert CORE_SIGPROC_KEYS.issubset(spphdr)
        assert (
            set(spphdr) <= CORE_SIGPROC_KEYS | COORD_SOURCE_KEYS | OPTIONAL_SIGPROC_KEYS
        )

    def test_to_sigproc_omits_unknown_optional_keys(self) -> None:
        header = _minimal_header()
        spphdr = header.to_sigproc()
        assert CORE_SIGPROC_KEYS.issubset(spphdr)
        assert COORD_SOURCE_KEYS.isdisjoint(spphdr)
        assert OPTIONAL_SIGPROC_KEYS.isdisjoint(spphdr)

    def test_to_sigproc_writes_known_optional_keys(self) -> None:
        header = _minimal_header().new_header({"dm": 5.0, "frame": "barycentric"})
        spphdr = header.to_sigproc()
        assert spphdr["refdm"] == 5.0
        assert spphdr["barycentric"] == 1
        # The source never recorded pulsarcentric, so it is not added.
        assert "pulsarcentric" not in spphdr
        assert "signed" not in spphdr

    def test_to_string(self, filfile_4bit: str) -> None:
        header = Header.from_sigproc(filfile_4bit)
        header_string = header.to_string()
        assert isinstance(header_string, str)
        assert "Signedness of 8-bit numbers" in header_string
        expected = "SIGNED" if header.signed else "UNSIGNED"
        assert expected in header_string

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
        assert type(header.fch1) is float
        assert type(header.foff) is float

    def test_from_pfits_otherfile(self, inffile: str) -> None:
        with np.testing.assert_raises(OSError):
            Header.from_pfits(inffile)


class TestHorizon:
    def test_pfits_angles_are_computed(self, fitsfile_4bit: str) -> None:
        hdr = Header.from_pfits(fitsfile_4bit)
        assert 0 < hdr.azimuth.deg < 360
        assert 0 < hdr.zenith.deg < 180
        out = hdr.to_sigproc()
        np.testing.assert_allclose(out["az_start"], hdr.azimuth.deg)
        np.testing.assert_allclose(out["za_start"], hdr.zenith.deg)

    def test_known_angles_are_kept(self, filfile_8bit_1: str) -> None:
        hdr = Header.from_sigproc(filfile_8bit_1)
        values = {"telescope": "Parkes", "coord": hdr.coord, "tstart": hdr.tstart}
        values["azimuth"] = Angle(12.5 * units.deg)
        values["zenith"] = Angle(30.0 * units.deg)
        known = frozenset({"telescope", "coord", "azimuth", "zenith"})
        assert _infer_horizon(values, known) == frozenset()
        assert values["azimuth"].deg == 12.5
        assert values["zenith"].deg == 30.0

    def test_unknown_site_returns_none(self) -> None:
        coord = SkyCoord(0, 0, unit="deg")
        assert utils.horizon_angles("NoSuchSite", coord, 60000.0) is None

    def test_site_lookup_failure_is_non_fatal(
        self,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mocker.patch(
            "sigpyproc.utils.EarthLocation.of_site",
            side_effect=OSError("offline"),
        )
        coord = SkyCoord(0, 0, unit="deg")
        assert utils.horizon_angles("Parkes", coord, 60000.0) is None
        assert "Could not look up site" in caplog.text

    def test_infer_horizon_reports_failure(self, mocker: MockerFixture) -> None:
        mocker.patch("sigpyproc.utils.horizon_angles", return_value=None)
        values = {
            "telescope": "Parkes",
            "coord": SkyCoord(0, 0, unit="deg"),
            "tstart": 60000.0,
        }
        known = frozenset({"telescope", "coord"})
        assert _infer_horizon(values, known) == frozenset()
        assert "azimuth" not in values

    def test_infer_horizon_fills_only_missing_angle(self) -> None:
        values = {
            "telescope": "Parkes",
            "coord": SkyCoord(0, 0, unit="deg"),
            "tstart": 60000.0,
            "azimuth": Angle(12.5 * units.deg),
        }
        known = frozenset({"telescope", "coord", "azimuth"})
        inferred = _infer_horizon(values, known)
        assert inferred == frozenset({"zenith"})
        assert values["azimuth"].deg == 12.5
        assert 0 < values["zenith"].deg < 180


class TestHeaderRewrite:
    def test_rewrite_adds_no_optional_keys(self, tmp_path: Path) -> None:
        src = _write_fil(tmp_path / "in.fil")
        out = tmp_path / "out.fil"
        with Header.from_sigproc(src).prep_outfile(str(out)):
            pass
        assert _schema_keys(str(out)) == _schema_keys(src)
        assert OPTIONAL_SIGPROC_KEYS.isdisjoint(sigproc.parse_header(out))

    def test_rewrite_is_stable(self, tmp_path: Path) -> None:
        src = _write_fil(tmp_path / "in.fil", barycentric=1, refdm=3.0)
        first = tmp_path / "first.fil"
        second = tmp_path / "second.fil"
        with Header.from_sigproc(src).prep_outfile(str(first)):
            pass
        with Header.from_sigproc(str(first)).prep_outfile(str(second)):
            pass
        assert _schema_keys(str(first)) == _schema_keys(str(second))
        assert _schema_keys(src) == _schema_keys(str(second))

    def test_rewrite_keeps_known_optional_keys(self, tmp_path: Path) -> None:
        src = _write_fil(tmp_path / "in.fil", barycentric=1, refdm=10.0, signed=1)
        out = tmp_path / "out.fil"
        with Header.from_sigproc(src).prep_outfile(str(out)):
            pass
        parsed = sigproc.parse_header(out)
        assert parsed["barycentric"] == 1
        assert parsed["refdm"] == 10.0
        assert parsed["signed"] == 1
        assert Header.from_sigproc(str(out)).frame == "barycentric"

    def test_rewrite_keeps_unknown_keys(self, tmp_path: Path) -> None:
        unknown = (("mystery", struct.pack("<I", 999)),)
        src = _write_fil(tmp_path / "in.fil", unknown_keys=unknown)
        out = tmp_path / "out.fil"
        assert Header.from_sigproc(src).unknown_keys == unknown
        with Header.from_sigproc(src).prep_outfile(str(out)):
            pass
        assert Header.from_sigproc(str(out)).unknown_keys == unknown
        assert "mystery" not in sigproc.parse_header(out)

    def test_optional_value_of_zero_is_read_back(self, tmp_path: Path) -> None:
        src = _write_fil(tmp_path / "in.fil", barycentric=0)
        hdr = Header.from_sigproc(src)
        assert hdr.frame == "topocentric"
        assert hdr.to_sigproc()["barycentric"] == 0

    @pytest.mark.parametrize(
        ("extra", "expected"),
        [
            ({}, None),
            ({"barycentric": 0}, "topocentric"),
            ({"barycentric": 1}, "barycentric"),
            ({"pulsarcentric": 1}, "pulsarcentric"),
            ({"barycentric": 0, "pulsarcentric": 1}, "pulsarcentric"),
        ],
    )
    def test_frame_from_sigproc(
        self,
        tmp_path: Path,
        extra: dict,
        expected: str | None,
    ) -> None:
        src = _write_fil(tmp_path / "in.fil", **extra)
        assert Header.from_sigproc(src).frame == expected

    def test_frame_from_fbh5(self, mocker: MockerFixture) -> None:
        values = _core_fields(filename="in.h5", nsamples=16, pulsarcentric=1)
        mocker.patch("sigpyproc.header.fbh5.parse_header", return_value=values)
        assert Header.from_fbh5("in.h5").frame == "pulsarcentric"

    def test_unknown_frame_is_topocentric_for_inf(self) -> None:
        inf = _minimal_header().make_inf()
        assert _inf_value(inf, "Barycentered") == "0"

    def test_barycentric_frame_for_inf(self, tmp_path: Path) -> None:
        inf = _minimal_header(frame="barycentric").make_inf()
        assert _inf_value(inf, "Barycentered") == "1"
        out = tmp_path / "bary.inf"
        Path(out).write_text(inf)
        assert Header.from_inffile(out).frame == "barycentric"

    def test_unknown_frame_not_written_to_sigproc(self) -> None:
        spphdr = _minimal_header().to_sigproc()
        assert "barycentric" not in spphdr
        assert "pulsarcentric" not in spphdr

    def test_string_representation_shows_frame(self) -> None:
        assert "(topocentric (derived))" in _minimal_header().to_string()

    def test_known_keys_follow_new_header(self) -> None:
        hdr = _minimal_header().new_header({"dm": 2.0})
        assert "dm" in hdr.known_keys
        assert "dm" in hdr.new_header({"nchans": 2}).known_keys


class TestFollowUpRules:
    """Coordinate, unknown-key, printed-header, PSRFITS, inf and fake rules."""

    def test_tiny_ra_loads(self, tmp_path: Path) -> None:
        src = _write_fil(tmp_path / "in.fil", src_raj=1e-8, src_dej=-300000.0)
        hdr = Header.from_sigproc(src)
        assert np.isfinite(hdr.coord.ra.deg)
        assert hdr.coord.dec.deg == pytest.approx(-30.0)

    @pytest.mark.parametrize(
        ("dec", "expected"), [(-30.5, -0.0084722), (-3000.0, -0.5)]
    )
    def test_sub_degree_negative_declination_keeps_sign(
        self,
        dec: float,
        expected: float,
    ) -> None:
        coord = sigproc.parse_radec(0.0, dec)
        assert coord.dec.deg == pytest.approx(expected, rel=1e-4)

    def test_unchanged_coordinate_is_written_back_exactly(self, tmp_path: Path) -> None:
        src = _write_fil(
            tmp_path / "in.fil", src_raj=123456.7890123456, src_dej=-654321.9
        )
        out = tmp_path / "out.fil"
        with Header.from_sigproc(src).prep_outfile(str(out)):
            pass
        parsed = sigproc.parse_header(out)
        assert parsed["src_raj"] == 123456.7890123456
        assert parsed["src_dej"] == -654321.9

    def test_replaced_coordinate_is_formatted_not_copied(self, tmp_path: Path) -> None:
        src = _write_fil(tmp_path / "in.fil", src_raj=123456.7, src_dej=-654321.9)
        new_coord = sigproc.parse_radec(10000.0, 200000.0)
        hdr = Header.from_sigproc(src).new_header({"coord": new_coord})
        assert hdr.sigproc_src_raj is None
        assert hdr.to_sigproc()["src_raj"] == pytest.approx(10000.0)

    def test_long_names_are_written_as_given(self, tmp_path: Path) -> None:
        name = "N" * 80
        src = _write_fil(tmp_path / "in.fil", source_name=name)
        out = tmp_path / "out.fil"
        with Header.from_sigproc(src).prep_outfile(str(out)):
            pass
        assert sigproc.parse_header(out)["source_name"] == name

    def test_string_and_adjacent_unknown_keys_survive(self, tmp_path: Path) -> None:
        fields = _core_fields()
        base = sigproc.encode_header(fields)
        end = base.rfind(sigproc._encode_string("HEADER_END"))
        extra = (
            sigproc._encode_string("projid")
            + sigproc._encode_string("hello")
            + sigproc._encode_string("mystery")
            + struct.pack("<I", 999)
        )
        src = tmp_path / "in.fil"
        src.write_bytes(base[:end] + extra + base[end:] + bytes(4 * 16))
        hdr = Header.from_sigproc(str(src))
        assert hdr.unknown_keys == (
            ("projid", sigproc._encode_string("hello")),
            ("mystery", struct.pack("<I", 999)),
        )
        out = tmp_path / "out.fil"
        with hdr.prep_outfile(str(out)):
            pass
        assert Header.from_sigproc(str(out)).unknown_keys == hdr.unknown_keys

    def test_inferred_angles_are_marked_derived(self) -> None:
        text = _minimal_header().to_string()
        assert "Start AZ (deg)" in text
        az_line = next(ln for ln in text.splitlines() if ln.startswith("Start AZ"))
        assert "(derived)" in az_line

    def test_stored_zero_angle_is_not_marked(self, filfile_8bit_1: str) -> None:
        text = Header.from_sigproc(filfile_8bit_1).to_string()
        az_line = next(ln for ln in text.splitlines() if ln.startswith("Start AZ"))
        assert "(derived)" not in az_line

    def test_pfits_does_not_copy_chan_dm(self, fitsfile_4bit: str) -> None:
        spphdr = Header.from_pfits(fitsfile_4bit).to_sigproc()
        assert "refdm" not in spphdr
        assert "signed" not in spphdr

    def test_pfits_copies_nbeam_when_present(
        self,
        mocker: MockerFixture,
        fitsfile_4bit: str,
    ) -> None:
        primary = SimpleNamespace(
            tstart=SimpleNamespace(mjd=60000.0),
            coord=SkyCoord(0, 0, unit="deg"),
            telescope="Parkes",
            backend=SimpleNamespace(name="PDFB"),
            source="J0000",
            header={"NBEAM": 4, "IBEAM": 1, "CHAN_DM": 56.7},
            ibeam=1,
        )
        mocker.patch("sigpyproc.header.pfits.PrimaryHdr", return_value=primary)
        spphdr = Header.from_pfits(fitsfile_4bit).to_sigproc()
        assert spphdr["nbeams"] == 4
        assert spphdr["ibeam"] == 1
        assert "refdm" not in spphdr

    def test_fake_dm_and_period_zero_are_not_written(self, tmp_path: Path) -> None:
        hdr = _minimal_header().new_header({"nsamples": 64, "nchans": 8})
        out = tmp_path / "noise.fil"
        fake_filterbank(hdr, PulsarParams(dm=0.0), outfile=str(out))
        parsed = sigproc.parse_header(out)
        assert "refdm" not in parsed
        assert "period" not in parsed


def _inf_value(inf: str, description: str) -> str:
    line = next(ln for ln in inf.splitlines() if description in ln)
    return line.split("=")[-1].strip()


class TestSigprocRules:
    """Rules for which keys are read, kept and written. See docs/header.md."""

    def test_parkes_rewrite_keeps_zero_angles(
        self, filfile_8bit_1: str, tmp_path: Path
    ) -> None:
        out = tmp_path / "parkes_rewrite.fil"
        with Header.from_sigproc(filfile_8bit_1).prep_outfile(str(out)):
            pass
        src = sigproc.parse_header(filfile_8bit_1)
        dst = sigproc.parse_header(out)
        assert dst["az_start"] == src["az_start"] == 0.0
        assert dst["za_start"] == src["za_start"] == 0.0

    def test_absent_angles_are_inferred_but_not_written(self, tmp_path: Path) -> None:
        src = _write_fil(
            tmp_path / "in.fil",
            telescope_id=4,
            src_raj=123000.0,
            src_dej=-600000.0,
        )
        hdr = Header.from_sigproc(src)
        assert 0 < hdr.azimuth.deg < 360
        assert "azimuth" not in hdr.known_keys
        out = tmp_path / "out.fil"
        with hdr.prep_outfile(str(out)):
            pass
        assert "az_start" not in sigproc.parse_header(out)
        assert "za_start" not in sigproc.parse_header(out)

    def test_unknown_ids_survive_rewrite(self, tmp_path: Path) -> None:
        src = _write_fil(tmp_path / "in.fil", telescope_id=1234, machine_id=777)
        hdr = Header.from_sigproc(src)
        assert hdr.telescope == "unknown"
        out = tmp_path / "out.fil"
        with hdr.prep_outfile(str(out)):
            pass
        parsed = sigproc.parse_header(out)
        assert parsed["telescope_id"] == 1234
        assert parsed["machine_id"] == 777

    def test_changing_telescope_name_replaces_id(self, tmp_path: Path) -> None:
        src = _write_fil(tmp_path / "in.fil", telescope_id=1234)
        hdr = Header.from_sigproc(src).new_header({"telescope": "Parkes"})
        assert hdr.to_sigproc()["telescope_id"] == 4

    def test_unknown_telescope_name_warns_and_writes_zero(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        spphdr = _minimal_header(telescope="NoSuchTelescope").to_sigproc()
        assert spphdr["telescope_id"] == 0
        assert "NoSuchTelescope" in caplog.text

    def test_source_name_written_only_when_known(self, tmp_path: Path) -> None:
        assert "source_name" not in _minimal_header().to_sigproc()
        assert (
            _minimal_header(source="J0534+2200").to_sigproc()["source_name"]
            == "J0534+2200"
        )
        src = _write_fil(tmp_path / "in.fil", source_name="J1807")
        assert Header.from_sigproc(src).to_sigproc()["source_name"] == "J1807"

    def test_empty_rawdatafile_is_refused(self) -> None:
        with pytest.raises(ValueError, match="rawdatafile"):
            _minimal_header().new_header({"rawdatafile": ""})

    def test_invalid_frame_is_refused(self) -> None:
        with pytest.raises(ValueError, match="frame"):
            _minimal_header().new_header({"frame": "bary"})

    def test_none_frame_does_not_clear_flags(self, tmp_path: Path) -> None:
        src = _write_fil(tmp_path / "in.fil", barycentric=1)
        hdr = Header.from_sigproc(src).new_header({"frame": None})
        assert hdr.to_sigproc()["barycentric"] == 1

    def test_both_flags_round_trip(self, tmp_path: Path) -> None:
        src = _write_fil(tmp_path / "in.fil", barycentric=1, pulsarcentric=1)
        out = tmp_path / "out.fil"
        with Header.from_sigproc(src).prep_outfile(str(out)):
            pass
        parsed = sigproc.parse_header(out)
        assert parsed["barycentric"] == 1
        assert parsed["pulsarcentric"] == 1

    def test_explicit_barycentric_clears_present_pulsarcentric(
        self, tmp_path: Path
    ) -> None:
        src = _write_fil(tmp_path / "in.fil", pulsarcentric=1)
        spphdr = (
            Header.from_sigproc(src).new_header({"frame": "barycentric"}).to_sigproc()
        )
        assert spphdr["barycentric"] == 1
        assert spphdr["pulsarcentric"] == 0

    def test_explicit_frame_adds_no_flag_the_source_lacks(self, tmp_path: Path) -> None:
        src = _write_fil(tmp_path / "in.fil")
        spphdr = (
            Header.from_sigproc(src).new_header({"frame": "topocentric"}).to_sigproc()
        )
        assert "barycentric" not in spphdr
        assert "pulsarcentric" not in spphdr

    def test_effective_frame_is_topocentric_when_unknown(self) -> None:
        hdr = _minimal_header()
        assert hdr.frame is None
        assert hdr.effective_frame == "topocentric"

    def test_signed_true_is_written_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        hdr = _minimal_header().new_header({"signed": True})
        assert hdr.to_sigproc()["signed"] == 1
        assert "signed=1 is rejected by some readers" in caplog.text

    def test_rewrite_keeps_known_signed_zero(self, tmp_path: Path) -> None:
        src = _write_fil(tmp_path / "in.fil", signed=0)
        assert Header.from_sigproc(src).to_sigproc()["signed"] == 0

    def test_pfits_rawdatafile_is_source_name(self, fitsfile_4bit: str) -> None:
        hdr = Header.from_pfits(fitsfile_4bit)
        assert hdr.to_sigproc()["rawdatafile"] == Path(fitsfile_4bit).name

    def test_pfits_inferred_angles_are_written(self, fitsfile_4bit: str) -> None:
        spphdr = Header.from_pfits(fitsfile_4bit).to_sigproc()
        assert "az_start" in spphdr
        assert "za_start" in spphdr

    def test_inf_accepts_both_barycentric_labels(
        self, inffile: str, tmp_path: Path
    ) -> None:
        bary_line = "Barycentered?           (1 yes, 0 no)  =  0"
        text = Path(inffile).read_text()
        assert bary_line in text
        for label, value in (
            ("Barycentered?           (1 yes, 0 no)", 1),
            ("Barycentered?           (1=yes, 0=no)", 1),
            ("Barycentered?           (1=yes, 0=no)", 0),
        ):
            path = tmp_path / f"{value}_{label.count('=')}.inf"
            path.write_text(text.replace(bary_line, f"{label}  =  {value}"))
            expected = "barycentric" if value else "topocentric"
            assert Header.from_inffile(path).frame == expected
        assert Header.from_inffile(inffile).frame == "topocentric"


@pytest.mark.parametrize("unit", ["Mhz", "MHz", "MHZ"])
def test_inf_frequency_unit_case(inffile: str, tmp_path: Path, unit: str) -> None:
    original = Header.from_inffile(inffile)
    text = Path(inffile).read_text().replace("MHz", unit).replace("Mhz", unit)
    path = tmp_path / "case.inf"
    path.write_text(text)
    loaded = Header.from_inffile(path)
    assert loaded.fch1 == original.fch1
    assert loaded.foff == original.foff


def test_inf_non_radio_metadata(inffile: str, tmp_path: Path) -> None:
    radio_fields = (
        "Central freq",
        "Total bandwidth",
        "Number of channels",
        "Channel bandwidth",
        "Dispersion measure",
    )
    text = "\n".join(
        line
        for line in Path(inffile).read_text().splitlines()
        if not any(label in line for label in radio_fields)
    )
    path = tmp_path / "xray.inf"
    path.write_text(text + "\n Type of observation (EM band) = X-ray\n")
    header = Header.from_inffile(path)
    assert header.fch1 == 0
    assert header.foff == 0
    assert header.dm == 0
    assert header.nchans == 1
