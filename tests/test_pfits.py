import numpy as np
from astropy.time import Time
from astropy.coordinates import Angle, EarthLocation, SkyCoord

from astropy.io import fits
from sigpyproc.io import pfits
from sigpyproc.io.bits import BitsInfo
from sigpyproc.utils import FrequencyChannels


class TestPrimaryHdr(object):
    def test_read(self, fitsfile_4bit):
        hdr = pfits.PrimaryHdr(fitsfile_4bit)
        assert isinstance(hdr.header, fits.Header)
        assert isinstance(hdr.location, EarthLocation)
        assert isinstance(hdr.coord, SkyCoord)
        assert isinstance(hdr.freqs, FrequencyChannels)
        assert isinstance(hdr.tstart, Time)
        np.testing.assert_equal(hdr.project_id, "P958")
        np.testing.assert_equal(hdr.telescope, "Parkes")


class TestSubintHdr(object):
    def test_read(self, fitsfile_4bit):
        hdr = pfits.SubintHdr(fitsfile_4bit)
        assert isinstance(hdr.header, fits.Header)
        assert isinstance(hdr.azimuth, Angle)
        assert isinstance(hdr.zenith, Angle)
        assert isinstance(hdr.freqs, FrequencyChannels)
        np.testing.assert_equal(hdr.poln_type, "AABBCRCI")
        np.testing.assert_equal(hdr.poln_state, "Coherence")
        np.testing.assert_equal(hdr.npol, 4)


class TestPFITSFile(object):
    def test_read(self, fitsfile_4bit):
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            assert isinstance(fitsfile.bitsinfo, BitsInfo)

    def test_read_freqs(self, fitsfile_4bit):
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            freqs = fitsfile.read_freqs(0)
            assert isinstance(freqs, FrequencyChannels)
            np.testing.assert_equal(freqs.nchans, fitsfile.sub_hdr.nchans)

    def test_read_weights(self, fitsfile_4bit):
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            weights = fitsfile.read_weights(0)
            np.testing.assert_equal(weights.size, fitsfile.sub_hdr.nchans)

    def test_read_scales(self, fitsfile_4bit):
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            scales = fitsfile.read_scales(0)
            np.testing.assert_equal(
                scales.shape, ((fitsfile.sub_hdr.npol, fitsfile.sub_hdr.nchans))
            )

    def test_read_offsets(self, fitsfile_4bit):
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            scales = fitsfile.read_offsets(0)
            np.testing.assert_equal(
                scales.shape, ((fitsfile.sub_hdr.npol, fitsfile.sub_hdr.nchans))
            )

    def test_read_subint(self, fitsfile_4bit):
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            data = fitsfile.read_subint(0)
            np.testing.assert_equal(data.shape, fitsfile.sub_hdr.subint_shape)

    def test_read_subint_pol(self, fitsfile_4bit):
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            data = fitsfile.read_subint_pol(0, poln_select=1)
            np.testing.assert_equal(
                data.shape, (fitsfile.sub_hdr.subint_samples, fitsfile.sub_hdr.nchans)
            )

    def test_read_subints(self, fitsfile_4bit):
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            nsub = 2
            data = fitsfile.read_subints(0, nsub)
            np.testing.assert_equal(
                data.shape,
                (fitsfile.sub_hdr.subint_samples * nsub, fitsfile.sub_hdr.nchans),
            )
