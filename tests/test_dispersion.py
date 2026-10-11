from __future__ import annotations

import numpy as np

from sigpyproc.core.dispersion import DMPlan, compute_dmdelays, compute_dmsmearing
from sigpyproc.header import Header

_GOLDEN_HEAD = np.array([0.0, 0.18078946447221292, 0.3615769462299683])


class TestDispersion:
    def test_compute_dmdelays(self) -> None:
        freqs = np.arange(1000, dtype=np.float32) * (-1.0) + 1400
        dm = 100
        tsamp = 0.001
        delays = compute_dmdelays(freqs, dm, tsamp, freqs[0], in_samples=False)
        assert delays.shape == (freqs.size,)
        assert delays.dtype == np.float32
        assert delays[0] == 0
        delays_samp = compute_dmdelays(freqs, dm, tsamp, freqs[0])
        assert delays_samp.shape == (freqs.size,)
        assert delays_samp.dtype == np.int32
        assert delays_samp[0] == 0

    def test_compute_dmdelays_vect(self) -> None:
        freqs = np.arange(1000, dtype=np.float32) * (-1.0) + 1400
        dm = np.linspace(0, 100, 50)
        tsamp = 0.001
        delays = compute_dmdelays(freqs, dm, tsamp, freqs[0], in_samples=False)
        assert delays.shape == (dm.size, freqs.size)
        assert delays.dtype == np.float32
        np.testing.assert_equal(delays[:, 0], np.zeros(dm.size))
        delays_samp = compute_dmdelays(freqs, dm, tsamp, freqs[0])
        assert delays_samp.shape == (dm.size, freqs.size)
        assert delays_samp.dtype == np.int32
        np.testing.assert_equal(delays_samp[:, 0], np.zeros(dm.size))

    def test_compute_dmsmearing(self) -> None:
        freqs = np.arange(1000, dtype=np.float32) * (-1.0) + 1400
        dm = 100
        tsamp = 0.001
        smearing = compute_dmsmearing(freqs, dm, tsamp, in_samples=False)
        assert smearing.shape == (freqs.size,)
        assert smearing.dtype == np.float32
        smearing_samp = compute_dmsmearing(freqs, dm, tsamp)
        assert smearing_samp.shape == (freqs.size,)
        assert smearing_samp.dtype == np.int32

    def test_compute_dmsmearing_vect(self) -> None:
        freqs = np.arange(1000, dtype=np.float32) * (-1.0) + 1400
        dm = np.linspace(0, 100, 50)
        tsamp = 0.001
        smearing = compute_dmsmearing(freqs, dm, tsamp, in_samples=False)
        assert smearing.shape == (dm.size, freqs.size)
        assert smearing.dtype == np.float32
        smearing_samp = compute_dmsmearing(freqs, dm, tsamp)
        assert smearing_samp.shape == (dm.size, freqs.size)
        assert smearing_samp.dtype == np.int32


class TestDMPlan:
    def test_from_sigproc_units_golden(self) -> None:
        plan = DMPlan.from_sigproc_units(
            0.0,
            100.0,
            tsamp_us=64.0,
            ti_us=40.0,
            bw_mhz=0.390625,
            cfreq_ghz=1.384,
            nchan=1024,
            tol=1.25,
        )
        assert len(plan) == 432
        np.testing.assert_allclose(plan.dms[:3], _GOLDEN_HEAD[:3], rtol=1e-5)
        np.testing.assert_allclose(plan.dms[1], 0.18078946447221292, rtol=1e-5)
        assert plan.dms[0] == 0.0
        assert plan.dms[-1] <= 100.0
        assert np.all(np.diff(plan.dms) > 0)

    def test_from_header_matches_units(self) -> None:
        fch1 = 1.384 * 1000.0 - (1024 / 2 - 0.5) * (-0.390625)
        header = Header(
            filename="synth.fil",
            data_type="filterbank",
            nchans=1024,
            foff=-0.390625,
            fch1=fch1,
            nbits=8,
            tsamp=64e-6,
            tstart=50000.0,
            nsamples=1024,
        )
        from_hdr = DMPlan.from_header(header, dm_max=100.0, ti=40e-6, tol=1.25)
        from_units = DMPlan.from_sigproc_units(
            0.0,
            100.0,
            tsamp_us=64.0,
            ti_us=40.0,
            bw_mhz=0.390625,
            cfreq_ghz=1.384,
            nchan=1024,
            tol=1.25,
        )
        np.testing.assert_allclose(from_hdr.dms, from_units.dms, rtol=1e-5)
