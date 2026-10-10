import numpy as np

from sigpyproc.core.dmplan import DMPlan
from sigpyproc.header import Header

# First trials from getDMtable.C mixed-unit recurrence
# (tsamp=64 µs, ti=40 µs, bw=0.390625 MHz, cfreq=1.384 GHz, nchan=1024, tol=1.25)
_GOLDEN_HEAD = np.array([0.0, 0.18078946447221292, 0.3615769462299683])


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
