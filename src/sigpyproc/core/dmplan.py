"""Levin/Cordes-style DM trial tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np

if TYPE_CHECKING:
    from sigpyproc.header import Header

K_SIGPROC = 8.3
"""Dispersion smearing constant used by SIGPROC ``getDMtable.C`` (µs, MHz, GHz)."""


@attrs.define(auto_attribs=True, slots=True)
class DMPlan:
    """A table of trial dispersion measures.

    Parameters
    ----------
    dms : ndarray
        Trial DMs in pc cm^-3.
    """

    dms: np.ndarray

    def __len__(self) -> int:
        return int(self.dms.size)

    @classmethod
    def from_sigproc_units(
        cls,
        dm_min: float,
        dm_max: float,
        tsamp_us: float,
        ti_us: float,
        bw_mhz: float,
        cfreq_ghz: float,
        nchan: int,
        tol: float = 1.25,
    ) -> DMPlan:
        """Build a DM table using SIGPROC mixed units (verbatim ``getDMtable.C``)."""
        if dm_max < dm_min:
            msg = f"dm_max ({dm_max}) must be >= dm_min ({dm_min})"
            raise ValueError(msg)
        if nchan <= 0:
            msg = "nchan must be positive"
            raise ValueError(msg)
        bw = abs(float(bw_mhz))
        dms: list[float] = []
        dm = float(dm_min)
        k = K_SIGPROC
        while dm <= dm_max + 1e-12:
            dms.append(dm)
            t00 = np.sqrt(tsamp_us**2 + ti_us**2 + (k * bw * dm / cfreq_ghz**3) ** 2)
            a = k * bw / cfreq_ghz**3
            b = k * nchan * bw / 4.0 / cfreq_ghz**3
            c = tol * tol * t00 * t00 - tsamp_us * tsamp_us - ti_us * ti_us
            disc = -a * a * b * b * dm * dm + a * a * c + b * b * c
            if disc < 0 or (a * a + b * b) == 0:
                msg = "DM plan recurrence failed (negative discriminant)"
                raise ValueError(msg)
            new_dm = (b * b * dm + np.sqrt(disc)) / (a * a + b * b)
            if new_dm <= dm:
                msg = f"DM plan recurrence stalled at DM={dm}"
                raise ValueError(msg)
            dm = float(new_dm)
            if len(dms) > 1_000_000:
                msg = "DM plan produced too many trials"
                raise ValueError(msg)
        return cls(dms=np.asarray(dms, dtype=np.float64))

    @classmethod
    def from_header(
        cls,
        header: Header,
        dm_max: float,
        dm_min: float = 0.0,
        *,
        ti: float = 40e-6,
        tol: float = 1.25,
        nsub: int = 1,
    ) -> DMPlan:
        """Build a DM table from a header.

        ``ti`` is in seconds. Internally converted to the mixed units used by
        SIGPROC ``getDMtable.C``.
        """
        if nsub < 1:
            msg = "nsub must be >= 1"
            raise ValueError(msg)
        tsamp_us = header.tsamp * 1e6
        ti_us = ti * 1e6
        bw_mhz = abs(header.foff)
        cfreq_ghz = (header.fch1 + (header.nchans / 2 - 0.5) * header.foff) / 1000.0
        nchan = header.nchans // nsub
        return cls.from_sigproc_units(
            dm_min,
            dm_max,
            tsamp_us,
            ti_us,
            bw_mhz,
            cfreq_ghz,
            nchan,
            tol,
        )
