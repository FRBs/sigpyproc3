"""Mock FRB population generation over physical parameter distributions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import attrs
import numpy as np

from sigpyproc.simulation.furby import FurbyGenerator, PulseParams
from sigpyproc.simulation.inject import Injector

if TYPE_CHECKING:
    from sigpyproc.core.custom_types import SpecSimulMethods
    from sigpyproc.header import Header
    from sigpyproc.simulation.furby import Furby


@attrs.frozen(auto_attribs=True, kw_only=True)
class BurstDistribution:
    """Distribution parameters for generating mock FRB populations.

    Parameters
    ----------
    dm_range : tuple[float, float], optional
        Range of dispersion measures [pc cm^-3], by default (10.0, 1000.0).
    width_range : tuple[float, float], optional
        Range of intrinsic pulse widths [s], by default (1e-3, 1e-2).
    snr_range : tuple[float, float], optional
        Range of dedispersed matched-filter SNRs, by default (10.0, 50.0).
    snr_law : Literal["uniform", "euclidean"], optional
        Distribution law for SNR: 'uniform' or 'euclidean' (cumulative N(>S) ~ S^-1.5),
        by default "uniform".
    tau_range : tuple[float, float], optional
        Range of scattering timescales at tau_ref_freq [s], by default (0.0, 0.0).
    tau_ref_freq : float, optional
        Reference frequency for scattering timescales [MHz], by default 1000.0.
    spec_kinds : tuple[str, ...], optional
        Allowed spectral structure types,
        by default ("flat", "power_law", "scintillation").
    spec_idx_range : tuple[float, float], optional
        Range of spectral indices, by default (-4.0, 0.0).
    time_range : tuple[float, float] | None, optional
        Arrival time range [s]. If None, drawn uniformly within the observation length,
        by default None.
    shape : str, optional
        Intrinsic pulse profile shape ("gaussian" or "boxcar"), by default "gaussian".
    dmsmear : bool, optional
        Whether to apply intra-channel dispersion smearing, by default True.
    scatt_idx : float, optional
        Frequency scaling exponent for scattering, by default -4.4.
    disp_ind : float, optional
        Frequency scaling exponent for dispersion, by default -2.0.
    os_fact : int, optional
        Oversampling factor for generation, by default 10.
    """

    dm_range: tuple[float, float] = (10.0, 1000.0)
    width_range: tuple[float, float] = (1e-3, 1e-2)
    snr_range: tuple[float, float] = (10.0, 50.0)
    snr_law: Literal["uniform", "euclidean"] = "uniform"
    tau_range: tuple[float, float] = (0.0, 0.0)
    tau_ref_freq: float = 1000.0
    spec_kinds: tuple[str, ...] = ("flat", "power_law", "scintillation")
    spec_idx_range: tuple[float, float] = (-4.0, 0.0)
    time_range: tuple[float, float] | None = None
    shape: str = "gaussian"
    dmsmear: bool = True
    scatt_idx: float = -4.4
    disp_ind: float = -2.0
    os_fact: int = 10

    def __attrs_post_init__(self) -> None:
        if self.snr_law not in {"uniform", "euclidean"}:
            msg = f"Unknown snr_law: {self.snr_law}. Expected 'uniform' or 'euclidean'."
            raise ValueError(msg)
        if not self.spec_kinds:
            msg = "spec_kinds must contain at least one spectral structure type"
            raise ValueError(msg)


class RandomBurstPopulation:
    """Generate mock FRB populations over parameter distributions.

    Parameters
    ----------
    distribution : BurstDistribution | None, optional
        Distribution configuration. If None, default BurstDistribution() is used.
    seed : int | None, optional
        Random seed for reproducibility, by default None.
    """

    def __init__(
        self,
        distribution: BurstDistribution | None = None,
        seed: int | None = None,
    ) -> None:
        self.distribution = (
            distribution if distribution is not None else BurstDistribution()
        )
        self.rng = np.random.default_rng(seed)

    def draw_bursts(
        self,
        count: int,
        tobs: float,
    ) -> list[tuple[float, PulseParams]]:
        """Draw parameter sets for a population of bursts.

        Parameters
        ----------
        count : int
            Number of bursts to draw.
        tobs : float
            Total observation duration [s].

        Returns
        -------
        list of tuple of (float, PulseParams)
            Pairs of (arrival_time, pulse_params).
        """
        if count <= 0:
            return []

        dist = self.distribution
        t_min, t_max = dist.time_range if dist.time_range is not None else (0.0, tobs)
        times = self.rng.uniform(t_min, t_max, size=count)
        dms = self.rng.uniform(dist.dm_range[0], dist.dm_range[1], size=count)
        widths = self.rng.uniform(dist.width_range[0], dist.width_range[1], size=count)

        if dist.snr_law == "uniform":
            snrs = self.rng.uniform(dist.snr_range[0], dist.snr_range[1], size=count)
        else:
            # Euclidean power law: N(>S) ~ S^(-1.5)
            s_min, s_max = dist.snr_range
            if np.isclose(s_min, s_max):
                snrs = np.full(count, s_min, dtype=np.float64)
            else:
                u = self.rng.uniform(0.0, 1.0, size=count)
                exponent = -1.5
                p_min = s_min**exponent
                p_max = s_max**exponent
                snrs = (p_min - u * (p_min - p_max)) ** (-2.0 / 3.0)

        if np.isclose(dist.tau_range[0], dist.tau_range[1]):
            taus = np.full(count, dist.tau_range[0], dtype=np.float64)
        else:
            taus = self.rng.uniform(dist.tau_range[0], dist.tau_range[1], size=count)

        spec_indices = self.rng.uniform(
            dist.spec_idx_range[0],
            dist.spec_idx_range[1],
            size=count,
        )
        spec_choices = self.rng.choice(dist.spec_kinds, size=count)

        bursts: list[tuple[float, PulseParams]] = []
        for i in range(count):
            params = PulseParams(
                dm=float(dms[i]),
                snr=float(snrs[i]),
                width=float(widths[i]),
                shape=dist.shape,
                dmsmear=dist.dmsmear,
                disp_ind=dist.disp_ind,
                scatt_idx=dist.scatt_idx,
                tau0=float(taus[i]),
                tau_ref_freq=dist.tau_ref_freq,
                spec_kind=cast("SpecSimulMethods", spec_choices[i]),
                spec_idx=float(spec_indices[i]),
                os_fact=dist.os_fact,
            )
            bursts.append((float(times[i]), params))
        return bursts

    def generate_rows(
        self,
        count: int,
        tobs: float,
    ) -> list[tuple[float, float, float, float, float, float, str]]:
        """Draw parameters formatted as table rows for catalogues or injectors.

        Parameters
        ----------
        count : int
            Number of bursts to draw.
        tobs : float
            Total observation duration [s].

        Returns
        -------
        list of tuple of (time, dm, width, snr, spec_idx, tau0, spec_kind)
        """
        bursts = self.draw_bursts(count, tobs)
        return [
            (
                t,
                p.dm,
                p.width,
                p.snr,
                p.spec_idx,
                p.tau0,
                str(p.spec_kind),
            )
            for t, p in bursts
        ]

    def generate_pulses(
        self,
        header: Header,
        count: int,
        *,
        ref_freq: str = "fch1",
    ) -> tuple[list[Furby], list[int]]:
        """Generate simulated Furby profiles and their arrival sample indices.

        Parameters
        ----------
        header : Header
            Observation header.
        count : int
            Number of bursts to generate.
        ref_freq : str, optional
            Reference frequency name ("fch1", "fcenter", "fmin", "fmax"),
            by default "fch1".

        Returns
        -------
        tuple of (list of Furby, list of int)
            Generated furbies and their integer arrival sample indices.
        """
        bursts = self.draw_bursts(count, header.tobs)
        furbies: list[Furby] = []
        arrivals: list[int] = []
        for time_s, params in bursts:
            furby_seed = int(self.rng.integers(0, 2**31 - 1))
            gen = FurbyGenerator(
                header,
                params,
                ref_freq=ref_freq,
                seed=furby_seed,
            )
            furbies.append(gen.generate())
            arrivals.append(int(np.rint(time_s / header.tsamp)))
        return furbies, arrivals

    def to_injector(
        self,
        header: Header,
        count: int,
        *,
        ref_freq: str = "fch1",
    ) -> Injector:
        """Create an Injector containing simulated bursts from the population.

        Parameters
        ----------
        header : Header
            Observation header.
        count : int
            Number of bursts.
        ref_freq : str, optional
            Reference frequency name, by default "fch1".

        Returns
        -------
        Injector
            Ready-to-inject population.
        """
        furbies, arrivals = self.generate_pulses(header, count, ref_freq=ref_freq)
        return Injector.from_furbies(furbies, arrivals)

    def write_catalogue(
        self,
        path: str | Path,
        count: int,
        tobs: float,
    ) -> str:
        """Draw bursts and write them to a text catalogue file.

        Parameters
        ----------
        path : str | Path
            Destination catalogue file path.
        count : int
            Number of bursts to generate.
        tobs : float
            Observation length [s].

        Returns
        -------
        str
            Output path as a string.
        """
        out_path = Path(path)
        rows = self.generate_rows(count, tobs)
        # Sort by arrival time for convenience
        rows.sort(key=lambda r: r[0])

        header_lines = [
            "# Mock FRB catalogue generated by sigpyproc RandomBurstPopulation",
            "# Columns:",
            "# 1: time (s)",
            "# 2: dm (pc cm^-3)",
            "# 3: width (s)",
            "# 4: snr",
            "# 5: spec_idx",
            f"# 6: tau0 (s, at {self.distribution.tau_ref_freq:g} MHz)",
            "# 7: spec_kind",
            f"# Total bursts: {count}",
            f"# Observation time: {tobs:.4f} s",
            "# -------------------------------------------------------------",
        ]
        body_lines = [
            f"{r[0]:10.4f}  {r[1]:8.2f}  {r[2]:10.4e}  {r[3]:6.2f}  "
            f"{r[4]:6.2f}  {r[5]:10.4e}  {r[6]}"
            for r in rows
        ]
        content = "\n".join(header_lines + body_lines) + "\n"
        out_path.write_text(content, encoding="ascii")
        return str(out_path)
