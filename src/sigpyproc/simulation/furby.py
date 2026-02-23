from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import h5py
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import polynomial

from sigpyproc import utils
from sigpyproc.block import FilterbankBlock
from sigpyproc.core import kernels
from sigpyproc.core.filters import MatchedFilter
from sigpyproc.header import Header

if TYPE_CHECKING:
    from collections.abc import Callable

    from sigpyproc.core.custom_types import SpecSimulMethods

logger = logging.getLogger(__name__)


@attrs.frozen(auto_attribs=True, kw_only=True)
class PulseParams:
    """Container object to handle pulse simulation parameters.

    Parameters
    ----------
    dm : float
        Dispersion measure of the FRB.
    snr : float
        Signal-to-noise ratio of the FRB.
    width : float
        Expected width of the pulse in seconds (FWHM in case of Gaussian).
    shape : str, optional
        The shape of the intrinsic pulse profile, by default 'gaussian'.
    disp_ind : float, optional
        Dispersion index of the ISM, by default -2.0.
    scatt_idx : float, optional
        Power-law index to scale the tau0 in each channel,
        by default -4.4 (Kolmogorov-like spectrum).
    tau0 : float, optional
        Decay timescale of the scattering kernel at central frequency,
        by default 1e-3 seconds.
    spec_kind : Literal["flat", "power_law", "smooth_envelope", "gaussian",
        "polynomial_peaks", "scintillation", "gaussian_blobs", "random"], optional
        Kind of the desired spectral structure, by default 'flat'.
    spec_idx : float, optional
        Spectral index for the power-law spectrum, by default -2.0.
    os_fact : int, optional
        Oversampling factor in time, by default 10.
    noise : float, optional
        rms of the noise onto which this burst would be added.
        Used to normalise the height of the simulated furby, by default 0.0.

    Notes
    -----
    Over-sampling factor is used to generate a more accurate representation of
    the pulse, especially in those cases where the requested width is of the
    order of ~1 sample.
    """

    dm: float
    snr: float
    width: float
    shape: str = attrs.field(default="gaussian")
    dmsmear: bool = attrs.field(default=True)
    disp_ind: float = attrs.field(default=-2.0)
    scatt_idx: float = attrs.field(default=-4.4)
    tau0: float = attrs.field(default=1e-3)
    spec_kind: SpecSimulMethods = attrs.field(default="flat")
    spec_idx: float = attrs.field(default=-2.0)
    os_fact: int = attrs.field(default=10, validator=attrs.validators.ge(1))  # ty: ignore[no-matching-overload]
    noise: float = attrs.field(default=1.0, validator=attrs.validators.ge(0.0))


@attrs.frozen(auto_attribs=True, kw_only=True)
class PulseStats:
    """Container object to store the statistics of the simulated pulse.

    Parameters
    ----------
    top_hat_width: float
        Width of the top-hat window in seconds.
    box_width: float
        Width of the best boxcar template in seconds.
    box_snr: float
        Signal-to-noise ratio for the best boxcar template.
    gaus_width: float
        Width of the best Gaussian template in seconds.
    gaus_snr: float
        Signal-to-noise ratio for the best Gaussian template.
    template_snr: float
        Signal-to-noise ratio for the input template.
    norm: float
        Normalisation factor for the simulated pulse.
    width_eff: float
        Effective width of the pulse after accounting for dispersion and smearing.
    """

    top_hat_width: float
    box_width: float
    box_snr: float
    gaus_width: float
    gaus_snr: float
    template_snr: float
    norm: float
    width_eff: float


@attrs.frozen(auto_attribs=True)
class Furby:
    """Container object to handle the simulated FRB signal.

    Parameters
    ----------
    block : :class:`~sigpyproc.block.FilterbankBlock`
        FilterbankBlock object containing the simulated FRB signal.
    params_hdr : PulseParams
        PulseParams object containing the simulation parameters.
    stats_hdr : PulseStats
        PulseStats object containing the computed statistics.

    Notes
    -----
    The simulated FRB signal is stored as a 2D array representing the time-frequency
    profile of the FRB. The object can be saved to a HDF5 file for later use.
    """

    block: FilterbankBlock = attrs.field(
        validator=attrs.validators.instance_of(FilterbankBlock),
    )
    params_hdr: PulseParams = attrs.field(
        validator=attrs.validators.instance_of(PulseParams),
    )
    stats_hdr: PulseStats = attrs.field(
        validator=attrs.validators.instance_of(PulseStats),
    )

    def save(self, filename: str | Path) -> str:
        """Save the simulated FRB signal to a HDF5 file.

        Parameters
        ----------
        filename : str | Path
            Path to the output HDF5 file.

        Returns
        -------
        str
            Saved HDF5 file name.
        """
        filepath = Path(filename)
        with h5py.File(filepath, "w") as f:
            block_group = f.create_group("block")
            block_dict = self.block.header.to_dict()
            for key, value in block_dict.items():
                if not isinstance(value, list | np.ndarray):
                    block_group.attrs[key] = value if value is not None else "None"
            block_group.create_dataset("data", data=self.block.data)

            params_group = f.create_group("params")
            for key, val in attrs.asdict(self.params_hdr).items():
                params_group.attrs[key] = val
            stats_group = f.create_group("stats")
            for key, val in attrs.asdict(self.stats_hdr).items():
                stats_group.attrs[key] = val
        return filepath.as_posix()

    def plot(self, figsize: tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot the Furby simulated FRB signal.

        Parameters
        ----------
        figsize : tuple[int, int], optional
            Figure size in inches, by default (12, 6).

        Returns
        -------
        Figure
            Matplotlib figure object containing the plot.
        """
        raise NotImplementedError

    @classmethod
    def from_file(cls, filename: str | Path) -> Furby:
        """Load a simulated FRB signal from a HDF5 file.

        Parameters
        ----------
        filename : str | Path
            Path to the input HDF5 file.

        Returns
        -------
        Furby
            Furby object containing the loaded FRB signal.
        """
        filepath = utils.validate_path(filename)
        with h5py.File(filepath, "r") as f:
            block_group = f["block"]
            data = block_group["data"][:]
            block_hdr = Header(**block_group.attrs)
            params_hdr = PulseParams(**f["params"].attrs)
            stats_hdr = PulseStats(**f["stats"].attrs)
        block = FilterbankBlock(data, block_hdr)
        return cls(block, params_hdr, stats_hdr)


class FurbyGenerator:
    """Class to generate mock FRB signals with noise-free templates.

    Simulates noise-free templates of mock FRB according to the
    requested pulse parameters. The class generates a 2D array
    representing the time-frequency profile of the FRB.

    Parameters
    ----------
    hdr : :class:`~sigpyproc.header.Header`
        Header object containing the observation metadata.
    params : PulseParams
        PulseParams object containing the simulation parameters.
    nsamps_out : int, optional
        Total number of samples desired in the output mock signal, by default None.

    References
    ----------
    .. [1] Gupta, V., et al. (2020), "Real-time injection of mock FRBs",
        https://arxiv.org/abs/2011.10191

    """

    def __init__(
        self,
        hdr: Header,
        params: PulseParams,
        nsamps_out: int | None = None,
    ) -> None:
        self._hdr = hdr
        self._params = params
        self._nsamps_out = nsamps_out
        self._hdr_os = self.hdr.new_header({"tsamp": self.tsamp_os})

    @property
    def hdr(self) -> Header:
        """Input header object.

        Returns
        -------
        :class:`~sigpyproc.header.Header`
            Header object.
        """
        return self._hdr

    @property
    def params(self) -> PulseParams:
        """Pulse parameters object.

        Returns
        -------
        PulseParams
            PulseParams object.
        """
        return self._params

    @property
    def nsamps_out(self) -> int | None:
        """Desired number of samples in the output data.

        Returns
        -------
        int
            Number of samples in the output data.
        """
        return self._nsamps_out

    @property
    def hdr_os(self) -> Header:
        """Header with the oversampled time resolution.

        Returns
        -------
        :class:`~sigpyproc.header.Header`
            Header object.
        """
        return self._hdr_os

    @property
    def tsamp_os(self) -> float:
        """Oversampled time resolution of the data.

        Returns
        -------
        float
            Time resolution of the data after oversampling.
        """
        return self.hdr.tsamp / self.params.os_fact

    def generate(self) -> Furby:
        """Generate a realistic 2D FRB signal with noise.

        Returns
        -------
        Furby
            Furby object containing the simulated FRB signal.
        """
        data_dedisp_os = self._gen_pulse()
        delays_os = self.hdr_os.get_dmdelays(self.params.dm, ref_freq="fcenter")
        # Need padding on both sides for dispersion kernel
        nsamps_disp_os = 2 * (data_dedisp_os.shape[1] + np.abs(delays_os).max())
        # Ensure minimum output length if specified
        min_nsamps_disp_os = (self.nsamps_out or 0) * self.params.os_fact
        nsamps_disp_os = max(nsamps_disp_os, min_nsamps_disp_os)
        # Ensure nsamps_disp_os is multiple of os_fact for downsampling
        nsamps_disp_os = utils.next_multiple(nsamps_disp_os, self.params.os_fact)

        # Calculate stats on the padded, oversampled, dedispersed time series
        ts_os = utils.pad_centre(data_dedisp_os.sum(axis=0), nsamps_disp_os)
        stats = self._compute_stats(ts_os)

        # Apply dispersion and downsample
        data_disp = kernels.disperse_block(
            data_dedisp_os,
            delays_os,
            nsamps_disp_os,
            self.params.os_fact,
        )
        # Correct number of samples after downsampling
        nsamps_final = data_disp.shape[1]
        block_hdr = self.hdr_os.new_header({"nsamples": nsamps_final})
        # Apply normalization factor calculated in stats
        frb_block = FilterbankBlock(data_disp * stats.norm, block_hdr)
        return Furby(frb_block, self.params, stats)

    def _gen_pulse(self) -> np.ndarray:
        """Generate a simple FRB profile with freq-time axis."""
        width_samps = max(1, int(self.params.width / self.tsamp_os))
        nsamps_pure = 2 * max(self.params.os_fact, int(2.5 * width_samps)) + 1
        if self.params.shape == "gaussian":
            x = np.arange(nsamps_pure, dtype=np.float32)
            signal = utils.gaussian(x, nsamps_pure // 2, width_samps)
        elif self.params.shape == "boxcar":
            x = np.ones(width_samps, dtype=np.float32)
            signal = utils.pad_centre(x, nsamps_pure)
        else:
            msg = f"Requested pulse shape : {self.params.shape} is not supported yet"
            raise ValueError(msg)
        spec_struct = SpectralStructure(self.hdr.chan_freqs, kind=self.params.spec_kind)
        spec = spec_struct.generate()

        # Calculate smearing and scattering timescales in oversampled units
        dm_smear = self.hdr_os.get_dmsmearing(self.params.dm)
        tau_nus = (
            int(self.params.tau0 / self.tsamp_os)
            * (self.hdr_os.chan_freqs / self.hdr_os.fcenter) ** self.params.scatt_idx
        )
        return kernels.simulate_ism(
            signal,
            spec,
            dm_smear,
            tau_nus,
            self.params.os_fact,
        )

    def _compute_stats(self, ts_os: np.ndarray) -> PulseStats:
        """Compute the statistics of the simulated pulse."""
        top_hat_width = max(
            self.hdr.tsamp,
            np.sum(ts_os) / np.max(ts_os) * self.tsamp_os,
        )

        ts = (
            kernels.downsample_1d_mean(ts_os, self.params.os_fact) * self.params.os_fact
        )
        noise_ts = self.params.noise * np.sqrt(self.hdr.nchans)

        nbins_max = min(len(ts), len(ts[ts > 0]))
        mf_norm = MatchedFilter(
            ts,
            loc_method="norm",
            scale_method="norm",
            temp_kind="boxcar",
            nbins_max=nbins_max,
            spacing_factor=1,
        )
        norm = self.params.snr / (mf_norm.snr / noise_ts)
        ts_norm = ts * norm
        mf_box = MatchedFilter(
            ts_norm,
            loc_method="norm",
            scale_method="norm",
            temp_kind="boxcar",
            nbins_max=nbins_max,
            spacing_factor=1,
        )
        mf_gaus = MatchedFilter(
            ts_norm,
            loc_method="norm",
            scale_method="norm",
            temp_kind="gaussian",
            nbins_max=nbins_max,
        )
        template_snr = np.sqrt(np.sum(ts_norm / noise_ts**2))

        t_dm = self.hdr.get_dmsmearing(self.params.dm, in_samples=False).max()
        width_eff = np.sqrt(
            self.params.tau0**2 + self.hdr.tsamp**2 + self.params.width**2 + t_dm**2,
        )
        return PulseStats(
            top_hat_width=top_hat_width,
            box_width=mf_box.best_temp.width,
            box_snr=mf_box.snr / noise_ts,
            gaus_width=mf_gaus.best_temp.width,
            gaus_snr=mf_gaus.snr / noise_ts,
            template_snr=template_snr,
            norm=norm,
            width_eff=width_eff,
        )


class SpectralStructure:
    """Class to simulate various spectral structures for the radio bursts.

    Generates spectral patterns with customisable characteristics,
    including frequency-dependent effects and various structural types.

    Parameters
    ----------
    freqs : numpy.ndarray
        1-D numpy array containing the frequency values of each channel (in MHz).
    kind : Literal["flat", "power_law", "smooth_envelope", "gaussian",
        "polynomial_peaks", "scintillation", "gaussian_blobs", "random"], optional
        The kind of the desired frequency structure, by default "scintillation".
    spec_index : float, optional
        Spectral index for the power-law component, by default -2.0.
    seed : int, optional
        Seed for the random number generator, by default None.

    Notes
    -----
    Supported spectral structures are:
    - flat: Equal gain, no evolution with frequency (unless spec_index != 0).
    - power_law: Gains follow a power-law with the given spectral index.
    - smooth_envelope: Smooth Gaussian-like envelope.
    - gaussian: Single Gaussian profile.
    - polynomial_peaks: Polynomial with random peaks, degree 2-5.
    - scintillation: Sinusoidal (scintillating) profile.
    - gaussian_blobs: Patchy profile with Gaussian blobs.
    - random: Randomly selects one of the above.

    The power-law weight (freqs / freqs[0]) ** spec_index is applied to all spectra.
    Set spec_index=0 for no power-law weighting.
    """

    def __init__(
        self,
        freqs: np.ndarray,
        kind: SpecSimulMethods = "scintillation",
        spec_index: float = -2.0,
        seed: int | None = None,
    ) -> None:
        self.freqs = np.asarray(freqs, dtype=np.float32)
        self.kind = kind
        self.spec_index = spec_index
        self.rng = np.random.default_rng(seed)
        self.power_law_wt = (self.freqs / self.freqs[0]) ** self.spec_index
        self._spec_generators: dict[str, Callable[[], np.ndarray]] = {
            "flat": self._spec_flat,
            "power_law": self._spec_power_law,
            "smooth_envelope": self._spec_smooth_envelope,
            "gaussian": self._spec_gaussian,
            "polynomial_peaks": self._spec_poly,
            "scintillation": self._spec_scint,
            "gaussian_blobs": self._spec_gaussian_blobs,
            "random": self._spec_random,
        }

    @property
    def nchans(self) -> int:
        """Number of frequency channels."""
        return len(self.freqs)

    @property
    def foff(self) -> float:
        """Frequency offset between two adjacent channels."""
        return self.freqs[1] - self.freqs[0]

    def generate(self) -> np.ndarray:
        """Generate the desired spectral structure.

        Returns
        -------
        ndarray
            1-D numpy array containing the simulated spectral structure.

        Notes
        -----
        The simulated spectrum is normalised to have a minimum of 0 and a maximum of 1.
        Then the spectrum is shifted to have a mean of 1 to conserve the total signal
        after averaging along the frequency axis.
        """
        spec = self._spec_generators[self.kind]() * self.power_law_wt
        return self._normalize_and_shift(spec)

    def plot(
        self,
        spec: np.ndarray | None = None,
        figsize: tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot the generated spectral structure.

        Parameters
        ----------
        spec : np.ndarray, optional
            1-D numpy array containing the spectral structure to plot, by default None.
        figsize : tuple[int, int], optional
            Figure size in inches, by default (8, 6).

        Returns
        -------
        Figure
            Matplotlib axes object containing the plot.
        """
        if spec is None:
            spec = self.generate()
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.freqs, spec)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Spectral Structure: {self.kind}")
        return fig

    def _spec_flat(self) -> np.ndarray:
        """Generate a flat spectrum."""
        return np.ones_like(self.freqs)

    def _spec_power_law(self) -> np.ndarray:
        """Generate a power-law spectrum."""
        return np.ones_like(self.freqs)

    def _spec_smooth_envelope(self) -> np.ndarray:
        """Generate a smooth envelope spectrum."""
        center = self.rng.uniform(self.freqs.min(), self.freqs.max())
        width = self.rng.uniform(np.ptp(self.freqs) / 4, np.ptp(self.freqs) / 2)
        roots = [center - width, center + width]
        spec = -polynomial.polyvalfromroots(self.freqs, roots)
        return spec.astype(np.float32)

    def _spec_gaussian(self) -> np.ndarray:
        """Generate a Gaussian spectrum."""
        center = self.rng.uniform(self.freqs.mean(), self.freqs.max())
        width = self.rng.uniform(np.ptp(self.freqs) / 10, np.ptp(self.freqs) / 2)
        return utils.gaussian(self.freqs, center, width)

    def _spec_poly(self) -> np.ndarray:
        """Generate a polynomial spectrum with random peaks."""
        degree = self.rng.integers(2, 6)  # Low degree to prevent overflow
        coeffs = self.rng.normal(size=degree + 1)
        bandwidth = self.freqs.max() - self.freqs.min()
        freqs_norm = (self.freqs - self.freqs.min()) / (bandwidth) * 2 - 1  # [-1,1]
        poly = np.poly1d(coeffs)
        spec = poly(freqs_norm)
        return spec.astype(np.float32)

    def _spec_scint(self) -> np.ndarray:
        """Generate a scintillating (sinusoid) spectrum."""
        nscints = self.rng.geometric(p=1 / 3)
        scint_phi = self.rng.uniform(0, 2 * np.pi)
        scint_width_idx = self.rng.uniform(-10, 10)
        scint_width_weight = (self.freqs / self.freqs[0]) ** scint_width_idx
        spec = np.cos(2 * np.pi * nscints * scint_width_weight + scint_phi)
        return np.maximum(spec, 0)

    def _spec_gaussian_blobs(self) -> np.ndarray:
        """Generate a patchy spectrum with Gaussian blobs (10 +- 5 MHz blobs)."""
        nblobs = self.rng.geometric(p=1 / 3)
        blobs_size = np.abs(self.rng.normal(loc=20 / nblobs, scale=10))
        spectrum = np.zeros_like(self.freqs)
        for _ in range(nblobs):
            center = self.rng.uniform(self.freqs.min(), self.freqs.max())
            width = self.rng.normal(blobs_size, 1)
            amp = self.rng.uniform(1, 3)
            spectrum += utils.gaussian(self.freqs, center, width, amp)
        return spectrum

    def _spec_random(self) -> np.ndarray:
        """Generate a random spectrum type."""
        random_kind = self.rng.choice(
            [k for k in self._spec_generators if k != "random"],
        )
        return self._spec_generators[random_kind]()

    @staticmethod
    def _normalize_and_shift(arr: np.ndarray) -> np.ndarray:
        """Normalize and shift the array to have a mean of 1."""
        arr_min, arr_max = arr.min(), arr.max()
        if arr_min == arr_max:
            normalized = np.ones_like(arr)
        else:
            normalized = (arr - arr_min) / (arr_max - arr_min)
        return normalized - normalized.mean() + 1
