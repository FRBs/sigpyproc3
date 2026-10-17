"""Compact mock FRB profiles.

A furby is a dedispersed, native-resolution pulse. Dispersion is a per-channel
delay applied when the pulse is injected or when :meth:`Furby.realize` builds
a waterfall. The profile is scaled so the best boxcar matched filter on the
frequency sum has the requested SNR in unit-variance channel noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import h5py
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from numpy.polynomial import polynomial

from sigpyproc import utils
from sigpyproc.block import FilterbankBlock
from sigpyproc.core.filters import MatchedFilter
from sigpyproc.header import Header
from sigpyproc.kernels.dedispersion import disperse_block
from sigpyproc.kernels.simulation import downsample_profiles, simulate_ism, smear_lead

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure

    from sigpyproc.core.custom_types import MatchFilterMethods, SpecSimulMethods

# Fine samples of intrinsic support. Wider pulses use a smaller oversampling
# factor so the fine grid stays under this length.
_FINE_BUDGET = 4096
_FORMAT = "compact-1"

SPEC_KINDS: tuple[str, ...] = (
    "flat",
    "power_law",
    "smooth_envelope",
    "gaussian",
    "polynomial_peaks",
    "scintillation",
    "gaussian_blobs",
    "random",
)


@attrs.frozen(auto_attribs=True, kw_only=True)
class PulseParams:
    """Pulse parameters for one mock FRB.

    Parameters
    ----------
    dm : float
        Dispersion measure.
    snr : float
        Matched-filter signal-to-noise ratio of the frequency-summed profile.
    width : float
        Intrinsic width in seconds. FWHM for a Gaussian.
    shape : str, optional
        ``"gaussian"`` or ``"boxcar"``, by default ``"gaussian"``.
    dmsmear : bool, optional
        Apply intra-channel dispersion smearing, by default True.
    disp_ind : float, optional
        Frequency exponent of the delay law, by default -2.
    scatt_idx : float, optional
        Frequency exponent of the scattering time, by default -4.4.
    tau0 : float, optional
        Decay timescale of the scattering kernel in seconds, by default 0 (off).
        Evaluated at ``tau_ref_freq``.
    tau_ref_freq : float, optional
        Reference frequency for ``tau0`` in MHz, by default 1000.0 (1 GHz).
    spec_kind : str, optional
        Spectral shape, by default ``"flat"``.
    spec_idx : float, optional
        Spectral index. Applied to ``"power_law"`` and, when non-zero, to the
        other structured shapes. ``"flat"`` ignores it. By default -2.
    os_fact : int, optional
        Requested time oversampling, by default 10. Wide pulses use less.
    noise : float, optional
        Per-channel rms the SNR is defined against, by default 1.
    """

    dm: float
    snr: float
    width: float
    shape: str = attrs.field(default="gaussian")
    dmsmear: bool = attrs.field(default=True)
    disp_ind: float = attrs.field(default=-2.0)
    scatt_idx: float = attrs.field(default=-4.4)
    tau0: float = attrs.field(default=0.0, validator=attrs.validators.ge(0.0))
    tau_ref_freq: float = attrs.field(
        default=1000.0,
        validator=attrs.validators.gt(0.0),
    )
    spec_kind: SpecSimulMethods = attrs.field(default="flat")
    spec_idx: float = attrs.field(default=-2.0)
    os_fact: int = attrs.field(default=10, validator=attrs.validators.ge(1))
    noise: float = attrs.field(default=1.0, validator=attrs.validators.gt(0.0))

    def __attrs_post_init__(self) -> None:
        if not np.isfinite(self.dm) or self.dm < 0:
            msg = f"dm must be finite and >= 0, got {self.dm}"
            raise ValueError(msg)
        if not np.isfinite(self.snr) or self.snr <= 0:
            msg = f"snr must be finite and positive, got {self.snr}"
            raise ValueError(msg)
        if not np.isfinite(self.width) or self.width <= 0:
            msg = f"width must be finite and positive, got {self.width}"
            raise ValueError(msg)
        if self.shape not in {"gaussian", "boxcar"}:
            msg = f"shape must be 'gaussian' or 'boxcar', got {self.shape!r}"
            raise ValueError(msg)
        if self.spec_kind not in SPEC_KINDS:
            msg = f"spec_kind must be one of {SPEC_KINDS}, got {self.spec_kind!r}"
            raise ValueError(msg)
        for name in (
            "disp_ind",
            "scatt_idx",
            "spec_idx",
            "tau0",
            "tau_ref_freq",
            "noise",
        ):
            value = getattr(self, name)
            if not np.isfinite(value):
                msg = f"{name} must be finite, got {value}"
                raise ValueError(msg)
        if isinstance(self.os_fact, bool) or not isinstance(self.os_fact, int):
            msg = f"os_fact must be an integer >= 1, got {self.os_fact!r}"
            raise TypeError(msg)


@attrs.frozen(auto_attribs=True, kw_only=True)
class PulseStats:
    """Measured widths and signal-to-noise ratios of a simulated pulse.

    Widths are in seconds. Signal-to-noise ratios are for the frequency sum
    in noise of rms :attr:`PulseParams.noise` per channel.
    """

    top_hat_width: float
    box_width: float
    box_snr: float
    gaus_width: float
    gaus_snr: float
    template_snr: float
    norm: float
    width_eff: float


def _header_payload(header: Header) -> dict[str, Any]:
    """JSON-safe fields enough to rebuild a profile header."""
    payload: dict[str, Any] = {
        "filename": header.filename,
        "data_type": header.data_type,
        "nchans": int(header.nchans),
        "foff": float(header.foff),
        "fch1": float(header.fch1),
        "nbits": int(header.nbits),
        "tsamp": float(header.tsamp),
        "tstart": float(header.tstart),
        "nsamples": int(header.nsamples),
        "source": header.source,
        "telescope": header.telescope,
        "backend": header.backend,
        "dm": float(header.dm),
        "nifs": int(header.nifs),
        "signed": bool(header.signed),
        "period": float(header.period),
        "accel": float(header.accel),
    }
    if header.frame:
        payload["frame"] = header.frame
    if header.rawdatafile:
        payload["rawdatafile"] = header.rawdatafile
    return payload


def _plain_mapping(values: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy scalars so the mapping can be stored as JSON."""
    plain: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, np.generic):
            plain[key] = value.item()
        else:
            plain[key] = value
    return plain


@attrs.frozen(auto_attribs=True)
class Furby:
    """Dedispersed native-resolution FRB profile.

    Parameters
    ----------
    block : FilterbankBlock
        Noise-free profile, shape ``(nchans, nsamples)``, at the input
        sampling time. Channels are aligned at the intrinsic epoch.
    params_hdr : PulseParams
        Parameters used to build the profile.
    stats_hdr : PulseStats
        Widths and signal-to-noise ratios after scaling.
    delays : ndarray
        Dispersion delay of each channel in native samples. Positive means
        that channel arrives later.
    ref_freq : str
        Reference frequency the delays and the arrival time use.
    sample0 : int
        Native sample of profile index 0 when the reference-frequency arrival
        is sample 0. Negative when the pulse starts before that arrival.
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
    delays: np.ndarray = attrs.field(
        eq=False,
        converter=lambda value: np.asarray(value, dtype=np.float64),
    )
    ref_freq: str = "fch1"
    sample0: int = 0

    @property
    def profile(self) -> FilterbankBlock:
        """Dedispersed profile. Same object as :attr:`block`."""
        return self.block

    def realize(self) -> FilterbankBlock:
        """Build the dispersed waterfall at native resolution.

        Returns
        -------
        FilterbankBlock
            One row per channel, shifted by the rounded delay.
        """
        shifts = np.rint(self.delays).astype(np.int64)
        data = disperse_block(self.block.data, shifts, 1, 1)
        header = self.block.header.new_header({"nsamples": int(data.shape[1])})
        return FilterbankBlock(data, header, dm=self.params_hdr.dm)

    def to_filterbank(self, filename: str | None = None) -> str:
        """Write :meth:`realize` as a 32-bit sigproc filterbank.

        Parameters
        ----------
        filename : str, optional
            Output path. The default uses the profile header name.

        Returns
        -------
        str
            Path written.
        """
        return self.realize().to_file(filename)

    def save(self, filename: str | Path) -> str:
        """Save the compact profile to HDF5.

        Parameters
        ----------
        filename : str | Path
            Output path.

        Returns
        -------
        str
            Path written.
        """
        filepath = Path(filename)
        header = _header_payload(self.block.header)
        params = _plain_mapping(attrs.asdict(self.params_hdr))
        stats = _plain_mapping(attrs.asdict(self.stats_hdr))
        with h5py.File(filepath, "w") as handle:
            handle.attrs["format"] = _FORMAT
            handle.attrs["header"] = json.dumps(header)
            handle.attrs["params"] = json.dumps(params)
            handle.attrs["stats"] = json.dumps(stats)
            handle.attrs["ref_freq"] = self.ref_freq
            handle.attrs["sample0"] = int(self.sample0)
            handle.create_dataset("profile", data=self.block.data)
            handle.create_dataset("delays", data=self.delays)
        return filepath.as_posix()

    def plot(self, figsize: tuple[float, float] = (12.0, 6.0)) -> Figure:
        """Plot the dedispersed profile.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size in inches, by default (12, 6).

        Returns
        -------
        Figure
            Matplotlib figure.
        """
        return self.block.plot(figsize=figsize)

    @classmethod
    def from_file(cls, filename: str | Path) -> Furby:
        """Load a compact profile written by :meth:`save`.

        Parameters
        ----------
        filename : str | Path
            HDF5 path.

        Returns
        -------
        Furby
            Loaded profile.

        Raises
        ------
        ValueError
            If the file is not a compact furby.
        """
        filepath = utils.validate_path(filename)
        with h5py.File(filepath, "r") as handle:
            if handle.attrs.get("format") != _FORMAT:
                msg = f"{filepath} is not a compact furby file"
                raise ValueError(msg)
            header = Header(**json.loads(handle.attrs["header"]))
            params = PulseParams(**json.loads(handle.attrs["params"]))
            stats = PulseStats(**json.loads(handle.attrs["stats"]))
            data = np.asarray(handle["profile"], dtype=np.float32)
            delays = np.asarray(handle["delays"], dtype=np.float64)
            ref_freq = str(handle.attrs["ref_freq"])
            sample0 = int(handle.attrs["sample0"])
        block = FilterbankBlock(data, header, dm=params.dm)
        return cls(
            block,
            params,
            stats,
            delays=delays,
            ref_freq=ref_freq,
            sample0=sample0,
        )


class FurbyGenerator:
    """Build a noise-free dedispersed FRB profile.

    Parameters
    ----------
    hdr : Header
        Band and sampling time. ``nsamples`` is not the profile length.
    params : PulseParams
        Pulse to simulate.
    nsamps_out : int, optional
        Minimum native samples in the profile. The profile is zero-padded
        at the end when it is shorter.
    ref_freq : str, optional
        Reference frequency for delays and the arrival epoch, by default
        ``"fch1"``.
    seed : int, optional
        Seed for the spectral structure. ``None`` draws from OS entropy.

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
        *,
        ref_freq: str = "fch1",
        seed: int | None = None,
    ) -> None:
        self._hdr = hdr
        self._params = params
        self._nsamps_out = nsamps_out
        self._ref_freq = ref_freq
        self._seed = seed
        self._hdr_os = self.hdr.new_header({"tsamp": self.tsamp_os})

    @property
    def hdr(self) -> Header:
        """Input header."""
        return self._hdr

    @property
    def params(self) -> PulseParams:
        """Pulse parameters."""
        return self._params

    @property
    def nsamps_out(self) -> int | None:
        """Minimum native profile length, if one was requested."""
        return self._nsamps_out

    @property
    def hdr_os(self) -> Header:
        """Header at the requested oversampled cadence.

        Wide pulses may be generated with a smaller factor than this header.
        """
        return self._hdr_os

    @property
    def tsamp_os(self) -> float:
        """Sampling time at the requested oversampling factor."""
        return self.hdr.tsamp / self.params.os_fact

    @property
    def ref_freq(self) -> str:
        """Reference frequency for delays and the arrival epoch."""
        return self._ref_freq

    def generate(self) -> Furby:
        """Generate the scaled dedispersed profile.

        Returns
        -------
        Furby
            Compact profile, delays, and statistics.
        """
        os_fact = self._resolved_os_fact()
        tsamp_os = self.hdr.tsamp / os_fact
        signal, t0_fine = self._intrinsic(os_fact, tsamp_os)
        smear = self._smear_samples(tsamp_os)
        lead = smear_lead(smear)
        fine = simulate_ism(
            signal,
            self._spectrum(),
            smear,
            self._scatter_samples(tsamp_os),
        )
        fine = _pad_time(fine, utils.next_multiple(fine.shape[1], os_fact))
        native = downsample_profiles(fine, os_fact)
        stats = self._compute_stats(native.sum(axis=0))
        scaled = np.ascontiguousarray(native * np.float32(stats.norm))
        if self.nsamps_out is not None:
            scaled = _pad_time(scaled, self.nsamps_out)
        header = self.hdr.new_header(
            {
                "nsamples": int(scaled.shape[1]),
                "tsamp": self.hdr.tsamp,
                "nbits": 32,
                "dm": self.params.dm,
            },
        )
        block = FilterbankBlock(scaled, header, dm=self.params.dm)
        delays = self._delays()
        t0_fine_effective = t0_fine + lead
        return Furby(
            block,
            self.params,
            stats,
            delays=delays,
            ref_freq=self.ref_freq,
            sample0=-(t0_fine_effective // os_fact),
        )

    def _resolved_os_fact(self) -> int:
        """Oversampling that keeps the intrinsic support under the fine budget."""
        requested = self.params.os_fact
        width = self.params.width
        if width <= 0.0:
            return requested
        limit = int(_FINE_BUDGET * self.hdr.tsamp / (2.5 * width))
        return max(1, min(requested, max(limit, 1)))

    def _intrinsic(self, os_fact: int, tsamp_os: float) -> tuple[np.ndarray, int]:
        """Return the fine intrinsic pulse and the index of its centre."""
        width_samps = max(1, round(self.params.width / tsamp_os))
        nsamps = 2 * max(os_fact, int(2.5 * width_samps)) + 1
        if self.params.shape == "gaussian":
            samples = np.arange(nsamps, dtype=np.float32)
            signal = utils.gaussian(samples, nsamps // 2, width_samps)
            return signal.astype(np.float32, copy=False), nsamps // 2
        if self.params.shape == "boxcar":
            pulse = np.ones(width_samps, dtype=np.float32)
            signal = utils.pad_centre(pulse, nsamps).astype(np.float32, copy=False)
            pad = nsamps - width_samps
            centre = (pad + 1) // 2 + width_samps // 2
            return signal, centre
        msg = f"pulse shape {self.params.shape!r} is not supported"
        raise ValueError(msg)

    def _spectrum(self) -> np.ndarray:
        structure = SpectralStructure(
            self.hdr.chan_freqs,
            kind=self.params.spec_kind,
            spec_index=self.params.spec_idx,
            seed=self._seed,
        )
        return structure.generate()

    def _smear_samples(self, tsamp_os: float) -> np.ndarray:
        zeros = np.zeros(self.hdr.nchans, dtype=np.float64)
        if not self.params.dmsmear or self.hdr.nchans < 2:
            return zeros
        seconds = self.hdr.get_dmsmearing(
            self.params.dm,
            in_samples=False,
            index=self.params.disp_ind,
        )
        return np.asarray(seconds, dtype=np.float64) / tsamp_os

    def _scatter_samples(self, tsamp_os: float) -> np.ndarray:
        if self.params.tau0 <= 0.0:
            return np.zeros(self.hdr.nchans, dtype=np.float64)
        freqs = np.asarray(self.hdr.chan_freqs, dtype=np.float64)
        seconds = self.params.tau0 * (freqs / float(self.params.tau_ref_freq)) ** (
            self.params.scatt_idx
        )
        return seconds / tsamp_os

    def _delays(self) -> np.ndarray:
        seconds = self.hdr.get_dmdelays(
            self.params.dm,
            ref_freq=self.ref_freq,
            in_samples=False,
            index=self.params.disp_ind,
        )
        return np.asarray(seconds, dtype=np.float64) / self.hdr.tsamp

    def _compute_stats(self, ts: np.ndarray) -> PulseStats:
        """Scale factor and widths for a dedispersed frequency sum."""
        peak = float(np.max(ts))
        if peak <= 0.0:
            msg = "simulated pulse has no positive samples"
            raise ValueError(msg)
        top_hat = max(self.hdr.tsamp, float(np.sum(ts) / peak) * self.hdr.tsamp)
        noise_ts = self.params.noise * float(np.sqrt(self.hdr.nchans))
        positive = int(np.count_nonzero(ts > 0.0))
        nbins_max = max(1, min(int(ts.size), positive or int(ts.size)))
        mf_norm = _matched(ts, "boxcar", nbins_max)
        if mf_norm.snr == 0.0:
            msg = "matched-filter SNR is zero"
            raise ValueError(msg)
        norm = self.params.snr / (mf_norm.snr / noise_ts)
        scaled = ts * norm
        mf_box = _matched(scaled, "boxcar", nbins_max)
        gaus_max = _gaussian_nbins(int(ts.size), nbins_max)
        if _gaussian_fits(int(ts.size), gaus_max):
            mf_gaus = _matched(scaled, "gaussian", gaus_max)
        else:
            mf_gaus = mf_box
        template_snr = float(np.sqrt(np.sum((scaled / noise_ts) ** 2)))
        return PulseStats(
            top_hat_width=top_hat,
            box_width=float(mf_box.best_temp.width) * self.hdr.tsamp,
            box_snr=mf_box.snr / noise_ts,
            gaus_width=float(mf_gaus.best_temp.width) * self.hdr.tsamp,
            gaus_snr=mf_gaus.snr / noise_ts,
            template_snr=template_snr,
            norm=float(norm),
            width_eff=self._width_eff(),
        )

    def _width_eff(self) -> float:
        if self.params.dmsmear and self.hdr.nchans > 1:
            smearing = self.hdr.get_dmsmearing(
                self.params.dm,
                in_samples=False,
                index=self.params.disp_ind,
            )
            t_dm = float(np.max(smearing))
        else:
            t_dm = 0.0
        tau_center = (
            self.params.tau0
            * (float(self.hdr.fcenter) / float(self.params.tau_ref_freq))
            ** self.params.scatt_idx
            if self.params.tau0 > 0.0
            else 0.0
        )
        return float(
            np.sqrt(
                tau_center**2 + self.hdr.tsamp**2 + self.params.width**2 + t_dm**2,
            ),
        )


def _pad_time(profiles: np.ndarray, nsamps: int) -> np.ndarray:
    """Zero-pad the time axis on the right to ``nsamps``."""
    if profiles.shape[1] >= nsamps:
        return profiles
    out = np.zeros((profiles.shape[0], nsamps), dtype=np.float32)
    out[:, : profiles.shape[1]] = profiles
    return out


def _gaussian_fits(nsamp: int, width: float) -> bool:
    """Return whether a Gaussian template of this FWHM fits in ``nsamp``."""
    stddev = float(gaussian_fwhm_to_sigma) * width
    size = int(np.ceil(3.5 * stddev))
    return 2 * size + 1 <= nsamp


def _gaussian_nbins(nsamp: int, nbins_max: int) -> int:
    """Widest Gaussian FWHM whose template fits in ``nsamp``."""
    limit = (nsamp - 1) / (2.0 * 3.5 * float(gaussian_fwhm_to_sigma))
    fitted = int(np.floor(limit))
    return max(1, min(nbins_max, fitted))


def _matched(
    ts: np.ndarray,
    kind: MatchFilterMethods,
    nbins_max: int,
) -> MatchedFilter:
    return MatchedFilter(
        ts,
        loc_method="norm",
        scale_method="norm",
        temp_kind=kind,
        nbins_max=nbins_max,
        spacing_factor=1.5,
    )


class SpectralStructure:
    """Spectral gain across a filterbank band.

    Parameters
    ----------
    freqs : ndarray
        Channel frequencies in MHz.
    kind : str, optional
        Shape name, by default ``"scintillation"``.
    spec_index : float, optional
        Power-law index, by default -2. ``"flat"`` does not use it. Other
        shapes are multiplied by the power law when the index is not 0.
    seed : int, optional
        Seed for shapes that draw random numbers.

    Notes
    -----
    For ``"power_law"``, weights are divided by their mean to strictly preserve
    physical flux density ratios across frequency. Other structured shapes are
    scaled to the range 0 to 1 and shifted to have a mean of 1.
    """

    def __init__(
        self,
        freqs: np.ndarray,
        kind: SpecSimulMethods = "scintillation",
        spec_index: float = -2.0,
        seed: int | None = None,
    ) -> None:
        self.freqs = np.asarray(freqs, dtype=np.float32)
        if self.freqs.ndim != 1 or self.freqs.size < 1:
            msg = "freqs must be a non-empty 1-D array"
            raise ValueError(msg)
        if not np.all(np.isfinite(self.freqs)) or np.any(self.freqs <= 0):
            msg = "freqs must be finite and positive"
            raise ValueError(msg)
        self.kind = kind
        self.spec_index = spec_index
        if not np.isfinite(self.spec_index):
            msg = f"spec_index must be finite, got {spec_index}"
            raise ValueError(msg)
        self.rng = np.random.default_rng(seed)
        if self.freqs.size < 2:
            # One channel has no bandwidth, so every shape is unity.
            self.power_law_wt = np.ones_like(self.freqs)
        else:
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
        if self.kind not in self._spec_generators:
            msg = f"unknown spectral kind {self.kind!r}"
            raise ValueError(msg)

    @property
    def nchans(self) -> int:
        """Number of frequency channels."""
        return len(self.freqs)

    @property
    def foff(self) -> float:
        """Frequency step between the first two channels."""
        if self.freqs.size < 2:
            return 0.0
        return float(self.freqs[1] - self.freqs[0])

    def generate(self) -> np.ndarray:
        """Generate one spectral gain curve, mean 1.

        Returns
        -------
        ndarray
            Gain of each channel. A single channel is unity for every kind.
        """
        if self.kind not in self._spec_generators:
            msg = f"unknown spectral kind {self.kind!r}"
            raise ValueError(msg)
        if self.freqs.size < 2:
            return np.ones(self.freqs.size, dtype=np.float32)
        if self.kind == "power_law":
            spec = self.power_law_wt
            return (spec / np.mean(spec)).astype(np.float32)
        if self.kind == "random":
            spec = self._spec_random()
        else:
            spec = self._apply_weight(self._spec_generators[self.kind](), self.kind)
        return self._normalize_and_shift(spec)

    def plot(
        self,
        spec: np.ndarray | None = None,
        figsize: tuple[int, int] = (8, 6),
    ) -> Figure:
        """Plot a spectral gain curve.

        Parameters
        ----------
        spec : ndarray, optional
            Curve to plot. The default generates a new one.
        figsize : tuple[int, int], optional
            Figure size in inches, by default (8, 6).

        Returns
        -------
        Figure
            Matplotlib figure.
        """
        from matplotlib import pyplot as plt

        if spec is None:
            spec = self.generate()
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.freqs, spec)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Spectral Structure: {self.kind}")
        return fig

    def _apply_weight(self, spec: np.ndarray, kind: str) -> np.ndarray:
        if kind == "flat" or self.spec_index == 0.0:
            return spec
        return spec * self.power_law_wt

    def _spec_flat(self) -> np.ndarray:
        return np.ones_like(self.freqs)

    def _spec_power_law(self) -> np.ndarray:
        return np.ones_like(self.freqs)

    def _spec_smooth_envelope(self) -> np.ndarray:
        center = self.rng.uniform(self.freqs.min(), self.freqs.max())
        width = self.rng.uniform(np.ptp(self.freqs) / 4, np.ptp(self.freqs) / 2)
        roots = [center - width, center + width]
        spec = -polynomial.polyvalfromroots(self.freqs, roots)
        return spec.astype(np.float32)

    def _spec_gaussian(self) -> np.ndarray:
        center = self.rng.uniform(self.freqs.mean(), self.freqs.max())
        width = self.rng.uniform(np.ptp(self.freqs) / 10, np.ptp(self.freqs) / 2)
        return utils.gaussian(self.freqs, center, width)

    def _spec_poly(self) -> np.ndarray:
        degree = self.rng.integers(2, 6)
        coeffs = self.rng.normal(size=int(degree) + 1)
        bandwidth = self.freqs.max() - self.freqs.min()
        freqs_norm = (self.freqs - self.freqs.min()) / bandwidth * 2 - 1
        spec = np.poly1d(coeffs)(freqs_norm)
        return spec.astype(np.float32)

    def _spec_scint(self) -> np.ndarray:
        nscints = self.rng.geometric(p=1 / 3)
        scint_phi = self.rng.uniform(0, 2 * np.pi)
        scint_width_idx = self.rng.uniform(-10, 10)
        bandwidth = float(np.ptp(self.freqs))
        chan_norm = (
            np.zeros_like(self.freqs)
            if bandwidth == 0.0
            else (self.freqs - self.freqs.min()) / bandwidth
        )
        scint_width_weight = (self.freqs / self.freqs[0]) ** scint_width_idx
        spec = np.cos(2 * np.pi * nscints * chan_norm * scint_width_weight + scint_phi)
        return np.maximum(spec, 0.0)

    def _spec_gaussian_blobs(self) -> np.ndarray:
        nblobs = self.rng.geometric(p=1 / 3)
        blobs_size = float(np.abs(self.rng.normal(loc=20 / nblobs, scale=10)))
        spectrum = np.zeros_like(self.freqs)
        min_width = max(abs(float(self.foff)), 1e-3)
        for _ in range(int(nblobs)):
            center = self.rng.uniform(self.freqs.min(), self.freqs.max())
            width = max(float(self.rng.normal(blobs_size, 1)), min_width)
            amp = self.rng.uniform(1, 3)
            spectrum += utils.gaussian(self.freqs, center, width, amp)
        return spectrum

    def _spec_random(self) -> np.ndarray:
        names = [
            key for key in self._spec_generators if key not in {"random", "power_law"}
        ]
        kind = str(self.rng.choice(names))
        return self._apply_weight(self._spec_generators[kind](), kind)

    @staticmethod
    def _normalize_and_shift(arr: np.ndarray) -> np.ndarray:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_min == arr_max:
            normalized = np.ones_like(arr, dtype=np.float64)
        else:
            normalized = (arr - arr_min) / (arr_max - arr_min)
        return normalized - normalized.mean() + 1
