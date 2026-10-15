"""Block classes for handling 2D data blocks.

This module contains the base class for handling 2D data blocks and the
FilterbankBlock and DMTBlock classes for handling filterbank and DM-time
transform data blocks, respectively.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

import numpy as np

from sigpyproc import kernels
from sigpyproc.core import stats
from sigpyproc.header import Header
from sigpyproc.timeseries import TimeSeries

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from sigpyproc.core.custom_types import FilterMethods, LocMethods, ScaleMethods


class BaseBlock(ABC):
    """Base class for handling data blocks.

    Parameters
    ----------
    data : ndarray
        2-D array of shape (nchans, nsamples).
    header : :class:`~sigpyproc.header.Header`
        Header object containing metadata.

    Attributes
    ----------
    data
    header
    nsamples
    """

    def __init__(self, data: np.ndarray, header: Header) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._header = header
        self._check_input()

    @property
    def data(self) -> NDArray[np.float32]:
        """Block data array.

        Returns
        -------
        NDArray[float32]
            2-D array of shape (nchans, nsamples).
        """
        return self._data

    @property
    def header(self) -> Header:
        """Metadata header object.

        Returns
        -------
        :class:`~sigpyproc.header.Header`
            Header object containing metadata.
        """
        return self._header

    @property
    def nsamples(self) -> int:
        """Number of samples.

        Returns
        -------
        int
            Number of samples in the data block.
        """
        return self.data.shape[1]

    def normalise(
        self,
        loc_method: LocMethods = "mean",
        scale_method: ScaleMethods = "std",
        axis: int | None = 1,
    ) -> Self:
        """Normalise/standardise the data block.

        Normalisation is performed by subtracting the loc estimate,
        and dividing by the scale estimate along the given axis.

        Parameters
        ----------
        loc_method : {"mean", "median"}, optional
            Method to estimate location to subtract, by default "mean".
        scale_method : {"std", "iqr", "mad"}, optional
            Method to estimate scale to divide by, by default "std".
        axis : int, optional
            Axis along which to perform normalisation, by default 1.

        Returns
        -------
        BaseBlock
            Normalised data block.
        """
        zscore_re = stats.estimate_zscore(self.data, loc_method, scale_method, axis)
        return self._spawn(zscore_re.data, self.header.new_header())

    def pad_samples(
        self,
        nsamps_final: int,
        offset: int,
        pad_mode: LocMethods = "median",
    ) -> Self:
        """Pad the data block with the given mode.

        Parameters
        ----------
        nsamps_final : int
            Number of time samples in the final padded data block.
        offset : int
            Offset to start padding.
        pad_mode : {"mean", "median"}, optional
            Mode for padding the data, by default "median".

        Returns
        -------
        BaseBlock
            Padded data block.

        Raises
        ------
        ValueError
            If the ``pad_mode`` is not "mean" or "median".
        """
        if pad_mode not in {"mean", "median"}:
            msg = f"pad_mode must be 'mean' or 'median', got {pad_mode}"
            raise ValueError(msg)
        if offset < 0 or nsamps_final < offset + self.nsamples:
            msg = (
                f"pad offset {offset} and length {nsamps_final} "
                f"cannot hold {self.nsamples} samples"
            )
            raise ValueError(msg)
        np_op = getattr(np, pad_mode)
        pad_values = np_op(self.data, axis=1)
        data_pad = np.ones((self.data.shape[0], nsamps_final), dtype=self.data.dtype)
        data_pad *= pad_values[:, None]
        data_pad[:, offset : offset + self.data.shape[1]] = self.data
        header = self.header.new_header(
            {
                "nsamples": nsamps_final,
                "tstart": self.header.mjd_after_nsamps(-offset),
            },
        )
        return self._spawn(data_pad, header)

    @abstractmethod
    def plot(self, *args, **kwargs) -> Figure:  # noqa: ANN002, ANN003
        """Plot the data block."""

    def _check_input(self) -> None:
        if not isinstance(self.header, Header):
            msg = "Input header is not a Header instance"
            raise TypeError(msg)
        if self.data.ndim != 2:
            msg = "Input data is not 2 dimensional"
            raise ValueError(msg)
        if self.nsamples != self.header.nsamples:
            msg = (
                f"Input data length ({self.nsamples}) does not match "
                f"header nsamples ({self.header.nsamples})"
            )
            raise ValueError(msg)

    def _spawn(self, data: np.ndarray, header: Header) -> Self:
        """Return a block of this type carrying the same metadata."""
        return self.__class__(data, header)


class FilterbankBlock(BaseBlock):
    """A class to handle a block of filterbank data in time-major order.

    Parameters
    ----------
    data : ndarray
        2-D array of shape (nchans, nsamples).
    header : :class:`~sigpyproc.header.Header`
        Header object containing metadata.
    dm : float, optional
        Dispersion measure of the data, by default 0.

    Attributes
    ----------
    dm
    nchans
    """

    def __init__(self, data: np.ndarray, header: Header, dm: float = 0) -> None:
        super().__init__(data, header)
        self._dm = float(dm)
        if self.data.shape[0] != self.header.nchans:
            msg = (
                f"Input has {self.data.shape[0]} channels, "
                f"header nchans is {self.header.nchans}"
            )
            raise ValueError(msg)
        if not np.isfinite(self._dm):
            msg = f"dm must be finite, got {dm}"
            raise ValueError(msg)

    def _spawn(self, data: np.ndarray, header: Header) -> FilterbankBlock:
        return FilterbankBlock(data, header, dm=self._dm)

    @property
    def dm(self) -> float:
        """Dispersion measure of the data.

        Returns
        -------
        float
            Dispersion measure.
        """
        return self._dm

    @property
    def nchans(self) -> int:
        """Number of frequency channels.

        Returns
        -------
        int
            Number of frequency channels.
        """
        return self.data.shape[0]

    def plot(
        self,
        scaler: str = "zscore",
        cmap: str = "magma",
        *,
        show_ts: bool = True,
        show_bp: bool = True,
        figsize: tuple[float, float] = (10, 7),
    ) -> Figure:
        """Plot the filterbank block with time series and bandpass panels.

        Parameters
        ----------
        scaler : str, optional
            Scaler name ('zscore', 'minmax', 'zscale', 'percentile', 'clip').
            Default is 'zscore'.
        cmap : str, optional
            Colormap name, by default 'magma'.
        show_ts : bool, optional
            Whether to show the top time series panel, by default True.
        show_bp : bool, optional
            Whether to show the right bandpass panel, by default True.
        figsize : tuple[float, float], optional
            Figure size in inches, by default (10, 7).

        Returns
        -------
        matplotlib.figure.Figure
            Rendered matplotlib figure.
        """
        import matplotlib.pyplot as plt

        from sigpyproc.viz.scaling import SCALERS, frequency_oriented

        active_scaler = next((s for s in SCALERS if s.name == scaler), SCALERS[0])
        prep = active_scaler.prepare(self.data)
        vmin, vmax = active_scaler.clim(prep)

        t0 = 0.0
        t1 = self.nsamples * self.header.tsamp
        freqs = self.header.chan_freqs
        image, f_lo, f_hi = frequency_oriented(prep, freqs, foff=self.header.foff)
        extent = (t0, t1, f_lo, f_hi)

        fig = plt.figure(figsize=figsize, layout="constrained")
        if show_ts and show_bp:
            gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])
            ax_ts = fig.add_subplot(gs[0, 0])
            ax_im = fig.add_subplot(gs[1, 0], sharex=ax_ts)
            ax_bp = fig.add_subplot(gs[1, 1], sharey=ax_im)

            ts = np.nanmean(self.data, axis=0)
            bp = np.nanmean(self.data, axis=1)
            t_axis = np.linspace(t0, t1, len(ts), endpoint=False)

            ax_ts.plot(t_axis, ts, color="#1f77b4", lw=1.0)
            ax_ts.set_ylabel("Power")
            ax_ts.tick_params(labelbottom=False)

            im = ax_im.imshow(
                image,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax_im.set_xlabel("Time (s)")
            ax_im.set_ylabel("Frequency (MHz)")

            ax_bp.plot(bp, freqs, color="#1f77b4", lw=1.0)
            ax_bp.set_xlabel("Mean")
            ax_bp.tick_params(labelleft=False)

            fig.colorbar(im, ax=ax_bp, orientation="vertical", pad=0.02)
        else:
            ax_im = fig.add_subplot(1, 1, 1)
            im = ax_im.imshow(
                image,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax_im.set_xlabel("Time (s)")
            ax_im.set_ylabel("Frequency (MHz)")
            fig.colorbar(im, ax=ax_im, orientation="vertical", pad=0.02)

        return fig

    def downsample(
        self,
        ffactor: int = 1,
        tfactor: int = 1,
        filter_method: FilterMethods = "mean",
    ) -> FilterbankBlock:
        """Downsample data block in frequency and/or time.

        Parameters
        ----------
        ffactor : int, optional
            Factor by which to downsample in frequency, by default 1.
        tfactor : int, optional
            Factor by which to downsample in time, by default 1.
        filter_method : {"mean", "median"}, optional
            Method to downsample, by default 'mean'.

        Returns
        -------
        FilterbankBlock
            Downsampled data block.
        """
        new_ar = stats.downsample_2d(self.data, (ffactor, tfactor), filter_method)
        changes = {
            "tsamp": self.header.tsamp * tfactor,
            "foff": self.header.foff * ffactor,
            "fch1": self.header.fch1 + self.header.foff * (ffactor - 1) / 2,
            "nsamples": self.header.nsamples // tfactor,
            "nchans": self.header.nchans // ffactor,
        }
        return self._spawn(new_ar, self.header.new_header(changes))

    def get_tim(self) -> TimeSeries:
        """Sum across all frequencies for each time sample.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            Sum of all channels as timeseries.
        """
        ts = self.data.sum(axis=0)
        return TimeSeries(ts, self.header.dedispersed_header(dm=self.dm))

    def get_bandpass(self) -> np.ndarray:
        """Sum across all time samples for each channel.

        Returns
        -------
        ndarray
            Bandpass of the data block.
        """
        return self.data.sum(axis=1)

    def dedisperse(
        self,
        dm: float,
        ref_freq: str = "fch1",
        *,
        only_valid_samples: bool = False,
    ) -> FilterbankBlock:
        """Dedisperse the block.

        Frequency dependent delays are applied as rotations to each
        channel in the block with respect to the reference frequency.

        Parameters
        ----------
        dm : float
            Dispersion measure to dedisperse to.
        ref_freq : str, optional
            Reference frequency to dedisperse to, by default "fch1".
        only_valid_samples : bool, optional
            Return a FilterbankBlock for valid time samples, by default False.

        Returns
        -------
        FilterbankBlock
            Dedispersed data block.

        Raises
        ------
        ValueError
            If there are not enough time samples to dedisperse.
        """
        delays = self.header.get_dmdelays(dm, ref_freq=ref_freq)
        if only_valid_samples:
            new_ar = kernels.roll_block_valid(self.data, -delays)
        else:
            new_ar = kernels.roll_block(self.data, -delays)
        return FilterbankBlock(
            new_ar,
            self.header.new_header({"nsamples": new_ar.shape[1]}),
            dm,
        )

    def dmt_transform(
        self,
        dm: float,
        dmsteps: int = 512,
        ref_freq: str | float = "fch1",
        *,
        only_valid_samples: bool = False,
    ) -> DMTBlock:
        """Compute the DM-time transform.

        The transform is computed by dedispersing data block at adjacent DMs.

        Parameters
        ----------
        dm : float
            Central DM to dedisperse to.
        dmsteps : int, optional
            Number of adjacent DMs to dedisperse to, by default 512.
        ref_freq : str | float, optional
            Reference frequency to dedisperse to, by default "fch1".
        only_valid_samples : bool, optional
            Return a DMTBlock for valid time samples, by default False.

        Returns
        -------
        DMTBlock
            DM-time transform block.
        """
        dm_arr = dm + np.linspace(-dm, dm, dmsteps)
        dm_delays = self.header.get_dmdelays(dm_arr, ref_freq=ref_freq)
        if only_valid_samples:
            new_ar = kernels.dmt_block_valid(self.data, dm_delays)
            dropped = max(0, int(np.max(-dm_delays)))
            tstart = self.header.mjd_after_nsamps(dropped)
        else:
            new_ar = kernels.dmt_block(self.data, dm_delays)
            tstart = self.header.tstart
        return DMTBlock(
            new_ar,
            self.header.new_header(
                {"nchans": 1, "nsamples": int(new_ar.shape[1]), "tstart": tstart},
            ),
            dm_arr,
        )

    def to_file(self, filename: str | None = None) -> str:
        """Write the data to file.

        Parameters
        ----------
        filename : str, optional
            Name of the output file, by default ``basename_split_start_to_end.fil``.

        Returns
        -------
        str
            Name of the output file.
        """
        if filename is None:
            mjd_after = self.header.mjd_after_nsamps(self.data.shape[1])
            filename = (
                f"{self.header.basename}_{self.header.tstart:.12f}_"
                f"to_{mjd_after:.12f}.fil"
            )
        updates = {"nbits": 32}
        with self.header.prep_outfile(filename, updates=updates, nbits=32) as out_file:
            out_file.cwrite(self.data.transpose().ravel())
        return filename


class DMTBlock(BaseBlock):
    """A class to handle a DM-time transform block in time-major order.

    Parameters
    ----------
    data : ndarray
        2-D array of shape (ndms, nsamples).
    header : :class:`~sigpyproc.header.Header`
        Header object containing metadata.
    dms : ndarray
        Array of DM values corresponding to each row of data.

    Attributes
    ----------
    dms
    ndms
    """

    def __init__(self, data: np.ndarray, header: Header, dms: np.ndarray) -> None:
        super().__init__(data, header)
        self._dms = np.asarray(dms, dtype=np.float32)
        self._check_dm_input()

    def _spawn(self, data: np.ndarray, header: Header) -> DMTBlock:
        return DMTBlock(data, header, self._dms)

    @property
    def dms(self) -> np.ndarray:
        """Array of DM values corresponding to each row of data.

        Returns
        -------
        ndarray
            DM array.
        """
        return self._dms

    @property
    def ndms(self) -> int:
        """Number of DM values.

        Returns
        -------
        int
            Number of DMs.
        """
        return self.data.shape[0]

    def plot(
        self,
        cmap: str = "magma",
        figsize: tuple[float, float] = (10, 6),
    ) -> Figure:
        """Plot the DM-time transform block with peak SNR curve.

        Parameters
        ----------
        cmap : str, optional
            Colormap name, by default 'magma'.
        figsize : tuple[float, float], optional
            Figure size in inches, by default (10, 6).

        Returns
        -------
        matplotlib.figure.Figure
            Rendered matplotlib figure.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])
        ax_dmt = fig.add_subplot(gs[0, 0])
        ax_snr = fig.add_subplot(gs[0, 1], sharey=ax_dmt)

        t0 = 0.0
        t1 = self.nsamples * self.header.tsamp
        extent = (t0, t1, float(np.min(self.dms)), float(np.max(self.dms)))

        im = ax_dmt.imshow(
            self.data,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap,
        )
        ax_dmt.set_xlabel("Time (s)")
        ax_dmt.set_ylabel("DM ($pc\\,cm^{-3}$)")

        peak_snrs = np.nanmax(self.data, axis=1)
        ax_snr.plot(peak_snrs, self.dms, color="#d62728", lw=1.5)
        ax_snr.set_xlabel("Peak")
        ax_snr.tick_params(labelleft=False)

        fig.colorbar(im, ax=ax_snr, orientation="vertical", pad=0.02)
        return fig

    def _check_dm_input(self) -> None:
        if self.ndms != self.dms.size:
            msg = (
                f"Number of DMs ({self.ndms}) does not match number of DM values "
                f"({self.dms.size})"
            )
            raise ValueError(msg)
