from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from sigpyproc import fourierseries
from sigpyproc.core import kernels, stats
from sigpyproc.foldedcube import FoldedData
from sigpyproc.header import Header
from sigpyproc.utils import validate_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from sigpyproc.core.custom_types import FilterMethods, LocMethods, ScaleMethods


class TimeSeries:
    """Container for 1-D time series data.

    Parameters
    ----------
    data : ArrayLike
        1-D time series array.
    header : :class:`~sigpyproc.header.Header`
        Header object containing metadata.

    Attributes
    ----------
    data
    header
    nsamples
    """

    def __init__(self, data: npt.ArrayLike, header: Header) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._header = header
        self._check_input()

    @property
    def data(self) -> npt.NDArray[np.float32]:
        """Time series data array.

        Returns
        -------
        NDArray[float32]
            1-D time series array.
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
            Number of samples in the time series.
        """
        return len(self.data)

    def normalise(
        self,
        loc_method: LocMethods = "mean",
        scale_method: ScaleMethods = "std",
    ) -> TimeSeries:
        """Normalise/standardise the time series.

        Normalisation is performed by subtracting the loc estimate,
        and dividing by the scale estimate of the data.

        Parameters
        ----------
        loc_method : {"mean", "median"}, optional
            Method to estimate location to subtract, by default "mean".
        scale_method : {"std", "iqr", "mad"}, optional
            Method to estimate scale to divide by, by default "std".

        Returns
        -------
        TimeSeries
            Normalised time series.
        """
        zscore_re = stats.estimate_zscore(self.data, loc_method, scale_method)
        return TimeSeries(zscore_re.data, self.header.new_header())

    def downsample(
        self,
        factor: int,
        filter_method: FilterMethods = "mean",
    ) -> TimeSeries:
        """Downsample the time series.

        Returned time series is of size ``nsamples // factor``.

        Parameters
        ----------
        factor : int
            Factor by which to downsample the time series.
        filter_method : {"mean", "median"}, optional
            Method to downsample, by default 'mean'.

        Returns
        -------
        TimeSeries
            Downsampled time series.
        """
        if factor == 1:
            return self
        tim_data = stats.downsample_1d(self.data, factor, method=filter_method)
        hdr_changes = {"tsamp": self.header.tsamp * factor, "nsamples": len(tim_data)}
        return TimeSeries(tim_data, self.header.new_header(hdr_changes))

    def pad(self, npad: int, mode: str = "mean", **pad_kwargs: dict) -> TimeSeries:
        """Pad a time series with mean valued data.

        Parameters
        ----------
        npad : int
            Number of bins to add at the end of the time series.
        mode : str, optional
            Padding mode (as used by `numpy.pad`), by default 'mean'.
        **pad_kwargs : dict
            Keyword arguments for `numpy.pad`.

        Returns
        -------
        TimeSeries
            Padded time series.
        """
        tim_data = np.pad(self.data, (0, npad), mode=mode, **pad_kwargs)  # type: ignore[call-overload]
        hdr_changes = {"nsamples": len(tim_data)}
        return TimeSeries(tim_data, self.header.new_header(hdr_changes))

    def deredden(
        self,
        method: FilterMethods = "mean",
        window: float = 0.5,
        *,
        fast: bool = False,
    ) -> TimeSeries:
        """Remove low-frequency red noise using a moving filter.

        Parameters
        ----------
        method : {'mean', 'median'}, optional
            Moving filter function to use, by default 'mean'.
        window : int, optional
            Width of moving filter window in seconds, by default 0.5 seconds.
        fast : bool, optional
            Use a faster but less accurate method, by default False.

        Returns
        -------
        TimeSeries
            De-reddened time series.

        Raises
        ------
        ValueError
            If ``window < 0``.
        """
        if window < 0:
            msg = "Window size must be greater than 0"
            raise ValueError(msg)
        window_bins = int(round(window / self.header.tsamp))
        if fast:
            tim_filter = stats.running_filter_fast(
                self.data,
                window_bins,
                method=method,
            )
        else:
            tim_filter = stats.running_filter(
                self.data,
                window_bins,
                method=method,
            )
        tim_deredden = self.data - tim_filter
        return TimeSeries(tim_deredden, self.header)

    def fold(
        self,
        period: float,
        accel: float = 0,
        nbins: int = 50,
        nints: int = 32,
    ) -> FoldedData:
        """Fold time series into discrete phase and sub-integration bins.

        Parameters
        ----------
        period : float
            Period to fold (in seconds).
        accel : float, optional
            Acceleration to fold, by default 0.
        nbins : int, optional
            Number of phase bins in output, by default 50.
        nints : int, optional
            Number of sub-integrations in output, by default 32.

        Returns
        -------
        :class:`~sigpyproc.foldedcube.FoldedData`
            Data cube containing the folded data.

        Raises
        ------
        ValueError
            If ``nbins * nints`` is too large for length of the data.
        """
        if self.data.size // (nbins * nints) < 10:
            msg = "Data length is too short for requested number of bins"
            raise ValueError(msg)
        fold_ar = np.zeros(nbins * nints, dtype=np.float32)
        count_ar = np.zeros(nbins * nints, dtype=np.int32)
        kernels.fold(
            self.data,
            fold_ar,
            count_ar,
            np.array([0], dtype=np.int32),
            0,
            self.header.tsamp,
            period,
            accel,
            self.data.size,
            self.data.size,
            1,
            nbins,
            nints,
            1,
            0,
        )
        fold_ar /= count_ar
        fold_ar = fold_ar.reshape(nints, 1, nbins)
        return FoldedData(
            fold_ar,
            self.header.new_header(),
            period,
            self.header.dm,
            accel,
        )

    def rfft(
        self,
        fftn: Callable[[np.ndarray, int], np.ndarray] | None = None,
    ) -> fourierseries.FourierSeries:
        """Perform 1-D real to complex forward FFT.

        Time series is zero-padded to the next good size for the FFT.

        Parameters
        ----------
        fftn : Callable[[np.ndarray], np.ndarray], optional
            The fft function to use. Own fft implementations can be used,
            e.g, `pyfftw.interfaces.numpy_fft.rfft`, or
            ``mkl_fft.interfaces.numpy_fft.rfft``, by default None.

        Returns
        -------
        :class:`~sigpyproc.fourierseries.FourierSeries`
            Fourier transform of the time series.
        """
        if fftn is None:
            fftn = kernels.nb_rfft
        if not callable(fftn):
            msg = f"Input fftn is not callable: {fftn}"
            raise TypeError(msg)
        n_good = kernels.nb_fft_good_size(self.nsamples, real=True)
        hdr_changes = {"nsamples": n_good}
        return fourierseries.FourierSeries(
            fftn(self.data, n_good),
            self.header.new_header(hdr_changes),
        )

    def apply_boxcar(self, width: int) -> TimeSeries:
        """Apply a square-normalized boxcar filter to the time series.

        Parameters
        ----------
        width : int
            Width of boxcar to apply in bins.

        Returns
        -------
        TimeSeries
            Filtered time series.

        Raises
        ------
        ValueError
            If ``width < 1``.

        Notes
        -----
        Time series returned is normalized in units of S/N.
        """
        if width < 1:
            msg = f"invalid boxcar width: {width}"
            raise ValueError(msg)
        mean_ar = stats.running_filter(self.data, width, method="mean")
        mean_ar_norm = mean_ar * np.sqrt(width)
        ref_bin = -width // 2 + 1 if width % 2 else -width // 2
        boxcar_ar = np.roll(mean_ar_norm, ref_bin)
        return TimeSeries(boxcar_ar, self.header.new_header())

    def resample(self, accel: float) -> TimeSeries:
        """Perform time domain resampling to remove acceleration.

        Parameters
        ----------
        accel : float
            Acceleration to remove from the time series.

        Returns
        -------
        TimeSeries
            Resampled time series.
        """
        tim_ar = kernels.resample_tim(self.data, accel, self.header.tsamp)
        hdr_changes = {"nsamples": tim_ar.size, "accel": accel}
        return TimeSeries(tim_ar, self.header.new_header(hdr_changes))

    def correlate(self, other: TimeSeries | np.ndarray) -> TimeSeries:
        """Perform cross correlation with another time series using FFTs.

        This method implements correlation equivalent to
        `scipy.signal.correlate` with ``mode='full'`` and ``method='fft'``.
        Correlation lags will be ``np.arange(-len(other) + 1, nsamples)``.

        Parameters
        ----------
        other : TimeSeries | np.ndarray
            Array to correlate with.

        Returns
        -------
        TimeSeries
            Time series containing the full correlation.

        Raises
        ------
        OSError
            If input array ``other`` is not `TimeSeries` or `ndarray`.

        See Also
        --------
        scipy.signal.correlate
        scipy.signal.correlation_lags
        """
        if isinstance(other, TimeSeries):
            other_data = other.data
        elif isinstance(other, np.ndarray):
            other_data = other.astype(np.float32)
        else:
            msg = "Input data is not array like"
            raise OSError(msg)
        other_data_conj = np.conj(other_data[::-1])
        corr_ar = kernels.fftconvolve(self.data, other_data_conj)
        header_changes = {
            "nsamples": corr_ar.size,
        }
        return TimeSeries(corr_ar, self.header.new_header(header_changes))

    def to_dat(self, basename: str | None = None) -> str:
        """Write time series in presto ``.dat`` format.

        Parameters
        ----------
        basename : str, optional
            File basename for output ``.dat`` and ``.inf`` files, by default None.

        Returns
        -------
        str
            Output ``.dat`` file name.

        Notes
        -----
        Method also writes a corresponding .inf file from the header data.
        """
        if basename is None:
            basename = self.header.basename
        self.header.make_inf(outfile=f"{basename}.inf")
        out_filename = f"{basename}.dat"
        with self.header.prep_outfile(out_filename, nbits=32) as outfile:
            outfile.cwrite(self.data)
        return out_filename

    def to_tim(self, filename: str | None = None) -> str:
        """Write time series in sigproc format.

        Parameters
        ----------
        filename : str, optional
            Name of file to write to, by default ``basename.tim``.

        Returns
        -------
        str
            Output ``.tim`` file name.
        """
        if filename is None:
            filename = f"{self.header.basename}.tim"
        with self.header.prep_outfile(filename, nbits=32) as outfile:
            outfile.cwrite(self.data)
        return filename

    @classmethod
    def from_dat(
        cls,
        datfile: str | Path,
        inffile: str | Path | None = None,
    ) -> TimeSeries:
        """Read a Presto format ``.dat`` file.

        Parameters
        ----------
        datfile : str | Path
            Name of the ``.dat`` file to read.
        inffile : str | Path, optional
            Name of the corresponding ``.inf`` file, by default None.

        Returns
        -------
        TimeSeries
            TimeSeries object.

        Notes
        -----
        If ``inffile`` is None, then the associated .inf file must be in
        the same directory.
        """
        datpath = validate_path(datfile)
        if inffile is None:
            inffile = datpath.with_suffix(".inf")
        data = np.fromfile(datpath, dtype=np.float32)
        inf_hdr = Header.from_inffile(inffile)
        hdr_changes = {"nsamples": data.size, "filename": datpath.as_posix()}
        return cls(data, inf_hdr.new_header(hdr_changes))

    @classmethod
    def from_tim(cls, timfile: str | Path) -> TimeSeries:
        """Read a sigproc format ``.tim`` file.

        Parameters
        ----------
        timfile : str | Path
            Name of the ``.tim`` file to read.

        Returns
        -------
        TimeSeries
            TimeSeries object.
        """
        header = Header.from_sigproc(timfile)
        data = np.fromfile(
            timfile,
            dtype=header.dtype,
            offset=header.stream_info.entries[0].hdrlen,
        )
        return cls(data, header)

    def _check_input(self) -> None:
        if not isinstance(self.header, Header):
            msg = "Input header is not a Header instance"
            raise TypeError(msg)
        if self.data.ndim != 1:
            msg = "Input data is not 1D"
            raise ValueError(msg)
        if self.data.size == 0:
            msg = "Input data is empty"
            raise ValueError(msg)
        if len(self.data) != self.header.nsamples:
            msg = (
                f"Input data length ({len(self.data)}) does not match "
                f"header nsamples ({self.header.nsamples})"
            )
            raise ValueError(msg)
