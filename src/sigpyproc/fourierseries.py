"""Classes for handling 1D Fourier series data.

This module contains the PowerSpectrum and FourierSeries classes for handling
Fourier transformed data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from sigpyproc import kernels, timeseries
from sigpyproc.algorithms import period
from sigpyproc.foldedcube import Profile
from sigpyproc.header import Header
from sigpyproc.utils import validate_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import ArrayLike, NDArray

    from sigpyproc.core.candidates import CandidateTable


class PowerSpectrum:
    """An array class to handle Fourier Power spectrum.

    Parameters
    ----------
    data : ArrayLike
        1-D power spectrum array.
    header : :class:`~sigpyproc.header.Header`
        Header object containing metadata.

    Attributes
    ----------
    data
    header
    """

    def __init__(self, data: ArrayLike, header: Header) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._header = header
        self._check_input()

    @property
    def data(self) -> NDArray[np.float32]:
        """Power spectrum data array.

        Returns
        -------
        NDArray[float32]
            1-D power spectrum array.
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

    def bin2freq(self, r: int) -> float:
        """Compute centre frequency of a given bin.

        Parameters
        ----------
        r : int
            Fourier bin number.

        Returns
        -------
        float
            Frequency of the given bin.
        """
        if r < 0 or r >= self.data.size:
            msg = f"Fourier bin number {r} out of range"
            raise ValueError(msg)
        return r / self.header.tobs

    def bin2period(self, r: int) -> float:
        """Compute centre period of a given bin.

        Parameters
        ----------
        r : int
            Fourier bin number.

        Returns
        -------
        float
            Period of the given bin.
        """
        return 1 / self.bin2freq(r)

    def freq2bin(self, freq: float) -> int:
        """Compute nearest bin to a given frequency.

        Parameters
        ----------
        freq : float
            Frequency.

        Returns
        -------
        int
            Nearest bin to frequency.
        """
        return round(freq * self.header.tobs)

    def period2bin(self, period: float) -> int:
        """Compute nearest bin to a given periodicity.

        Parameters
        ----------
        period : float
            Period.

        Returns
        -------
        int
            Nearest bin to period.
        """
        return self.freq2bin(1 / period)

    def harmonic_fold(
        self,
        nfolds: int = 1,
        *,
        method: Literal["lyne", "simple", "presto", "greedy"] = "lyne",
    ) -> list[PowerSpectrum]:
        """Fold harmonics back onto the fundamental Fourier bins.

        Parameters
        ----------
        nfolds : int, optional
            Number of power-of-two folds. Fold ``i`` uses ``H = 2**(i+1)``
            harmonics (2, 4, 8, …), by default 1.
        method : {"lyne", "simple", "presto", "greedy"}, optional
            ``lyne`` is the SIGPROC Lyne-Ashworth stretch. ``simple`` sums
            integer multiples ``h*n`` and ignores drift. ``presto`` is the
            stock PRESTO sum (even harmonics, nearest bin). ``greedy`` follows
            the louder neighbouring bin as the harmonic order increases
            (Adámek, Roy & Armour 2022). By default ``lyne``.

        Returns
        -------
        list[PowerSpectrum]
            One spectrum per fold, on the original frequency grid.

        Raises
        ------
        ValueError
            If ``nfolds`` is less than 1 or ``method`` is unknown.
        """
        if nfolds < 1:
            msg = "nfolds must be at least 1"
            raise ValueError(msg)
        kernels_by_method = {
            "lyne": kernels.sum_harmonics_lyne,
            "simple": kernels.sum_harmonics_simple,
            "presto": kernels.sum_harmonics_presto,
            "greedy": kernels.sum_harmonics_greedy,
        }
        kernel = kernels_by_method.get(method)
        if kernel is None:
            msg = f"method must be one of {tuple(kernels_by_method)}, got {method!r}"
            raise ValueError(msg)
        summed = kernel(self.data, nfolds)
        return [
            PowerSpectrum(summed[ifold], self.header.new_header())
            for ifold in range(nfolds)
        ]

    def find_peaks(
        self,
        *,
        npeaks: int = 100,
        snr_min: float = 6.0,
        pmax: float = 10.0,
        dm: float = 0.0,
        accel: float = 0.0,
        nharm: int = 1,
    ) -> CandidateTable:
        """Pick local maxima in a whitened power spectrum.

        SNR in bin ``i`` uses :func:`sigpyproc.core.stats.running_robust_standardize`
        over a running window of ``max(128, nfreq // 64)`` bins. The DC bin and
        every bin with period ``> pmax`` are skipped. A local maximum is
        strictly greater than both neighbours.

        Folded spectra are still sampled at the original ``tobs``. Harmonic
        power has already been summed into the fundamental bin. For a
        non-sinusoidal signal the brightest row can still be a harmonic of
        the true period (see the search tutorial).

        Parameters
        ----------
        npeaks : int, optional
            Maximum number of peaks to return, by default 100.
        snr_min : float, optional
            Minimum SNR, by default 6.0.
        pmax : float, optional
            Longest period to keep, in seconds, by default 10.0.
        dm : float, optional
            Dispersion measure stored on each row, by default 0.0.
        accel : float, optional
            Acceleration stored on each row, by default 0.0.
        nharm : int, optional
            Harmonic fold that produced this spectrum (1 = unfolded),
            by default 1.

        Returns
        -------
        CandidateTable
            Up to ``npeaks`` rows with ``kind="fft"``, sorted by SNR.

        Raises
        ------
        ValueError
            If ``npeaks``, ``pmax``, or ``nharm`` is not positive.
        """
        if npeaks < 1:
            msg = "npeaks must be at least 1"
            raise ValueError(msg)
        if pmax <= 0:
            msg = "pmax must be positive"
            raise ValueError(msg)
        if nharm < 1:
            msg = "nharm must be at least 1"
            raise ValueError(msg)

        return period.pick_spectral_peaks(
            self.data,
            self.header.tobs,
            npeaks=npeaks,
            snr_min=snr_min,
            pmax=pmax,
            dm=dm,
            accel=accel,
            nharm=nharm,
        )

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

    @classmethod
    def from_spec(cls, filename: str | Path) -> PowerSpectrum:
        """Read a sigpyproc format ``.spec`` file as a power spectrum.

        Parameters
        ----------
        filename : str | Path
            Name of the ``.spec`` file to read.

        Returns
        -------
        PowerSpectrum
            Power spectrum object.
        """
        header = Header.from_sigproc(filename)
        data = np.fromfile(
            filename,
            dtype=np.float32,
            offset=header.stream_info.entries[0].hdrlen,
        )
        return cls(data, header)

    def view(self, **kwargs: Any) -> Any:  # noqa: ANN401
        """Open the interactive power spectrum viewer.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to :class:`~sigpyproc.viz.spectrum.SpectrumViewer`.

        Returns
        -------
        SpectrumViewer
            Interactive spectrum viewer instance.
        """
        from sigpyproc.viz.series import SpectrumViewer

        viewer = SpectrumViewer(self, **kwargs)
        if not kwargs.get("batch"):
            viewer.show()
        return viewer


class FourierSeries:
    """An array class to handle complex Fourier series.

    Parameters
    ----------
    data : ArrayLike
        1-D fourier series array.
    header : :class:`~sigpyproc.header.Header`
        Header object containing metadata.

    Attributes
    ----------
    header
    data
    binwidth
    """

    def __init__(self, data: ArrayLike, header: Header) -> None:
        self._data = np.asarray(data, dtype=np.complex64)
        self._header = header
        self._check_input()

    @property
    def data(self) -> NDArray[np.complex64]:
        """Fourier series data array.

        Returns
        -------
        NDArray[complex64]
            1-D fourier series array.
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
    def binwidth(self) -> float:
        """Fourier bin width.

        Returns
        -------
        float
            Width of each frequency bin.
        """
        return 1 / self.header.tobs

    def ifft(
        self,
        ifftn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> timeseries.TimeSeries:
        """Perform 1-D complex to real inverse FFT.

        Parameters
        ----------
        ifftn : Callable[[np.ndarray, int], np.ndarray], optional
            The ifft function to use. Own ifft implementation can be used,
            e.g. `pyfftw.interfaces.numpy_fft.irfft`, by default None.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            The real-valued time series.
        """
        if ifftn is None:
            ifftn = kernels.nb_irfft
        if not callable(ifftn):
            msg = f"Input ifftn is not callable: {ifftn}"
            raise TypeError(msg)
        tim_ar = ifftn(self.data)
        return timeseries.TimeSeries(tim_ar, self.header.new_header())

    def form_spec(self, *, interpolate: bool = False) -> PowerSpectrum:
        """Form the power spectrum.

        Parameters
        ----------
        interpolate : bool, optional
            Interpolate using nearest bins, by default False.

        Returns
        -------
        PowerSpectrum
            The power spectrum.
        """
        if interpolate:
            spec_ar = kernels.form_interp_mspec(self.data)
        else:
            spec_ar = kernels.form_mspec(self.data)
        return PowerSpectrum(spec_ar, self.header.new_header())

    def deredden(
        self,
        start_width: int = 6,
        end_width: int = 100,
        end_freq: float = 6.0,
    ) -> FourierSeries:
        """Perform rednoise removal via Presto-style method.

        Parameters
        ----------
        start_width : int, optional
            Initial window size in the range (2, 50), by default 6.
        end_width : int, optional
            Final window size in the range (50, 500), by default 100.
        end_freq : float, optional
            The highest frequency where the windowing increases in
            the range (0.1, 10), by default 6.0.

        Returns
        -------
        FourierSeries
            Whitened fourier series.
        """
        end_freq_bin = round(end_freq / self.binwidth)
        out_ar = kernels.fs_running_median(
            self.data,
            start_width,
            end_width,
            end_freq_bin,
        )
        return FourierSeries(out_ar, self.header.new_header())

    def recon_prof(self, freq: float, nharms: int = 32) -> Profile:
        """Reconstruct the time-domain pulse profile.

        Parameters
        ----------
        freq : float
            Frequency of signal to reconstruct.
        nharms : int, optional
            Number of harmonics to use in reconstruction, by default 32.

        Returns
        -------
        :class:`~sigpyproc.foldedcube.Profile`
            The reconstructed pulse profile.
        """
        if nharms < 1:
            msg = "nharms must be at least 1"
            raise ValueError(msg)
        freq_bin = round(freq * self.header.tobs)
        if freq_bin < 1:
            msg = f"frequency {freq} is below one Fourier bin"
            raise ValueError(msg)
        spec_ids = np.arange(1, nharms + 1) * freq_bin
        if int(spec_ids[-1]) >= self.data.size:
            msg = (
                f"harmonic {nharms} of frequency {freq} falls outside the spectrum "
                f"({int(spec_ids[-1])} >= {self.data.size})"
            )
            raise ValueError(msg)
        nprof = 2 * nharms
        spec = np.zeros(nprof, dtype=np.complex64)
        spec[1 : nharms + 1] = self.data[spec_ids]
        if nharms > 1:
            spec[nprof - nharms + 1 :] = np.conj(self.data[spec_ids[:0:-1]])
        profile = np.fft.ifft(spec).real.astype(np.float32, copy=False)
        return Profile(profile, tsamp=self.header.tsamp)

    def multiply(self, other: FourierSeries | ArrayLike) -> FourierSeries:
        """Multiply two Fourier series together.

        Parameters
        ----------
        other : FourierSeries | ArrayLike
            Fourier series to multiply with.

        Returns
        -------
        FourierSeries
            The product of the two Fourier series.
        """
        if not isinstance(other, FourierSeries):
            other = FourierSeries(other, self.header.new_header())
        return FourierSeries(self.data * other.data, self.header.new_header())

    def to_fft(self, basename: str | None = None) -> str:
        """Write Fourier series to file in presto ``.fft`` format.

        Parameters
        ----------
        basename : str, optional
            File basename for output ``.fft`` and ``.inf`` file, by default None.

        Returns
        -------
        str
            Output ``.fft`` file name.

        Notes
        -----
        Method also writes a corresponding .inf file from the header data.
        """
        if basename is None:
            basename = self.header.basename
        self.header.make_inf(outfile=f"{basename}.inf")
        out_filename = f"{basename}.fft"
        self.data.view(np.float32).tofile(out_filename)
        return out_filename

    def to_spec(self, filename: str | None = None) -> str:
        """Write Fourier series in sigproc format.

        Parameters
        ----------
        filename : str, optional
            Name of file to write to, by default ``basename.spec``.

        Returns
        -------
        str
            Output ``.spec`` file name.
        """
        if filename is None:
            filename = f"{self.header.basename}.spec"
        with self.header.prep_outfile(filename, nbits=32) as outfile:
            outfile.cwrite(self.data.view(np.float32))
        return filename

    @classmethod
    def from_fft(
        cls,
        fftfile: str | Path,
        inffile: str | Path | None = None,
    ) -> FourierSeries:
        """Read a Presto format ``.fft`` file.

        Parameters
        ----------
        fftfile : str | Path
            Name of the ``.fft`` file to read.
        inffile : str | Path, optional
            Name of the corresponding ``.inf`` file, by default None.

        Returns
        -------
        FourierSeries
            Fourier series object.

        Notes
        -----
        If ``inffile`` is None, then the associated .inf file must be in
        the same directory.
        """
        fftpath = validate_path(fftfile)
        if inffile is None:
            inffile = fftpath.with_suffix(".inf")
        data = np.fromfile(fftpath, dtype=np.float32)
        inf_hdr = Header.from_inffile(inffile)
        hdr_changes = {"filename": fftpath.as_posix()}
        return cls(data.view(np.complex64), inf_hdr.new_header(hdr_changes))

    @classmethod
    def from_spec(cls, filename: str | Path) -> FourierSeries:
        """Read a sigpyproc format ``.spec`` file.

        Parameters
        ----------
        filename : str | Path
            Name of the ``.spec`` file to read.

        Returns
        -------
        FourierSeries
            Fourier series object.

        Notes
        -----
        This is not setup to handle ``.spec`` files such as are
        created by Sigproc seek module. To do this would require
        a new header parser for that file format.
        """
        header = Header.from_sigproc(filename)
        data = np.fromfile(
            filename,
            dtype=np.float32,
            offset=header.stream_info.entries[0].hdrlen,
        )
        return cls(data.view(np.complex64), header)

    def view(self, **kwargs: Any) -> Any:  # noqa: ANN401
        """Open the interactive spectrum viewer for this Fourier series.

        Forms a power spectrum and launches
        :class:`~sigpyproc.viz.spectrum.SpectrumViewer`.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to :class:`~sigpyproc.viz.spectrum.SpectrumViewer`.

        Returns
        -------
        SpectrumViewer
            Interactive spectrum viewer instance.
        """
        spec = self.form_spec()
        return spec.view(**kwargs)

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
