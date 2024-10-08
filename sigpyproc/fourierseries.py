from __future__ import annotations

import pathlib

import numpy as np
from numpy import typing as npt

try:
    from pyfftw.interfaces import numpy_fft
except ModuleNotFoundError:
    from numpy import fft as numpy_fft


from sigpyproc import timeseries
from sigpyproc.core import kernels
from sigpyproc.foldedcube import Profile
from sigpyproc.header import Header


class PowerSpectrum:
    """An array class to handle Fourier Power spectrum.

    Parameters
    ----------
    data : :py:obj:`~numpy.typing.ArrayLike`
        1 dimensional power spectrum
    header : :class:`~sigpyproc.header.Header`
        header object containing metadata

    Returns
    -------
    :py:obj:`~numpy.ndarray`
        1 dimensional power spectrum with header metadata

    Notes
    -----
    Data is converted to 32 bits regardless of original type.
    """

    def __init__(self, data: npt.ArrayLike, hdr: Header) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._hdr = hdr
        self._check_input()

    @property
    def header(self) -> Header:
        """Observational metadata."""
        return self._hdr

    @property
    def data(self) -> npt.NDArray[np.float32]:
        """Power spectrum data."""
        return self._data

    def bin2freq(self, r: int) -> float:
        """Return centre frequency of a given bin.

        Parameters
        ----------
        r : int
            Fourier bin number

        Returns
        -------
        float
            frequency of the given bin
        """
        if r < 0 or r >= self.data.size:
            msg = f"Fourier bin number {r} out of range"
            raise ValueError(msg)
        return r / self.header.tobs

    def bin2period(self, r: int) -> float:
        """Return centre period of a given bin.

        Parameters
        ----------
        r : int
            Fourier bin number

        Returns
        -------
        float
            period of the given bin
        """
        return 1 / self.bin2freq(r)

    def freq2bin(self, freq: float) -> int:
        """Return nearest bin to a given frequency.

        Parameters
        ----------
        freq : float
            frequency

        Returns
        -------
        int
            nearest bin to frequency
        """
        return round(freq * self.header.tobs)

    def period2bin(self, period: float) -> int:
        """Return nearest bin to a given periodicity.

        Parameters
        ----------
        period : float
            periodicity

        Returns
        -------
        int
            nearest bin to period
        """
        return self.freq2bin(1 / period)

    def harmonic_fold(self, nfolds: int = 1) -> list[PowerSpectrum]:
        """Perform Lyne-Ashworth harmonic folding of the power spectrum.

        Parameters
        ----------
        nfolds : int, optional
            number of harmonic folds to perform, by default 1

        Returns
        -------
        list of :class:`~sigpyproc.fourierseries.PowerSpectrum`
            A list of folded spectra where the i :sup:`th` element
            is the spectrum folded i times.
        """
        sum_arr = kernels.sum_harmonics(self.data, nfolds)
        nharms = 2 ** np.arange(1, nfolds + 1)
        folds = []
        for ii, nharm in enumerate(nharms):
            new_header = self.header.new_header({"tsamp": self.header.tsamp * nharm})
            folds.append(PowerSpectrum(sum_arr[ii], new_header))
        return folds

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


class FourierSeries:
    """An array class to handle Fourier series.

    Parameters
    ----------
    data : :py:obj:`~numpy.typing.ArrayLike`
        1 dimensional fourier series
    header : :class:`~sigpyproc.header.Header`
        header object containing metadata

    Returns
    -------
    :py:obj:`~numpy.ndarray`
        1 dimensional fourier series with header metadata

    Notes
    -----
    Data is converted to 64-bit complex regardless of original type.

    """

    def __init__(self, data: npt.ArrayLike, hdr: Header) -> None:
        self._data = np.asarray(data, dtype=np.complex64)
        self._hdr = hdr
        self._check_input()

    @property
    def header(self) -> Header:
        """Observational metadata."""
        return self._hdr

    @property
    def data(self) -> npt.NDArray[np.complex64]:
        """Fourier series data."""
        return self._data

    @property
    def binwidth(self) -> float:
        """Width of each frequency bin."""
        return 1 / self.header.tobs

    def ifft(self) -> timeseries.TimeSeries:
        """Perform 1-D complex to real inverse FFT.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            a time series
        """
        tim_ar = numpy_fft.irfft(self.data)
        return timeseries.TimeSeries(tim_ar, self.header.new_header())

    def form_spec(self, *, interpolate: bool = False) -> PowerSpectrum:
        """Form power spectrum.

        Parameters
        ----------
        interpolated : bool, optional
            interpolate the power spectrum using nearest bins, by default False

        Returns
        -------
        :class:`~sigpyproc.fourierseries.PowerSpectrum`
            The power spectrum
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
        """Perform rednoise removal via Presto style method.

        Parameters
        ----------
        start_width : int, optional
            Initial window size in the range (2, 50), by default 6
        end_width : int, optional
            Final window size in the range (50, 500), by default 100
        end_freq : float, optional
            The highest frequency where the windowing increases in
            the range (0.1, 10), by default 6.0

        Returns
        -------
        :class:`~sigpyproc.fourierseries.FourierSeries`
            whitened fourier series
        """
        end_freq_bin = int(round(end_freq / self.binwidth))
        out_ar = kernels.fs_running_median(
            self.data,
            start_width,
            end_width,
            end_freq_bin,
        )
        return FourierSeries(out_ar, self.header.new_header())

    def recon_prof(self, freq: float, nharms: int = 32) -> Profile:
        """Reconstruct the time domain pulse profile from a signal and its harmonics.

        Parameters
        ----------
        freq : float
            frequency of signal to reconstruct
        nharms : int, optional
            number of harmonics to use in reconstruction, by default 32

        Returns
        -------
        :class:`~sigpyproc.foldedcube.Profile`
            a pulse profile
        """
        freq_bin = round(freq * self.header.tobs)
        spec_ids = np.arange(1, nharms + 1) * 2 * freq_bin
        harms = self.data[spec_ids]
        harm_ar = np.hstack((harms, np.conj(harms[1:][::-1])))
        return Profile(np.abs(numpy_fft.ifft(harm_ar)), tsamp=self.header.tsamp)

    def multiply(self, other: FourierSeries | npt.ArrayLike) -> FourierSeries:
        """Multiply two Fourier series together.

        Parameters
        ----------
        other : FourierSeries or :py:obj:`~numpy.typing.ArrayLike`
            Fourier series to multiply with

        Returns
        -------
        :class:`~sigpyproc.fourierseries.FourierSeries`
            The product of the two Fourier series
        """
        if not isinstance(other, FourierSeries):
            other = FourierSeries(other, self.header.new_header())
        return FourierSeries(self.data * other.data, self.header.new_header())

    def to_spec(self, filename: str | None = None) -> str:
        """Write Fourier series to file in sigproc format.

        Parameters
        ----------
        filename : str, optional
            name of file to write to, by default ``basename.spec``

        Returns
        -------
        str
            output ``.spec`` file name
        """
        if filename is None:
            filename = f"{self.header.basename}.spec"
        with self.header.prep_outfile(filename, nbits=32) as outfile:
            outfile.cwrite(self.data.view(np.float32))
        return filename

    def to_file(self, basename: str | None = None) -> str:
        """Write Fourier series to file in presto ``.fft`` format.

        Parameters
        ----------
        basename : str, optional
            file basename for output ``.fft`` and ``.inf`` file, by default None

        Returns
        -------
        str
            output ``.fft`` file name

        Notes
        -----
        Method also writes a corresponding .inf file from the header data

        """
        if basename is None:
            basename = self.header.basename
        self.header.make_inf(outfile=f"{basename}.inf")
        out_filename = f"{basename}.fft"
        self.data.view(np.float32).tofile(out_filename)
        return out_filename

    @classmethod
    def from_file(cls, fftfile: str, inffile: str | None = None) -> FourierSeries:
        """Read a presto format ``.fft`` file.

        Parameters
        ----------
        fftfile : str
            the name of the ``.fft`` file to read
        inffile : str, optional
            the name of the corresponding ``.inf`` file, by default None

        Returns
        -------
        :class:`~sigpyproc.fourierseries.FourierSeries`
            a new fourier series object

        Raises
        ------
        IOError
            If no ``.inf`` file found in the same directory of ``.fft`` file.

        Notes
        -----
        If inf is None, then the associated .inf file must be in the same directory.
        """
        fftpath = pathlib.Path(fftfile).resolve()
        if inffile is None:
            inffile = fftpath.with_suffix(".inf").as_posix()
        if not pathlib.Path(inffile).is_file():
            msg = "No corresponding .inf file found"
            raise FileNotFoundError(msg)
        data = np.fromfile(fftfile, dtype=np.float32)
        header = Header.from_inffile(inffile)
        header.filename = fftfile
        return cls(data.view(np.complex64), header)

    @classmethod
    def from_spec(cls, filename: str) -> FourierSeries:
        """Read a sigpyproc format ``.spec`` file.

        Parameters
        ----------
        filename : str
            the name of the ``.spec`` file to read

        Returns
        -------
        :class:`~sigpyproc.fourierseries.FourierSeries`
            a new fourier series object

        Notes
        -----
        This is not setup to handle ``.spec`` files such as are
        created by Sigprocs seek module. To do this would require
        a new header parser for that file format.
        """
        header = Header.from_sigproc(filename)
        data = np.fromfile(
            filename,
            dtype=np.float32,
            offset=header.stream_info.entries[0].hdrlen,
        )
        return cls(data.view(np.complex64), header)

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
