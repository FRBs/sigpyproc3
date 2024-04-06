from __future__ import annotations
import pathlib
import numpy as np

from numpy import typing as npt

try:
    from pyfftw.interfaces import numpy_fft  # noqa: WPS433
except ModuleNotFoundError:
    from numpy import fft as numpy_fft  # noqa: WPS433


from sigpyproc import timeseries
from sigpyproc.header import Header
from sigpyproc.foldedcube import Profile
from sigpyproc.core import kernels


class PowerSpectrum(np.ndarray):
    """An array class to handle pulsar power spectrum.

    Parameters
    ----------
    input_array : :py:obj:`~numpy.typing.ArrayLike`
        1 dimensional array of shape (nsamples)
    header : :class:`~sigpyproc.header.Header`
        observational metadata

    Returns
    -------
    :py:obj:`~numpy.ndarray`
        1 dimensional power spectra with header

    Notes
    -----
    Data is converted to 32 bits regardless of original type.
    """

    def __new__(cls, input_array: npt.ArrayLike, header: Header) -> PowerSpectrum:
        obj = np.asarray(input_array).astype(np.float32, copy=False).view(cls)
        obj.header = header
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.header = getattr(obj, "header", None)

    def bin2freq(self, bin_num: int) -> float:
        """Return centre frequency of a given bin.

        Parameters
        ----------
        bin_num : int
            Bin number

        Returns
        -------
        float
            frequency of the given bin
        """
        return bin_num / self.header.tobs

    def bin2period(self, bin_num: int) -> float:
        """Return centre period of a given bin.

        Parameters
        ----------
        bin_num : int
            Bin number

        Returns
        -------
        float
            period of bin
        """
        return 1 / self.bin2freq(bin_num)

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
        sum_ar = self.copy()

        nfold1 = 0  # int(self.header.tsamp*2*self.size/maxperiod)
        folds = []
        for ii in range(nfolds):
            nharm = 2 ** (ii + 1)
            nfoldi = int(max(1, min(nharm * nfold1 - nharm // 2, self.size)))
            harm_ar = np.array(
                [
                    int(kk * ll / float(nharm))
                    for ll in range(nharm)
                    for kk in range(1, nharm, 2)
                ]
            ).astype("int32")

            facts_ar = np.array(
                [(kk * nfoldi + nharm // 2) / nharm for kk in range(1, nharm, 2)]
            ).astype("int32")

            kernels.sum_harms(self, sum_ar, harm_ar, facts_ar, nharm, self.size, nfoldi)

            new_header = self.header.new_header({"tsamp": self.header.tsamp * nharm})
            folds.append(PowerSpectrum(sum_ar, new_header))
        return folds


class FourierSeries(np.ndarray):
    """An array class to handle Fourier series with headers.

    Parameters
    ----------
    input_array : :py:obj:`~numpy.typing.ArrayLike`
        1 dimensional array of shape (nsamples)
    header : :class:`~sigpyproc.header.Header`
        observational metadata

    Returns
    -------
    :py:obj:`~numpy.ndarray`
        1 dimensional fourier series with header
    """

    def __new__(cls, input_array: npt.ArrayLike, header: Header) -> FourierSeries:
        obj = np.asarray(input_array).astype(np.complex64, copy=False).view(cls)
        obj.header = header
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.header = getattr(obj, "header", None)

    def ifft(self) -> timeseries.TimeSeries:
        """Perform 1-D complex to real inverse FFT.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            a time series
        """
        tim_ar = numpy_fft.irfft(self)
        return timeseries.TimeSeries(tim_ar, self.header.new_header())

    def form_spec(self, interpolated: bool = True) -> PowerSpectrum:
        """Form power spectrum.

        Parameters
        ----------
        interpolated : bool, optional
            flag to set nearest bin interpolation, by default True

        Returns
        -------
        :class:`~sigpyproc.fourierseries.PowerSpectrum`
            a power spectrum
        """
        spec_ar = kernels.form_spec(self.view(np.float32), interpolated=interpolated)
        return PowerSpectrum(spec_ar, self.header.new_header())

    def remove_rednoise(
        self, startwidth: int = 6, endwidth: int = 100, endfreq: float = 1.0
    ) -> FourierSeries:
        """Perform rednoise removal via Presto style method.

        Parameters
        ----------
        startwidth : int, optional
            size of initial array for median calculation, by default 6
        endwidth : int, optional
            size of largest array for median calculation, by default 100
        endfreq : float, optional
            remove rednoise up to this frequency, by default 1.0

        Returns
        -------
        :class:`~sigpyproc.fourierseries.FourierSeries`
            whitened fourier series
        """
        out_ar = kernels.remove_rednoise(
            self.view(np.float32), startwidth, endwidth, endfreq, self.header.tsamp
        )
        return FourierSeries(out_ar.view(np.complex64), self.header.new_header())

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
        harms = self[spec_ids]
        harm_ar = np.hstack((harms, np.conj(harms[1:][::-1])))
        return Profile(np.abs(numpy_fft.ifft(harm_ar)))

    def to_file(self, filename: str | None = None) -> str:
        """Write Fourier series to file in sigproc format.

        Parameters
        ----------
        filename : str, optional
            name of file to write to, by default ``basename.spec``

        Returns
        -------
        str
            output file name
        """
        if filename is None:
            filename = f"{self.header.basename}.spec"
        with self.header.prep_outfile(filename, nbits=32) as outfile:
            outfile.cwrite(self.view(np.float32))
        return filename

    def to_fftfile(self, basename: str | None = None) -> tuple[str, str]:
        """Write spectrum to file in presto ``.fft`` format.

        Parameters
        ----------
        basename : str, optional
            basename of ``.fft`` and ``.inf`` file to be written, by default None

        Returns
        -------
        tuple of str
            name of files written to
        """
        if basename is None:
            basename = self.header.basename
        self.header.make_inf(outfile=f"{basename}.inf")
        self.view(np.float32).tofile(f"{basename}.fft")
        return f"{basename}.fft", f"{basename}.inf"

    @classmethod
    def read_fft(cls, fftfile: str, inffile: str | None = None) -> FourierSeries:
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
            an array containing the whole file contents

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
            raise IOError("No corresponding .inf file found")
        data = np.fromfile(fftfile, dtype=np.float32)
        header = Header.from_inffile(inffile)
        header.filename = fftfile
        return cls(data.view(np.complex64), header)

    @classmethod
    def read_spec(cls, filename: str) -> FourierSeries:
        """Read a sigpyproc format ``.spec`` file.

        Parameters
        ----------
        filename : str
            the name of the ``.spec`` file to read

        Returns
        -------
        :class:`~sigpyproc.fourierseries.FourierSeries`
            an array containing the whole file contents

        Notes
        -----
        This is not setup to handle ``.spec`` files such as are
        created by Sigprocs seek module. To do this would require
        a new header parser for that file format.
        """
        header = Header.from_sigproc(filename)
        data = np.fromfile(
            filename, dtype=np.float32, offset=header.stream_info.entries[0].hdrlen
        )
        return cls(data.view(np.complex64), header)
