from __future__ import annotations
import numpy as np

from typing import Optional

from sigpyproc.Utils import rollArray
from sigpyproc.FoldedData import FoldedData
from sigpyproc.TimeSeries import TimeSeries
from sigpyproc import libSigPyProc as lib


class Filterbank(object):
    """Class exporting methods for the manipulation of frequency-major order pulsar data.

    Notes
    -----
    The Filterbank class should never be instantiated directly. Instead it
    should be inherited by data reading classes.
    """

    def __init__(self) -> None:
        self.chan_means = None
        self.chan_vars = None
        self.chan_stdevs = None
        self.chan_skews = None
        self.chan_kurts = None
        self.chan_maxima = None
        self.chan_minima = None

    @property
    def omp_threads(self):
        """Number of omp threads (`int`)."""
        return lib.get_omp_threads()

    @omp_threads.setter
    def omp_threads(self, nthreads: Optional[int] = None):
        """Set the number of threads available to OpenMP.

        Parameters
        ----------
        nthreads : int, optional
            number of threads to use, by default 4
        """
        if nthreads is None:
            nthreads = 4
        lib.set_omp_threads(nthreads)

    def collapse(
        self, gulp: int = 512, start: int = 0, nsamps: int = None, **kwargs
    ) -> TimeSeries:
        """Sum across all frequencies for each time sample.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 512
        start : int, optional
            start sample, by default 0
        nsamps : int, optional
            number of samples to read, by default all

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            A zero-DM time series
        """
        if nsamps is None:
            size = self.header.nsamples - start
        else:
            size = nsamps
        timar = np.zeros(size, dtype="float32")
        for nsamp, ii, data in self.readPlan(
            gulp,
            start=start,
            nsamps=nsamps,
            **kwargs,
        ):
            lib.getTim(data, timar, self.header.nchans, nsamp, ii * gulp)
        return TimeSeries(timar, self.header.newHeader({"nchans": 1, "refdm": 0}))

    def invert_freq(
        self,
        gulp: int = 512,
        start: int = 0,
        nsamps: int = None,
        filename: str = None,
        back_compatible: bool = True,
        **kwargs,
    ) -> str:
        """Invert the frequency ordering of the data and write new data to a new file.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 512
        start : int, optional
            start sample, by default 0
        nsamps : int, optional
            number of samples to read, by default all
        filename : str, optional
            name of output file, by default ``basename_inverted.fil``
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        str
            name of output file
        """
        if filename is None:
            filename = f"{self.header.basename}_inverted.fil"

        if nsamps is None:
            size = self.header.nsamples - start
        else:
            size = nsamps
        out_ar = np.empty(size * self.header.nchans, dtype=self.header.dtype)
        sign = 1 if self.header.foff >= 0 else -1

        changes = {
            "fch1": self.header.fch1 + sign * (self.header.nchans - 1) * self.header.foff,
            "foff": self.header.foff * sign * (-1.0),
        }
        # NB bandwidth is +ive by default
        out_file = self.header.prepOutfile(
            filename, changes, nbits=self.header.nbits, back_compatible=back_compatible
        )
        for nsamp, _ii, data in self.readPlan(
            gulp,
            start=start,
            nsamps=nsamps,
            **kwargs,
        ):
            lib.invertFreq(data, out_ar, self.header.nchans, nsamp)
            out_file.cwrite(out_ar[: nsamp * self.header.nchans])
        out_file.close()
        return out_file.name

    def bandpass(self, gulp=512, **kwargs):
        """Average across each time sample for all frequencies.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 512

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            the bandpass of the data
        """
        bpass_ar = np.zeros(self.header.nchans, dtype="float64")
        num_samples = 0
        for nsamps, _ii, data in self.readPlan(gulp, **kwargs):
            lib.getBpass(data, bpass_ar, self.header.nchans, nsamps)
            num_samples += nsamps
        bpass_ar /= num_samples
        return TimeSeries(bpass_ar, self.header.newHeader({"nchans": 1}))

    def dedisperse(self, dm, gulp=10000, **kwargs):
        """Dedisperse the data to a time series.

        Parameters
        ----------
        dm : float
            dispersion measure to dedisperse to
        gulp : int, optional
            number of samples in each read, by default 10000

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            a dedispersed time series

        Notes
        -----
        If gulp < maximum dispersion delay, gulp is taken to be twice the
        maximum dispersion delay.
        """
        chan_delays = self.header.getDMdelays(dm)
        max_delay = int(chan_delays.max())
        gulp = max(2 * max_delay, gulp)
        tim_len = self.header.nsamples - max_delay
        tim_ar = np.zeros(tim_len, dtype="float32")
        for nsamps, ii, data in self.readPlan(gulp, skipback=max_delay, **kwargs):
            lib.dedisperse(
                data,
                tim_ar,
                chan_delays,
                max_delay,
                self.header.nchans,
                nsamps,
                ii * (gulp - max_delay),
            )
        return TimeSeries(tim_ar, self.header.newHeader({"nchans": 1, "refdm": dm}))

    def subband(self, dm, nsub, filename=None, gulp=10000, **kwargs):
        """Produce a set of dedispersed subbands from the data.

        Parameters
        ----------
        dm : float
            the DM of the subbands
        nsub : int
            the number of subbands to produce
        filename : str, optional
            output file name of subbands, by default ``basename_DM.subbands``
        gulp : int, optional
            number of samples in each read, by default 10000

        Returns
        -------
        str
            name of output subbands file
        """
        subfactor = self.header.nchans // nsub
        chan_delays = self.header.getDMdelays(dm)
        max_delay = int(chan_delays.max())
        gulp = max(2 * max_delay, gulp)
        # must be memset to zero in c code
        out_ar = np.empty((gulp - max_delay) * nsub, dtype="float32")
        new_foff = self.header.foff * self.header.nchans // nsub
        new_fch1 = self.header.ftop - new_foff / 2
        chan_to_sub = np.arange(self.header.nchans, dtype="int32") // subfactor
        changes = {
            "fch1": new_fch1,
            "foff": new_foff,
            "refdm": dm,
            "nchans": nsub,
            "nbits": 32,
        }
        if filename is None:
            filename = f"{self.header.basename}_DM{dm:06.2f}.subbands"
        out_file = self.header.prepOutfile(
            filename, changes, nbits=32, back_compatible=True
        )

        for nsamps, _ii, data in self.readPlan(gulp, skipback=max_delay, **kwargs):
            lib.subband(
                data,
                out_ar,
                chan_delays,
                chan_to_sub,
                max_delay,
                self.header.nchan,
                nsub,
                nsamps,
            )
            out_file.cwrite(out_ar[: (nsamps - max_delay) * nsub])
        return filename

    def upTo8bit(self, filename=None, gulp=512, back_compatible=True, **kwargs):
        """Convert 1-,2- or 4-bit data to 8-bit data and write to file.

        Parameters
        ----------
        filename : str, optional
            name of file to write to, by default ``basename_8bit.fil``
        gulp : int, optional
            number of samples in each read, by default 512
        back_compatible : bool, optional
            sigproc compatibility flag, by default True

        Returns
        -------
        str
            name of output file
        """
        if filename is None:
            filename = f"{self.header.basename}_8bit.fil"

        out_file = self.header.prepOutfile(
            filename, {"nbits": 8}, nbits=8, back_compatible=back_compatible
        )
        for _nsamps, _ii, data in self.readPlan(gulp, **kwargs):
            out_file.cwrite(data)
        return out_file.name

    def applyChannelMask(
        self, chanmask, outfilename=None, gulp=512, back_compatible=True, **kwargs
    ):
        """Set the data in the given channels to zero.

        Parameters
        ----------
        chanmask : list
            binary channel mask (0 for bad channel, 1 for good)
        outfilename : str, optional
            name of the output filterbank file, by default ``basename_masked.fil``
        gulp : int, optional
            number of samples in each read, by default 512
        back_compatible : bool, optional
            sigproc compatibility flag, by default True

        Returns
        -------
        str
            name of output file
        """
        if outfilename is None:
            outfilename = f"{self.header.basename}_masked.fil"
        mask = np.array(chanmask).astype("ubyte")
        out_file = self.header.prepOutfile(outfilename, back_compatible=back_compatible)
        for nsamps, _ii, data in self.readPlan(gulp, **kwargs):
            lib.maskChannels(data, mask, self.header.nchans, nsamps)
            out_file.cwrite(data)
        return out_file.name

    def downsample(
        self,
        tfactor=1,
        ffactor=1,
        gulp=512,
        filename=None,
        back_compatible=True,
        **kwargs,
    ):
        """Downsample data in time and/or frequency and write to file.

        Parameters
        ----------
        tfactor : int, optional
            factor by which to downsample in time, by default 1
        ffactor : int, optional
            factor by which to downsample in frequency, by default 1
        gulp : int, optional
            number of samples in each read, by default 512
        filename : str, optional
            name of file to write to, by default ``basename_tfactor_ffactor.fil``
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        str
            output file name

        Raises
        ------
        ValueError
            If number of channels is not divisible by `ffactor`.
        """
        if filename is None:
            filename = f"{self.header.basename}_f{ffactor:d}_t{tfactor:d}.fil"
        if self.header.nchans % ffactor != 0:
            raise ValueError("Bad frequency factor given")

        # Gulp must be a multiple of tfactor
        gulp = int(np.ceil(gulp / tfactor) * tfactor)

        out_file = self.header.prepOutfile(
            filename,
            {
                "tsamp": self.header.tsamp * tfactor,
                "nchans": self.header.nchans // ffactor,
                "foff": self.header.foff * ffactor,
            },
            back_compatible=back_compatible,
        )

        write_ar = np.zeros(
            gulp * self.header.nchans // ffactor // tfactor,
            dtype=self.header.dtype,
        )
        for nsamps, _ii, data in self.readPlan(gulp, **kwargs):
            lib.downsample(data, write_ar, tfactor, ffactor, self.header.nchans, nsamps)
            out_file.cwrite(write_ar[: nsamps * self.header.nchans // ffactor // tfactor])
        return out_file.name

    def fold(
        self, period, dm, accel=0, nbins=50, nints=32, nbands=32, gulp=10000, **kwargs
    ):
        """Fold data into discrete phase, subintegration and subband bins.

        Parameters
        ----------
        period : float
            period in seconds to fold with
        dm : float
            dispersion measure to dedisperse to
        accel : float, optional
            acceleration in m/s/s to fold with, by default 0
        nbins : int, optional
            number of phase bins in output, by default 50
        nints : int, optional
            number of subintegrations in output, by default 32
        nbands : int, optional
            number of subbands in output, by default 32
        gulp : int, optional
            number of samples in each read, by default 10000

        Returns
        -------
        :class:`~sigpyproc.FoldedData.FoldedData`
            3 dimensional data cube

        Raises
        ------
        ValueError
            If `nbands * nints * nbins` is too large

        Notes
        -----
        If gulp < maximum dispersion delay, gulp is taken to be twice the
        maximum dispersion delay.
        """
        if np.modf(period / self.header.tsamp)[0] < 0.001:
            print("WARNING: Foldng interval is an integer multiple of the sampling time")
        if nbins > period / self.header.tsamp:
            print("WARNING: Number of phase bins is greater than period/sampling time")
        if (self.header.nsamples * self.header.nchans) // (nbands * nints * nbins) < 10:
            raise ValueError("nbands x nints x nbins is too large.")
        nbands = min(nbands, self.header.nchans)
        chan_delays = self.header.getDMdelays(dm)
        max_delay = int(chan_delays.max())
        gulp = max(2 * max_delay, gulp)
        fold_ar = np.zeros(nbins * nints * nbands, dtype="float32")
        count_ar = np.zeros(nbins * nints * nbands, dtype="int32")
        for nsamps, ii, data in self.readPlan(gulp, skipback=max_delay, **kwargs):
            lib.foldFil(
                data,
                fold_ar,
                count_ar,
                chan_delays,
                max_delay,
                self.header.tsamp,
                period,
                accel,
                self.header.nsamples,
                nsamps,
                self.header.nchans,
                nbins,
                nints,
                nbands,
                ii * (gulp - max_delay),
            )
        fold_ar /= count_ar
        fold_ar = fold_ar.reshape(nints, nbands, nbins)
        return FoldedData(fold_ar, self.header.newHeader(), period, dm, accel)

    def getChan(self, chan, gulp=512, **kwargs):
        """Retrieve a single frequency channel from the data.

        Parameters
        ----------
        chan : int
            channel to retrieve (0 is the highest frequency channel)
        gulp : int, optional
            number of samples in each read, by default 512

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            selected channel as a time series

        Raises
        ------
        ValueError
            If chan is out of range (chan < 0 or chan > total channels).
        """
        if chan >= self.header.nchans or chan < 0:
            raise ValueError("Selected channel out of range.")
        tim_ar = np.empty(self.header.nsamples, dtype="float32")
        for nsamps, ii, data in self.readPlan(gulp, **kwargs):
            lib.getChan(data, tim_ar, chan, self.header.nchans, nsamps, ii * gulp)
        return TimeSeries(
            tim_ar, self.header.newHeader({"channel": chan, "refdm": 0.0, "nchans": 1})
        )

    def split(
        self, start, nsamps, filename=None, gulp=1024, back_compatible=True, **kwargs
    ):
        """Split data in time.

        Parameters
        ----------
        start : int
            start sample of split
        nsamps : int
            number of samples in split
        filename : str, optional
            name of output file, by default ``basename_start_start+nsamps.fil``
        gulp : int, optional
            number of samples in each read, by default 1024
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        str
            name of new file
        """
        if filename is None:
            filename = f"{self.header.basename}_{start:d}_{start+nsamps:d}.fil"
        new_tstart = self.header.tstart + ((self.header.tsamp * start) / 86400.0)
        out_file = self.header.prepOutfile(
            filename, updates={"tstart": new_tstart}, nbits=self.header.nbits
        )
        for _count, _ii, data in self.readPlan(
            gulp,
            start=start,
            nsamps=nsamps,
            **kwargs,
        ):
            out_file.cwrite(data)
        out_file.close()
        return out_file.name

    def splitToChans(self, gulp=1024, back_compatible=True, **kwargs):
        """Split the data into component channels and write each to file.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 1024
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        list of str
            names of all files written to disk

        Notes
        -----
        Time series are written to disk with names based on channel number.
        """
        tim_ar = np.empty([self.header.nchans, gulp], dtype="float32")
        out_files = [
            self.header.prepOutfile(
                f"{self.header.basename}_chan{ii:04d}.tim",
                {"nchans": 1, "nbits": 32, "data_type": 2},
                back_compatible=back_compatible,
                nbits=32,
            )
            for ii in range(self.header.nchans)
        ]
        for nsamps, _ii, data in self.readPlan(gulp, **kwargs):
            lib.splitToChans(data, tim_ar, self.header.nchans, nsamps, gulp)
            for ifile, out_file in enumerate(out_files):
                out_file.cwrite(tim_ar[ifile][:nsamps])

        for out_file in out_files:
            out_file.close()

        return [out_file.name for out_file in out_files]

    def splitToBands(
        self, chanpersub, chanstart=0, gulp=1024, back_compatible=True, **kwargs
    ):
        """Split the data into component Sub-bands and write to disk.

        Parameters
        ----------
        chanpersub : int
            number of channels in each sub-band
        chanstart : int, optional
            start channel of split, by default 0
        gulp : int, optional
            number of samples in each read, by default 1024
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        list of str
            names of all files written to disk

        Notes
        -----
        Filterbanks are written to disk with names based on sub-band number.
        """
        nsub = (self.header.nchans - chanstart) // chanpersub
        fstart = self.header.fch1 + chanstart * self.header.foff

        out_files = [
            self.header.prepOutfile(
                f"{self.header.basename}_sub{ii:02d}.fil",
                {
                    "nchans": chanpersub,
                    "fch1": fstart + ii * chanpersub * self.header.foff,
                },
                back_compatible=back_compatible,
                nbits=self.header.nbits,
            )
            for ii in range(nsub)
        ]

        for nsamps, _ii, data in self.readPlan(gulp, **kwargs):
            for iband, out_band in enumerate(out_files):
                data = data.reshape(nsamps, self.header.nchans)
                subband_ar = data[
                    :,
                    chanstart + chanpersub * iband : chanstart + chanpersub * (iband + 1),
                ]
                out_band.cwrite(subband_ar.ravel())

        for out_file in out_files:
            out_file.close()

        return [out_file.name for out_file in out_files]

    def getStats(self, gulp=512, **kwargs):
        """Retrieve channelwise statistics of data.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 512

        Function creates following instance attributes:

           * :attr:`chan_means`: the mean value of each channel
           * :attr:`chan_vars`: the variance of each channel
           * :attr:`chan_stdevs`: the standard deviation of each channel
           * :attr:`chan_skews`: the skewness of each channel
           * :attr:`chan_kurts`: the kurtosis of each channel
           * :attr:`chan_max`: the maximum value of each channel
           * :attr:`chan_min`: the minimum value of each channel
        """
        maxima_ar = np.zeros(self.header.nchans, dtype="float32")
        minima_ar = np.zeros(self.header.nchans, dtype="float32")
        count_ar = np.zeros(self.header.nchans, dtype="int64")
        M1_ar = np.zeros(self.header.nchans, dtype="float32")
        M2_ar = np.zeros(self.header.nchans, dtype="float32")
        M3_ar = np.zeros(self.header.nchans, dtype="float32")
        M4_ar = np.zeros(self.header.nchans, dtype="float32")
        for nsamps, ii, data in self.readPlan(gulp, **kwargs):
            lib.getStats(
                data,
                M1_ar,
                M2_ar,
                M3_ar,
                M4_ar,
                maxima_ar,
                minima_ar,
                count_ar,
                self.header.nchans,
                nsamps,
                ii,
            )

        means_ar = M1_ar
        var_ar = M2_ar / self.header.nsamples
        stdev_ar = np.sqrt(var_ar)

        M2_ar[M2_ar == 0] = np.nan
        skew_ar = M3_ar / np.power(M2_ar, 1.5) * np.sqrt(self.header.nsamples)
        kurt_ar = M4_ar / np.power(M2_ar, 2.0) * self.header.nsamples - 3.0

        stdev_ar[np.isnan(stdev_ar)] = 0
        skew_ar[np.isnan(skew_ar)] = 0
        kurt_ar[np.isnan(kurt_ar)] = -3.0

        self.chan_means = means_ar.astype("float32")
        self.chan_vars = var_ar.astype("float32")
        self.chan_stdevs = stdev_ar.astype("float32")
        self.chan_skews = skew_ar.astype("float32")
        self.chan_kurts = kurt_ar.astype("float32")
        self.chan_maxima = maxima_ar
        self.chan_minima = minima_ar

    def removeBandpass(self, gulp=512, filename=None, back_compatible=True, **kwargs):
        """Remove the bandpass from the data and write new data to a new file.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 512
        filename : str, optional
            name of output file, by default ``basename_bpcorr.fil``
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        str
            name of output file

        Raises
        ------
        TypeError
            If `nbits` of input file is < 8.
        """
        if self.header.nbits < 8:
            raise TypeError(f"{self.header.nbits}-bit filterbank not supported yet!")
        if filename is None:
            filename = f"{self.header.basename}_bpcorr.fil"

        if self.chan_stdevs is None:
            self.getStats(gulp=gulp, **kwargs)

        out_ar = np.empty(
            self.header.nsamples * self.header.nchans,
            dtype=self.header.dtype,
        )
        out_file = self.header.prepOutfile(
            filename, nbits=self.header.nbits, back_compatible=back_compatible
        )
        for nsamps, _ii, data in self.readPlan(gulp, **kwargs):
            lib.removeBandpass(
                data,
                out_ar,
                self.chan_means,
                self.chan_stdevs,
                self.header.nchans,
                nsamps,
            )
            out_file.cwrite(out_ar[: nsamps * self.header.nchans])
        out_file.close()
        return out_file.name

    def removeZeroDM(self, gulp=512, filename=None, back_compatible=True, **kwargs):
        """Remove the channel-weighted zero-DM from the data and write to disk.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 512
        filename : str, optional
            name of output file , by default ``basename_noZeroDM.fil``
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        str
            name of output file

        Notes
        -----
        Based on Presto implementation of Eatough, Keane & Lyne 2009
        """
        if filename is None:
            filename = f"{self.header.basename}_noZeroDM.fil"

        bpass = self.bandpass(gulp=gulp, **kwargs)
        chanwts = bpass / bpass.sum()
        out_ar = np.empty(
            self.header.nsamples * self.header.nchans,
            dtype=self.header.dtype,
        )
        out_file = self.header.prepOutfile(
            filename, nbits=self.header.nbits, back_compatible=back_compatible
        )
        for nsamps, _ii, data in self.readPlan(gulp, **kwargs):
            lib.removeZeroDM(data, out_ar, bpass, chanwts, self.header.nchans, nsamps)
            out_file.cwrite(out_ar[: nsamps * self.header.nchans])
        out_file.close()
        return out_file.name


class FilterbankBlock(np.ndarray):
    """Class to handle a discrete block of data in time-major order.

    Parameters
    ----------
    input_array : :py:obj:`numpy.ndarray`
        2 dimensional array of shape (nchans, nsamples)
    header : :class:`~sigpyproc.Header.Header`
        observational metadata

    Returns
    -------
    :py:obj:`numpy.ndarray`
        2 dimensional array of shape (nchans, nsamples) with header metadata

    Notes
    -----
    Data is converted to 32 bits regardless of original type.
    """

    def __new__(cls, input_array, header):
        obj = input_array.astype("float32").view(cls)
        obj.header = header
        obj.dm = 0.0
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        if hasattr(obj, "header"):
            self.header = obj.header
        self.dm = getattr(obj, "dm", 0.0)

    def downsample(self, tfactor=1, ffactor=1):
        """Downsample data block in frequency and/or time.

        Parameters
        ----------
        tfactor : int, optional
            factor by which to downsample in time, by default 1
        ffactor : int, optional
            factor by which to downsample in frequency, by default 1

        Returns
        -------
        :class:`~sigpyproc.Filterbank.FilterbankBlock`
            2 dimensional array of downsampled data

        Raises
        ------
        ValueError
            If number of channels is not divisible by `ffactor`.
        """
        if self.shape[0] % ffactor != 0:
            raise ValueError("Bad frequency factor given")
        newnsamps = self.shape[1] - self.shape[1] % tfactor
        new_ar = np.empty(
            newnsamps * self.shape[0] // ffactor // tfactor,
            dtype="float32",
        )
        ar = self.transpose().ravel().copy()
        lib.downsample(ar, new_ar, tfactor, ffactor, self.shape[0], newnsamps)
        new_ar = new_ar.reshape(
            newnsamps // tfactor, self.shape[0] // ffactor
        ).transpose()
        new_tsamp = self.header.tsamp * tfactor
        new_nchans = self.header.nchans // ffactor
        new_header = self.header.newHeader({"tsamp": new_tsamp, "nchans": new_nchans})
        return FilterbankBlock(new_ar, new_header)

    def toFile(self, filename=None, back_compatible=True):
        """Write the data to file.

        Parameters
        ----------
        filename : str, optional
            name of the output file, by default ``basename_split_start_to_end.fil``
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        str
            name of output file
        """
        if filename is None:
            mjd_after = self.header.mjdAfterNsamps(self.shape[1])
            filename = (
                f"{self.header.basename}_{self.header.tstart:d}_" f"to_{mjd_after:d}.fil"
            )
        new_header = {"nbits": 32}
        out_file = self.header.prepOutfile(
            filename, new_header, nbits=32, back_compatible=back_compatible
        )
        out_file.cwrite(self.transpose().ravel())
        return filename

    def normalise(self):
        """Divide each frequency channel by its average.

        Returns
        -------
        :class:`~sigpyproc.Filterbank.FilterbankBlock`
            normalised version of the data
        """
        return self / self.mean(axis=1).reshape(self.shape[0], 1)

    def get_tim(self):
        """Sum across all frequencies for each time sample

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            a timeseries
        """
        return self.sum(axis=0)

    def get_bandpass(self):
        """Average across each time sample for all frequencies.

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            the bandpass of the data
        """
        return self.sum(axis=1)

    def dedisperse(self, dm, only_valid_samples=False):
        """Dedisperse the block.

        Parameters
        ----------
        dm : float
            dm to dedisperse to
        only_valid_samples : bool, optional
            return a FilterbankBlock with only time samples that
            contain the full bandwidth, by default False

        Returns
        -------
        :class:`~sigpyproc.Filterbank.FilterbankBlock`
            a dedispersed version of the block

        Raises
        ------
        ValueError
            If there are not enough time samples to dedisperse.

        Notes
        -----
        Frequency dependent delays are applied as rotations to each
        channel in the block.
        """
        delays = self.header.getDMdelays(dm)
        if only_valid_samples:
            if self.shape[1] < delays[-1]:
                raise ValueError(
                    f"Insufficient time samples to dedisperse to {dm} (requires "
                    f"at least {delays[-1]} samples, given {self.shape[1]})."
                )
            new_ar = FilterbankBlock(
                np.zeros(
                    (self.header.nchans, self.shape[1] - delays[-1]), dtype=self.dtype
                ),
                self.header,
            )
            end_samples = delays + new_ar.shape[1]

            slices = [
                np.arange(delay, end_sample)
                for delay, end_sample in zip(delays, end_samples)
            ]
            for idx, timeSlice in enumerate(slices):
                new_ar[idx, :] = self[idx, timeSlice]
        else:
            new_ar = self.copy()
            for ii in range(self.shape[0]):
                new_ar[ii] = rollArray(self[ii], delays[ii] % self.shape[1], 0)

        new_ar.dm = dm
        return new_ar
