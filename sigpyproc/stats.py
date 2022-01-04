import numpy as np
from numba import njit, prange, types


@njit(
    "void(f4[:], f4[:], f4[:], f4[:], f4[:], i4[:], i4, i4, i4)",
    cache=True,
    parallel=True,
    locals={"val": types.f8},
)
def compute_online_moments_basic(
    array, m1, m2, maxbuffer, minbuffer, count, nchans, nsamps, startflag
):
    if startflag == 0:
        for ii in range(nchans):
            maxbuffer[ii] = array[ii]
            minbuffer[ii] = array[ii]

    for ichan in prange(nchans):
        for isamp in range(nsamps):
            val = array[isamp * nchans + ichan]
            count[ichan] += 1
            nn = count[ichan]

            delta = val - m1[ichan]
            delta_n = delta / nn
            m1[ichan] += delta_n
            m2[ichan] += delta * delta_n * (nn - 1)

            maxbuffer[ichan] = max(maxbuffer[ichan], val)
            minbuffer[ichan] = min(minbuffer[ichan], val)


@njit(
    "void(f4[:], f4[:], f4[:], f4[:], f4[:], f4[:], f4[:], i4[:], i4, i4, i4)",
    cache=True,
    parallel=True,
    locals={"val": types.f8},
)
def compute_online_moments(
    array, m1, m2, m3, m4, maxbuffer, minbuffer, count, nchans, nsamps, startflag
):
    """Computing central moments in one pass through the data.

    The algorithm is numerically stable and accurate.
    Reference:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    https://www.johndcook.com/blog/skewness_kurtosis/
    https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf
    """
    if startflag == 0:
        for ii in range(nchans):
            maxbuffer[ii] = array[ii]
            minbuffer[ii] = array[ii]

    for ichan in prange(nchans):
        for isamp in range(nsamps):
            val = array[isamp * nchans + ichan]
            count[ichan] += 1
            nn = count[ichan]

            delta = val - m1[ichan]
            delta_n = delta / nn
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * (nn - 1)
            m1[ichan] += delta_n
            m4[ichan] += (
                term1 * delta_n2 * (nn * nn - 3 * nn + 3)
                + 6 * delta_n2 * m2[ichan]
                - 4 * delta_n * m3[ichan]
            )
            m3[ichan] += term1 * delta_n * (nn - 2) - 3 * delta_n * m2[ichan]
            m2[ichan] += term1

            maxbuffer[ichan] = max(maxbuffer[ichan], val)
            minbuffer[ichan] = min(minbuffer[ichan], val)


class OnlineStats(object):
    def __init__(self, nchans, nsamps):
        self._nchans = nchans
        self._nsamps = nsamps
        self._m1 = np.zeros(nchans, dtype=np.float32)
        self._m2 = np.zeros(nchans, dtype=np.float32)
        self._m3 = np.zeros(nchans, dtype=np.float32)
        self._m4 = np.zeros(nchans, dtype=np.float32)
        self._min = np.zeros(nchans, dtype=np.float32)
        self._max = np.zeros(nchans, dtype=np.float32)
        self._count = np.zeros(nchans, dtype=np.int32)

    @property
    def nchans(self):
        return self._nchans

    @property
    def nsamps(self):
        return self._nsamps

    @property
    def min_ar(self):
        return self._min

    @property
    def max_ar(self):
        return self._max

    @property
    def mean(self):
        return self._m1

    @property
    def stdev(self):
        return np.sqrt(self._m2 / self.nsamps)

    @property
    def skew(self):
        return (
            np.divide(
                self._m3,
                np.power(self._m2, 1.5),
                out=np.zeros_like(self._m3),
                where=self._m2 != 0,
            )
            * np.sqrt(self.nsamps)
        )

    @property
    def kurtosis(self):
        return (
            np.divide(
                self._m4,
                np.power(self._m2, 2.0),
                out=np.zeros_like(self._m4),
                where=self._m2 != 0,
            )
            * self.nsamps
            - 3.0
        )

    def increment(self, array, gulp_size, ii, mode="basic"):
        if mode == "basic":
            compute_online_moments_basic(
                array,
                self._m1,
                self._m2,
                self._max,
                self._min,
                self._count,
                self.nchans,
                gulp_size,
                ii,
            )
        else:
            compute_online_moments(
                array,
                self._m1,
                self._m2,
                self._m3,
                self._m4,
                self._max,
                self._min,
                self._count,
                self.nchans,
                gulp_size,
                ii,
            )
