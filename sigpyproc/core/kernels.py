import numpy as np
from numba import njit, prange, generated_jit, types
from numba.experimental import jitclass
from scipy import constants


@njit("u1[:](u1[:])", cache=True, parallel=True)
def unpack1_8(array):
    bitfact = 8
    unpacked = np.zeros(shape=array.size * bitfact, dtype=np.uint8)
    for ii in prange(array.size):
        unpacked[ii * bitfact + 0] = (array[ii] >> 7) & 1
        unpacked[ii * bitfact + 1] = (array[ii] >> 6) & 1
        unpacked[ii * bitfact + 2] = (array[ii] >> 5) & 1
        unpacked[ii * bitfact + 3] = (array[ii] >> 4) & 1
        unpacked[ii * bitfact + 4] = (array[ii] >> 3) & 1
        unpacked[ii * bitfact + 5] = (array[ii] >> 2) & 1
        unpacked[ii * bitfact + 6] = (array[ii] >> 1) & 1
        unpacked[ii * bitfact + 7] = (array[ii] >> 0) & 1
    return unpacked


@njit("u1[:](u1[:])", cache=True, parallel=True)
def unpack2_8(array):
    bitfact = 8 // 2
    unpacked = np.zeros(shape=array.size * bitfact, dtype=np.uint8)
    for ii in prange(array.size):
        unpacked[ii * bitfact + 0] = (array[ii] & 0xC0) >> 6
        unpacked[ii * bitfact + 1] = (array[ii] & 0x30) >> 4
        unpacked[ii * bitfact + 2] = (array[ii] & 0x0C) >> 2
        unpacked[ii * bitfact + 3] = (array[ii] & 0x03) >> 0
    return unpacked


@njit("u1[:](u1[:])", cache=True, parallel=True)
def unpack4_8(array):
    bitfact = 8 // 4
    unpacked = np.zeros(shape=array.size * bitfact, dtype=np.uint8)
    for ii in prange(array.size):
        unpacked[ii * bitfact + 0] = (array[ii] & 0xF0) >> 4
        unpacked[ii * bitfact + 1] = (array[ii] & 0x0F) >> 0

    return unpacked


@njit("u1[:](u1[:])", cache=True, parallel=True)
def pack2_8(array):
    bitfact = 8 // 2
    packed = np.zeros(shape=array.size // bitfact, dtype=np.uint8)
    for ii in prange(array.size // bitfact):
        packed[ii] = (
            (array[ii * 4] << 6)
            | (array[ii * 4 + 1] << 4)
            | (array[ii * 4 + 2] << 2)
            | array[ii * 4 + 3]
        )

    return packed


@njit("u1[:](u1[:])", cache=True, parallel=True)
def pack4_8(array):
    bitfact = 8 // 4
    packed = np.zeros(shape=array.size // bitfact, dtype=np.uint8)
    for ii in prange(array.size // bitfact):
        packed[ii] = (array[ii * 2] << 4) | array[ii * 2 + 1]

    return packed


@njit(cache=True)
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in {0, 1}
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for ii in range(arr.shape[1]):
            result[ii] = func1d(arr[:, ii])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for ii in range(arr.shape[0]):
            result[ii] = func1d(arr[ii, :])
    return result


@njit(cache=True)
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@generated_jit(nopython=True, cache=True)
def downcast(intype, result):
    if isinstance(intype, types.Integer):
        return lambda intype, result: np.uint8(result)
    return lambda intype, result: np.float32(result)


@njit(cache=True)
def downsample_1d(array, factor):
    reshaped_ar = np.reshape(array, (array.size // factor, factor))
    return np_mean(reshaped_ar, 1)


@njit(
    ["u1[:](u1[:], i4, i4, i4, i4)", "f4[:](f4[:], i4, i4, i4, i4)"],
    cache=True,
    parallel=True,
    locals={"temp": types.f8},
)
def downsample_2d(array, tfactor, ffactor, nchans, nsamps):
    nsamps_new = nsamps // tfactor
    nchans_new = nchans // ffactor
    totfactor = ffactor * tfactor
    result = np.empty(nsamps * nchans // totfactor, dtype=array.dtype)
    for isamp in prange(nsamps_new):
        for ichan in range(nchans_new):
            pos = nchans * isamp * tfactor + ichan * ffactor
            temp = 0
            for ifactor in range(tfactor):
                ipos = pos + ifactor * nchans
                for jfactor in range(ffactor):
                    temp += array[ipos + jfactor]
            result[nchans_new * isamp + ichan] = downcast(array[0], temp / totfactor)
    return result


@njit(
    ["void(u1[:], f4[:], i4, i4, i4)", "void(f4[:], f4[:], i4, i4, i4)"],
    cache=True,
    parallel=True,
)
def extract_tim(inarray, outarray, nchans, nsamps, index):
    for isamp in prange(nsamps):
        for ichan in range(nchans):
            outarray[index + isamp] += inarray[nchans * isamp + ichan]


@njit(
    ["void(u1[:], f4[:], i4, i4)", "void(f4[:], f4[:], i4, i4)"],
    cache=True,
    parallel=True,
)
def extract_bpass(inarray, outarray, nchans, nsamps):
    for ichan in prange(nchans):
        for isamp in range(nsamps):
            outarray[ichan] += inarray[nchans * isamp + ichan]


@njit(
    ["void(u1[:], u1[:], i4, i4)", "void(f4[:], u1[:], i4, i4)"],
    cache=True,
    parallel=True,
)
def mask_channels(array, mask, nchans, nsamps):
    for ichan in prange(nchans):
        if mask[ichan] == 0:
            for isamp in range(nsamps):
                array[nchans * isamp + ichan] = 0


@njit(
    [
        "void(u1[:], f4[:], i4[:], i4, i4, i4, i4)",
        "void(f4[:], f4[:], i4[:], i4, i4, i4, i4)",
    ],
    cache=True,
    parallel=True,
)
def dedisperse(inarray, outarray, delays, maxdelay, nchans, nsamps, index):
    for isamp in prange(nsamps - maxdelay):
        for ichan in range(nchans):
            outarray[index + isamp] += inarray[nchans * (isamp + delays[ichan]) + ichan]


@njit(
    ["u1[:](u1[:], i4, i4)", "f4[:](f4[:], i4, i4)"],
    cache=True,
    parallel=True,
)
def invert_freq(array, nchans, nsamps):
    outarray = np.empty_like(array)
    for isamp in prange(nsamps):
        for ichan in range(nchans):
            outarray[nchans * isamp + ichan] = array[
                nchans * isamp + (nchans - ichan - 1)
            ]
    return outarray


@njit(
    [
        "void(u1[:], f4[:], i4[:], i4[:], i4, i4, i4, i4)",
        "void(f4[:], f4[:], i4[:], i4[:], i4, i4, i4, i4)",
    ],
    cache=True,
    parallel=True,
)
def subband(inarray, outarray, delays, chan_to_sub, maxdelay, nchans, nsubs, nsamps):
    for isamp in prange(nsamps - maxdelay):
        for ichan in range(nchans):
            outarray[nsubs * isamp + chan_to_sub[ichan]] += inarray[
                nchans * (isamp + delays[ichan]) + ichan
            ]


@njit(
    [
        "void(u1[:], f4[:], i4[:], i4[:], i4, f4, f4, f4, i4, i4, i4, i4, i4, i4, i4)",
        "void(f4[:], f4[:], i4[:], i4[:], i4, f4, f4, f4, i4, i4, i4, i4, i4, i4, i4)",
    ],
    cache=True,
)
def fold(
    inarray,
    fold_ar,
    count_ar,
    delays,
    maxdelay,
    tsamp,
    period,
    accel,
    total_nsamps,
    nsamps,
    nchans,
    nbins,
    nints,
    nsubs,
    index,
):
    factor1 = total_nsamps / nints
    factor2 = nchans / nsubs
    tobs = total_nsamps * tsamp
    for isamp in range(nsamps - maxdelay):
        tj = (isamp + index) * tsamp
        phase = nbins * tj * (1 + accel * (tj - tobs) / (2 * constants.c)) / period + 0.5
        phasebin = abs(int(phase)) % nbins
        subint = (isamp + index) // factor1
        pos1 = (subint * nbins * nsubs) + phasebin

        for ichan in range(nchans):
            sub_band = ichan // factor2
            pos2 = int(pos1 + (sub_band * nbins))
            val = inarray[nchans * (isamp + delays[ichan]) + ichan]
            fold_ar[pos2] += val
            count_ar[pos2] += 1


@njit("f4[:](f4[:], f4, f4)")
def resample_tim(array, accel, tsamp):
    nsamps = len(array) - 1 if accel > 0 else len(array)
    resampled = np.zeros(nsamps, dtype=array.dtype)

    partial_calc = (accel * tsamp) / (2 * constants.c)
    tot_drift = partial_calc * (nsamps // 2) ** 2
    last_bin = 0
    for ii in range(nsamps):
        index = int(ii + partial_calc * (ii - nsamps // 2) ** 2 - tot_drift)
        resampled[index] = array[ii]
        if index - last_bin > 1:
            resampled[index - 1] = array[ii]
        last_bin = index
    return resampled


@njit(
    [
        "void(u1[:], u1[:], f4[:], f4[:], i4, i4)",
        "void(f4[:], f4[:], f4[:], f4[:], i4, i4)",
    ],
    cache=True,
    parallel=True,
    locals={"zerodm": types.f8},
)
def remove_zerodm(inarray, outarray, bpass, chanwts, nchans, nsamps):
    for isamp in prange(nsamps):
        zerodm = 0
        for ichan in range(nchans):
            zerodm += inarray[nchans * isamp + ichan]

        for ichan in range(nchans):
            pos = nchans * isamp + ichan
            result = (inarray[pos] - zerodm * chanwts[ichan]) + bpass[ichan]
            outarray[pos] = downcast(inarray[0], result)


@njit("f4[:](f4[:], b1)", cache=True)
def form_spec(fft_ar, interpolated=False):
    specsize = fft_ar.size // 2
    spec_arr = np.zeros(shape=specsize, dtype=fft_ar.dtype)
    if interpolated:
        rl = 0
        il = 0
        for ispec in range(specsize):
            rr = fft_ar[2 * ispec]
            ii = fft_ar[2 * ispec + 1]
            aa = rr ** 2 + ii ** 2
            bb = ((rr - rl) ** 2 + (ii - il) ** 2) / 2
            spec_arr[ispec] = np.sqrt(max(aa, bb))

            rl = rr
            il = ii
    else:
        for ispec in range(specsize):
            spec_arr[ispec] = np.sqrt(fft_ar[2 * ispec] ** 2 + fft_ar[2 * ispec + 1] ** 2)
    return spec_arr


@njit("f4[:](f4[:], i4, i4, f4, f4)", cache=True, locals={"norm": types.f4})
def remove_rednoise(fftbuffer, startwidth, endwidth, endfreq, tsamp):
    outbuffer = np.zeros_like(fftbuffer, dtype=np.float32)
    oldinbuf = np.empty(2 * endwidth, dtype=np.float32)
    newinbuf = np.empty(2 * endwidth, dtype=np.float32)
    realbuffer = np.empty(endwidth, dtype=np.float32)
    nsamps = fftbuffer.size // 2
    binnum = 1

    # Set DC bin to 1.0
    outbuffer[0] = 1.0
    windex = 2
    rindex = 2
    numread_old = startwidth

    # transfer numread_old complex samples to oldinbuf
    oldinbuf[: 2 * numread_old] = fftbuffer[rindex : rindex + 2 * numread_old]
    rindex += 2 * numread_old

    # calculate powers for oldinbuf
    for ispec in range(numread_old):
        realbuffer[ispec] = oldinbuf[2 * ispec] ** 2 + oldinbuf[2 * ispec + 1] ** 2

    # calculate first median of our data and determine next bufflen
    mean_old = np.median(realbuffer) / np.log(2.0)
    binnum += numread_old
    bufflen = round(startwidth * np.log(binnum))

    while rindex // 2 < nsamps:
        numread_new = min(bufflen, nsamps - rindex // 2)

        # transfer numread_new complex samples to newinbuf
        newinbuf[: 2 * numread_new] = fftbuffer[rindex : rindex + 2 * numread_new]
        rindex += 2 * numread_new

        # calculate powers for newinbuf
        for ispec in range(numread_new):
            realbuffer[ispec] = newinbuf[2 * ispec] ** 2 + newinbuf[2 * ispec + 1] ** 2

        mean_new = np.median(realbuffer) / np.log(2.0)
        slope = (mean_new - mean_old) / (numread_old + numread_new)

        for ispec in range(numread_old):
            norm = np.sqrt(mean_old + slope * ((numread_old + numread_new) / 2 - ispec))
            outbuffer[2 * ispec + windex] = oldinbuf[2 * ispec] / norm
            outbuffer[2 * ispec + 1 + windex] = oldinbuf[2 * ispec + 1] / norm

        windex += 2 * numread_old
        binnum += numread_new
        if binnum / (nsamps * tsamp) < endfreq:
            bufflen = round(startwidth * np.log(binnum))
        else:
            bufflen = endwidth
        numread_old = numread_new
        mean_old = mean_new

        oldinbuf[: 2 * numread_new] = newinbuf[: 2 * numread_new]

    outbuffer[windex : windex + 2 * numread_old] = oldinbuf[: 2 * numread_old] / np.sqrt(
        mean_old
    )
    return outbuffer


@njit("void(f4[:], f4[:], i4[:], i4[:], i4, i4, i4)", cache=True)
def sum_harms(spec_arr, sum_arr, harm_arr, fact_arr, nharms, nsamps, nfold):
    for ifold in range(nfold, nsamps - (nharms - 1), nharms):
        for iharm in range(nharms):
            for kk in range(nharms // 2):
                sum_arr[ifold + iharm] += spec_arr[
                    fact_arr[kk] + harm_arr[iharm * nharms // 2 + kk]
                ]
        for kk in range(nharms // 2):
            fact_arr[kk] += 2 * kk + 1


MomentsBagSpec = [
    ("nchans", types.i4),
    ("m1", types.f4[:]),
    ("m2", types.f4[:]),
    ("m3", types.f4[:]),
    ("m4", types.f4[:]),
    ("min", types.f4[:]),
    ("max", types.f4[:]),
    ("count", types.i4[:]),
]


@jitclass(MomentsBagSpec)
class MomentsBag(object):
    def __init__(self, nchans):
        self.nchans = nchans
        self.m1 = np.zeros(nchans, dtype=np.float32)
        self.m2 = np.zeros(nchans, dtype=np.float32)
        self.m3 = np.zeros(nchans, dtype=np.float32)
        self.m4 = np.zeros(nchans, dtype=np.float32)
        self.min = np.zeros(nchans, dtype=np.float32)
        self.max = np.zeros(nchans, dtype=np.float32)
        self.count = np.zeros(nchans, dtype=np.int32)


@njit(cache=True, parallel=True, locals={"val": types.f8})
def compute_online_moments_basic(array, bag, nsamps, startflag):
    if startflag == 0:
        for ii in range(bag.nchans):
            bag.max[ii] = array[ii]
            bag.min[ii] = array[ii]

    for ichan in prange(bag.nchans):
        for isamp in range(nsamps):
            val = array[isamp * bag.nchans + ichan]
            bag.count[ichan] += 1
            nn = bag.count[ichan]

            delta = val - bag.m1[ichan]
            delta_n = delta / nn
            bag.m1[ichan] += delta_n
            bag.m2[ichan] += delta * delta_n * (nn - 1)

            bag.max[ichan] = max(bag.max[ichan], val)
            bag.min[ichan] = min(bag.min[ichan], val)


@njit(cache=True, parallel=True, locals={"val": types.f8})
def compute_online_moments(array, bag, nsamps, startflag):
    """Computing central moments in one pass through the data."""
    if startflag == 0:
        for ii in range(bag.nchans):
            bag.max[ii] = array[ii]
            bag.min[ii] = array[ii]

    for ichan in prange(bag.nchans):
        for isamp in range(nsamps):
            val = array[isamp * bag.nchans + ichan]
            bag.count[ichan] += 1
            nn = bag.count[ichan]

            delta = val - bag.m1[ichan]
            delta_n = delta / nn
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * (nn - 1)
            bag.m1[ichan] += delta_n
            bag.m4[ichan] += (
                term1 * delta_n2 * (nn * nn - 3 * nn + 3)
                + 6 * delta_n2 * bag.m2[ichan]
                - 4 * delta_n * bag.m3[ichan]
            )
            bag.m3[ichan] += term1 * delta_n * (nn - 2) - 3 * delta_n * bag.m2[ichan]
            bag.m2[ichan] += term1

            bag.max[ichan] = max(bag.max[ichan], val)
            bag.min[ichan] = min(bag.min[ichan], val)


@njit(cache=True, parallel=False, locals={"val": types.f8})
def add_online_moments(bag_a, bag_b, bag_c):
    bag_c.count = bag_a.count + bag_b.count
    delta = bag_b.m1 - bag_a.m1
    delta2 = delta * delta
    delta3 = delta * delta2
    delta4 = delta2 * delta2

    bag_c.m1 = (bag_a.count * bag_a.m1 + bag_b.count * bag_b.m1) / bag_c.count
    bag_c.m2 = bag_a.m2 + bag_b.m2 + delta2 * bag_a.count * bag_b.count / bag_c.count

    bag_c.m3 = (
        bag_a.m3
        + bag_b.m3
        + delta3
        * bag_a.count
        * bag_b.count
        * (bag_a.count - bag_b.count)
        / (bag_c.count ** 2)
    )
    bag_c.m3 += (
        3 * delta * (bag_a.count * bag_b.m2 - bag_b.count * bag_a.m2) / bag_c.count
    )

    bag_c.m4 = (
        bag_a.m4
        + bag_b.m4
        + delta4
        * bag_a.count
        * bag_b.count
        * (bag_a.count ** 2 - bag_a.count * bag_b.count + bag_b.count ** 2)
        / (bag_c.count ** 3)
    )
    bag_c.m4 += (
        6
        * delta2
        * (bag_a.count * bag_a.count * bag_b.m2 + bag_b.count * bag_b.count * bag_a.m2)
        / (bag_c.count ** 2)
    )
    bag_c.m4 += (
        4 * delta * (bag_a.count * bag_b.m3 - bag_b.count * bag_a.m3) / bag_c.count
    )

    bag_c.max = np.maximum(bag_a.max, bag_b.max)
    bag_c.min = np.minimum(bag_c.min, bag_c.min)
