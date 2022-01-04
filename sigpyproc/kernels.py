import numpy as np
from numba import njit, prange, generated_jit, types
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
    ["void(u1[:], f4[:], i4, i4, i4, i4)", "void(f4[:], f4[:], i4, i4, i4, i4)"],
    cache=True,
    parallel=True,
)
def extract_channel(inarray, outarray, ichannel, nchans, nsamps, index):
    for isamp in prange(nsamps):
        outarray[index + isamp] = inarray[nchans * isamp + ichannel]


@njit(
    ["void(u1[:], f4[:], i4, i4, i4)", "void(f4[:], f4[:], i4, i4, i4)"],
    cache=True,
    parallel=True,
)
def split_to_channels(inarray, outarray, nchans, nsamps, gulp):
    for isamp in prange(nsamps):
        for ichan in range(nchans):
            outarray[ichan * gulp + isamp] = inarray[nchans * isamp + ichan]


@njit(
    ["void(u1[:], u1[:], i4, i4)", "void(f4[:], f4[:], i4, i4)"],
    cache=True,
    parallel=True,
)
def invert_freq(inarray, outarray, nchans, nsamps):
    for isamp in prange(nsamps):
        for ichan in range(nchans):
            outarray[nchans * isamp + ichan] = inarray[
                nchans * isamp + (nchans - ichan - 1)
            ]


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
            pos2 = pos1 + (sub_band * nbins)
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


@njit("f4[:](f4[:], i4, b1)", cache=True)
def form_spec(fft, specsize, interpolated=False):
    spec_arr = np.zeros(shape=specsize, dtype=fft.dtype)
    if interpolated:
        rl = 0
        il = 0
        for ispec in range(specsize):
            rr = fft[2 * ispec]
            ii = fft[2 * ispec + 1]
            aa = rr ** 2 + ii ** 2
            bb = (rr - rl) ** 2 + (ii - il) ** 2
            spec_arr[ispec] = np.sqrt(max(aa, bb))

            rl = rr
            il = ii
    else:
        for ispec in range(specsize):
            spec_arr = np.sqrt(fft[2 * ispec] ** 2 + fft[2 * ispec + 1] ** 2)
    return spec_arr


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
