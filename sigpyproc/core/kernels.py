from __future__ import annotations

import numpy as np
import numpy.typing as npt
import rocket_fft
from numba import njit, prange, types

CONST_C_VAL = 299792458.0  # Speed of light in m/s (astropy.constants.c.value)


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def unpack1_8_big(
    array: npt.NDArray[np.uint8],
    unpacked: npt.NDArray[np.uint8],
) -> None:
    for ii in range(array.size):
        pos = ii * 8
        for jj in range(8):
            unpacked[pos + (7 - jj)] = (array[ii] >> jj) & 1


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def unpack1_8_little(
    array: npt.NDArray[np.uint8],
    unpacked: npt.NDArray[np.uint8],
) -> None:
    for ii in range(array.size):
        pos = ii * 8
        for jj in range(8):
            unpacked[pos + jj] = (array[ii] >> jj) & 1


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def unpack2_8_big(
    array: npt.NDArray[np.uint8],
    unpacked: npt.NDArray[np.uint8],
) -> None:
    for ii in range(array.size):
        pos = ii * 4
        unpacked[pos + 3] = (array[ii] & 0x03) >> 0
        unpacked[pos + 2] = (array[ii] & 0x0C) >> 2
        unpacked[pos + 1] = (array[ii] & 0x30) >> 4
        unpacked[pos + 0] = (array[ii] & 0xC0) >> 6


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def unpack2_8_little(
    array: npt.NDArray[np.uint8],
    unpacked: npt.NDArray[np.uint8],
) -> None:
    for ii in range(array.size):
        pos = ii * 4
        unpacked[pos + 0] = (array[ii] & 0x03) >> 0
        unpacked[pos + 1] = (array[ii] & 0x0C) >> 2
        unpacked[pos + 2] = (array[ii] & 0x30) >> 4
        unpacked[pos + 3] = (array[ii] & 0xC0) >> 6


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def unpack4_8_big(
    array: npt.NDArray[np.uint8],
    unpacked: npt.NDArray[np.uint8],
) -> None:
    for ii in range(array.size):
        pos = ii * 2
        unpacked[pos + 1] = (array[ii] & 0x0F) >> 0
        unpacked[pos + 0] = (array[ii] & 0xF0) >> 4


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def unpack4_8_little(
    array: npt.NDArray[np.uint8],
    unpacked: npt.NDArray[np.uint8],
) -> None:
    for ii in range(array.size):
        pos = ii * 2
        unpacked[pos + 0] = (array[ii] & 0x0F) >> 0
        unpacked[pos + 1] = (array[ii] & 0xF0) >> 4


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True)
def pack1_8_big(array: npt.NDArray[np.uint8], packed: npt.NDArray[np.uint8]) -> None:
    for ii in range(packed.size):
        pos = ii * 8
        packed[ii] = (
            (array[pos + 0] << 7)
            | (array[pos + 1] << 6)
            | (array[pos + 2] << 5)
            | (array[pos + 3] << 4)
            | (array[pos + 4] << 3)
            | (array[pos + 5] << 2)
            | (array[pos + 6] << 1)
            | array[pos + 7]
        )


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True)
def pack1_8_little(array: npt.NDArray[np.uint8], packed: npt.NDArray[np.uint8]) -> None:
    for ii in range(packed.size):
        pos = ii * 8
        packed[ii] = (
            (array[pos + 7] << 7)
            | (array[pos + 6] << 6)
            | (array[pos + 5] << 5)
            | (array[pos + 4] << 4)
            | (array[pos + 3] << 3)
            | (array[pos + 2] << 2)
            | (array[pos + 1] << 1)
            | array[pos + 0]
        )


@njit("void(u1[::1], u1[::1], b1)", cache=True, fastmath=True)
def pack1_8_vect(
    array: npt.NDArray[np.uint8],
    packed: npt.NDArray[np.uint8],
    *,
    big_endian: bool = True,
) -> None:
    mask = types.uint64(0x0101010101010101)
    if big_endian:
        magic = types.uint64(0x8040201008040201)
    else:
        magic = types.uint64(0x0102040810204080)
    shift = types.uint64(56)

    array_uint64 = array.view(np.uint64)
    nwords = array_uint64.size
    batch_size = 16
    i = 0
    while i < nwords - batch_size:
        for j in range(batch_size):
            x = array_uint64[i + j]
            x &= mask
            x *= magic
            packed[i + j] = types.uint8(x >> shift)
        i += batch_size

    # Handle the remaining elements
    for j in range(i, nwords):
        x = array_uint64[j]
        x &= mask
        x *= magic
        packed[j] = types.uint8(x >> shift)


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def pack2_8_big(array: npt.NDArray[np.uint8], packed: npt.NDArray[np.uint8]) -> None:
    for ii in range(packed.size):
        pos = ii * 4
        packed[ii] = (
            (array[pos + 0] << 6)
            | (array[pos + 1] << 4)
            | (array[pos + 2] << 2)
            | array[pos + 3]
        )


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def pack2_8_little(array: npt.NDArray[np.uint8], packed: npt.NDArray[np.uint8]) -> None:
    for ii in range(packed.size):
        pos = ii * 4
        packed[ii] = (
            (array[pos + 3] << 6)
            | (array[pos + 2] << 4)
            | (array[pos + 1] << 2)
            | array[pos + 0]
        )


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def pack4_8_big(array: npt.NDArray[np.uint8], packed: npt.NDArray[np.uint8]) -> None:
    for ii in range(packed.size):
        pos = ii * 2
        packed[ii] = (array[pos + 0] << 4) | array[pos + 1]


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def pack4_8_little(array: npt.NDArray[np.uint8], packed: npt.NDArray[np.uint8]) -> None:
    for ii in range(packed.size):
        pos = ii * 2
        packed[ii] = (array[pos + 1] << 4) | array[pos + 0]


@njit(cache=True, fastmath=True)
def downsample_1d_mean(array: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 1D array by averaging over bins.

    Parameters
    ----------
    array : ndarray
        Input 1D array to be downsampled.
    factor : int
        Downsampling factor. Must be a positive integer.

    Returns
    -------
    ndarray
        Downsampled array

    Notes
    -----
    Uses float64 accumulator to avoid overflow.
    """
    nsamps_new = len(array) // factor
    result = np.empty(nsamps_new, dtype=array.dtype)
    for isamp in prange(nsamps_new):
        temp = np.float64(0.0)
        start = isamp * factor
        for ifactor in range(factor):
            temp += array[start + ifactor]
        result[isamp] = temp / factor
    return result


@njit(cache=True, fastmath=True)
def downsample_2d_mean_flat(
    array: np.ndarray,
    factor1: int,
    factor2: int,
    dim1: int,
    dim2: int,
) -> np.ndarray:
    """Downsample a flattened 2D array by averaging over bins in both dimensions.

    Parameters
    ----------
    array : np.ndarray
        Input flattened 2D array to be downsampled.
    factor1 : int
        Downsampling factor for the first dimension. Must be a positive integer.
    factor2 : int
        Downsampling factor for the second dimension. Must be a positive integer.
    dim1 : int
        Number of bins in the first dimension.
    dim2 : int
        Number of bins in the second dimension.

    Returns
    -------
    np.ndarray
        Downsampled flattened 2D array

    Notes
    -----
    dim2 must ve the fastest varying dimension.
    """
    new_dim1 = dim1 // factor1
    new_dim2 = dim2 // factor2
    totfactor = factor1 * factor2
    result = np.empty(new_dim1 * new_dim2, dtype=array.dtype)
    for i in prange(new_dim1):
        for j in range(new_dim2):
            temp = np.float64(0.0)
            pos = dim2 * i * factor1 + j * factor2
            for ifactor in range(factor1):
                ipos = pos + ifactor * dim2
                for ifactor2 in range(factor2):
                    temp += array[ipos + ifactor2]
            result[new_dim2 * i + j] = temp / totfactor
    return result


@njit(
    ["void(u1[::1], f4[::1], i8, i8, i8)", "void(f4[::1], f4[::1], i8, i8, i8)"],
    cache=True,
    fastmath=True,
)
def extract_tim(
    in_arr: np.ndarray,
    out_arr: np.ndarray,
    nchans: int,
    nsamps: int,
    out_start_index: int,
) -> None:
    # \in_arr.reshape(nsamps, nchans).sum(axis=1)
    in_arr_2d = in_arr.reshape(nsamps, nchans)
    for isamp in prange(nsamps):
        summ = np.float64(0.0)
        for ichan in range(nchans):
            summ += in_arr_2d[isamp, ichan]
        out_arr[out_start_index + isamp] = summ


@njit(
    ["void(u1[::1], f4[::1], i8, i8)", "void(f4[::1], f4[::1], i8, i8)"],
    cache=True,
    fastmath=True,
)
def extract_bpass(
    in_arr: np.ndarray,
    out_arr: np.ndarray,
    nchans: int,
    nsamps: int,
) -> None:
    # \in_arr.reshape(nsamps, nchans).sum(axis=0)
    in_arr_2d = in_arr.reshape(nsamps, nchans)
    for isamp in range(nsamps):
        for ichan in range(nchans):
            out_arr[ichan] += in_arr_2d[isamp, ichan]


@njit(
    ["void(u1[::1], f4[::1], i8, i8)", "void(f4[::1], f4[::1], i8, i8)"],
    cache=True,
    fastmath=True,
    parallel=True,
)
def extract_bpass_par(
    in_arr: np.ndarray,
    out_arr: np.ndarray,
    nchans: int,
    nsamps: int,
) -> None:
    in_arr_2d = in_arr.reshape(nsamps, nchans).T
    for ichan in prange(nchans):
        summ = np.float64(0.0)
        for isamp in range(nsamps):
            summ += in_arr_2d[ichan, isamp]
        out_arr[ichan] += summ


@njit(
    ["void(u1[::1], b1[::1], u1, i8, i8)", "void(f4[::1], b1[::1], f4, i8, i8)"],
    cache=True,
    fastmath=True,
)
def mask_channels(
    in_arr: np.ndarray,
    chan_mask: np.ndarray,
    mask_value: float,
    nchans: int,
    nsamps: int,
) -> None:
    in_arr_2d = in_arr.reshape(nsamps, nchans)
    # Build mask indices first (to avoid if statements in loop)
    masked_chans = np.empty(nchans, dtype=np.int64)
    n_masked = 0
    for ichan in range(nchans):
        if chan_mask[ichan]:
            masked_chans[n_masked] = ichan
            n_masked += 1
    for isamp in prange(nsamps):
        for i in range(n_masked):
            ichan = masked_chans[i]
            in_arr_2d[isamp, ichan] = mask_value


@njit(
    [
        "void(u1[::1], f4[::1], i8[::1], i8, i8, i8)",
        "void(f4[::1], f4[::1], i8[::1], i8, i8, i8)",
    ],
    cache=True,
    fastmath=True,
)
def dedisperse(
    in_arr: np.ndarray,
    out_arr: np.ndarray,
    delays: np.ndarray,
    nchans: int,
    nsamps: int,
    out_start_index: int,
) -> None:
    max_delay = int(np.max(np.abs(delays)))
    in_arr_2d = in_arr.reshape(nsamps, nchans)
    for isamp in prange(nsamps - max_delay):
        summ = np.float64(0.0)
        for ichan in range(nchans):
            summ += in_arr_2d[isamp + delays[ichan], ichan]
        out_arr[out_start_index + isamp] = summ


@njit(
    ["u1[::1](u1[::1], i8, i8)", "f4[::1](f4[::1], i8, i8)"],
    cache=True,
    fastmath=True,
    parallel=True,
)
def invert_freq(in_arr: np.ndarray, nchans: int, nsamps: int) -> np.ndarray:
    in_arr_2d = in_arr.reshape(nsamps, nchans)
    out_arr_2d = np.empty((nsamps, nchans), dtype=in_arr.dtype)
    for isamp in prange(nsamps):
        out_arr_2d[isamp, :] = in_arr_2d[isamp, ::-1]
    return out_arr_2d.ravel()


@njit(
    [
        "void(u1[:], f4[:], i4[:], i4[:], i4, i4, i4, i4)",
        "void(f4[:], f4[:], i4[:], i4[:], i4, i4, i4, i4)",
    ],
    cache=True,
    parallel=True,
)
def subband(
    inarray: np.ndarray,
    outarray: np.ndarray,
    delays: np.ndarray,
    chan_to_sub: np.ndarray,
    maxdelay: int,
    nchans: int,
    nsubs: int,
    nsamps: int,
) -> None:
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
    inarray: np.ndarray,
    fold_ar: np.ndarray,
    count_ar: np.ndarray,
    delays: np.ndarray,
    maxdelay: int,
    tsamp: float,
    period: float,
    accel: float,
    total_nsamps: int,
    nsamps: int,
    nchans: int,
    nbins: int,
    nints: int,
    nsubs: int,
    index: int,
) -> None:
    factor1 = total_nsamps / nints
    factor2 = nchans / nsubs
    tobs = total_nsamps * tsamp
    for isamp in range(nsamps - maxdelay):
        tj = (isamp + index) * tsamp
        phase = (
            nbins * tj * (1 + accel * (tj - tobs) / (2 * CONST_C_VAL)) / period + 0.5
        )
        phasebin = abs(int(phase)) % nbins
        subint = (isamp + index) // factor1
        pos1 = (subint * nbins * nsubs) + phasebin

        for ichan in range(nchans):
            sub_band = ichan // factor2
            pos2 = int(pos1 + (sub_band * nbins))
            val = inarray[nchans * (isamp + delays[ichan]) + ichan]
            fold_ar[pos2] += val
            count_ar[pos2] += 1


@njit("f4[:](f4[:], f4, f4)", cache=True)
def resample_tim(array: np.ndarray, accel: float, tsamp: float) -> np.ndarray:
    nsamps = len(array) - 1 if accel > 0 else len(array)
    resampled = np.zeros(nsamps, dtype=array.dtype)

    partial_calc = (accel * tsamp) / (2 * CONST_C_VAL)
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
        "void(u1[::1], u1[::1], f4[::1], i8, i8)",
        "void(f4[::1], f4[::1], f4[::1], i8, i8)",
    ],
    cache=True,
    fastmath=True,
)
def remove_zerodm(
    in_arr: np.ndarray,
    out_arr: np.ndarray,
    bpass: np.ndarray,
    nchans: int,
    nsamps: int,
) -> None:
    """Remove zero DM from an input array.

    Parameters
    ----------
    in_arr : np.ndarray
        Input 2D flattened array to remove zero DM from.
    out_arr : np.ndarray
        Output 2D flattened array with zero DM removed.
    bpass : np.ndarray
        Bandpass to be added back to the data.
    nchans : int
        Number of frequency channels in the 2D data.
    nsamps : int
        Number of samples in the 2D data.
    """
    in_arr_2d = in_arr.reshape(nsamps, nchans)
    out_arr_2d = out_arr.reshape(nsamps, nchans)
    chanwts = bpass / bpass.sum()
    for isamp in prange(nsamps):
        zerodm = np.float64(0.0)
        for ichan in range(nchans):
            zerodm += in_arr_2d[isamp, ichan]
        for ichan in range(nchans):
            out_arr_2d[isamp, ichan] = (
                in_arr_2d[isamp, ichan] - zerodm * chanwts[ichan] + bpass[ichan]
            )


@njit("f4[::1](c8[::1])", cache=True, fastmath=True)
def form_mspec(fspec: np.ndarray) -> np.ndarray:
    nfreq = len(fspec)
    mspec = np.zeros(nfreq, dtype=np.float32)
    for i in range(nfreq):
        mspec[i] = np.sqrt(fspec[i].real ** 2 + fspec[i].imag ** 2)
    return mspec


@njit("f4[::1](c8[::1])", cache=True, fastmath=True)
def form_interp_mspec(fspec: np.ndarray) -> np.ndarray:
    nfreq = len(fspec)
    mspec = np.zeros(nfreq, dtype=np.float32)
    re_prev = 0
    im_prev = 0
    for i in range(nfreq):
        re, im = fspec[i].real, fspec[i].imag
        ampsq = re**2 + im**2
        ampsq_diff = 0.5 * ((re - re_prev) ** 2 + (im - im_prev) ** 2)
        mspec[i] = np.sqrt(max(ampsq, ampsq_diff))
        re_prev, im_prev = re, im
    return mspec


# Note: No caching, as the py_func is same.
downsample_1d_mean_par = njit(downsample_1d_mean.py_func, parallel=True, fastmath=True)
downsample_2d_mean_par = njit(
    downsample_2d_mean_flat.py_func,
    parallel=True,
    fastmath=True,
)
extract_tim_par = njit(extract_tim.py_func, parallel=True, fastmath=True)
mask_channels_par = njit(mask_channels.py_func, parallel=True, fastmath=True)
invert_freq_par = njit(invert_freq.py_func, parallel=True, fastmath=True)
dedisperse_par = njit(dedisperse.py_func, parallel=True, fastmath=True)
remove_zerodm_par = njit(remove_zerodm.py_func, parallel=True, fastmath=True)


@njit("c8[:](c8[:], i8, i8, i8)", cache=True, fastmath=True)
def fs_running_median(
    spec_arr: np.ndarray,
    start_width: int,
    end_width: int,
    end_freq_bin: int,
) -> np.ndarray:
    nspecs = len(spec_arr)
    out_arr = np.zeros_like(spec_arr)
    powbuf = np.empty(nspecs, dtype=np.float32)
    for ii in range(nspecs):
        powbuf[ii] = spec_arr[ii].real ** 2 + spec_arr[ii].imag ** 2

    # Set DC bin to 1.0
    out_arr[0] = 1.0 + 0j
    rindex, windex, binnum = 1, 1, 1
    buflen = start_width

    numread_old = buflen
    mean_old = np.median(powbuf[rindex : rindex + numread_old]) / np.log(2)
    rindex += numread_old
    binnum += numread_old
    buflen = round(start_width * np.log(binnum))

    while rindex < nspecs:
        numread_new = min(buflen, nspecs - rindex)
        mean_new = np.median(powbuf[rindex : rindex + numread_new]) / np.log(2)
        rindex += numread_new

        slope = (mean_new - mean_old) / (numread_old + numread_new)
        for i in range(numread_old):
            norm = 1 / np.sqrt(mean_old + slope * ((numread_old + numread_new) / 2 - i))
            out_arr[windex + i] = spec_arr[windex + i] * norm

        windex += numread_old
        binnum += numread_new
        if binnum < end_freq_bin:
            buflen = round(start_width * np.log(binnum))
        else:
            buflen = end_width
        numread_old, mean_old = numread_new, mean_new

    # Remaining samples
    norm = 1 / np.sqrt(mean_old)
    out_arr[windex : windex + numread_old] = (
        spec_arr[windex : windex + numread_old] * norm
    )
    return out_arr


@njit("f4[:,:](f4[:], i8)", cache=True, fastmath=True)
def sum_harmonics(pow_spec: np.ndarray, nfolds: int) -> np.ndarray:
    nfreqs = len(pow_spec)
    sum_arr = np.zeros((nfolds, nfreqs), dtype=np.float32)
    harm_sum = pow_spec.copy()
    nfold1 = 0  # int(self.header.tsamp*2*self.size/maxperiod)
    for iff in range(nfolds):
        nharm = 2 ** (iff + 1)
        nfoldi = int(max(1, min(nharm * nfold1 - nharm // 2, nfreqs)))
        harm_arr = np.array(
            [kk * ll // nharm for ll in range(nharm) for kk in range(1, nharm, 2)],
            dtype=np.int32,
        )

        facts_ar = np.array(
            [(kk * nfoldi + nharm // 2) // nharm for kk in range(1, nharm, 2)],
            dtype=np.int32,
        )
        for ifold in range(nfoldi, nfreqs - (nharm - 1), nharm):
            for iharm in range(nharm):
                for kk in range(nharm // 2):
                    harm_sum[ifold + iharm] += pow_spec[
                        facts_ar[kk] + harm_arr[iharm * nharm // 2 + kk]
                    ]
            for kk in range(nharm // 2):
                facts_ar[kk] += 2 * kk + 1
        sum_arr[iff] = harm_sum
    return sum_arr


moments_dtype = np.dtype(
    [
        ("count", np.int32),
        ("m1", np.float32),
        ("m2", np.float32),
        ("m3", np.float32),
        ("m4", np.float32),
        ("min", np.float32),
        ("max", np.float32),
    ],
    align=True,
)


@njit(cache=True, fastmath=True)
def update_moments(
    val: float,
    m1: float,
    m2: float,
    m3: float,
    m4: float,
    n: int,
) -> tuple[float, float, float, float, int]:
    n += 1
    delta = val - m1
    delta_n = delta / n
    delta_n2 = delta_n * delta_n
    term = delta * delta_n * (n - 1)

    m1 += delta_n
    m4 += term * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3
    m3 += term * delta_n * (n - 2) - 3 * delta_n * m2
    m2 += term
    return m1, m2, m3, m4, n


@njit(cache=True, fastmath=True)
def update_moments_basic(
    val: float,
    m1: float,
    m2: float,
    n: int,
) -> tuple[float, float, int]:
    n += 1
    delta = val - m1
    delta_n = delta / n

    m1 += delta_n
    m2 += delta * delta_n * (n - 1)
    return m1, m2, n


@njit(cache=True, parallel=True, fastmath=True, locals={"val": types.f4})
def compute_online_moments(
    array: np.ndarray,
    moments: np.ndarray,
    startflag: int = 0,
) -> None:
    """Compute central moments in one pass through the data."""
    nchans = moments.shape[0]
    nsamps = array.shape[0] // nchans

    if startflag == 0:
        for ichan in range(nchans):
            moments[ichan]["min"] = array[ichan]
            moments[ichan]["max"] = array[ichan]

    for ichan in prange(nchans):
        m1, m2, m3, m4 = (
            moments[ichan]["m1"],
            moments[ichan]["m2"],
            moments[ichan]["m3"],
            moments[ichan]["m4"],
        )
        count = moments[ichan]["count"]
        min_val, max_val = moments[ichan]["min"], moments[ichan]["max"]

        for isamp in range(nsamps):
            val = array[isamp * nchans + ichan]
            m1, m2, m3, m4, count = update_moments(val, m1, m2, m3, m4, count)
            min_val = min(min_val, val)
            max_val = max(max_val, val)
        (
            moments[ichan]["m1"],
            moments[ichan]["m2"],
            moments[ichan]["m3"],
            moments[ichan]["m4"],
        ) = m1, m2, m3, m4
        moments[ichan]["count"] = count
        moments[ichan]["min"], moments[ichan]["max"] = min_val, max_val


@njit(cache=True, parallel=True, fastmath=True, locals={"val": types.f4})
def compute_online_moments_basic(
    array: np.ndarray,
    moments: np.ndarray,
    startflag: int = 0,
) -> None:
    """Compute central moments in one pass through the data."""
    nchans = moments.shape[0]
    nsamps = array.shape[0] // nchans

    if startflag == 0:
        for ichan in range(nchans):
            moments[ichan]["min"] = array[ichan]
            moments[ichan]["max"] = array[ichan]

    for ichan in prange(nchans):
        m1, m2 = moments[ichan]["m1"], moments[ichan]["m2"]
        count = moments[ichan]["count"]
        min_val, max_val = moments[ichan]["min"], moments[ichan]["max"]

        for isamp in range(nsamps):
            val = array[isamp * nchans + ichan]
            m1, m2, count = update_moments_basic(val, m1, m2, count)
            min_val = min(min_val, val)
            max_val = max(max_val, val)
        moments[ichan]["m1"], moments[ichan]["m2"] = m1, m2
        moments[ichan]["count"] = count
        moments[ichan]["min"], moments[ichan]["max"] = min_val, max_val


@njit(cache=True, fastmath=True)
def add_online_moments(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    c["count"][:] = a["count"] + b["count"]
    delta = b["m1"] - a["m1"]
    delta2 = delta * delta
    delta3 = delta * delta2
    delta4 = delta2 * delta2

    c["m1"][:] = (a["count"] * a["m1"] + b["count"] * b["m1"]) / c["count"]
    c["m2"][:] = a["m2"] + b["m2"] + delta2 * a["count"] * b["count"] / c["count"]
    c["m3"][:] = (
        a["m3"]
        + b["m3"]
        + delta3
        * a["count"]
        * b["count"]
        * (a["count"] - b["count"])
        / (c["count"] ** 2)
    )
    c["m3"][:] += 3 * delta * (a["count"] * b["m2"] - b["count"] * a["m2"]) / c["count"]
    c["m4"][:] = (
        a["m4"]
        + b["m4"]
        + delta4
        * a["count"]
        * b["count"]
        * (a["count"] ** 2 - a["count"] * b["count"] + b["count"] ** 2)
        / (c["count"] ** 3)
    )
    c["m4"][:] += (
        6
        * delta2
        * (a["count"] ** 2 * b["m2"] + b["count"] ** 2 * a["m2"])
        / (c["count"] ** 2)
    )
    c["m4"][:] += 4 * delta * (a["count"] * b["m3"] - b["count"] * a["m3"]) / c["count"]
    c["max"][:] = np.maximum(a["max"], b["max"])
    c["min"][:] = np.minimum(a["min"], b["min"])


@njit("f4[::1](f4[::1])", cache=True, fastmath=True)
def detrend_1d(arr: np.ndarray) -> np.ndarray:
    """Detrend a 1D array using a linear fit.

    Similar to scipiy.signal.detrend. Currently for 1d arrays only.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array.

    Returns
    -------
    np.ndarray
        Detrended array.

    Raises
    ------
    ValueError
        If the input array is empty.
    """
    m = len(arr)
    if m == 0:
        msg = "Input array must be non-empty."
        raise ValueError(msg)
    if m == 1:
        return np.zeros(1, dtype=arr.dtype)
    x_sum = m * (m - 1) / 2
    y_sum = 0.0
    x_sq_sum = m * (m - 1) * (2 * m - 1) / 6
    x_y_sum = 0.0
    denominator = m * x_sq_sum - x_sum**2
    if denominator == 0:
        return arr - np.mean(arr, dtype=np.float32)
    for i in range(m):
        y_sum += arr[i]
        x_y_sum += i * arr[i]
    slope = (m * x_y_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / m
    result = np.empty(m, dtype=np.float32)
    for i in range(m):
        result[i] = arr[i] - (slope * i + intercept)
    return result


@njit("f4[:, ::1](f4[:, ::1])", cache=True, fastmath=True)
def detrend_2d(arr: np.ndarray) -> np.ndarray:
    """Detrend a 2D array using a linear fit.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D array.

    Returns
    -------
    np.ndarray
        Detrended array.
    """
    result = np.empty_like(arr)
    for i in range(arr.shape[0]):
        result[i] = detrend_1d(arr[i])
    return result


@njit(cache=True, fastmath=True)
def convolve_templates(
    data: np.ndarray,
    temp_bank: types.List[types.Array],
    ref_bin: types.List[int],
) -> np.ndarray:
    """
    Convolve the data with the templates in the template bank.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    temp_bank : list[np.ndarray]
        List of template arrays.
    ref_bin : list[int]
        List of reference bin indices.

    Returns
    -------
    np.ndarray
        Convolved array.

    Notes
    -----
    The reference bin is aligned to the index 0 and the template is time-reversed
    (to perform convolution rather than correlation). The template is then
    normalised to zero mean and unity power.
    """
    nbins = len(data)
    ntemps = len(temp_bank)
    convs = np.empty((ntemps, nbins), dtype=data.dtype)
    data_pad = circular_pad_goodsize(data)
    data_fft = np.fft.rfft(data_pad)
    for itemp in range(ntemps):
        temp_kernel = temp_bank[itemp]
        temp_pad = np.zeros_like(data_pad)
        temp_pad[: len(temp_kernel)] = temp_kernel
        # Align the reference bin to the index 0
        temp_pad = np.roll(temp_pad, -ref_bin[itemp])
        # Time reverse the template (for convolution)
        temp_pad = np.roll(temp_pad[::-1], 1)
        temp_norm = normalize_template(temp_pad)
        conv = np.fft.irfft(data_fft * np.fft.rfft(temp_norm))
        convs[itemp, :] = conv[:nbins]
    return convs


@njit(cache=True, fastmath=True)
def nb_fft_good_size(n: int, real: bool = False) -> int:  # noqa: FBT001, FBT002
    """Get the good size for FFT.

    Parameters
    ----------
    n : int
        Input size.

    real : bool, optional
        If True, the input is real, by default False

    Returns
    -------
    int
        Good size for FFT.
    """
    return rocket_fft.good_size(n, real=real)


@njit(cache=True, fastmath=True)
def nb_rfft(arr: np.ndarray, n: int | None = None) -> np.ndarray:
    """Compute the 1D FFT of a real array.

    Parameters
    ----------
    arr : np.ndarray
        Input array (real).
    n : int | None, optional
        Length of the FFT, by default None

    Returns
    -------
    np.ndarray
        Real part of the FFT.
    """
    return np.fft.rfft(arr, n)


@njit(cache=True, fastmath=True)
def nb_irfft(arr: np.ndarray, n: int | None = None) -> np.ndarray:
    return np.fft.irfft(arr, n)


@njit(cache=True, fastmath=True)
def nb_fft(arr: np.ndarray, n: int | None = None) -> np.ndarray:
    return np.fft.fft(arr, n)


@njit(cache=True, fastmath=True)
def nb_ifft(arr: np.ndarray, n: int | None = None) -> np.ndarray:
    return np.fft.ifft(arr, n)


@njit(cache=True, fastmath=True)
def fftconvolve(in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
    """Convolve two 1D arrays using FFT in mode "full".

    Parameters
    ----------
    in1 : np.ndarray
        First input array.
    in2 : np.ndarray
        Second input array.

    Returns
    -------
    np.ndarray
        Convolved array in mode "full".

    Notes
    -----
    Return full discrete linear convolution of in1 and in2.
    """
    if in1.ndim != 1 or in2.ndim != 1:
        msg = "Input arrays must be 1D."
        raise ValueError(msg)
    n1 = len(in1)
    n2 = len(in2)
    if n1 == 0 or n2 == 0:
        return np.zeros(0, dtype=in1.dtype)
    n = n1 + n2 - 1
    n_good = nb_fft_good_size(n, real=True)
    sp1 = np.fft.rfft(in1, n_good)
    sp2 = np.fft.rfft(in2, n_good)
    ret = np.fft.irfft(sp1 * sp2, n_good)
    return ret[:n]


@njit(cache=True, fastmath=True)
def circular_pad_goodsize(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    n_good = nb_fft_good_size(n, real=True)
    result = np.empty(n_good, dtype=arr.dtype)
    for i in range(n_good):
        result[i] = arr[i % n]
    return result


@njit(cache=True, fastmath=True)
def normalize_template(arr: np.ndarray) -> np.ndarray:
    """Normalize the template to have zero mean and unit power.

    Parameters
    ----------
    arr : np.ndarray
        Template array.

    Returns
    -------
    np.ndarray
        Normalized template array.
    """
    mean = np.mean(arr)
    arr_norm = arr - mean
    norm = np.sqrt(np.sum(arr_norm**2))
    if norm == 0:
        return arr_norm
    return arr_norm / norm


@njit(cache=True, nogil=True, fastmath=True)
def nb_roll(
    arr: np.ndarray,
    shift: int | tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Roll array elements along a given axis.

    This is a Numba-compiled wrapper around `numpy.roll`, implemented via `rocket-fft`
    to support the `axis` argument.

    Parameters
    ----------
    arr : ndarray
        Input array.
    shift : int | tuple[int, ...]
        Number of positions to shift. Positive shifts right/down, negative left/up.
        If a tuple, must match the length of `axis`.
    axis : int | tuple[int, ...] | None, optional
        Axis or axes to roll along. If None, flattens array and rolls all elements.

    Returns
    -------
    np.ndarray
        Rolled array with the same shape as ``arr``.
    """
    return np.roll(arr, shift, axis)


@njit(cache=True, fastmath=True)
def roll_block(arr: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Roll each row of a 2D array along columns by per-row shifts.

    Applies a circular shift to each row independently, wrapping elements around
    the column axis. Positive shifts move elements right, negative shifts move left.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D array of shape (nrows, ncols).
    shifts : np.ndarray
        1D array of integer shifts, length equal to nrows. Can be positive or negative.

    Returns
    -------
    np.ndarray
        Rolled 2D array with the same shape as `arr`.

    Raises
    ------
    ValueError
        If `arr` is not 2D or `shifts` length does not match number of rows.

    """
    if arr.ndim != 2:
        msg = "Input array must be 2D."
        raise ValueError(msg)
    if len(shifts) != arr.shape[0]:
        msg = "Number of shifts must be equal to the number of rows."
        raise ValueError(msg)
    nrows, ncols = arr.shape
    res = np.empty_like(arr)
    for irow in range(nrows):
        shift = shifts[irow] % ncols
        if shift == 0:
            res[irow] = arr[irow]
        else:
            res[irow, shift:] = arr[irow, : ncols - shift]
            res[irow, :shift] = arr[irow, ncols - shift :]
    return res


@njit(cache=True, fastmath=True)
def roll_block_valid(arr: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Roll each row of a 2D array by per-row shifts, keeping only valid columns.

    Similar to `roll_block` but only keeps the valid region where no wrapping occurs.
    Positive shifts move elements right, negative shifts move elements left.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D array of shape (nrows, ncols).
    shifts : np.ndarray
        1D array of integer shifts, length equal to nrows. Can be positive or negative.

    Returns
    -------
    np.ndarray
        Rolled 2D array with shape (nrows, ncols - shift_range).

    Raises
    ------
    ValueError
        If `arr` is not 2D or `shifts` length doesn't match number of rows.
        If the shift range exceeds the number of columns.

    """
    if arr.ndim != 2:
        msg = "Input array must be 2D."
        raise ValueError(msg)
    if len(shifts) != arr.shape[0]:
        msg = "Number of shifts must be equal to the number of rows."
        raise ValueError(msg)
    nrows, ncols = arr.shape
    max_pos_shift = max(0, np.max(shifts))
    min_neg_shift = min(0, np.min(shifts))

    # Calculate the valid region size
    start_col = max_pos_shift
    end_col = ncols + min_neg_shift
    valid_cols = end_col - start_col
    if valid_cols <= 0:
        msg = (
            f"Not enough samples. Required at least {max_pos_shift - min_neg_shift} "
            f"samples, given {ncols}."
        )
        raise ValueError(msg)
    res = np.empty((nrows, valid_cols), dtype=arr.dtype)
    for irow in range(nrows):
        shift = shifts[irow]
        res[irow, :] = arr[irow, start_col - shift : end_col - shift]
    return res


@njit(cache=True, fastmath=True)
def dmt_block(arr: np.ndarray, dm_delays: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or dm_delays.ndim != 2:
        msg = "Input array and delays must be 2D."
        raise ValueError(msg)
    if arr.shape[0] != dm_delays.shape[1]:
        msg = "Number of chans must be same in both arrays."
        raise ValueError(msg)
    _, nsamps = arr.shape
    ndms, _ = dm_delays.shape
    res = np.empty((ndms, nsamps), dtype=arr.dtype)
    for idm in range(ndms):
        res[idm] = np.sum(roll_block(arr, dm_delays[idm]), axis=0)
    return res


@njit(cache=True, fastmath=True)
def dmt_block_valid(arr: np.ndarray, dm_delays: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or dm_delays.ndim != 2:
        msg = "Input array and delays must be 2D."
        raise ValueError(msg)
    if arr.shape[0] != dm_delays.shape[1]:
        msg = "Number of chans must be same in both arrays."
        raise ValueError(msg)
    _, nsamps = arr.shape
    ndms, _ = dm_delays.shape
    max_pos_shift = max(0, np.max(dm_delays))
    min_neg_shift = min(0, np.min(dm_delays))
    valid_samples = nsamps + min_neg_shift - max_pos_shift
    if valid_samples <= 0:
        msg = (
            f"Not enough samples. Required at least {max_pos_shift - min_neg_shift} "
            f"samples, given {nsamps}."
        )
        raise ValueError(msg)
    res = np.empty((ndms, valid_samples), dtype=arr.dtype)
    for idm in range(ndms):
        res[idm] = np.sum(roll_block_valid(arr, dm_delays[idm]), axis=0)
    return res


@njit(cache=True, fastmath=True)
def disperse_block(
    arr: np.ndarray,
    shifts: np.ndarray,
    nsamps_out: int = 1,
    tfactor: int = 1,
) -> np.ndarray:
    """Roll the 2D array along the second axis.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D array.
    shifts : np.ndarray
        Array of shifts for each row.
    nsamps_out : int, optional
        Desired minimum number of samples in the output, by default 1.
    tfactor : int, optional
        Time factor to decimate the output, by default 1.

    Returns
    -------
    np.ndarray
        Dispersed 2D array.
    """
    if arr.ndim != 2:
        msg = "Input array must be 2D."
        raise ValueError(msg)
    if len(shifts) != arr.shape[0]:
        msg = "Number of shifts must be equal to the number of rows."
        raise ValueError(msg)
    nrows, ncols = arr.shape
    valid_samps = max(2 * (ncols + np.abs(shifts).max()), nsamps_out)
    res = np.zeros((nrows, valid_samps), dtype=arr.dtype)
    start = valid_samps // 2 - ncols // 2
    end = start + ncols
    for irow in range(nrows):
        res[irow, start + shifts[irow] : end + shifts[irow]] = arr[irow]
    return res


@njit(cache=True, fastmath=True, parallel=True)
def simulate_ism(
    signal: np.ndarray,
    spectrum: np.ndarray,
    dm_smear: np.ndarray,
    tau_nus: np.ndarray,
    over_sampling: int = 1,
) -> np.ndarray:
    """Convolve the input signal with the ISM effects.

    Parameters
    ----------
    signal : np.ndarray
        1D pulse profile template.
    spectrum : np.ndarray
        Spectrum of the pulse profile (length = number of frequency channels).
    dm_smear : np.ndarray
        DM smearing (in samples) for each frequency channel.
    tau_nus : np.ndarray
        Scattering timescale (in samples) for each frequency channel.
    over_sampling : int, optional
        Oversampling factor for higher time resolution, by default 1.

    Returns
    -------
    np.ndarray
        Convolved 2D pulse dynamic spectrum (nchans x nsamps).

    Notes
    -----
    The function performs the following steps for each frequency channel:
    - Replicate the signal across all frequency channels.
    - Apply DM smearing via convolution with a boxcar function.
    - Apply scattering via convolution with an exponential decay function.
    - Normalize the scattering kernel to maintain the signal area.
    """
    nsamps_pure = len(signal)
    nchans = len(spectrum)
    nsamps_smear = int(max(over_sampling, np.ceil(dm_smear.max())))
    nsamps_scat = int(max(over_sampling, np.ceil(int(6 * tau_nus.max()))))
    max_len = nsamps_pure + nsamps_smear + nsamps_scat - 2
    final_arr = np.zeros((nchans, max_len), dtype=signal.dtype)

    x_scat = np.arange(nsamps_scat, dtype=signal.dtype)
    do_smear = dm_smear.max() > 0
    do_scatter = tau_nus.max() > 0

    for ichan in prange(nchans):
        chan_data = signal * spectrum[ichan]
        # Apply dm smearing
        if do_smear:
            dm_smear_samps = int(max(1, np.ceil(dm_smear[ichan])))
            dm_smear_prof = np.zeros(nsamps_smear, dtype=signal.dtype)
            dm_smear_prof[:dm_smear_samps] = 1
            dm_smear_prof /= dm_smear_prof.sum()
            chan_data = fftconvolve(chan_data, dm_smear_prof)
        # Apply scattering
        if do_scatter:
            scat_prof = np.exp(-x_scat / tau_nus[ichan])
            scat_prof /= scat_prof.sum()
            chan_data = fftconvolve(chan_data, scat_prof)
        final_arr[ichan] = chan_data[:max_len]
    return final_arr
