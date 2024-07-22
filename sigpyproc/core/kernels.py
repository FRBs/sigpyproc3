from __future__ import annotations

import numpy as np
from numba import njit, prange, types

CONST_C_VAL = 299792458.0  # Speed of light in m/s (astropy.constants.c.value)


def packunpack_njit(func: types.FunctionType) -> types.FunctionType:
    return njit(
        "void(u1[::1], u1[::1])",
        cache=True,
        parallel=True,
        fastmath=True,
        locals={"pos": types.i8},
    )(func)


def packunpack_njit_serial(func: types.FunctionType) -> types.FunctionType:
    return njit(
        "void(u1[::1], u1[::1])",
        cache=True,
        parallel=False,
        fastmath=True,
        locals={"pos": types.i8},
    )(func)


@packunpack_njit
def unpack1_8_big(array: np.ndarray, unpacked: np.ndarray) -> None:
    for ii in prange(array.size):
        pos = ii * 8
        for jj in range(8):
            unpacked[pos + (7 - jj)] = (array[ii] >> jj) & 1


@packunpack_njit
def unpack1_8_little(array: np.ndarray, unpacked: np.ndarray) -> None:
    for ii in prange(array.size):
        pos = ii * 8
        for jj in range(8):
            unpacked[pos + jj] = (array[ii] >> jj) & 1


@packunpack_njit
def unpack2_8_big(array: np.ndarray, unpacked: np.ndarray) -> None:
    for ii in prange(array.size):
        pos = ii * 4
        unpacked[pos + 3] = (array[ii] & 0x03) >> 0
        unpacked[pos + 2] = (array[ii] & 0x0C) >> 2
        unpacked[pos + 1] = (array[ii] & 0x30) >> 4
        unpacked[pos + 0] = (array[ii] & 0xC0) >> 6


@packunpack_njit
def unpack2_8_little(array: np.ndarray, unpacked: np.ndarray) -> None:
    for ii in prange(array.size):
        pos = ii * 4
        unpacked[pos + 0] = (array[ii] & 0x03) >> 0
        unpacked[pos + 1] = (array[ii] & 0x0C) >> 2
        unpacked[pos + 2] = (array[ii] & 0x30) >> 4
        unpacked[pos + 3] = (array[ii] & 0xC0) >> 6


@packunpack_njit
def unpack4_8_big(array: np.ndarray, unpacked: np.ndarray) -> None:
    for ii in prange(array.size):
        pos = ii * 2
        unpacked[pos + 1] = (array[ii] & 0x0F) >> 0
        unpacked[pos + 0] = (array[ii] & 0xF0) >> 4


@packunpack_njit
def unpack4_8_little(array: np.ndarray, unpacked: np.ndarray) -> None:
    for ii in prange(array.size):
        pos = ii * 2
        unpacked[pos + 0] = (array[ii] & 0x0F) >> 0
        unpacked[pos + 1] = (array[ii] & 0xF0) >> 4


@packunpack_njit
def pack1_8_big(array: np.ndarray, packed: np.ndarray) -> None:
    for ii in prange(packed.size):
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


@packunpack_njit
def pack1_8_little(array: np.ndarray, packed: np.ndarray) -> None:
    for ii in prange(packed.size):
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


@packunpack_njit
def pack2_8_big(array: np.ndarray, packed: np.ndarray) -> None:
    for ii in prange(packed.size):
        pos = ii * 4
        packed[ii] = (
            (array[pos + 0] << 6)
            | (array[pos + 1] << 4)
            | (array[pos + 2] << 2)
            | array[pos + 3]
        )


@packunpack_njit
def pack2_8_little(array: np.ndarray, packed: np.ndarray) -> None:
    for ii in prange(packed.size):
        pos = ii * 4
        packed[ii] = (
            (array[pos + 3] << 6)
            | (array[pos + 2] << 4)
            | (array[pos + 1] << 2)
            | array[pos + 0]
        )


@packunpack_njit
def pack4_8_big(array: np.ndarray, packed: np.ndarray) -> None:
    for ii in prange(packed.size):
        pos = ii * 2
        packed[ii] = (array[pos + 0] << 4) | array[pos + 1]


@packunpack_njit
def pack4_8_little(array: np.ndarray, packed: np.ndarray) -> None:
    for ii in prange(packed.size):
        pos = ii * 2
        packed[ii] = (array[pos + 1] << 4) | array[pos + 0]


unpack1_8_big_serial = packunpack_njit_serial(unpack1_8_big.py_func)
unpack1_8_little_serial = packunpack_njit_serial(unpack1_8_little.py_func)
unpack2_8_big_serial = packunpack_njit_serial(unpack2_8_big.py_func)
unpack2_8_little_serial = packunpack_njit_serial(unpack2_8_little.py_func)
unpack4_8_big_serial = packunpack_njit_serial(unpack4_8_big.py_func)
unpack4_8_little_serial = packunpack_njit_serial(unpack4_8_little.py_func)
pack1_8_big_serial = packunpack_njit_serial(pack1_8_big.py_func)
pack1_8_little_serial = packunpack_njit_serial(pack1_8_little.py_func)
pack2_8_big_serial = packunpack_njit_serial(pack2_8_big.py_func)
pack2_8_little_serial = packunpack_njit_serial(pack2_8_little.py_func)
pack4_8_big_serial = packunpack_njit_serial(pack4_8_big.py_func)
pack4_8_little_serial = packunpack_njit_serial(pack4_8_little.py_func)


@njit(
    [
        "u1[:](u1[:], i8)",
        "f4[:](f4[:], i8)",
        "f8[:](f8[:], i8)",
    ],
    cache=True,
    parallel=True,
    fastmath=True,
    locals={"temp": types.f8},
)
def downsample_1d(array: np.ndarray, factor: int) -> np.ndarray:
    nsamps_new = len(array) // factor
    result = np.empty(nsamps_new, dtype=array.dtype)
    for isamp in prange(nsamps_new):
        temp = 0
        for ifactor in range(factor):
            temp += array[isamp * factor + ifactor]
        result[isamp] = temp / factor
    return result


@njit(
    [
        "u1[:](u1[:], i8, i8, i8, i8)",
        "f4[:](f4[:], i8, i8, i8, i8)",
        "f8[:](f8[:], i8, i8, i8, i8)",
    ],
    cache=True,
    parallel=True,
    fastmath=True,
    locals={"temp": types.f8},
)
def downsample_2d(
    array: np.ndarray,
    tfactor: int,
    ffactor: int,
    nchans: int,
    nsamps: int,
) -> np.ndarray:
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
                temp += np.sum(array[ipos : ipos + ffactor])
            result[nchans_new * isamp + ichan] = temp / totfactor
    return result


@njit(
    ["void(u1[:], f4[:], i8, i8, i8)", "void(f4[:], f4[:], i8, i8, i8)"],
    cache=True,
    parallel=True,
    fastmath=True,
)
def extract_tim(
    inarray: np.ndarray,
    outarray: np.ndarray,
    nchans: int,
    nsamps: int,
    index: int,
) -> None:
    for isamp in prange(nsamps):
        for ichan in range(nchans):
            outarray[index + isamp] += inarray[nchans * isamp + ichan]


@njit(
    ["void(u1[:], f4[:], i4, i4)", "void(f4[:], f4[:], i4, i4)"],
    cache=True,
    parallel=True,
    fastmath=True,
)
def extract_bpass(
    inarray: np.ndarray,
    outarray: np.ndarray,
    nchans: int,
    nsamps: int,
) -> None:
    for ichan in prange(nchans):
        for isamp in range(nsamps):
            outarray[ichan] += inarray[nchans * isamp + ichan]


@njit(
    ["void(u1[:], b1[:], u1, i8, i8)", "void(f4[:], b1[:], f4, i8, i8)"],
    cache=True,
    parallel=True,
    fastmath=True,
)
def mask_channels(
    array: np.ndarray,
    mask: np.ndarray,
    maskvalue: float,
    nchans: int,
    nsamps: int,
) -> None:
    for ichan in prange(nchans):
        if mask[ichan]:
            for isamp in range(nsamps):
                array[nchans * isamp + ichan] = maskvalue


@njit(
    [
        "void(u1[:], f4[:], i4[:], i4, i4, i4, i4)",
        "void(f4[:], f4[:], i4[:], i4, i4, i4, i4)",
    ],
    cache=True,
    parallel=True,
    fastmath=True,
)
def dedisperse(
    inarray: np.ndarray,
    outarray: np.ndarray,
    delays: np.ndarray,
    maxdelay: int,
    nchans: int,
    nsamps: int,
    index: int,
) -> None:
    for isamp in prange(nsamps - maxdelay):
        for ichan in range(nchans):
            outarray[index + isamp] += inarray[nchans * (isamp + delays[ichan]) + ichan]


@njit(
    ["u1[:](u1[:], i4, i4)", "f4[:](f4[:], i4, i4)"],
    cache=True,
    parallel=True,
    fastmath=True,
)
def invert_freq(array: np.ndarray, nchans: int, nsamps: int) -> np.ndarray:
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
        "void(u1[:], u1[:], f4[:], f4[:], i8, i8)",
        "void(f4[:], f4[:], f4[:], f4[:], i8, i8)",
    ],
    cache=True,
    parallel=True,
    fastmath=True,
    locals={"zerodm": types.f8, "result": types.f8},
)
def remove_zerodm(
    inarray: np.ndarray,
    outarray: np.ndarray,
    bpass: np.ndarray,
    chanwts: np.ndarray,
    nchans: int,
    nsamps: int,
) -> None:
    for isamp in prange(nsamps):
        zerodm = 0
        for ichan in range(nchans):
            zerodm += inarray[nchans * isamp + ichan]

        for ichan in range(nchans):
            pos = nchans * isamp + ichan
            result = (inarray[pos] - zerodm * chanwts[ichan]) + bpass[ichan]
            outarray[pos] = result


@njit("void(f4[:], f4[:])", cache=True)
def form_spec(fft_ar: np.ndarray, spec_ar: np.ndarray) -> None:
    specsize = len(spec_ar)
    for ispec in range(specsize):
        spec_ar[ispec] = np.sqrt(fft_ar[2 * ispec] ** 2 + fft_ar[2 * ispec + 1] ** 2)


@njit("void(f4[:], f4[:])", cache=True, locals={"re_l": types.f4, "im_l": types.f4})
def form_spec_interpolated(fft_ar: np.ndarray, spec_ar: np.ndarray) -> None:
    specsize = len(spec_ar)
    re_l = 0
    im_l = 0
    for ispec in range(specsize):
        re = fft_ar[2 * ispec]
        im = fft_ar[2 * ispec + 1]
        ampsq = re**2 + im**2
        ampsq_diff = 0.5 * ((re - re_l) ** 2 + (im - im_l) ** 2)
        spec_ar[ispec] = np.sqrt(max(ampsq, ampsq_diff))
        re_l = re
        im_l = im


@njit(
    "void(f4[:], f4[:], i4, i4, i4)",
    cache=True,
    locals={"norm": types.f4, "slope": types.f4},
)
def remove_rednoise_presto(
    fft_ar: np.ndarray,
    out_ar: np.ndarray,
    start_width: int,
    end_width: int,
    end_freq_bin: int,
) -> None:
    nspecs = len(fft_ar) // 2
    oldinbuf = np.empty(2 * end_width, dtype=np.float32)
    newinbuf = np.empty(2 * end_width, dtype=np.float32)
    realbuffer = np.empty(end_width, dtype=np.float32)
    binnum = 1
    bufflen = start_width

    # Set DC bin to 1.0
    out_ar[0] = 1.0
    windex = 2
    rindex = 2

    # transfer bufflen complex samples to oldinbuf
    oldinbuf[: 2 * bufflen] = fft_ar[rindex : rindex + 2 * bufflen]
    numread_old = bufflen
    rindex += 2 * bufflen

    # calculate powers for oldinbuf
    for ispec in range(numread_old):
        realbuffer[ispec] = oldinbuf[2 * ispec] ** 2 + oldinbuf[2 * ispec + 1] ** 2

    # calculate first median of our data and determine next bufflen
    mean_old = np.median(realbuffer[:numread_old]) / np.log(2.0)
    binnum += numread_old
    bufflen = round(start_width * np.log(binnum))

    while rindex // 2 < nspecs:
        numread_new = min(bufflen, nspecs - rindex // 2)

        # transfer numread_new complex samples to newinbuf
        newinbuf[: 2 * numread_new] = fft_ar[rindex : rindex + 2 * numread_new]
        rindex += 2 * numread_new

        # calculate powers for newinbuf
        for ispec in range(numread_new):
            realbuffer[ispec] = newinbuf[2 * ispec] ** 2 + newinbuf[2 * ispec + 1] ** 2

        mean_new = np.median(realbuffer[:numread_new]) / np.log(2.0)
        slope = (mean_new - mean_old) / (numread_old + numread_new)

        for ispec in range(numread_old):
            norm = 1 / np.sqrt(
                mean_old + slope * ((numread_old + numread_new) / 2 - ispec),
            )
            out_ar[2 * ispec + windex] = oldinbuf[2 * ispec] * norm
            out_ar[2 * ispec + 1 + windex] = oldinbuf[2 * ispec + 1] * norm

        windex += 2 * numread_old
        binnum += numread_new
        if binnum < end_freq_bin:
            bufflen = int(start_width * np.log(binnum))
        else:
            bufflen = end_width
        numread_old = numread_new
        mean_old = mean_new
        oldinbuf[: 2 * numread_new] = newinbuf[: 2 * numread_new]

    # Remaining samples
    norm = 1 / np.sqrt(mean_old)
    out_ar[windex : windex + 2 * numread_old] = oldinbuf[: 2 * numread_old] * norm


@njit("void(f4[:], f4[:], i4[:], i4[:], i4, i4, i4)", cache=True)
def sum_harms(
    spec_arr: np.ndarray,
    sum_arr: np.ndarray,
    harm_arr: np.ndarray,
    fact_arr: np.ndarray,
    nharms: int,
    nsamps: int,
    nfold: int,
) -> None:
    for ifold in range(nfold, nsamps - (nharms - 1), nharms):
        for iharm in range(nharms):
            for kk in range(nharms // 2):
                sum_arr[ifold + iharm] += spec_arr[
                    fact_arr[kk] + harm_arr[iharm * nharms // 2 + kk]
                ]
        for kk in range(nharms // 2):
            fact_arr[kk] += 2 * kk + 1


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


def add_online_moments(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    c["count"] = a["count"] + b["count"]
    delta = b["m1"] - a["m1"]
    delta2 = delta * delta
    delta3 = delta * delta2
    delta4 = delta2 * delta2

    c["m1"] = (a["count"] * a["m1"] + b["count"] * b["m1"]) / c["count"]
    c["m2"] = a["m2"] + b["m2"] + delta2 * a["count"] * b["count"] / c["count"]
    c["m3"] = (
        a["m3"]
        + b["m3"]
        + delta3
        * a["count"]
        * b["count"]
        * (a["count"] - b["count"])
        / (c["count"] ** 2)
    )
    c["m3"] += 3 * delta * (a["count"] * b["m2"] - b["count"] * a["m2"]) / c["count"]
    c["m4"] = (
        a["m4"]
        + b["m4"]
        + delta4
        * a["count"]
        * b["count"]
        * (a["count"] ** 2 - a["count"] * b["count"] + b["count"] ** 2)
        / (c["count"] ** 3)
    )
    c["m4"] += (
        6
        * delta2
        * (a["count"] ** 2 * b["m2"] + b["count"] ** 2 * a["m2"])
        / (c["count"] ** 2)
    )
    c["m4"] += 4 * delta * (a["count"] * b["m3"] - b["count"] * a["m3"]) / c["count"]
    c["max"] = np.maximum(a["max"], b["max"])
    c["min"] = np.minimum(a["min"], b["min"])
