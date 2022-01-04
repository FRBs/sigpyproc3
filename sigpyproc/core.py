import numpy as np
import bottleneck as bn

from sigpyproc import kernels


def running_median(array, window):
    """
    Calculate the running median of an array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to calculate the running median of.

    Returns
    -------
    numpy.ndarray
        The running median of the array.

    """
    pad_size = (
        (window // 2, window // 2) if window % 2 else (window // 2, window // 2 - 1)
    )
    padded = np.pad(array, pad_size, "symmetric")

    median = bn.move_median(padded, window)
    return median[window - 1 :]


def running_mean(array, window):
    """
    Calculate the running mean of an array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to calculate the running mean of.

    Returns
    -------
    numpy.ndarray
        The running mean of the array.

    """
    pad_size = (
        (window // 2, window // 2) if window % 2 else (window // 2, window // 2 - 1)
    )
    padded = np.pad(array, pad_size, "symmetric")

    mean = bn.move_mean(padded, window)
    return mean[window - 1 :]


def unpack(array, nbits):
    """Unpack 1, 2 and 4 bit array. Only unpacks in big endian bit ordering.

    Parameters
    ----------
    array : ndarray
        Array to unpack.
    nbits : int
        Number of bits to unpack.

    Returns
    -------
    ndarray
        Unpacked array.

    Raises
    ------
    ValueError
        if nbits is not 1, 2, or 4
    """
    assert array.dtype == np.uint8, "Array must be uint8"
    if nbits == 1:
        unpacked = np.unpackbits(array, bitorder="big")
    elif nbits == 2:
        unpacked = kernels.unpack2_8(array)
    elif nbits == 4:
        unpacked = kernels.unpack4_8(array)
    else:
        raise ValueError("nbits must be 1, 2, or 4")
    return unpacked


def pack(array, nbits):
    """Pack 1, 2 and 4 bit array. Only packs in big endian bit ordering.

    Parameters
    ----------
    array : ndarray
        Array to pack.
    nbits : int
        Number of bits to pack.

    Returns
    -------
    ndarray
        Packed array.

    Raises
    ------
    ValueError
        if nbits is not 1, 2, or 4
    """
    assert array.dtype == np.uint8, "Array must be uint8"
    if nbits == 1:
        packed = np.packbits(array, bitorder="big")
    elif nbits == 2:
        packed = kernels.pack2_8(array)
    elif nbits == 4:
        packed = kernels.pack4_8(array)
    else:
        raise ValueError("nbits must be 1, 2, or 4")
    return packed


def fold_tim(array, tsamp, period, accel, nsamps, nbins, nints):
    fold_ar = np.zeros(nbins * nints, dtype=np.float64)
    count_ar = np.zeros(nbins * nints, dtype=np.int32)
    kernels.fold(
        array,
        fold_ar,
        count_ar,
        np.array([0]),
        0,
        tsamp,
        period,
        accel,
        nsamps,
        nsamps,
        1,
        nbins,
        nints,
        1,
        0,
    )
    fold_ar /= count_ar
    return fold_ar.reshape(nints, 1, nbins)
