import numpy as np
import logging


def rollArray(y, shift, axis):
    """Roll the elements in the array by 'shift' positions along the
    given axis.

    Parameters
    ----------
    y : :py:obj:`numpy.ndarray`
        array to roll
    shift : int
        number of bins to shift by
    axis : int
        axis to roll along

    Returns
    -------
    :py:obj:`numpy.ndarray`
        shifted numpy array
    """
    y = np.asanyarray(y)
    n = y.shape[axis]
    shift %= n
    return y.take(np.concatenate((np.arange(shift, n), np.arange(shift))), axis)


def _flattenList(n):
    new = []
    repack = lambda x: [new.append(int(y)) for y in x]
    for elem in n:
        if hasattr(elem, "__iter__"):
            repack(elem)
        else:
            new.append(int(elem))
    return new


def stackRecarrays(arrays):
    """Wrapper for stacking :py:obj:`numpy.recarrays`
    """
    return arrays[0].__array_wrap__(np.hstack(arrays))


def nearestFactor(n, val):
    """Find nearest factor.

    :param n: number that we wish to factor
    :type n: int

    :param val: number that we wish to find nearest factor to
    :type val: int

    :return: nearest factor
    :rtype: int
    """
    fact  = [1, n]
    check = 2
    rootn = np.sqrt(n)
    while check < rootn:
        if n % check == 0:
            fact.append(check)
            fact.append(n // check)
        check += 1
    if rootn == check:
        fact.append(check)
    fact.sort()
    return fact[np.abs(np.array(fact) - val).argmin()]


class CustomHandler(logging.StreamHandler):
    def emit(self, record):
        messages = record.msg.splitlines()
        for message in messages:
            record.msg = message
            super().emit(record)


def get_logger(name, formatter=None, level=logging.INFO, logfile=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if formatter is None:
        logformat = "%(asctime)s.%(msecs)03d - %(name)s - %(message)s"
        formatter = logging.Formatter(fmt=logformat, datefmt="%Y-%m-%d %H:%M:%S")

    if not logger.hasHandlers():
        if logfile is None:
            handler = CustomHandler()
        else:
            handler = logging.FileHandler(logfile)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
