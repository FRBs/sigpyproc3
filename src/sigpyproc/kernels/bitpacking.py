"""Numba kernels for bit packing and unpacking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, types

if TYPE_CHECKING:
    from numpy.typing import NDArray


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def unpack1_8_big(array: NDArray[np.uint8], unpacked: NDArray[np.uint8]) -> None:
    for ii in range(array.size):
        pos = ii * 8
        for jj in range(8):
            unpacked[pos + (7 - jj)] = (array[ii] >> jj) & 1


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def unpack1_8_little(array: NDArray[np.uint8], unpacked: NDArray[np.uint8]) -> None:
    for ii in range(array.size):
        pos = ii * 8
        for jj in range(8):
            unpacked[pos + jj] = (array[ii] >> jj) & 1


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def unpack2_8_big(array: NDArray[np.uint8], unpacked: NDArray[np.uint8]) -> None:
    for ii in range(array.size):
        pos = ii * 4
        unpacked[pos + 3] = (array[ii] & 0x03) >> 0
        unpacked[pos + 2] = (array[ii] & 0x0C) >> 2
        unpacked[pos + 1] = (array[ii] & 0x30) >> 4
        unpacked[pos + 0] = (array[ii] & 0xC0) >> 6


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def unpack2_8_little(array: NDArray[np.uint8], unpacked: NDArray[np.uint8]) -> None:
    for ii in range(array.size):
        pos = ii * 4
        unpacked[pos + 0] = (array[ii] & 0x03) >> 0
        unpacked[pos + 1] = (array[ii] & 0x0C) >> 2
        unpacked[pos + 2] = (array[ii] & 0x30) >> 4
        unpacked[pos + 3] = (array[ii] & 0xC0) >> 6


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def unpack4_8_big(array: NDArray[np.uint8], unpacked: NDArray[np.uint8]) -> None:
    for ii in range(array.size):
        pos = ii * 2
        unpacked[pos + 1] = (array[ii] & 0x0F) >> 0
        unpacked[pos + 0] = (array[ii] & 0xF0) >> 4


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def unpack4_8_little(array: NDArray[np.uint8], unpacked: NDArray[np.uint8]) -> None:
    for ii in range(array.size):
        pos = ii * 2
        unpacked[pos + 0] = (array[ii] & 0x0F) >> 0
        unpacked[pos + 1] = (array[ii] & 0xF0) >> 4


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True)
def pack1_8_big(array: NDArray[np.uint8], packed: NDArray[np.uint8]) -> None:
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
def pack1_8_little(array: NDArray[np.uint8], packed: NDArray[np.uint8]) -> None:
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
    array: NDArray[np.uint8],
    packed: NDArray[np.uint8],
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

    for j in range(i, nwords):
        x = array_uint64[j]
        x &= mask
        x *= magic
        packed[j] = types.uint8(x >> shift)


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def pack2_8_big(array: NDArray[np.uint8], packed: NDArray[np.uint8]) -> None:
    for ii in range(packed.size):
        pos = ii * 4
        packed[ii] = (
            (array[pos + 0] << 6)
            | (array[pos + 1] << 4)
            | (array[pos + 2] << 2)
            | array[pos + 3]
        )


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def pack2_8_little(array: NDArray[np.uint8], packed: NDArray[np.uint8]) -> None:
    for ii in range(packed.size):
        pos = ii * 4
        packed[ii] = (
            (array[pos + 3] << 6)
            | (array[pos + 2] << 4)
            | (array[pos + 1] << 2)
            | array[pos + 0]
        )


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def pack4_8_big(array: NDArray[np.uint8], packed: NDArray[np.uint8]) -> None:
    for ii in range(packed.size):
        pos = ii * 2
        packed[ii] = (array[pos + 0] << 4) | array[pos + 1]


@njit("void(u1[::1], u1[::1])", cache=True, fastmath=True, locals={"pos": types.i8})
def pack4_8_little(array: NDArray[np.uint8], packed: NDArray[np.uint8]) -> None:
    for ii in range(packed.size):
        pos = ii * 2
        packed[ii] = (array[pos + 1] << 4) | array[pos + 0]


_UNPACK_MAP = {
    (1, "big"): unpack1_8_big,
    (1, "little"): unpack1_8_little,
    (2, "big"): unpack2_8_big,
    (2, "little"): unpack2_8_little,
    (4, "big"): unpack4_8_big,
    (4, "little"): unpack4_8_little,
}

_PACK_MAP = {
    (1, "big"): pack1_8_big,
    (1, "little"): pack1_8_little,
    (2, "big"): pack2_8_big,
    (2, "little"): pack2_8_little,
    (4, "big"): pack4_8_big,
    (4, "little"): pack4_8_little,
}


def unpack_array(
    array: NDArray[np.uint8],
    nbits: int,
    unpacked: NDArray[np.uint8],
    bitorder_str: str,
) -> None:
    func = _UNPACK_MAP.get((nbits, bitorder_str))
    if func is None:
        msg = f"No unpack kernel for nbits={nbits}, bitorder={bitorder_str}"
        raise ValueError(msg)
    func(array, unpacked)


def pack_array(
    array: NDArray[np.uint8],
    nbits: int,
    packed: NDArray[np.uint8],
    bitorder_str: str,
) -> None:
    func = _PACK_MAP.get((nbits, bitorder_str))
    if func is None:
        msg = f"No pack kernel for nbits={nbits}, bitorder={bitorder_str}"
        raise ValueError(msg)
    func(array, packed)
