"""Kernels that build and inject compact pulse profiles."""

from __future__ import annotations

import numpy as np

from sigpyproc.core.dispersion import compute_dmdelays, compute_dmsmearing
from sigpyproc.kernels.dedispersion import disperse_block
from sigpyproc.kernels.simulation import (
    downsample_profiles,
    inject_pulses,
    simulate_ism,
)


def test_boxcar_smear_matches_rect() -> None:
    signal = np.zeros(8, dtype=np.float32)
    signal[2] = 1.0
    spectrum = np.ones(2, dtype=np.float32)
    dm_smear = np.array([4.0, 2.0], dtype=np.float64)
    tau = np.zeros(2, dtype=np.float64)
    out = simulate_ism(signal, spectrum, dm_smear, tau)
    # Channel 0: width 4 centered around sample 2 (spread across indices 2..5)
    np.testing.assert_allclose(out[0, 2:6], 0.25, atol=1e-5)
    np.testing.assert_allclose(out[0, :2], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[0, 6:], 0.0, atol=1e-6)

    # Channel 1: width 2 centered around sample 2 (spread across indices 3..4)
    np.testing.assert_allclose(out[1, 3:5], 0.5, atol=1e-5)
    np.testing.assert_allclose(out[1, :3], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[1, 5:], 0.0, atol=1e-6)

    # Both channels share the same centroid
    indices = np.arange(out.shape[1])
    centroid0 = float(np.sum(indices * out[0]) / np.sum(out[0]))
    centroid1 = float(np.sum(indices * out[1]) / np.sum(out[1]))
    np.testing.assert_allclose(centroid0, centroid1, atol=1e-5)


def test_scatter_matches_truncated_exponential() -> None:
    signal = np.zeros(4, dtype=np.float32)
    signal[0] = 1.0
    tau = 4.0
    length = int(np.ceil(6.0 * tau))
    kernel = np.exp(-np.arange(length) / tau)
    kernel /= kernel.sum()
    expected = np.convolve(signal, kernel)
    out = simulate_ism(
        signal,
        np.ones(1, dtype=np.float32),
        np.zeros(1, dtype=np.float64),
        np.array([tau], dtype=np.float64),
    )
    np.testing.assert_allclose(out[0, : expected.size], expected, atol=1e-4)
    np.testing.assert_allclose(out[0, expected.size :], 0.0, atol=1e-6)


def test_short_kernels_leave_the_scaled_signal() -> None:
    signal = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    spectrum = np.array([2.0, 1.0], dtype=np.float32)
    out = simulate_ism(
        signal,
        spectrum,
        np.ones(2, dtype=np.float64),
        np.zeros(2, dtype=np.float64),
    )
    np.testing.assert_allclose(out[0, : signal.size], signal * 2.0)
    np.testing.assert_allclose(out[1, : signal.size], signal)


def test_downsample_profiles_averages_groups() -> None:
    fine = np.full((3, 8), 2.0, dtype=np.float32)
    fine[1, :] = np.arange(8, dtype=np.float32)
    out = downsample_profiles(fine, 4)
    np.testing.assert_allclose(out[0], 2.0)
    np.testing.assert_allclose(out[1], [1.5, 5.5])
    assert out.shape == (3, 2)


def test_inject_pulses_places_fractional_delay_and_overlap() -> None:
    block = np.zeros((12, 3), dtype=np.float32)
    profiles = np.zeros((1, 3, 1), dtype=np.float32)
    profiles[0, :, 0] = 1.0
    delays = np.array([[0.0, 1.0, 2.5]], dtype=np.float64)
    inject_pulses(
        block,
        profiles,
        delays,
        np.array([5], dtype=np.int64),
        np.array([2.0], dtype=np.float32),
        np.array([1], dtype=np.int64),
    )
    np.testing.assert_allclose(block[5, 0], 2.0)
    np.testing.assert_allclose(block[6, 1], 2.0)
    np.testing.assert_allclose(block[7, 2], 1.0)
    np.testing.assert_allclose(block[8, 2], 1.0)
    assert np.count_nonzero(block) == 4

    window = np.zeros((5, 1), dtype=np.float32)
    pulse = np.arange(1, 11, dtype=np.float32).reshape(1, 1, 10)
    inject_pulses(
        window,
        pulse,
        np.zeros((1, 1), dtype=np.float64),
        np.array([-3], dtype=np.int64),
        np.array([1.0], dtype=np.float32),
        np.array([10], dtype=np.int64),
    )
    np.testing.assert_allclose(window[:, 0], [4, 5, 6, 7, 8])


def test_disperse_block_shifts_and_downsamples() -> None:
    arr = np.zeros((2, 4), dtype=np.float32)
    arr[:, 0] = 1.0
    out = disperse_block(arr, np.array([0, 2], dtype=np.int32), 1, 1)
    assert out[0, 1] == np.float32(1.0)
    assert out[1, 3] == np.float32(1.0)
    assert out.shape[1] == 4 + 2 + 2

    ones = np.ones((1, 4), dtype=np.float32)
    down = disperse_block(ones, np.array([0], dtype=np.int32), 1, 2)
    np.testing.assert_allclose(down[0, 1:3], 1.0)
    np.testing.assert_allclose(down[0, [0, 3]], 0.0)


def test_dispersion_index_matches_cold_plasma() -> None:
    freqs = np.linspace(1500.0, 1000.0, 64)
    dm = 50.0
    tsamp = 64e-6
    ref = float(freqs[0])
    delays = compute_dmdelays(freqs, dm, tsamp, ref, index=-2.0)
    plain = compute_dmdelays(freqs, dm, tsamp, ref)
    np.testing.assert_array_equal(delays, plain)
    smear = compute_dmsmearing(freqs, dm, tsamp, index=-2.0)
    np.testing.assert_array_equal(smear, compute_dmsmearing(freqs, dm, tsamp))

    other = compute_dmdelays(freqs, dm, tsamp, ref, in_samples=False, index=-3.0)
    scale = -2.0 / -3.0
    expected = dm * 4.148808e3 * scale * (freqs**-3 - ref**-3)
    np.testing.assert_allclose(other, expected, rtol=1e-6)
