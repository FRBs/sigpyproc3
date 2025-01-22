from __future__ import annotations

import numpy as np

from sigpyproc import params


class TestParams:
    def test_compute_dmdelays(self) -> None:
        freqs = np.arange(1000, dtype=np.float32) * (-1.0) + 1400
        dm = 100
        tsamp = 0.001
        delays = params.compute_dmdelays(freqs, dm, tsamp, freqs[0], in_samples=False)
        assert delays.shape == (freqs.size,)
        assert delays.dtype == np.float32
        assert delays[0] == 0
        delays_samp = params.compute_dmdelays(freqs, dm, tsamp, freqs[0])
        assert delays_samp.shape == (freqs.size,)
        assert delays_samp.dtype == np.int32
        assert delays_samp[0] == 0

    def test_compute_dmdelays_vect(self) -> None:
        freqs = np.arange(1000, dtype=np.float32) * (-1.0) + 1400
        dm = np.linspace(0, 100, 50)
        tsamp = 0.001
        delays = params.compute_dmdelays(freqs, dm, tsamp, freqs[0], in_samples=False)
        assert delays.shape == (dm.size, freqs.size)
        assert delays.dtype == np.float32
        np.testing.assert_equal(delays[:, 0], np.zeros(dm.size))
        delays_samp = params.compute_dmdelays(freqs, dm, tsamp, freqs[0])
        assert delays_samp.shape == (dm.size, freqs.size)
        assert delays_samp.dtype == np.int32
        np.testing.assert_equal(delays_samp[:, 0], np.zeros(dm.size))

    def test_compute_dmsmearing(self) -> None:
        freqs = np.arange(1000, dtype=np.float32) * (-1.0) + 1400
        dm = 100
        tsamp = 0.001
        smearing = params.compute_dmsmearing(freqs, dm, tsamp, in_samples=False)
        assert smearing.shape == (freqs.size,)
        assert smearing.dtype == np.float32
        smearing_samp = params.compute_dmsmearing(freqs, dm, tsamp)
        assert smearing_samp.shape == (freqs.size,)
        assert smearing_samp.dtype == np.int32

    def test_compute_dmsmearing_vect(self) -> None:
        freqs = np.arange(1000, dtype=np.float32) * (-1.0) + 1400
        dm = np.linspace(0, 100, 50)
        tsamp = 0.001
        smearing = params.compute_dmsmearing(freqs, dm, tsamp, in_samples=False)
        assert smearing.shape == (dm.size, freqs.size)
        assert smearing.dtype == np.float32
        smearing_samp = params.compute_dmsmearing(freqs, dm, tsamp)
        assert smearing_samp.shape == (dm.size, freqs.size)
        assert smearing_samp.dtype == np.int32
