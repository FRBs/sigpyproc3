# mypy: disable_error_code="arg-type"

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from matplotlib import pyplot as plt

from sigpyproc.block import FilterbankBlock
from sigpyproc.header import Header
from sigpyproc.simulation import furby

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module", autouse=True)
def dummy_params() -> furby.PulseParams:
    return furby.PulseParams(dm=10.0, snr=10.0, width=2e-3)


@pytest.fixture(scope="module", autouse=True)
def dummy_header() -> Header:
    return Header(
        filename="dummy.fil",
        data_type="filterbank",
        nchans=512,
        foff=-1.0,
        fch1=1500.0,
        nbits=32,
        tsamp=6.4e-05,
        tstart=0.0,
        nsamples=1000,
        source="dummy_source",
    )


@pytest.fixture(scope="module", autouse=True)
def freqs() -> np.ndarray:
    return np.arange(1500, 1000, -1).astype(np.float32)


class TestFurbyGenerator:
    def test_init(
        self,
        dummy_header: Header,
        dummy_params: furby.PulseParams,
    ) -> None:
        gen = furby.FurbyGenerator(dummy_header, dummy_params)
        assert isinstance(gen.hdr, Header)
        assert isinstance(gen.hdr_os, Header)
        assert isinstance(gen.params, furby.PulseParams)
        assert np.isclose(gen.tsamp_os, dummy_header.tsamp / dummy_params.os_fact)

    def test_generate(
        self,
        dummy_header: Header,
        dummy_params: furby.PulseParams,
    ) -> None:
        gen = furby.FurbyGenerator(dummy_header, dummy_params)
        furby_obj = gen.generate()
        assert isinstance(furby_obj, furby.Furby)
        assert isinstance(furby_obj.block, FilterbankBlock)
        assert furby_obj.profile is furby_obj.block
        assert furby_obj.params_hdr == dummy_params
        assert isinstance(furby_obj.stats_hdr, furby.PulseStats)
        assert furby_obj.block.header.tsamp == pytest.approx(dummy_header.tsamp)
        assert furby_obj.block.header.nsamples < 1000
        assert furby_obj.delays.shape == (dummy_header.nchans,)
        np.testing.assert_allclose(
            furby_obj.stats_hdr.box_snr,
            dummy_params.snr,
            rtol=1e-3,
        )

    def test_flat_spectrum_has_equal_fluence(
        self,
        dummy_header: Header,
    ) -> None:
        params = furby.PulseParams(
            dm=10.0,
            snr=10.0,
            width=2e-3,
            spec_kind="flat",
            spec_idx=-2.0,
            dmsmear=False,
        )
        profile = furby.FurbyGenerator(dummy_header, params, seed=1).generate()
        fluence = profile.block.data.sum(axis=1)
        np.testing.assert_allclose(fluence, fluence[0], rtol=1e-4)

    def test_power_law_is_not_flat(self, dummy_header: Header) -> None:
        params = furby.PulseParams(
            dm=10.0,
            snr=10.0,
            width=2e-3,
            spec_kind="power_law",
            spec_idx=-2.0,
            dmsmear=False,
        )
        profile = furby.FurbyGenerator(dummy_header, params, seed=1).generate()
        fluence = profile.block.data.sum(axis=1)
        assert fluence[-1] > fluence[0] * 1.2

    def test_tau_ref_freq_scaling(self, dummy_header: Header) -> None:
        params_1ghz = furby.PulseParams(
            dm=10.0,
            snr=10.0,
            width=2e-3,
            tau0=1e-3,
            tau_ref_freq=1000.0,
            scatt_idx=-4.4,
        )
        params_1500mhz = furby.PulseParams(
            dm=10.0,
            snr=10.0,
            width=2e-3,
            tau0=1e-3,
            tau_ref_freq=1500.0,
            scatt_idx=-4.4,
        )
        gen1 = furby.FurbyGenerator(dummy_header, params_1ghz)
        gen2 = furby.FurbyGenerator(dummy_header, params_1500mhz)
        scat1 = gen1._scatter_samples(gen1.tsamp_os)
        scat2 = gen2._scatter_samples(gen2.tsamp_os)
        ratio = scat1 / scat2
        expected_ratio = (1500.0 / 1000.0) ** (-4.4)
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-4)

    def test_seeded_spectrum_is_reproducible(self, dummy_header: Header) -> None:
        params = furby.PulseParams(
            dm=10.0,
            snr=5.0,
            width=2e-3,
            spec_kind="scintillation",
            spec_idx=0.0,
            dmsmear=False,
        )
        first = furby.FurbyGenerator(dummy_header, params, seed=4).generate()
        second = furby.FurbyGenerator(dummy_header, params, seed=4).generate()
        other = furby.FurbyGenerator(dummy_header, params, seed=5).generate()
        np.testing.assert_allclose(first.block.data, second.block.data)
        assert not np.allclose(first.block.data, other.block.data)

    def test_realized_sweep_follows_delays(
        self,
        dummy_header: Header,
        dummy_params: furby.PulseParams,
    ) -> None:
        profile = furby.FurbyGenerator(dummy_header, dummy_params).generate()
        waterfall = profile.realize()
        peaks = waterfall.data.argmax(axis=1)
        delays = np.rint(profile.delays).astype(int)
        np.testing.assert_allclose(peaks - peaks[0], delays - delays[0], atol=3)

        aligned = waterfall.dedisperse(
            dummy_params.dm,
            ref_freq="fch1",
            only_valid_samples=True,
        )
        aligned_peaks = aligned.data.argmax(axis=1)
        assert int(np.ptp(aligned_peaks)) <= 5

    def test_hdf5_roundtrip(
        self,
        dummy_header: Header,
        dummy_params: furby.PulseParams,
        tmp_path: Path,
    ) -> None:
        profile = furby.FurbyGenerator(dummy_header, dummy_params, seed=2).generate()
        path = profile.save(tmp_path / "pulse.h5")
        loaded = furby.Furby.from_file(path)
        np.testing.assert_allclose(loaded.block.data, profile.block.data)
        np.testing.assert_allclose(loaded.delays, profile.delays)
        assert loaded.params_hdr == profile.params_hdr
        assert loaded.sample0 == profile.sample0
        assert loaded.block.header.tsamp == pytest.approx(profile.block.header.tsamp)

    def test_plot(
        self,
        dummy_header: Header,
        dummy_params: furby.PulseParams,
    ) -> None:
        profile = furby.FurbyGenerator(dummy_header, dummy_params).generate()
        fig = profile.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSpectralStructure:
    def test_init(self, freqs: np.ndarray) -> None:
        spec_struct = furby.SpectralStructure(freqs, kind="gaussian", spec_index=-1.5)
        assert spec_struct.kind == "gaussian"
        assert spec_struct.spec_index == -1.5
        assert spec_struct.nchans == len(freqs)
        assert np.isclose(spec_struct.foff, -1.0)

    @pytest.mark.parametrize(
        "kind",
        [
            "flat",
            "power_law",
            "smooth_envelope",
            "gaussian",
            "polynomial_peaks",
            "scintillation",
            "gaussian_blobs",
            "random",
        ],
    )
    def test_generate_and_normalize(self, freqs: np.ndarray, kind: str) -> None:
        spec_struct = furby.SpectralStructure(freqs, kind=kind, spec_index=0.0)
        spec = spec_struct.generate()
        assert spec.shape[0] == freqs.shape[0]
        assert np.all(np.isfinite(spec))
        assert np.all(spec >= 0)
        np.testing.assert_allclose(spec.mean(), 1.0, atol=1e-6)

    @pytest.mark.parametrize("kind", ["flat", "scintillation"])
    def test_plot(self, freqs: np.ndarray, kind: str) -> None:
        spec_struct = furby.SpectralStructure(freqs, kind=kind)
        fig = spec_struct.plot()
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        assert fig.axes[0].get_xlabel() == "Frequency (MHz)"
        assert fig.axes[0].get_ylabel() == "Amplitude"
        plt.close(fig)

    def test_invalid_kind(self, freqs: np.ndarray) -> None:
        with pytest.raises(ValueError, match="unknown spectral kind"):
            furby.SpectralStructure(freqs, kind="invalid")

    def test_power_law_flux_ratio(self, freqs: np.ndarray) -> None:
        spec_struct = furby.SpectralStructure(freqs, kind="power_law", spec_index=-2.0)
        spec = spec_struct.generate()
        ratio = spec[-1] / spec[0]
        expected_ratio = (freqs[-1] / freqs[0]) ** (-2.0)
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-5)


@pytest.mark.parametrize(
    "changes",
    [
        {"dm": -1.0},
        {"dm": np.inf},
        {"snr": 0.0},
        {"snr": np.nan},
        {"width": 0.0},
        {"width": np.inf},
        {"shape": "triangle"},
        {"spec_kind": "unknown"},
        {"disp_ind": np.nan},
        {"scatt_idx": np.inf},
        {"spec_idx": -np.inf},
    ],
)
def test_pulse_parameters_reject_invalid_physics(changes: dict) -> None:
    values = {"dm": 10.0, "snr": 12.0, "width": 0.003, **changes}
    with pytest.raises(ValueError):
        furby.PulseParams(**values)


@pytest.mark.parametrize("oversampling", [True, 1.5])
def test_pulse_oversampling_requires_an_integer(oversampling: object) -> None:
    with pytest.raises(TypeError):
        furby.PulseParams(dm=10.0, snr=12.0, width=0.003, os_fact=oversampling)


@pytest.mark.parametrize(
    "frequencies",
    [
        np.array([]),
        np.ones((2, 2)),
        np.array([0.0]),
        np.array([np.nan]),
        np.array([-1.0]),
    ],
)
def test_spectral_structures_require_physical_frequency_coordinates(
    frequencies: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        furby.SpectralStructure(frequencies)


def test_spectral_structure_requires_a_finite_index() -> None:
    with pytest.raises(ValueError):
        furby.SpectralStructure(np.array([1000.0, 1100.0]), spec_index=np.nan)


def test_unknown_spectral_structure_cannot_generate_samples() -> None:
    with pytest.raises(ValueError):
        furby.SpectralStructure(np.array([1000.0, 1100.0]), kind="unknown").generate()


@pytest.mark.parametrize("field", ["tau0", "tau_ref_freq", "noise"])
def test_pulse_broadening_and_noise_must_be_finite(field: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        furby.PulseParams(dm=10, snr=10, width=0.001, **{field: np.inf})
