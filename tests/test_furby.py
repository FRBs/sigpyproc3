# mypy: disable_error_code="arg-type"

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import pyplot as plt

from sigpyproc.block import FilterbankBlock
from sigpyproc.header import Header
from sigpyproc.simulation import furby


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
    return np.arange(1500, 1000, -1)


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
        assert furby_obj.params_hdr == dummy_params
        assert isinstance(furby_obj.stats_hdr, furby.PulseStats)


class TestSpectralStructure:
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
        assert spec_struct.freqs.dtype == np.float32
        assert spec_struct.kind == kind
        assert spec_struct.nchans == freqs.shape[0]
        assert spec_struct.foff == -1.0
        spec = spec_struct.generate()
        assert spec.shape[0] == freqs.shape[0]
        np.testing.assert_allclose(spec.mean(), 1.0, atol=1e-6)

    @pytest.mark.parametrize("kind", ["flat", "scintillation"])
    def test_plot(self, freqs: np.ndarray, kind: str) -> None:
        spec_struct = furby.SpectralStructure(freqs, kind=kind)
        fig = spec_struct.plot()
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        plt.close(fig)
