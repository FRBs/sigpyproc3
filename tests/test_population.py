from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from sigpyproc.header import Header
from sigpyproc.simulation.furby import Furby
from sigpyproc.simulation.inject import Injector, read_pulse_catalogue
from sigpyproc.simulation.population import BurstDistribution, RandomBurstPopulation

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def dummy_header() -> Header:
    return Header(
        filename="dummy.fil",
        data_type="filterbank",
        nchans=64,
        foff=-1.0,
        fch1=1400.0,
        nbits=32,
        tsamp=1e-3,
        tstart=50000.0,
        nsamples=2000,
    )


class TestBurstDistribution:
    def test_default_values(self) -> None:
        dist = BurstDistribution()
        assert dist.dm_range == (10.0, 1000.0)
        assert dist.snr_range == (10.0, 50.0)
        assert dist.snr_law == "uniform"
        assert dist.tau_ref_freq == 1000.0
        assert dist.shape == "gaussian"
        assert dist.dmsmear is True

    def test_invalid_snr_law(self) -> None:
        with pytest.raises(ValueError, match="Unknown snr_law"):
            BurstDistribution(snr_law="exponential")  # type: ignore[arg-type]

    def test_empty_spec_kinds(self) -> None:
        with pytest.raises(ValueError, match="spec_kinds must contain at least one"):
            BurstDistribution(spec_kinds=())


class TestRandomBurstPopulation:
    def test_draw_bursts_uniform(self) -> None:
        dist = BurstDistribution(
            dm_range=(50.0, 100.0),
            snr_range=(15.0, 30.0),
            snr_law="uniform",
            width_range=(0.002, 0.005),
        )
        pop = RandomBurstPopulation(dist, seed=42)
        bursts = pop.draw_bursts(20, tobs=5.0)
        assert len(bursts) == 20
        for t, params in bursts:
            assert 0.0 <= t <= 5.0
            assert 50.0 <= params.dm <= 100.0
            assert 15.0 <= params.snr <= 30.0
            assert 0.002 <= params.width <= 0.005
            assert params.tau_ref_freq == 1000.0

    def test_draw_bursts_euclidean(self) -> None:
        dist = BurstDistribution(
            snr_range=(10.0, 100.0),
            snr_law="euclidean",
        )
        pop = RandomBurstPopulation(dist, seed=123)
        bursts = pop.draw_bursts(100, tobs=10.0)
        snrs = [params.snr for _, params in bursts]
        assert all(10.0 <= s <= 100.0 for s in snrs)
        # In a -1.5 cumulative distribution, lower SNRs are much more frequent
        median_snr = float(np.median(snrs))
        assert median_snr < 30.0  # deutlich below the midpoint 55.0

    def test_generate_rows(self) -> None:
        pop = RandomBurstPopulation(seed=10)
        rows = pop.generate_rows(5, tobs=2.0)
        assert len(rows) == 5
        for row in rows:
            assert len(row) == 7
            t, dm, width, snr, spec_idx, tau0, spec_kind = row
            assert 0.0 <= t <= 2.0
            assert 10.0 <= dm <= 1000.0
            assert 1e-3 <= width <= 1e-2
            assert 10.0 <= snr <= 50.0
            assert -4.0 <= spec_idx <= 0.0
            assert tau0 == 0.0
            assert spec_kind in pop.distribution.spec_kinds

    def test_generate_pulses_and_injector(self, dummy_header: Header) -> None:
        pop = RandomBurstPopulation(seed=99)
        pulses, arrivals = pop.generate_pulses(dummy_header, 3)
        assert len(pulses) == 3
        assert len(arrivals) == 3
        assert all(isinstance(p, Furby) for p in pulses)
        assert all(isinstance(a, int) for a in arrivals)

        injector = pop.to_injector(dummy_header, 3)
        assert isinstance(injector, Injector)
        assert len(injector.pulses) == 3

    def test_write_and_read_catalogue(self, tmp_path: Path) -> None:
        cat_file = tmp_path / "population_cat.txt"
        pop = RandomBurstPopulation(
            BurstDistribution(
                dm_range=(100.0, 200.0),
                tau_range=(0.001, 0.005),
                tau_ref_freq=1400.0,
            ),
            seed=7,
        )
        out_path = pop.write_catalogue(cat_file, 8, tobs=4.0)
        assert out_path == str(cat_file)
        assert cat_file.is_file()

        content = cat_file.read_text()
        assert "# 6: tau0 (s, at 1400 MHz)" in content
        assert "# 7: spec_kind" in content

        rows = read_pulse_catalogue(cat_file)
        assert len(rows) == 8
        for row in rows:
            assert len(row) == 7
            t, dm, _width, _snr, _spec_idx, tau0, spec_kind = row
            assert 0.0 <= t <= 4.0
            assert 100.0 <= dm <= 200.0
            assert 0.001 <= tau0 <= 0.005
            assert spec_kind in pop.distribution.spec_kinds
