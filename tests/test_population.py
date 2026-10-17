from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from sigpyproc.header import Header
from sigpyproc.simulation.furby import Furby, PulseParams, SpectralStructure
from sigpyproc.simulation.inject import InjectedPulse, Injector, read_pulse_catalogue
from sigpyproc.simulation.population import BurstDistribution, RandomBurstPopulation
from sigpyproc.simulation.pulsar import PulsarParams

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


def test_simulation_rejects_invalid_physical_values() -> None:
    with pytest.raises(ValueError, match="width"):
        PulseParams(dm=1.0, snr=5.0, width=-1e-3)
    with pytest.raises(ValueError, match="count"):
        RandomBurstPopulation().draw_bursts(-3, tobs=1.0)
    with pytest.raises(ValueError, match="positive"):
        BurstDistribution(snr_range=(0.0, 10.0))
    with pytest.raises(ValueError, match="pdot and accel"):
        PulsarParams(period=0.1, pdot=1e-12, accel=2.0)
    spec = SpectralStructure(np.array([1400.0]), kind="gaussian", seed=1)
    np.testing.assert_array_equal(spec.generate(), [1.0])
    assert spec.foff == 0.0
    with pytest.raises(ValueError, match="delays"):
        InjectedPulse(
            profile=np.ones((2, 4), dtype=np.float32),
            delays=np.array([0.0, np.nan]),
            start=0,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"dm_range": (10.0, 1.0)},
        {"dm_range": (-1.0, 10.0)},
        {"width_range": (0.0, 0.01)},
        {"snr_range": (np.nan, 20.0)},
        {"tau_range": (-1.0, 0.0)},
        {"spec_idx_range": (-1.0, np.inf)},
        {"time_range": (-1.0, 2.0)},
        {"tau_ref_freq": np.inf},
        {"tau_ref_freq": 0.0},
        {"spec_kinds": ("unknown",)},
    ],
)
def test_population_distributions_reject_invalid_ranges(changes: dict) -> None:
    with pytest.raises(ValueError):
        BurstDistribution(**changes)


@pytest.mark.parametrize("count", [-1, -20])
def test_population_rejects_negative_count(count: int) -> None:
    with pytest.raises(ValueError):
        RandomBurstPopulation(seed=1).draw_bursts(count, tobs=1.0)


@pytest.mark.parametrize("tobs", [0.0, -1.0, np.nan, np.inf])
def test_population_rejects_invalid_observation_duration(tobs: float) -> None:
    with pytest.raises(ValueError):
        RandomBurstPopulation(seed=1).draw_bursts(1, tobs=tobs)


@pytest.mark.parametrize("field", ["scatt_idx", "disp_ind"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_population_scalars_reject_nonfinite_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        BurstDistribution(**{field: value})


@pytest.mark.parametrize(
    ("os_fact", "exc"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-2, ValueError)],
)
def test_population_oversampling_must_be_a_positive_integer(
    os_fact: object, exc: type[Exception]
) -> None:
    with pytest.raises(exc, match="os_fact"):
        BurstDistribution(os_fact=os_fact)  # type: ignore[arg-type]


def test_population_rejects_unknown_pulse_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        BurstDistribution(shape="triangle")


def test_valid_population_still_draws_physical_bursts() -> None:
    population = RandomBurstPopulation(
        BurstDistribution(scatt_idx=-4.0, disp_ind=-2.0, os_fact=4, shape="boxcar"),
        seed=3,
    )
    bursts = population.draw_bursts(2, tobs=1.0)
    assert len(bursts) == 2
    assert all(np.isfinite(params.dm) and params.os_fact == 4 for _, params in bursts)


def test_degenerate_euclidean_distribution_preserves_fixed_snr() -> None:
    population = RandomBurstPopulation(
        BurstDistribution(snr_law="euclidean", snr_range=(12.0, 12.0)), seed=9
    )
    bursts = population.draw_bursts(3, tobs=1.0)
    assert [params.snr for _, params in bursts] == [12.0] * 3
    assert population.draw_bursts(0, tobs=1.0) == []
