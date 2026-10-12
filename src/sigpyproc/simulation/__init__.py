"""Simulation module for mock pulsars, FRBs, and filterbanks."""

from sigpyproc.simulation.furby import (
    Furby,
    FurbyGenerator,
    PulseParams,
    PulseStats,
    SpectralStructure,
)
from sigpyproc.simulation.inject import (
    InjectedPulse,
    Injector,
    inject_file,
    read_pulse_catalogue,
)
from sigpyproc.simulation.population import (
    BurstDistribution,
    RandomBurstPopulation,
)
from sigpyproc.simulation.pulsar import (
    PulsarParams,
    fake_filterbank,
    make_header,
)

__all__ = [
    "BurstDistribution",
    "Furby",
    "FurbyGenerator",
    "InjectedPulse",
    "Injector",
    "PulsarParams",
    "PulseParams",
    "PulseStats",
    "RandomBurstPopulation",
    "SpectralStructure",
    "fake_filterbank",
    "inject_file",
    "make_header",
    "read_pulse_catalogue",
]
