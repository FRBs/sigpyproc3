# sigpyproc

```{div} .lead
A modern, object-oriented Python library for FRB and pulsar data analysis.
```

[![GitHub CI](https://github.com/FRBs/sigpyproc3/workflows/GitHub%20CI/badge.svg)](https://github.com/FRBs/sigpyproc3/actions)
[![Docs](https://readthedocs.org/projects/sigpyproc3/badge/?version=latest)](https://sigpyproc3.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/FRBs/sigpyproc3/branch/main/graph/badge.svg)](https://codecov.io/gh/FRBs/sigpyproc3)
[![License](https://img.shields.io/github/license/FRBs/sigpyproc3)](https://github.com/FRBs/sigpyproc3/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

---

**sigpyproc** provides high-performance tools for manipulating pulsar and fast radio burst (FRB) data, combining a clean object-oriented API with [Numba](https://numba.pydata.org/)-accelerated kernels for time-critical workloads. Originally developed as a Python alternative to the
[SIGPROC filterbank](http://sigproc.sourceforge.net) toolbox, the project has evolved into an independent framework for structured pulsar/FRB data manipulation.

It is a modern extension of SIGPROC, not a clone. Search lives on `TimeSeries` / `PowerSpectrum` / `CandidateTable` (`TimeSeries.search`, `find_pulses`, `ffa`) with an `spp_search` CLI and CSV output. That is enough for exploratory FFT, FFA, and single-pulse work; it is not a drop-in survey pipeline (no `.prd` / `.bst` files, birdie lists, or PGPLOT). The library still focuses on:

- Precise data inspection, data format conversion and header manipulation
- Fine-grained control/visualization of filterbank and time series data
- Rapid experimentation — including a first-class candidate table — prior to large-scale pipeline execution

This makes it particularly well suited for exploratory research, validation work, and a simple plug-and-play system with new modules and extensions.

---

## Features

- **Object-Oriented Design** — Explicit data objects for [SIGPROC filterbank](http://sigproc.sourceforge.net), [PSRFITS](https://www.atnf.csiro.au/research/pulsar/psrfits_definition/Psrfits.html), `TimeSeries`, `FourierSeries`, and `CandidateTable`.
- **Search, as an API** — FFT + harmonic folds, Fast Folding Analysis, and a boxcar single-pulse search, all returning one CSV-friendly candidate table. See {doc}`/tutorials/search`.
- **High Performance** — Critical paths implemented with Numba for speed without sacrificing clarity.
- **Structured Metadata** — Explicit header schemas for metadata handling.
- **Modern Stack** — Built with `uv`, enforced with `ruff`, and maintained with strict `ty` type checking.

---

## Get Started

::::{grid} 3
:gutter: 3

:::{grid-item-card} 🚀 Installation
:link: install
:link-type: doc

Environment setup and a first import.
:::

:::{grid-item-card} 📘 Filterbank quickstart
:link: tutorials/quickstart
:link-type: doc

Load a `.fil`, inspect the header, dedisperse, and fold.
:::

:::{grid-item-card} 🔎 Search a time series
:link: tutorials/search
:link-type: doc

FFT, FFA, single pulses, and `CandidateTable` — how to use the search API.
:::

:::{grid-item-card} 🖥️ Command line tools
:link: cmd
:link-type: doc

`spp_header`, `spp_search`, `spp_fake`, and the other `spp_*` scripts.
:::

:::{grid-item-card} 📚 API reference
:link: generated/sigpyproc
:link-type: doc

Complete documentation of core classes, data structures, and kernels.
:::

:::{grid-item-card} 🛠️ Development guide
:link: dev
:link-type: doc

Contributing, extending kernels, and architectural details.
:::
::::

---

```{toctree}
:caption: User Guide
:hidden:
:maxdepth: 1

install
cmd
generated/sigpyproc
dev
changes
```

```{toctree}
:caption: Tutorials
:hidden:
:maxdepth: 1

tutorials/quickstart
tutorials/search
tutorials/header
```
