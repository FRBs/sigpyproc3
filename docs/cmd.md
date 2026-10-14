# Command line utilities

Thin wrappers around the `sigpyproc` API. Each tool is a separate
`spp_*` script — there is no umbrella `spp` command and no SIGPROC
binary-name aliases.

Algorithms live on the data objects
({py:class}`~sigpyproc.base.Filterbank`,
{py:class}`~sigpyproc.timeseries.TimeSeries`,
{py:class}`~sigpyproc.core.candidates.CandidateTable`). These pages document
flags; {doc}`/tutorials/quickstart`, {doc}`/tutorials/search`, and
{doc}`/tutorials/fake` show how to use the same methods from Python.

## Logging

Every `spp_*` tool accepts global logging flags:

- **`-v` / `--verbose`** — repeat for more detail (`-v` sets DEBUG).
- **`-q` / `--quiet`** — only warnings and errors (`-q` overrides `-v`).

Default output is INFO-level messages with Rich formatting. On **group**
commands (`spp_header`, `spp_extract`, `spp_search`, …), put the flags
**before** the subcommand, for example:

```bash
spp_header -v print my_test.fil
spp_search -q fft observation.tim -o fft.csv
```

From Python, call `sigpyproc.core.logging.configure_logging()` for the same
Rich styling, or rely on your application's own `logging` configuration.

| Script | Role |
| --- | --- |
| `spp_header` | Print, get, or edit a header |
| `spp_extract` | Cut samples, channels, or a mask |
| `spp_stats` | Channel statistics |
| `spp_decimate` | Time / frequency downsample |
| `spp_bandpass` | Mean bandpass |
| `spp_zerodm` | Remove the time-mean (Eatough / PRESTO-style, not SIGPROC 8-bit) |
| `spp_clean` | RFI cleaning and zap |
| `spp_merge` | Concatenate in time |
| `spp_splice` | Stack in frequency |
| `spp_dedisperse` | `tim` / `subband` / `tree` / `plan` |
| `spp_fold` | Fold to `.npz` + ASCII profile |
| `spp_fake` | `noise` / `pulsar` / `pulse` / `furby` filterbanks, and `inject` |
| `spp_search` | `fft` / `pulses` / `cluster` on a `.tim` |
| `spp_digifil` | Convert PSRFITS / filterbank to N-bit filterbank (dspsr `digifil` numerics) |
| `spp_view` | Interactive viewer & batch renderer for filterbank, time-series, and spectrum |

`spp_search` writes **CSV** (or stdout with `-o -`). Periods on the
Python API are in seconds.

```{eval-rst}
.. click:: sigpyproc.apps.spp_header:main
   :prog: spp_header
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_extract:main
   :prog: spp_extract
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_stats:main
   :prog: spp_stats
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_decimate:main
   :prog: spp_decimate
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_bandpass:main
   :prog: spp_bandpass
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_zerodm:main
   :prog: spp_zerodm
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_merge:main
   :prog: spp_merge
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_splice:main
   :prog: spp_splice
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_dedisperse:main
   :prog: spp_dedisperse
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_fold:main
   :prog: spp_fold
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_clean:main
   :prog: spp_clean
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_fake:main
   :prog: spp_fake
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_search:main
   :prog: spp_search
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_digifil:main
   :prog: spp_digifil
   :nested: full
```

```{eval-rst}
.. click:: sigpyproc.apps.spp_view:main
   :prog: spp_view
   :nested: full
```

