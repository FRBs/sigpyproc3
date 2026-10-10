# Command line utilities

Thin wrappers around the `sigpyproc` API. Each tool is a separate
`spp_*` script — there is no umbrella `spp` command and no SIGPROC
binary-name aliases.

Algorithms live on the data objects
({py:class}`~sigpyproc.base.Filterbank`,
{py:class}`~sigpyproc.timeseries.TimeSeries`,
{py:class}`~sigpyproc.candidates.CandidateTable`). These pages document
flags; {doc}`/tutorials/quickstart` and {doc}`/tutorials/search` show how
to use the same methods from Python.

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
| `spp_fake` | `noise` / `pulsar` / `pulse` filterbanks |
| `spp_search` | `fft` / `pulses` / `ffa` / `cluster` on a `.tim` |
| `spp_digifil` | Convert PSRFITS / filterbank to N-bit filterbank (dspsr `digifil` numerics) |

`spp_search` writes **CSV** (or stdout with `-o -`). Periods on the
Python API are in seconds; `spp_search ffa --pmin-ms` / `--pmax-ms`
accept milliseconds for SIGPROC muscle memory and convert immediately.

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
