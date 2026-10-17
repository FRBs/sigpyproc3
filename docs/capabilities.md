# Supported operations

sigpyproc is a Python toolbox for the filterbank and time-series work that
SIGPROC covered. It does not promise byte-for-byte historical output, PGPLOT,
or the old binary names.

| Operation | What 2.0 does |
| --- | --- |
| SIGPROC header inspection and edit | `Header` and `spp_header`. In-place edits stay on the requested file. A streaming transform will not overwrite its own input. |
| Sample, channel, and band extraction | `extract_samps`, `extract_chans`, `extract_bands`, and `spp_io extract`. Output epochs follow the selected samples. Signed 8-bit values are preserved. |
| Decimation, bandpass, statistics, zero-DM, RFI | Library methods and the matching `spp_*` commands. Zero-DM restores the mean bandpass. Channel moments use the samples actually read. |
| Dedispersion, sub-bands, Taylor tree, DM plan | `dedisperse`, `subband`, `tree_dedisperse`, and `DMPlan`. One-channel delay arrays keep a frequency axis. Trials that collide at three decimals are rejected before any file is opened. |
| Folding | `Filterbank.fold` and `spp_fold`. Phase-bin duration is `period / nbins`. DM and period updates are measured from the original fold. Archives reload with `FoldedData.from_npz`. A selected sample range is not yet partitioned across sub-integrations. |
| Periodic search | `TimeSeries.search` and `spp_search fft`. The search is an FFT with harmonic summing. Lyne sums are placed on the fundamental bin. Clustering keeps the loudest harmonic, not necessarily the fundamental. |
| Single pulses | PRESTO-style boxcars, a circular matched filter, and a CPU Heimdall path. `kind="pulse"` does not say whether `bin` is the boxcar centre or its start; `bin_origin` does. GPU Heimdall parity is not claimed. |
| FFA | Not part of this library. |
| Simulation and conversion | `spp_fake` and `spp_digifil`. Frequency scrunch factors must divide the channel count. A selected digifil window keeps that start time. |
| PSRFITS | Single-file search-mode reading and conversion to SIGPROC. Fold mode, writing, and multi-file PSRFITS are rejected. |
| PRESTO `.dat`, `.inf`, `.fft` | Read and write with explicit little-endian layouts. Spectrum files store the real transform length. |
| HDF5 | Filterbank header parsing, plus RFI-mask and furby products. There is no general HDF5 filterbank stream. |
| Viewers | Matplotlib time series, spectrum, and waterfall views. `--batch` writes files and does not call `show`. |
| 16-bit data | Numerical streaming promotes each `uint16` block to float32, which holds every 16-bit value exactly. Raw copies stay 16-bit. |
