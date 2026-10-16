# Sigproc header rules

These rules set which Sigproc keys sigpyproc reads, keeps and writes. They match
dspsr's reader and writer, and the sigproc specification.

A `Header` has two kinds of values:

- **Source values** are the keys in the file. They are written only under the rules
  below.
- **Derived values** are what `Header` adds for use in code: an inferred azimuth
  and zenith, a frame of `None` treated as topocentric, `nbeams` defaulting to 1.
  They are written only if they are known (table below).

## Rewriting a Sigproc file

A `.fil` or `.tim` file is read and written again with these rules:

1. Every key in the file keeps its value and type. Explicit zeros are kept.
2. No key is added that the file does not have. The one exception is `telescope_id`
   and `machine_id`, which are written as 0 when the file lacks them. For readers, 0
   means fake data, the same as absence.
3. Unknown keys are kept byte-for-byte, including string values and adjacent unknown
   keys.
4. An operation changes only the keys it must change, such as `nchans`, `foff`,
   `fch1`, `tsamp`, `nbits`, `nifs` or `tstart`.
5. For filterbanks and time series, readers derive `nsamples` from file size.
   Spectrum files explicitly store the real transform length in `nsamples`;
   their bin payload alone cannot distinguish odd and even real lengths.
6. `src_raj` and `src_dej` are written back as stored when the coordinate is unchanged.

## Creating a Sigproc file from another format or from code

These core keys are always written: `data_type`, `telescope_id`, `machine_id`,
`tstart`, `tsamp`, `nbits`, `fch1`, `foff`, `nchans`, `nifs`.

The optional keys below are written only when the header knows the value. Known means
it came from the source, was set explicitly, or was inferred from complete inputs.

| Key | Written when |
| --- | --- |
| `source_name` | A non-empty source name is known |
| `src_raj`, `src_dej` | Source coordinates are known. A coordinate read from the file is written back as stored. A replaced coordinate is written in sexagesimal form, which may change the last digits |
| `az_start`, `za_start` | Read from the file, set explicitly, or inferred (see below) |
| `refdm` | A DM is set, for example by `subband` or `dedisperse` |
| `period` | A period is set to a non-default value, for example by `fake` |
| `signed` | Read from the file or set explicitly. Readers before 2011 (dspsr) reject this key, so a warning is logged |
| `ibeam`, `nbeams` | The source states them. PSRFITS supplies `IBEAM` and `NBEAM` when those cards are present |
| `rawdatafile` | The name of the raw data file is known. PSRFITS input uses its file name |
| `barycentric`, `pulsarcentric` | The source states the frame, or `frame` is set |

**Inferred angles.** `az_start` and `za_start` are inferred only when the telescope is
a known site, the source coordinates are present and `tstart` is present. Inferred
angles are written when creating a file from PSRFITS, HDF5 or PRESTO `.inf` input. A
rewritten Sigproc file keeps its own angles, and an angle absent from it is not added.

## Frame

- `frame` is `None` when the source does not state it. Such a frame is never written.
- Code that needs a frame uses `Header.effective_frame`, which returns
  `"topocentric"` for `None`.
- An explicit `frame="barycentric"` or `"pulsarcentric"` sets that flag to 1. The
  other flag is written only if the source had it.
- `frame="topocentric"` writes only the flags the source had, as 0. It never adds a
  flag.

## Telescope and machine IDs

- An ID read from a file is written back unchanged, even if its name is not in the
  Sigproc table.
- Otherwise the ID comes from the name. An unknown name writes 0 and logs a warning.
- A file without the ID gets 0 (see rule 2 under rewriting).
- Changing `telescope` or `backend` clears the stored ID, so the new name is used.

## Editing a file in place

`spp_header update` and `sigproc.edit_headers` change keys without changing the header
size. The rules are:

- A string keeps its stored length. A shorter value is padded with spaces, and a longer
  one is refused. Names are never truncated.
- A key absent from the file cannot be added.
- `nbits`, `nchans` and `nifs` cannot be changed in place, because they set how the
  samples are stored.
- A declared `nsamples` is kept unless it is edited.
- Every value is checked, and the new header is read back, before the file is changed.
  Any failure leaves the file as it was.

See {doc}`tutorials/header` for examples.
