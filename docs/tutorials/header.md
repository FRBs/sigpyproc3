(header-tutorial)=

# Printing and editing headers

SIGPROC stores metadata in a fixed binary header at the start of each
`.fil`. This page shows how to **read** that header, **try** an edit, and
**apply** it with `spp_header`. The full rules for what each key means live in
the [header rules](https://github.com/FRBs/sigpyproc3/blob/main/docs/header.md)
in the repository.

We use the sample file shipped in the test data. Copy it first, because editing
changes the file in place:

```bash
# from a git checkout; tests/data is not on the PyPI wheel
cp tests/data/parkes_8bit_1.fil my_test.fil
```

## Print the header

```bash
spp_header print my_test.fil
```

This prints a summary. Values marked `(derived)` were not in the file. They are
inferred or defaulted by sigpyproc, so treat those as hints, not
telescope truth. Add `--long` for the full `Header` object pretty-printed.

## Read one key

```bash
spp_header get my_test.fil -k fch1
spp_header get my_test.fil -k source_name
```

`-k` accepts any SIGPROC key plus a few derived names (`telescope`, `tobs`, …).
To see the full list:

```bash
spp_header get --keys-help
```

A typo in `-k` fails with a message that lists valid keys.

## Preview an edit

Always dry-run first. Nothing is written until you omit `--dry-run`:

```bash
spp_header update my_test.fil -i source_name J0437-4715 --dry-run
```

Changed keys show in bold. Several `-i` flags apply multiple keys at once:

```bash
spp_header update my_test.fil -i source_name J0437-4715 -i ibeam 2 --dry-run
```

## Apply an edit

```bash
spp_header update my_test.fil -i source_name J0437-4715 -i ibeam 2
```

Values are validated and the new header is read back before bytes on disk change.
If anything is rejected, the file stays untouched.

## What an edit can and cannot do

Edits never change the size of the header, so the data stays where it is. This
has three consequences:

- **Strings** keep their slot width. Shorter values are space-padded; longer
  ones are refused — nothing is silently truncated.
- **Missing keys** cannot be added through `update`; new keys would change the
  header length. Write a new file from Python if you need extra metadata.
- **Layout keys** (`nbits`, `nchans`, `nifs`) are refused because they describe
  how samples are packed. Change bit depth with `spp_digifil` instead.

Numeric values are checked against what a sigproc file can store. Examples are
`nbits` in {1, 2, 4, 8, 16, 32}, flags 0 or 1, angles within range, and valid
sexagesimal coordinates such as `src_raj 123456.7`.

## Edit from Python

The same checks are available in Python. `Header` objects are read-only, so
make a new header and write it to a new file with `prep_outfile`:

```python
import numpy as np
from sigpyproc.header import Header

hdr = Header.from_sigproc("my_test.fil")
new = hdr.new_header({"source": "J0437-4715"})
data = np.zeros((hdr.nchans, 1024), dtype=np.uint8)  # (nchans, nsamps)
with new.prep_outfile("renamed.fil") as out:
    out.cwrite(data.transpose().ravel())  # file order is time-major
```

To change one or more keys in place, using the same checks as `spp_header`:

```python
from sigpyproc.io import sigproc

before, after = sigproc.prepare_header_edits(
    "my_test.fil", {"source_name": "J0437-4715", "ibeam": 2}
)  # checks only, nothing is written
sigproc.edit_headers("my_test.fil", {"source_name": "J0437-4715", "ibeam": 2})
```

`prepare_header_edits` returns the values before and after the edit, so you can
show a preview. `edit_headers` does the same checks and then writes the file.
