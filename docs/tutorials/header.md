(header-tutorial)=

# Printing and editing headers

This tutorial shows how to read, check and change the header of a Sigproc
filterbank with `spp_header`, and the equivalent Python calls. The rules behind
the output are in {doc}`../header`.

We use the sample file shipped in the test data. Copy it first, because editing
changes the file in place:

```bash
cp tests/data/parkes_8bit_1.fil my_test.fil
```

## Print the header

```bash
spp_header print my_test.fil
```

This prints a summary. Values marked `(derived)` were not in the file. They are
inferred or defaulted by sigpyproc, so do not treat them as facts about the
observation. For the full object, use `--long`.

## Read one key

```bash
spp_header get my_test.fil -k fch1
spp_header get my_test.fil -k source_name
```

`-k` accepts any SIGPROC key, and a few derived values such as `telescope` or
`tobs`. To list the valid keys and what each one means:

```bash
spp_header get --keys-help
```

An unknown key stops with an error that lists the valid ones.

## Preview an edit

Always preview first. `--dry-run` shows the values that would be written and
writes nothing:

```bash
spp_header update my_test.fil -i source_name J0437-4715 --dry-run
```

Keys you asked to change are shown in bold. Use `-i` once per key to change several at once:

```bash
spp_header update my_test.fil -i source_name J0437-4715 -i ibeam 2 --dry-run
```

## Apply an edit

```bash
spp_header update my_test.fil -i source_name J0437-4715 -i ibeam 2
```

All the values are checked and the new header is read back before the file is
touched. If any value is refused, nothing is written.

## What an edit can and cannot do

Edits never change the size of the header, so the data stays where it is. This
has three consequences:

- **Strings** keep their stored length. A shorter value is padded with spaces; a
  longer one is refused. Names are never truncated.
- **Keys absent from the file** cannot be added. Adding a key changes the header
  size. To add metadata, write a new file from Python (see below).
- **Layout keys** `nbits`, `nchans` and `nifs` are refused. They say how the
  samples are stored, so changing them would make the data unreadable. To change
  the bit depth, use `spp_digifil`.

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
