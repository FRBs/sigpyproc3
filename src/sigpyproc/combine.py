"""Combine filterbank files along time or frequency."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import numpy as np

from sigpyproc.readers import FilReader

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def merge_filterbanks(
    filenames: Sequence[str | Path],
    outfile_name: str,
    *,
    axis: Literal["auto", "time", "freq"] = "auto",
    gulp: int = 16384,
) -> str:
    """Concatenate filterbanks in time or frequency and write ``outfile_name``.

    ``axis="time"`` requires matching frequency axes and contiguous start times
    (the existing :class:`~sigpyproc.readers.FilReader` multi-file stream).
    ``axis="freq"`` requires matching ``tstart`` / ``tsamp`` / ``nbits`` and a
    uniformly spaced frequency axis after concatenation.
    ``axis="auto"`` picks time if the frequency axes match, else frequency.
    """
    if len(filenames) < 2:
        msg = "Need at least two filterbank files"
        raise ValueError(msg)
    if axis == "auto":
        headers = [FilReader(name).header for name in filenames]
        same_freq = all(
            h.nchans == headers[0].nchans
            and h.fch1 == headers[0].fch1
            and h.foff == headers[0].foff
            for h in headers[1:]
        )
        axis = "time" if same_freq else "freq"
    if axis == "time":
        return _merge_time(filenames, outfile_name, gulp=gulp)
    return splice_filterbanks(filenames, outfile_name, gulp=gulp)


def splice_filterbanks(
    filenames: Sequence[str | Path],
    outfile_name: str,
    gulp: int = 16384,
) -> str:
    """Concatenate simultaneous files along the frequency axis.

    Files are sorted by ``fch1`` (descending if ``foff < 0``). The spliced
    axis must be uniformly spaced; irregular ``FREQUENCY_START`` output is
    not written.
    """
    if len(filenames) < 2:
        msg = "Need at least two filterbank files"
        raise ValueError(msg)
    fils = _open_sorted_fils(filenames)

    nchans = sum(fil.header.nchans for fil in fils)
    updates = {
        "nchans": nchans,
        "fch1": fils[0].header.fch1,
        "foff": fils[0].header.foff,
    }
    out_file = fils[0].header.prep_outfile(outfile_name, updates=updates)
    nsamps = fils[0].header.nsamples
    nread = 0
    while nread < nsamps:
        take = min(gulp, nsamps - nread)
        chunks = []
        for fil in fils:
            for nsamps_r, _, data in fil.read_plan(
                gulp=take,
                start=nread,
                nsamps=take,
                quiet=True,
            ):
                chunks.append(
                    data[: nsamps_r * fil.header.nchans].reshape(nsamps_r, -1),
                )
                break
        stacked = np.concatenate(chunks, axis=1)
        out_file.cwrite(stacked.ravel())
        nread += take
    out_file.close()
    return outfile_name


def _open_sorted_fils(filenames: Sequence[str | Path]) -> list[FilReader]:
    fils = [FilReader(name) for name in filenames]
    ref = fils[0].header
    for fil in fils[1:]:
        hdr = fil.header
        if abs(hdr.tstart - ref.tstart) > 0.5 * ref.tsamp:
            msg = f"tstart mismatch: {hdr.tstart} vs {ref.tstart}"
            raise ValueError(msg)
        if hdr.tsamp != ref.tsamp or hdr.nbits != ref.nbits or hdr.nifs != ref.nifs:
            msg = "tsamp, nbits, and nifs must match across files"
            raise ValueError(msg)
        if hdr.nsamples != ref.nsamples:
            msg = "nsamples must match across files"
            raise ValueError(msg)
        if hdr.foff != ref.foff:
            msg = "foff must match across files for a linear frequency axis"
            raise ValueError(msg)
    descending = ref.foff < 0
    fils.sort(key=lambda fil: fil.header.fch1, reverse=descending)
    for prev, cur in pairwise(fils):
        expected = prev.header.fch1 + prev.header.nchans * prev.header.foff
        if not np.isclose(cur.header.fch1, expected, atol=1e-6):
            msg = (
                "Frequency axes do not adjoin "
                f"({prev.header.fch1} + {prev.header.nchans}*{prev.header.foff} "
                f"!= {cur.header.fch1})"
            )
            raise ValueError(msg)
    return fils


def _merge_time(
    filenames: Sequence[str | Path],
    outfile_name: str,
    gulp: int,
) -> str:
    fil = FilReader(filenames, check_contiguity=True)
    out_file = fil.header.prep_outfile(outfile_name)
    for _, _, data in fil.read_plan(gulp=gulp):
        out_file.cwrite(data)
    out_file.close()
    return outfile_name
