"""16-bit streaming stays exact above 255, including values near 65535."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sigpyproc.core.dispersion import DMPlan
from sigpyproc.header import Header
from sigpyproc.readers import FilReader
from sigpyproc.simulation.pulsar import make_header

if TYPE_CHECKING:
    from pathlib import Path

CHANNELS = np.array([300, 400, 1000, 65535], dtype=np.uint16)


def _write_u16(path: Path, nsamps: int = 64) -> str:
    data = np.broadcast_to(CHANNELS, (nsamps, CHANNELS.size)).copy()
    tsamp = 1e-3
    header = make_header(
        filename=str(path),
        nchans=CHANNELS.size,
        nbits=16,
        tsamp=tsamp,
        tobs=nsamps * tsamp,
        tstart=50000.0,
        fch1=1400.0,
        foff=-1.0,
    )
    with header.prep_outfile(str(path), nbits=16, sources=()) as writer:
        writer.cwrite(data.ravel())
    return str(path)


def _write_signed(path: Path, samples: np.ndarray) -> str:
    nsamps, nchans = samples.shape
    header = Header(
        filename=str(path),
        data_type="filterbank",
        nchans=nchans,
        foff=-1.0,
        fch1=1400.0,
        nbits=8,
        signed=True,
        tsamp=1e-3,
        tstart=50000.0,
        nsamples=nsamps,
    )
    with header.prep_outfile(str(path), sources=()) as writer:
        writer.cwrite(samples.ravel())
    return str(path)


def test_uint16_streaming_matches_the_samples(tmp_path: Path) -> None:
    path = _write_u16(tmp_path / "u16.fil")
    fil = FilReader(path)
    nsamps = fil.header.nsamples
    sample = CHANNELS.astype(np.float32)
    total = float(sample.sum())

    gulp, _index, raw = next(iter(fil.read_plan(gulp=8)))
    assert gulp == 8
    assert raw.dtype == np.float32
    assert raw.reshape(gulp, -1)[0, -1] == 65535

    copied = tmp_path / "copy.fil"
    fil.extract_samps(4, 10, outfile_name=str(copied), gulp=3)
    assert FilReader(copied).read_block(0, 10).data[3, 0] == 65535

    for parallel in (False, True):
        collapsed = fil.collapse(method="sum", gulp=8, parallel=parallel)
        np.testing.assert_allclose(collapsed.data, np.full(nsamps, total))
        band = fil.bandpass(method="mean", gulp=8, parallel=parallel)
        np.testing.assert_allclose(band.data, sample)

    dedispersed = fil.dedisperse(0.0, gulp=8)
    np.testing.assert_allclose(dedispersed.data, np.full(nsamps, total))
    plan_paths = fil.dedisperse_plan(
        DMPlan(dms=np.array([0.0])),
        gulp=8,
        outfile_base=str(tmp_path / "u16"),
    )
    planned = FilReader(plan_paths[0]).read_block(0, nsamps).data.ravel()
    np.testing.assert_allclose(planned, np.full(nsamps, total))

    sub_path = fil.subband(0.0, nsub=2, outfile_name=str(tmp_path / "sub.fil"), gulp=8)
    sub = FilReader(sub_path).read_block(0, nsamps).data
    np.testing.assert_allclose(sub[0], np.full(nsamps, 700.0))
    np.testing.assert_allclose(sub[1], np.full(nsamps, 66535.0))

    folded = fil.fold(period=8 * fil.header.tsamp, dm=0.0, nbins=8, nints=1, nbands=1)
    np.testing.assert_allclose(folded.data, total / CHANNELS.size)

    masked = fil.apply_channel_mask(
        np.array([False, True, False, False]),
        mask_value=7,
        outfile_name=str(tmp_path / "mask.fil"),
        gulp=8,
        parallel=True,
    )
    mask_data = FilReader(masked).read_block(0, nsamps).data
    np.testing.assert_allclose(mask_data[0], 300)
    np.testing.assert_allclose(mask_data[1], 7)
    np.testing.assert_allclose(mask_data[3], 65535)

    inverted = fil.invert_freq(outfile_name=str(tmp_path / "inv.fil"), gulp=8)
    np.testing.assert_allclose(
        FilReader(inverted).read_block(0, nsamps).data[:, 0],
        sample[::-1],
    )

    fil.compute_stats(gulp=8)
    assert fil.chan_stats is not None
    np.testing.assert_allclose(fil.chan_stats.mean, sample)
    np.testing.assert_allclose(fil.chan_stats.maxima, sample)

    converted = fil.requantize(
        32,
        outfile_name=str(tmp_path / "f32.fil"),
        rescale_seconds=0.0,
        gulp=8,
    )
    np.testing.assert_allclose(
        FilReader(converted).read_block(0, nsamps).data[:, 0],
        sample,
    )

    down = fil.downsample(
        tfactor=1,
        ffactor=2,
        outfile_name=str(tmp_path / "down.fil"),
        gulp=8,
    )
    down_data = FilReader(down).read_block(0, nsamps).data
    np.testing.assert_allclose(down_data[0], 350)
    np.testing.assert_allclose(down_data[1], np.rint((1000 + 65535) / 2))

    zerodm = fil.remove_zerodm(outfile_name=str(tmp_path / "zdm.fil"), gulp=8)
    np.testing.assert_allclose(
        FilReader(zerodm).read_block(0, nsamps).data[:, 0],
        sample,
    )


def test_signed8_mask_keeps_the_replacement(tmp_path: Path) -> None:
    samples = np.array([[-40, -1, 10, 20], [-40, -1, 10, 20]], dtype=np.float32)
    path = _write_signed(tmp_path / "signed.fil", samples)
    fil = FilReader(path)
    out = fil.apply_channel_mask(
        np.array([True, False, False, False]),
        mask_value=-20,
        outfile_name=str(tmp_path / "signed_mask.fil"),
        gulp=1,
    )
    got = FilReader(out).read_block(0, 2).data
    np.testing.assert_allclose(got[0], -20)
    np.testing.assert_allclose(got[1], -1)

    flat = np.full((16, 4), -5, dtype=np.float32)
    src = _write_signed(tmp_path / "signed_flat.fil", flat)
    zpath = FilReader(src).remove_zerodm(
        outfile_name=str(tmp_path / "signed_zdm.fil"),
        gulp=4,
    )
    np.testing.assert_allclose(FilReader(zpath).read_block(0, 16).data, -5)
