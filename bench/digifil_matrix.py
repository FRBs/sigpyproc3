"""Compare sigpyproc ``spp_digifil`` with dspsr ``digifil`` over a matrix of options.

This is a manual validation and benchmarking tool. For every case in a matrix of
digifil options it runs dspsr digifil and ``spp_digifil`` on one PSRFITS file, both
single-threaded, then compares the outputs: data byte for byte, and header fields
one by one. It reports the wall-clock time of each.

Usage::

    python bench/digifil_matrix.py INFILE [--preset quick|full] [--dspsr PATH]
        [--repeat N] [--keep DIR]

The dspsr binary is taken from ``--dspsr``, the ``DSPSR_DIGIFIL`` environment
variable, or ``digifil`` on ``PATH``. Run it with the sigpyproc environment active,
for example ``PYTHONPATH=src python bench/digifil_matrix.py data.sf``.

Notes on the comparison:

- Header fields are compared exactly.
- ``filename``, ``filelen``, ``hdrlen`` and ``datalen`` are derived by the header
  parser and are not compared.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

from sigpyproc.io.sigproc import parse_header

DERIVED_KEYS = {"filename", "filelen", "hdrlen", "datalen"}
ANGLE_KEYS = {"az_start", "za_start"}
ANGLE_TOLERANCE_DEG = 0.01
SINGLE_THREAD_ENV = {
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


@dataclasses.dataclass(frozen=True)
class Case:
    """One digifil configuration, expressed in both dspsr and sigpyproc terms."""

    bits: int
    rescale: float
    tscrunch: int = 1
    fscrunch: int = 1
    npol: int = 1
    poln: int = -1
    scloffs: bool = False

    @property
    def label(self) -> str:
        parts = [f"b{self.bits}", f"I{self.rescale:g}"]
        if self.tscrunch > 1:
            parts.append(f"t{self.tscrunch}")
        if self.fscrunch > 1:
            parts.append(f"f{self.fscrunch}")
        if self.npol != 1:
            parts.append(f"d{self.npol}")
        if self.poln >= 0:
            parts.append(f"P{self.poln}")
        if self.scloffs:
            parts.append("scloffs")
        return " ".join(parts)


def build_cases(preset: str) -> list[Case]:
    """Return the matrix of cases for a preset."""
    if preset == "quick":
        return [
            Case(bits=bits, rescale=rescale, tscrunch=4)
            for bits in (2, 8)
            for rescale in (0.0, 10.0)
        ]
    cases = [
        Case(bits=bits, rescale=rescale, tscrunch=tscrunch)
        for bits in (1, 2, 4, 8, 16, -32)
        for rescale in (0.0, 10.0)
        for tscrunch in (1, 4)
    ]
    cases += [
        Case(bits=8, rescale=0.0, poln=0),
        Case(bits=8, rescale=0.0, scloffs=True),
        Case(bits=8, rescale=10.0, tscrunch=4, fscrunch=2),
    ]
    return cases


def dspsr_command(case: Case, dspsr: str, infile: str, outfile: str) -> list[str]:
    """Build the dspsr digifil command line for a case."""
    args = [dspsr, "-threads", "1", "-q", "-B", "1024", "-d", str(case.npol)]
    if case.bits < 0:
        args.append(f"-b{case.bits}")
    else:
        args += ["-b", str(case.bits)]
    args += ["-I", f"{case.rescale:g}"]
    if case.tscrunch > 1:
        args += ["-t", str(case.tscrunch)]
    if case.fscrunch > 1:
        args += ["-f", str(case.fscrunch)]
    if case.poln >= 0:
        args += ["-P", str(case.poln)]
    if case.scloffs:
        args.append("-scloffs")
    args += ["-o", outfile, infile]
    return args


def sigpyproc_command(case: Case, infile: str, outfile: str) -> list[str]:
    """Build the spp_digifil command line for a case."""
    args = [
        sys.executable,
        "-m",
        "sigpyproc.apps.spp_digifil",
        infile,
        f"--nbits={case.bits}",
        f"--rescale_seconds={case.rescale:g}",
        f"--tscrunch_factor={case.tscrunch}",
        f"--fscrunch_factor={case.fscrunch}",
        f"--npol={case.npol}",
        f"--poln_select={case.poln}",
        "--output_filename",
        outfile,
    ]
    if case.scloffs:
        args.append("--scloffs")
    return args


def run_timed(args: list[str], env: dict[str, str]) -> tuple[float, int, str]:
    """Run a command, returning (wall seconds, return code, stderr tail)."""
    start = time.perf_counter()
    proc = subprocess.run(args, capture_output=True, text=True, env=env, check=False)
    elapsed = time.perf_counter() - start
    return elapsed, proc.returncode, proc.stderr.strip()[-300:]


def read_data(path: Path) -> tuple[dict, np.ndarray]:
    """Return the parsed header and the raw data bytes of a sigproc file."""
    header = parse_header(path)
    nbytes = header["nchans"] * header["nsamples"] * header["nifs"]
    nbytes = nbytes * abs(header["nbits"]) // 8
    raw = np.fromfile(path, dtype=np.uint8)
    return header, raw[raw.size - nbytes :]


def header_differences(a: dict, b: dict) -> list[str]:
    """Return the header keys that differ between two parsed headers."""
    diffs = []
    for key in sorted((set(a) | set(b)) - DERIVED_KEYS):
        x, y = a.get(key), b.get(key)
        if (
            key in ANGLE_KEYS
            and isinstance(x, float)
            and isinstance(y, float)
            and abs(x - y) <= ANGLE_TOLERANCE_DEG
        ):
            continue

        if x != y:
            diffs.append(key)
    return diffs


@dataclasses.dataclass
class Result:
    case: Case
    dspsr_s: float | None = None
    ours_s: float | None = None
    mismatches: int | None = None
    header_diffs: list[str] = dataclasses.field(default_factory=list)
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error is not None:
            return "ERROR"
        if self.mismatches == 0 and not self.header_diffs:
            return "PASS"
        return "FAIL"

    @property
    def speedup(self) -> float | None:
        if self.dspsr_s is None or self.ours_s is None or self.ours_s == 0:
            return None
        return self.dspsr_s / self.ours_s


def evaluate(
    case: Case,
    dspsr: str,
    infile: str,
    workdir: Path,
    index: int,
    repeat: int,
) -> Result:
    """Run one case with both programs and compare their outputs."""
    env = {**os.environ, **SINGLE_THREAD_ENV}
    dspsr_out = workdir / f"dspsr_{index:03d}.fil"
    ours_out = workdir / f"ours_{index:03d}.fil"
    result = Result(case=case)
    dspsr_times: list[float] = []
    ours_times: list[float] = []
    for _ in range(repeat):
        for path in (dspsr_out, ours_out):
            path.unlink(missing_ok=True)
        d_time, d_rc, d_err = run_timed(
            dspsr_command(case, dspsr, infile, str(dspsr_out)),
            env,
        )
        if d_rc != 0 or not dspsr_out.exists():
            result.error = f"dspsr failed (rc={d_rc}): {d_err}"
            return result
        o_time, o_rc, o_err = run_timed(
            sigpyproc_command(case, infile, str(ours_out)),
            env,
        )
        if o_rc != 0 or not ours_out.exists():
            result.error = f"spp_digifil failed (rc={o_rc}): {o_err}"
            return result
        dspsr_times.append(d_time)
        ours_times.append(o_time)
    result.dspsr_s = statistics.median(dspsr_times)
    result.ours_s = statistics.median(ours_times)

    dspsr_header, dspsr_data = read_data(dspsr_out)
    ours_header, ours_data = read_data(ours_out)
    result.header_diffs = header_differences(dspsr_header, ours_header)
    if dspsr_data.size != ours_data.size:
        result.error = f"data size {dspsr_data.size} (dspsr) != {ours_data.size} (ours)"
    else:
        result.mismatches = int((dspsr_data != ours_data).sum())
    return result


def dspsr_version(dspsr: str) -> str:
    """Return the first line of ``digifil -i``, or ``unknown``."""
    try:
        proc = subprocess.run(
            [dspsr, "-i"], capture_output=True, text=True, check=False
        )
    except OSError:
        return "unknown"
    lines = (proc.stdout or proc.stderr).strip().splitlines()
    return lines[0] if lines else "unknown"


def resolve_dspsr(path_arg: str | None) -> str:
    """Return the dspsr digifil executable, or exit with a clear message."""
    msg = (
        "dspsr digifil not found: pass --dspsr, set DSPSR_DIGIFIL, "
        "or put digifil on PATH"
    )
    candidate = path_arg or os.environ.get("DSPSR_DIGIFIL") or "digifil"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise SystemExit(msg)
    return str(Path(resolved).resolve())


def render(console: Console, results: list[Result], infile: str, version: str) -> None:
    console.print(
        Panel.fit(
            f"input: [bold]{infile}[/]\n"
            f"dspsr: {version}\n"
            "single-threaded: dspsr -threads 1, numba/BLAS 1 thread",
            title="digifil matrix",
        ),
    )
    table = Table(show_lines=False, header_style="bold")
    table.add_column("case", no_wrap=True)
    table.add_column("dspsr [s]", justify="right")
    table.add_column("ours [s]", justify="right")
    table.add_column("speedup", justify="right")
    table.add_column("data mismatches", justify="right")
    table.add_column("header diffs")
    table.add_column("status", justify="center")
    for res in results:
        style = {"PASS": "green", "FAIL": "red", "ERROR": "magenta"}[res.status]
        table.add_row(
            res.case.label,
            "-" if res.dspsr_s is None else f"{res.dspsr_s:.2f}",
            "-" if res.ours_s is None else f"{res.ours_s:.2f}",
            "-" if res.speedup is None else f"{res.speedup:.1f}x",
            "-" if res.mismatches is None else f"{res.mismatches:,}",
            ", ".join(res.header_diffs) if res.header_diffs else "-",
            f"[{style}]{res.status}[/]",
        )
    console.print(table)
    for res in results:
        if res.error:
            console.print(f"[magenta]{res.case.label}[/]: {res.error}")

    passed = sum(res.status == "PASS" for res in results)
    timed = [res for res in results if res.speedup is not None]
    total_dspsr = sum(res.dspsr_s or 0.0 for res in timed)
    total_ours = sum(res.ours_s or 0.0 for res in timed)
    overall = total_dspsr / total_ours if total_ours else float("nan")
    console.print(
        Panel.fit(
            f"{passed}/{len(results)} cases pass\n"
            f"total dspsr {total_dspsr:.1f} s, total ours {total_ours:.1f} s, "
            f"overall speedup {overall:.1f}x",
            title="summary",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("infile", help="PSRFITS input file")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--dspsr", help="path to dspsr digifil")
    parser.add_argument("--repeat", type=int, default=1, help="timed repeats per case")
    parser.add_argument("--keep", help="directory to keep the output files in")
    args = parser.parse_args()

    infile = str(Path(args.infile).resolve())
    if not Path(infile).exists():
        parser.error(f"input file not found: {infile}")
    dspsr = resolve_dspsr(args.dspsr)
    cases = build_cases(args.preset)
    console = Console()

    workdir = (
        Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="digifil_"))
    )
    workdir.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    try:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("cases", total=len(cases))
            for index, case in enumerate(cases):
                progress.update(task, description=case.label)
                results.append(
                    evaluate(case, dspsr, infile, workdir, index, args.repeat),
                )
                progress.advance(task)
        render(console, results, infile, dspsr_version(dspsr))
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
    return 0 if all(res.status == "PASS" for res in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
