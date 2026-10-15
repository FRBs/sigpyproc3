"""Search candidate tables.

A modern stand-in for SIGPROC ``.prd`` / ``.bst`` bookkeeping: candidates
from FFT search and single-pulse detection live in one
:class:`~astropy.table.QTable` with SI units. There is no ``.prd`` writer
and no PGPLOT.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np
from astropy import units as u
from astropy.table import QTable, Table, vstack

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import TextIO

CANDIDATE_DTYPE: list[tuple[str, str]] = [
    ("period_s", "f8"),
    ("freq_hz", "f8"),
    ("snr", "f4"),
    ("dm", "f8"),
    ("accel", "f8"),
    ("nharm", "i4"),
    ("bin", "i8"),
    ("width_s", "f8"),
    ("kind", "U8"),
]
HEIMDALL_CANDIDATE_NAMES = ("filter_index", "begin_bin", "end_bin", "members")
PULSE_CANDIDATE_NAMES = ("dm", "snr", "time_s", "bin", "width_bins", "width_s")
CANDIDATE_NAMES: tuple[str, ...] = tuple(name for name, _ in CANDIDATE_DTYPE)
_COLUMN_UNITS: dict[str, u.UnitBase] = {
    "period_s": u.s,
    "freq_hz": u.Hz,
    "dm": u.pc / u.cm**3,
    "width_s": u.s,
    "accel": u.m / u.s**2,
}
_KIND_MAX = 8


def _cell_float(row: Mapping[str, object], key: str) -> float | None:
    if key not in row or row[key] is None:
        return None
    array = np.asarray(row[key])
    if array.size != 1:
        msg = f"{key} must be a scalar, got shape {array.shape}"
        raise ValueError(msg)
    return float(array.reshape(-1)[0])


def _scalar_in(value: object, unit: u.UnitBase, name: str) -> float:
    """Return one number in ``unit``.

    A quantity is converted. A bare number is already in ``unit``. A
    non-scalar cell is rejected.
    """
    if isinstance(value, u.Quantity):
        if value.size != 1:
            msg = f"{name} must be a scalar, got shape {value.shape}"
            raise ValueError(msg)
        try:
            return float(value.to_value(unit))
        except u.UnitConversionError as exc:
            msg = f"{name} unit {value.unit} is not compatible with {unit}"
            raise ValueError(msg) from exc
    array = np.asarray(value)
    if array.size != 1:
        msg = f"{name} must be a scalar, got shape {array.shape}"
        raise ValueError(msg)
    return float(array.reshape(-1)[0])


def _optional_unit(
    row: Mapping[str, object],
    key: str,
    unit: u.UnitBase,
) -> float | None:
    if key not in row or row[key] is None:
        return None
    return _scalar_in(row[key], unit, key)


def _cell_int(row: Mapping[str, object], key: str, default: int) -> int:
    value = _cell_float(row, key)
    return default if value is None else int(value)


def _extra_names(rows: Sequence[Mapping[str, object]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in CANDIDATE_NAMES and key not in names:
                names.append(key)
    return names


def _extra_column(rows: Sequence[Mapping[str, object]], name: str) -> np.ndarray:
    raw = [row.get(name) for row in rows]
    if any(isinstance(item, str) for item in raw):
        text = ["" if item is None else str(item) for item in raw]
        return np.array(text, dtype="U32")
    out = np.empty(len(rows), dtype=np.float64)
    for i, item in enumerate(raw):
        out[i] = np.nan if item is None else _cell_float({name: item}, name)
    return out


def _plain(col: object) -> np.ndarray:
    value = getattr(col, "value", col)
    return np.asarray(value)


def _empty_qtable() -> QTable:
    table = QTable()
    for name, dtype in CANDIDATE_DTYPE:
        data = np.array([], dtype=dtype)
        unit = _COLUMN_UNITS.get(name)
        table[name] = data * unit if unit is not None else data
    return table


def _attach_units(table: QTable) -> QTable:
    """Store schema columns in seconds, Hz, pc cm^-3, and m s^-2.

    CSV files keep the numeric value in that unit and do not store the unit
    object. ``bin_origin``, when present, is ``center`` or ``start`` and is
    not implied by ``kind``.
    """
    for name, unit in _COLUMN_UNITS.items():
        if name not in table.colnames:
            continue
        column = table[name]
        if isinstance(column, u.Quantity):
            try:
                converted = column.to(unit)
            except u.UnitConversionError as exc:
                msg = f"column {name} has unit {column.unit}, expected {unit}"
                raise ValueError(msg) from exc
            values = np.asarray(converted.value)
        else:
            values = np.asarray(column)
        if values.ndim > 1:
            msg = f"column {name} must be 1-D, got shape {values.shape}"
            raise ValueError(msg)
        table[name] = values * unit
    return table


def _validate_columns(table: QTable) -> None:
    missing = [name for name in CANDIDATE_NAMES if name not in table.colnames]
    if missing:
        msg = f"Candidate table is missing columns: {missing}"
        raise ValueError(msg)


def _as_qtable(value: QTable | Table) -> QTable:
    if isinstance(value, QTable):
        return value
    return QTable(value)


@attrs.define(auto_attribs=True, slots=True)
class CandidateTable:
    """Table of periodic or single-pulse search candidates.

    Parameters
    ----------
    table : ~astropy.table.QTable
        Table whose columns include ``period_s``, ``freq_hz``, ``snr``,
        ``dm``, ``accel``, ``nharm``, ``bin``, ``width_s``, and ``kind``.
        ``period_s``, ``freq_hz``, ``dm``, ``width_s``, and ``accel`` are
        stored in seconds, Hz, pc cm^-3, seconds, and m s^-2. A compatible
        quantity is converted; an incompatible unit is rejected. CSV export
        writes those canonical numbers and does not keep the unit object.
        Optional columns, including ``bin_origin`` (``center`` or ``start``),
        are kept. ``kind`` does not encode the bin convention.

    Attributes
    ----------
    table
    """

    table: QTable = attrs.field(converter=_as_qtable)

    def __attrs_post_init__(self) -> None:
        _validate_columns(self.table)
        _attach_units(self.table)

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return f"CandidateTable(n={len(self)})"

    def __getitem__(
        self,
        key: int | slice | np.ndarray,
    ) -> CandidateTable:
        """Return a row subset as a new :class:`CandidateTable`.

        Parameters
        ----------
        key : int | slice | ndarray
            Row index, slice, or boolean / integer mask.

        Returns
        -------
        CandidateTable
            Selected rows. An integer key yields a one-row table.
        """
        if isinstance(key, (int, np.integer)):
            return CandidateTable(QTable(self.table[[int(key)]]))
        return CandidateTable(QTable(self.table[key]))

    def __add__(self, other: CandidateTable) -> CandidateTable:
        if not isinstance(other, CandidateTable):
            return NotImplemented
        return self.concatenate([self, other])

    @classmethod
    def empty(cls) -> CandidateTable:
        """Return an empty candidate table."""
        return cls(_empty_qtable())

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, object]]) -> CandidateTable:
        """Build a table from row mappings.

        Missing ``period_s`` / ``freq_hz`` are inferred from each other.
        Other omitted fields use defaults (``dm=0``, ``accel=0``,
        ``nharm=1``, ``bin=-1``, ``width_s=NaN``, ``kind="fft"``).

        Parameters
        ----------
        rows : sequence of dict
            One mapping per candidate.

        Returns
        -------
        CandidateTable
            New table containing ``rows``.
        """
        if not rows:
            return cls.empty()
        period = np.empty(len(rows), dtype=np.float64)
        freq = np.empty(len(rows), dtype=np.float64)
        snr = np.empty(len(rows), dtype=np.float32)
        dm = np.empty(len(rows), dtype=np.float64)
        accel = np.empty(len(rows), dtype=np.float64)
        nharm = np.empty(len(rows), dtype=np.int32)
        bins = np.empty(len(rows), dtype=np.int64)
        width = np.empty(len(rows), dtype=np.float64)
        kind = np.empty(len(rows), dtype="U8")
        for i, row in enumerate(rows):
            period_s = _optional_unit(row, "period_s", u.s)
            freq_hz = _optional_unit(row, "freq_hz", u.Hz)
            if period_s is None and freq_hz is not None:
                freq[i] = freq_hz
                period[i] = (1.0 / freq_hz) if freq_hz != 0.0 else np.nan
            elif freq_hz is None and period_s is not None:
                period[i] = period_s
                freq[i] = (1.0 / period_s) if period_s != 0.0 else np.nan
            else:
                period[i] = np.nan if period_s is None else period_s
                freq[i] = np.nan if freq_hz is None else freq_hz
            snr_val = _cell_float(row, "snr")
            snr[i] = np.float32(np.nan if snr_val is None else snr_val)
            dm_val = _optional_unit(row, "dm", u.pc / u.cm**3)
            dm[i] = 0.0 if dm_val is None else dm_val
            accel_val = _optional_unit(row, "accel", u.m / u.s**2)
            accel[i] = 0.0 if accel_val is None else accel_val
            nharm[i] = _cell_int(row, "nharm", 1)
            bins[i] = _cell_int(row, "bin", -1)
            width_val = _optional_unit(row, "width_s", u.s)
            width[i] = np.nan if width_val is None else width_val
            raw_kind = "fft" if row.get("kind") is None else str(row["kind"])
            kind[i] = raw_kind[:_KIND_MAX]
        table = QTable()
        table["period_s"] = period * u.s
        table["freq_hz"] = freq * u.Hz
        table["snr"] = snr
        table["dm"] = dm * (u.pc / u.cm**3)
        table["accel"] = accel * (u.m / u.s**2)
        table["nharm"] = nharm
        table["bin"] = bins
        table["width_s"] = width * u.s
        table["kind"] = kind
        for name in _extra_names(rows):
            table[name] = _extra_column(rows, name)
        return cls(table)

    @classmethod
    def concatenate(cls, tables: Sequence[CandidateTable]) -> CandidateTable:
        """Stack candidate tables row-wise.

        Parameters
        ----------
        tables : sequence of CandidateTable
            Tables to concatenate.

        Returns
        -------
        CandidateTable
            Combined table, retaining optional columns even when all inputs
            are empty. No inputs returns :meth:`empty`.
        """
        if not tables:
            return cls.empty()
        nonempty = [item.table for item in tables if len(item) > 0]
        return cls(QTable(vstack(nonempty or [item.table for item in tables])))

    @classmethod
    def from_csv(cls, path: str | Path) -> CandidateTable:
        """Read a shared candidate CSV or pulse-specific CSV.

        Pulse CSVs written by :meth:`to_pulse_csv` are restored to the shared
        in-memory schema with kind="pulse" and unused periodic fields NaN.
        Derived event times and integer widths remain available in the CSV.

        Parameters
        ----------
        path : str | Path
            Input CSV path.

        Returns
        -------
        CandidateTable
            Loaded candidates.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        raw = Table.read(path, format="ascii.csv")
        if "kind" not in raw.colnames and set(PULSE_CANDIDATE_NAMES) <= set(
            raw.colnames
        ):
            pulses = cls.from_rows(
                [
                    {
                        "kind": "pulse",
                        "dm": row["dm"],
                        "snr": row["snr"],
                        "bin": row["bin"],
                        "width_s": row["width_s"],
                    }
                    for row in raw
                ]
            )
            for name in raw.colnames:
                if name not in PULSE_CANDIDATE_NAMES:
                    pulses.table[name] = raw[name]
            return pulses
        missing = [name for name in CANDIDATE_NAMES if name not in raw.colnames]
        if missing:
            msg = f"CSV is missing columns: {missing}"
            raise ValueError(msg)
        table = QTable()
        table["period_s"] = np.asarray(raw["period_s"], dtype=np.float64) * u.s
        table["freq_hz"] = np.asarray(raw["freq_hz"], dtype=np.float64) * u.Hz
        table["snr"] = np.asarray(raw["snr"], dtype=np.float32)
        table["dm"] = np.asarray(raw["dm"], dtype=np.float64) * (u.pc / u.cm**3)
        table["accel"] = np.asarray(raw["accel"], dtype=np.float64) * (u.m / u.s**2)
        table["nharm"] = np.asarray(raw["nharm"], dtype=np.int32)
        table["bin"] = np.asarray(raw["bin"], dtype=np.int64)
        table["width_s"] = np.asarray(raw["width_s"], dtype=np.float64) * u.s
        table["kind"] = np.asarray(raw["kind"], dtype="U8")
        for name in raw.colnames:
            if name not in CANDIDATE_NAMES:
                table[name] = raw[name]
        return cls(table)

    def to_csv(self, path: str | Path | TextIO) -> None:
        """Write candidates as a plain CSV (no units metadata).

        Column names carry the SI unit (``period_s``, ``freq_hz``, …) so
        the file is usable outside astropy. This is the public interchange
        format; there is no ``.prd`` writer.

        Parameters
        ----------
        path : str | Path | TextIO
            Output path (overwritten if it exists) or an open text stream.
        """
        plain = Table()
        for name in self.table.colnames:
            plain[name] = self.table[name]
            if name in _COLUMN_UNITS:
                plain[name] = _plain(self.table[name])
        if hasattr(path, "write"):
            plain.write(path, format="ascii.csv")
        else:
            plain.write(path, format="ascii.csv", overwrite=True)

    def to_pulse_csv(self, path: str | Path | TextIO, tsamp: float) -> None:
        """Write pulse event fields without periodic-search columns.

        Columns are dm, snr, time_s, bin, width_bins, and width_s. Event time
        is bin * tsamp in seconds since observation start. The bin convention
        follows the search method (center for PRESTO and Heimdall, start for
        matched filter).
        Heimdall's filter_index, begin_bin, end_bin, and members columns are
        also included when present. The shared base schema remains unchanged.

        Parameters
        ----------
        path : str | Path | TextIO
            Output CSV path or open text stream.
        tsamp : float
            Positive sampling interval in seconds.

        Raises
        ------
        ValueError
            If sampling interval is invalid, rows are not pulses, or widths
            are not positive whole numbers of samples.
        """
        if not np.isfinite(tsamp) or tsamp <= 0:
            msg = "tsamp must be finite and positive"
            raise ValueError(msg)
        if np.any(self.values("kind") != "pulse"):
            msg = "pulse export requires only pulse candidates"
            raise ValueError(msg)
        widths = self.values("width_s") / tsamp
        rounded = np.rint(widths)
        if (
            not np.all(np.isfinite(widths))
            or np.any(rounded < 1)
            or not np.allclose(widths, rounded, rtol=0, atol=1e-6)
        ):
            msg = "pulse widths must be positive whole numbers of samples"
            raise ValueError(msg)
        plain = Table(
            {
                "dm": self.values("dm"),
                "snr": self.values("snr"),
                "time_s": self.values("bin") * tsamp,
                "bin": self.values("bin"),
                "width_bins": rounded.astype(np.int64),
                "width_s": self.values("width_s"),
            }
        )
        for name in (*HEIMDALL_CANDIDATE_NAMES, "bin_origin"):
            if name in self.table.colnames:
                plain[name] = self.table[name]
        if hasattr(path, "write"):
            plain.write(path, format="ascii.csv")
        else:
            plain.write(path, format="ascii.csv", overwrite=True)

    def values(self, name: str) -> np.ndarray:
        """Return a column as a unitless :class:`numpy.ndarray`.

        Parameters
        ----------
        name : str
            Column name.

        Returns
        -------
        ndarray
            Numeric or string values with units stripped.
        """
        return _plain(self.table[name])

    def sort_by_snr(self) -> CandidateTable:
        """Return a copy sorted by decreasing SNR."""
        if len(self) == 0:
            return CandidateTable(QTable(self.table))
        order = np.argsort(-_plain(self.table["snr"]))
        return CandidateTable(QTable(self.table[order]))

    def cluster(
        self,
        *,
        period_tol: float = 1e-4,
        harm_max: int = 16,
        snr_min: float = 8.0,
    ) -> CandidateTable:
        """Keep the loudest member of each harmonic family.

        This is the bookkeeping half of SIGPROC ``best`` (no PGPLOT, no
        ``.bst`` / ``.fld``). A weaker candidate with period ``P_i`` is
        dropped when ``|P_i / P_j - k / n| <= period_tol`` for a louder
        kept period ``P_j`` and integers ``k, n <= harm_max``.

        Single-pulse rows (non-finite or non-positive period) are never
        treated as harmonics of each other; they are kept if they pass
        ``snr_min``. Slice the table first if you want to cluster only
        one ``kind`` or a DM window.

        Parameters
        ----------
        period_tol : float, optional
            Relative period match tolerance, by default ``1e-4``.
        harm_max : int, optional
            Largest integer ``k`` and ``n`` in the ``k / n`` grid, by
            default 16.
        snr_min : float, optional
            Discard candidates below this SNR first, by default 8.0.

        Returns
        -------
        CandidateTable
            Survivors, still sorted by decreasing SNR.

        Raises
        ------
        ValueError
            If ``period_tol`` is not positive or ``harm_max`` is ``< 1``.
        """
        if period_tol <= 0:
            msg = "period_tol must be positive"
            raise ValueError(msg)
        if harm_max < 1:
            msg = "harm_max must be at least 1"
            raise ValueError(msg)
        if len(self) == 0:
            return CandidateTable(QTable(self.table))

        snr = _plain(self.table["snr"]).astype(np.float64, copy=False)
        period = _plain(self.table["period_s"]).astype(np.float64, copy=False)
        eligible = np.isfinite(snr) & (snr >= snr_min)
        order = np.argsort(-snr)

        targets = np.unique(
            [k / n for n in range(1, harm_max + 1) for k in range(1, harm_max + 1)]
        )
        kept_period = np.empty(len(self), dtype=np.float64)
        nkept = 0
        keep = np.zeros(len(self), dtype=bool)
        for idx in order:
            if not eligible[idx]:
                continue
            p_i = period[idx]
            if not np.isfinite(p_i) or p_i <= 0:
                keep[idx] = True
                continue
            if nkept:
                ratio = p_i / kept_period[:nkept]
                pos = np.searchsorted(targets, ratio)
                below = targets[np.clip(pos - 1, 0, targets.size - 1)]
                above = targets[np.clip(pos, 0, targets.size - 1)]
                nearest = np.minimum(np.abs(ratio - below), np.abs(ratio - above))
                if np.any(nearest <= period_tol):
                    continue
            keep[idx] = True
            kept_period[nkept] = p_i
            nkept += 1
        return CandidateTable(QTable(self.table[keep])).sort_by_snr()
