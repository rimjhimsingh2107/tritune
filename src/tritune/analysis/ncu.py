"""Parse `ncu --csv --page raw` output."""

from __future__ import annotations

import csv
import io


def _number(text: str) -> float | None:
    text = text.strip().replace(",", "")
    if not text or text in ("n/a", "N/A"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_ncu_csv(text: str, wanted: dict[str, str]) -> tuple[dict[str, float], dict[str, str]]:
    """Return (metrics, units) for the first profiled kernel.

    `wanted` maps ncu metric names to the short names we store. ncu prints
    ==PROF== chatter and the program's own stdout around the CSV, so we look
    for the header row instead of assuming it's on line one.
    """
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"ID"')), None)
    if start is None:
        return {}, {}
    rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    if len(rows) < 2:
        return {}, {}
    header = rows[0]
    units_row = rows[1] if len(rows) > 2 and not rows[1][0].strip().isdigit() else None
    data = rows[2] if units_row else rows[1]

    metrics: dict[str, float] = {}
    units: dict[str, str] = {}
    for col, name in enumerate(header):
        if name not in wanted or col >= len(data):
            continue
        val = _number(data[col])
        if val is None:
            continue
        short = wanted[name]
        metrics[short] = val
        if units_row and col < len(units_row):
            units[short] = units_row[col]
    return metrics, units
