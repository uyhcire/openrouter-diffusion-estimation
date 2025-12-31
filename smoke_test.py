#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    return p.stdout


def _is_date_column(name: str) -> bool:
    n = name.lower()
    return n == "date" or n.endswith("_date")


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"{path} has no header")
        rows = list(r)
        return list(r.fieldnames), rows


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _parse_float(s: str) -> float:
    return float(s)


def _close(a: float, b: float, *, abs_tol: float, rel_tol: float) -> bool:
    return math.isclose(a, b, abs_tol=abs_tol, rel_tol=rel_tol)


def _compare_csv_snapshot(
    *,
    generated: Path,
    snapshot: Path,
    abs_tol: float,
    rel_tol: float,
    name: str,
) -> None:
    gen_cols, gen_rows = _read_csv_rows(generated)
    snap_cols, snap_rows = _read_csv_rows(snapshot)
    if gen_cols != snap_cols:
        raise RuntimeError(f"{name}: column mismatch\n  gen={gen_cols}\n  snap={snap_cols}")
    if len(gen_rows) != len(snap_rows):
        raise RuntimeError(f"{name}: row count mismatch gen={len(gen_rows)} snap={len(snap_rows)}")

    for i, (gr, sr) in enumerate(zip(gen_rows, snap_rows, strict=True)):
        for col in gen_cols:
            gv = (gr.get(col) or "").strip()
            sv = (sr.get(col) or "").strip()
            if _is_date_column(col):
                if _parse_date(gv) != _parse_date(sv):
                    raise RuntimeError(f"{name}: date mismatch row {i} col {col}: gen={gv} snap={sv}")
                continue
            # Strings that look numeric should match approximately.
            try:
                ga = _parse_float(gv)
                sa = _parse_float(sv)
            except ValueError:
                if gv != sv:
                    raise RuntimeError(f"{name}: value mismatch row {i} col {col}: gen={gv!r} snap={sv!r}")
                continue
            if not _close(ga, sa, abs_tol=abs_tol, rel_tol=rel_tol):
                raise RuntimeError(
                    f"{name}: float mismatch row {i} col {col}: gen={ga} snap={sa} "
                    f"(abs_tol={abs_tol} rel_tol={rel_tol})"
                )


def main() -> int:
    ap = argparse.ArgumentParser(description="Lightweight regression checks for figure extraction scripts.")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use (default: current).")
    ap.add_argument("--outdir", type=Path, default=Path("out"), help="Committed snapshot directory.")
    ap.add_argument("--min-fig15-union-iou", type=float, default=0.88)
    ap.add_argument("--min-fig15-dark-iou", type=float, default=0.88)
    ap.add_argument("--abs-tol", type=float, default=5e-4, help="Absolute tolerance for CSV snapshot comparisons.")
    ap.add_argument("--rel-tol", type=float, default=5e-4, help="Relative tolerance for CSV snapshot comparisons.")
    args = ap.parse_args()

    py = args.python
    snapdir = args.outdir

    with tempfile.TemporaryDirectory(prefix="smoke_out_") as td:
        tmpdir = Path(td)

        # Base extractions (snapshots are committed under `out/`).
        _run([py, "extract_figure15_red_timeseries.py", "--figure", "5", "--curve", "red_median", "--outdir", td])
        _run([py, "extract_figure15_red_timeseries.py", "--figure", "5", "--curve", "p90_frontier", "--outdir", td])
        _run([py, "extract_figure15_red_timeseries.py", "--figure", "6", "--curve", "black_frontier", "--outdir", td])
        _run([py, "extract_figure15_red_timeseries.py", "--figure", "15", "--curve", "red_median", "--outdir", td])
        _run([py, "extract_figure15_red_timeseries.py", "--figure", "15", "--percentiles", "--outdir", td])
        _run([py, "extract_figure15_red_timeseries.py", "--figure", "11", "--percentiles", "--outdir", td])

        # Figure 15 percentiles should stay high-quality (IoU regression guardrail).
        out = _run([py, "verify_figure15_percentiles.py", "--outdir", td])
        print(out, end="")

        def parse_iou(line: str, key: str) -> float:
            # Example: "UNION  IoU=0.9136  miss=..."
            if not line.startswith(key):
                raise ValueError("wrong key")
            i = line.index("IoU=") + 4
            j = line.find(" ", i)
            return float(line[i:j])

        union_iou = None
        dark_iou = None
        for line in out.splitlines():
            if line.startswith("UNION"):
                union_iou = parse_iou(line, "UNION")
            if line.startswith("DARK"):
                dark_iou = parse_iou(line, "DARK")
        if union_iou is None or dark_iou is None:
            raise RuntimeError("Failed to parse IoU from verify_figure15_percentiles output.")
        if union_iou < args.min_fig15_union_iou:
            raise RuntimeError(f"Figure 15 union IoU regression: {union_iou:.4f} < {args.min_fig15_union_iou:.4f}")
        if dark_iou < args.min_fig15_dark_iou:
            raise RuntimeError(f"Figure 15 dark IoU regression: {dark_iou:.4f} < {args.min_fig15_dark_iou:.4f}")

        # Derived outputs (depend on the base extractions).
        _run(
            [
                py,
                "estimate_token_weighted_model_age.py",
                "--fig5",
                (tmpdir / "figure5_red_median_timeseries.csv").as_posix(),
                "--fig15",
                (tmpdir / "figure15_red_median_timeseries.csv").as_posix(),
                "--outdir",
                td,
            ]
        )
        _run(
            [
                py,
                "estimate_token_weighted_model_age.py",
                "--fig5",
                (tmpdir / "figure5_p90_frontier_timeseries.csv").as_posix(),
                "--fig15",
                (tmpdir / "figure15_red_median_timeseries.csv").as_posix(),
                "--outdir",
                td,
            ]
        )
        _run(
            [
                py,
                "estimate_token_weighted_model_age.py",
                "--fig5",
                (tmpdir / "figure6_black_frontier_timeseries.csv").as_posix(),
                "--fig15",
                (tmpdir / "figure15_red_median_timeseries.csv").as_posix(),
                "--outdir",
                td,
            ]
        )
        _run(
            [
                py,
                "plot_dollar_weighted_mean_vintage_age.py",
                "--fig15",
                (tmpdir / "figure15_token_weighted_percentiles.csv").as_posix(),
                "--fig11",
                (tmpdir / "figure11_price_to_intelligence_ratio_percentiles.csv").as_posix(),
                "--frontier",
                (tmpdir / "figure6_black_frontier_timeseries.csv").as_posix(),
                "--outdir",
                td,
            ]
        )

        # Snapshot comparisons (CSVs).
        snapshots: list[str] = [
            "figure5_red_median_timeseries.csv",
            "figure5_p90_frontier_timeseries.csv",
            "figure6_black_frontier_timeseries.csv",
            "figure11_price_to_intelligence_ratio_percentiles.csv",
            "figure15_red_median_timeseries.csv",
            "figure15_token_weighted_percentiles.csv",
            "token_weighted_implied_model_age__fig5_red_median.csv",
            "token_weighted_implied_model_age__fig5_p90_frontier.csv",
            "token_weighted_implied_model_age__fig5_figure6_black_frontier.csv",
            "dollar_weighted_mean_vintage_age_timeseries.csv",
        ]
        for name in snapshots:
            gen = tmpdir / name
            snap = snapdir / name
            if not gen.exists():
                raise RuntimeError(f"Missing generated CSV: {gen}")
            if not snap.exists():
                raise RuntimeError(f"Missing snapshot CSV (commit it?): {snap}")
            _compare_csv_snapshot(
                generated=gen,
                snapshot=snap,
                abs_tol=args.abs_tol,
                rel_tol=args.rel_tol,
                name=name,
            )

    print("smoke_test.py: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
