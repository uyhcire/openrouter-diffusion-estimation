#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    return p.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description="Lightweight regression checks for figure extraction scripts.")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use (default: current).")
    ap.add_argument("--outdir", type=Path, default=Path("out"))
    ap.add_argument("--min-fig15-union-iou", type=float, default=0.88)
    ap.add_argument("--min-fig15-dark-iou", type=float, default=0.88)
    args = ap.parse_args()

    py = args.python

    # Figure 15 percentiles should stay high-quality.
    _run([py, "extract_figure15_red_timeseries.py", "--figure", "15", "--percentiles"])
    out = _run([py, "verify_figure15_percentiles.py"])
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

    # Quick Figure 11 smoke check: extraction should run and produce expected artifacts.
    _run([py, "extract_figure15_red_timeseries.py", "--figure", "11", "--percentiles"])
    for name in [
        "figure11_price_to_intelligence_ratio_percentiles.csv",
        "figure11_percentiles_overlay.png",
        "figure11_percentiles_timeseries.png",
    ]:
        path = args.outdir / name
        if not path.exists():
            raise RuntimeError(f"Missing expected output: {path}")

    print("smoke_test.py: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

