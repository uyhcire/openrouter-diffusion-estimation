#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def _read_series(path: Path) -> list[tuple[dt.date, float]]:
    rows: list[tuple[dt.date, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((dt.date.fromisoformat(row["date"]), float(row["intelligence_index"])))
    rows.sort(key=lambda t: t[0])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Scatterplot diagnostics for Figure 5 vs Figure 15 intelligence series.")
    ap.add_argument("--outdir", type=Path, default=Path("out"))
    ap.add_argument("--figure5", type=Path, default=Path("out/figure5_red_median_timeseries.csv"))
    ap.add_argument("--figure15", type=Path, default=Path("out/figure15_red_median_timeseries.csv"))
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path (default: out/model_age_scatter.png)")
    args = ap.parse_args()

    fig5 = _read_series(args.figure5)
    fig15 = _read_series(args.figure15)

    out = args.out or (args.outdir / "model_age_scatter.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Panel A: time-series overlay
    fig = plt.figure(figsize=(14, 8), dpi=200)
    ax_ts, ax_sc = fig.subplots(1, 2)

    d5 = [d for d, _ in fig5]
    v5 = [v for _, v in fig5]
    d15 = [d for d, _ in fig15]
    v15 = [v for _, v in fig15]

    ax_ts.plot(d5, v5, color="black", alpha=0.55, linewidth=1.0, label="Figure 5 (new-model median)")
    ax_ts.plot(d15, v15, color="red", linewidth=1.2, label="Figure 15 (token-weighted median)")
    ax_ts.set_title("Time series overlay")
    ax_ts.set_ylabel("Intelligence Index")
    ax_ts.grid(True, alpha=0.25)
    ax_ts.legend(loc="upper left", frameon=False)
    ax_ts.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30, ha="right")

    # Panel B: intelligence-to-date mapping (invertibility diagnostic)
    ax_sc.scatter(v5, d5, s=6, color="black", alpha=0.35, label="Figure 5 points")
    ax_sc.scatter(v15, d15, s=10, color="red", alpha=0.7, label="Figure 15 points")
    ax_sc.set_title("Intelligence → date (for inversion)")
    ax_sc.set_xlabel("Intelligence Index")
    ax_sc.grid(True, alpha=0.25)
    ax_sc.legend(loc="lower right", frameon=False)
    ax_sc.yaxis.set_major_locator(mdates.YearLocator())
    ax_sc.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(out.as_posix(), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

