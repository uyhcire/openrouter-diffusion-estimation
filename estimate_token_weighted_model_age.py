#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Series:
    dates: np.ndarray  # ordinal days, float64
    values: np.ndarray  # float64


def _read_series_csv(path: Path) -> Series:
    dates: list[float] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            d = dt.date.fromisoformat(row["date"]).toordinal()
            v = float(row["intelligence_index"])
            dates.append(float(d))
            values.append(v)
    order = np.argsort(np.array(dates))
    return Series(dates=np.array(dates, dtype=np.float64)[order], values=np.array(values, dtype=np.float64)[order])


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.empty_like(values, dtype=np.float64)
    for i in range(values.size):
        a = max(0, i - half)
        b = min(values.size, i + half + 1)
        out[i] = float(np.median(values[a:b]))
    return out


def _rolling_mad_sigma(values: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        # Fallback: robust global sigma.
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        return np.full(values.shape, 1.4826 * mad, dtype=np.float64)
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.empty_like(values, dtype=np.float64)
    for i in range(values.size):
        a = max(0, i - half)
        b = min(values.size, i + half + 1)
        seg = values[a:b]
        med = float(np.median(seg))
        mad = float(np.median(np.abs(seg - med)))
        out[i] = 1.4826 * mad
    return out


def _cummax(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    for i in range(1, out.size):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _compress_plateaus(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Keep one x per unique y, using the midpoint x of that plateau.
    if x.size == 0:
        return x, y
    y_round = np.round(y, 10)
    out_x: list[float] = []
    out_y: list[float] = []
    start = 0
    for i in range(1, y_round.size + 1):
        if i == y_round.size or y_round[i] != y_round[start]:
            mid = (x[start] + x[i - 1]) / 2.0
            out_x.append(float(mid))
            out_y.append(float(y_round[start]))
            start = i
    return np.array(out_x, dtype=np.float64), np.array(out_y, dtype=np.float64)


def _invert_monotone_x_of_y(x: np.ndarray, y: np.ndarray, y_query: np.ndarray) -> np.ndarray:
    # y must be nondecreasing; compress duplicates to make it strictly increasing for interp.
    x2, y2 = _compress_plateaus(x, y)
    if x2.size < 2:
        return np.full_like(y_query, float(x2[0] if x2.size else np.nan), dtype=np.float64)
    # Clamp queries to range.
    yq = np.clip(y_query, float(y2.min()), float(y2.max()))
    return np.interp(yq, y2, x2)


def _interp1(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    return np.interp(xq, x, y)


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate token-weighted model vintage/age by inverting Figure 5 against Figure 15.")
    ap.add_argument("--fig5", type=Path, default=Path("out/figure5_red_median_timeseries.csv"))
    ap.add_argument("--fig15", type=Path, default=Path("out/figure15_red_median_timeseries.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("out"))
    ap.add_argument("--fig5_smooth_days", type=int, default=21)
    ap.add_argument("--fig15_smooth_days", type=int, default=7)
    ap.add_argument("--noise_window_days", type=int, default=31)
    ap.add_argument("--slope_floor", type=float, default=5e-4, help="Minimum dI/dt (per day) to avoid exploding error bars.")
    ap.add_argument("--sigma_floor", type=float, default=0.002, help="Minimum intelligence sigma (accounts for digitization/quantization).")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    fig5 = _read_series_csv(args.fig5)
    fig15 = _read_series_csv(args.fig15)

    # Smooth Figure 5 (less volatile), then enforce monotonicity (new-model intelligence improves over time).
    fig5_smooth = _rolling_median(fig5.values, args.fig5_smooth_days)
    fig5_mono = _cummax(fig5_smooth)

    # Compute a slope proxy from the smoothed curve (before cummax), then clip.
    # Use central differences in ordinal-day space (days).
    t5 = fig5.dates
    dt5 = np.gradient(t5)
    dI5 = np.gradient(fig5_smooth)
    slope5 = dI5 / np.maximum(dt5, 1.0)

    # Smooth Figure 15 (more volatile) and estimate observation noise.
    fig15_smooth = _rolling_median(fig15.values, args.fig15_smooth_days)
    # Use first-difference volatility to avoid sigma collapsing to ~0 when the smoother tracks the jagged curve.
    dI15 = np.diff(fig15.values, prepend=fig15.values[0])
    sigma_dI = _rolling_mad_sigma(dI15, args.noise_window_days)
    sigma_I = sigma_dI / np.sqrt(2.0)
    sigma_I = np.maximum(sigma_I, float(args.sigma_floor))

    # Invert: for each token-weighted intelligence, infer the vintage date where new models had similar intelligence.
    t_vintage = _invert_monotone_x_of_y(fig5.dates, fig5_mono, fig15_smooth)
    # Constrain vintage to not be after the usage date.
    t_vintage = np.minimum(t_vintage, fig15.dates)

    # Propagate uncertainty from intelligence to vintage date via local slope of Figure 5.
    slope_at_vintage = _interp1(fig5.dates, np.abs(slope5), t_vintage)
    slope_at_vintage = np.maximum(slope_at_vintage, float(args.slope_floor))
    sigma_t_days = sigma_I / slope_at_vintage

    z90 = 1.645  # ~90% CI half-width
    vintage_lo = t_vintage - z90 * sigma_t_days
    vintage_hi = t_vintage + z90 * sigma_t_days
    # Clip to observed fig5 range and to <= usage date.
    t5_min = float(np.min(fig5.dates))
    t5_max = float(np.max(fig5.dates))
    vintage_lo = np.clip(vintage_lo, t5_min, fig15.dates)
    vintage_hi = np.clip(vintage_hi, t5_min, fig15.dates)

    age_days = fig15.dates - t_vintage
    age_lo = fig15.dates - vintage_hi
    age_hi = fig15.dates - vintage_lo

    # Write joined CSV (daily points at fig15 resolution).
    out_csv = args.outdir / "token_weighted_implied_model_age.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "date",
                "token_weighted_intelligence",
                "token_weighted_intelligence_smooth",
                "implied_vintage_date",
                "implied_vintage_date_p05",
                "implied_vintage_date_p95",
                "implied_age_days",
                "implied_age_days_p05",
                "implied_age_days_p95",
                "sigma_intelligence_est",
            ]
        )
        for i in range(fig15.dates.size):
            d = dt.date.fromordinal(int(fig15.dates[i]))
            vdate = dt.date.fromordinal(int(round(t_vintage[i])))
            vlo = dt.date.fromordinal(int(round(vintage_lo[i])))
            vhi = dt.date.fromordinal(int(round(vintage_hi[i])))
            w.writerow(
                [
                    d.isoformat(),
                    f"{fig15.values[i]:.6f}",
                    f"{fig15_smooth[i]:.6f}",
                    vdate.isoformat(),
                    vlo.isoformat(),
                    vhi.isoformat(),
                    f"{age_days[i]:.2f}",
                    f"{age_lo[i]:.2f}",
                    f"{age_hi[i]:.2f}",
                    f"{sigma_I[i]:.6f}",
                ]
            )

    # Plot: date-date scatter + band, and age(t) + band.
    dates15 = [dt.date.fromordinal(int(d)) for d in fig15.dates.tolist()]
    vintage_date = [dt.date.fromordinal(int(round(d))) for d in t_vintage.tolist()]
    vintage_lo_d = [dt.date.fromordinal(int(round(d))) for d in vintage_lo.tolist()]
    vintage_hi_d = [dt.date.fromordinal(int(round(d))) for d in vintage_hi.tolist()]

    fig = plt.figure(figsize=(14, 8), dpi=200)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)

    ax1.fill_between(dates15, vintage_lo_d, vintage_hi_d, color="tab:blue", alpha=0.15, label="Vintage 90% band")
    ax1.plot(dates15, vintage_date, color="tab:blue", linewidth=1.2, label="Implied vintage (median)")
    ax1.plot(dates15, dates15, color="black", alpha=0.25, linewidth=1.0, linestyle="--", label="y = x")
    ax1.set_ylabel("Implied model vintage date")
    ax1.set_title("Token-weighted implied model vintage (from Figure 15 inverted through Figure 5)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", frameon=False)
    ax1.yaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax2.fill_between(dates15, age_lo, age_hi, color="tab:orange", alpha=0.15, label="Age 90% band")
    ax2.plot(dates15, age_days, color="tab:orange", linewidth=1.2, label="Implied age (days)")
    ax2.set_ylabel("Implied token-weighted model age (days)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", frameon=False)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    out_png = args.outdir / "token_weighted_implied_model_age.png"
    fig.savefig(out_png.as_posix(), bbox_inches="tight")
    plt.close(fig)

    # Extra diagnostic: date-date scatter (usage date vs vintage date).
    fig2 = plt.figure(figsize=(10, 6), dpi=200)
    ax = fig2.add_subplot(1, 1, 1)
    ax.fill_between(dates15, vintage_lo_d, vintage_hi_d, color="tab:blue", alpha=0.12)
    ax.scatter(dates15, vintage_date, s=10, alpha=0.6, color="tab:blue")
    ax.plot(dates15, dates15, color="black", alpha=0.25, linewidth=1.0, linestyle="--")
    ax.set_title("Date–date scatter: usage date vs implied vintage date")
    ax.set_xlabel("Usage date (Figure 15 date)")
    ax.set_ylabel("Implied vintage date (from Figure 5)")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig2.autofmt_xdate(rotation=30, ha="right")
    fig2.tight_layout()
    out_sc = args.outdir / "token_weighted_vintage_date_scatter.png"
    fig2.savefig(out_sc.as_posix(), bbox_inches="tight")
    plt.close(fig2)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_sc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
