#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class DailyPoint:
    date: dt.date
    mean_capability_age_days: float


@dataclass(frozen=True)
class MethodConfig:
    """Configuration for a robustness variant of the estimation method."""
    name: str
    bands: tuple[tuple[str, str, float], ...]  # (lo_pct, hi_pct, weight)
    use_geometric_mean: bool  # True=geometric, False=arithmetic for price averaging
    assume_lognormal: bool = False  # If True, fit lognormal and use proper band expectations


# Z-scores for standard percentiles (for normal distribution fitting)
PERCENTILE_Z = {
    "p0": -3.5,   # approximate, truncated
    "p10": -1.2816,
    "p25": -0.6745,
    "p50": 0.0,
    "p75": 0.6745,
    "p90": 1.2816,
    "p100": 3.5,  # approximate, truncated
}


# Define method variants for robustness analysis
METHOD_BASELINE = MethodConfig(
    name="Baseline",
    bands=(("p10", "p25", 0.15), ("p25", "p50", 0.25), ("p50", "p75", 0.25), ("p75", "p90", 0.15)),
    use_geometric_mean=True,
)

METHOD_EQUAL_WEIGHTS = MethodConfig(
    name="Equal weights",
    bands=(("p10", "p25", 0.25), ("p25", "p50", 0.25), ("p50", "p75", 0.25), ("p75", "p90", 0.25)),
    use_geometric_mean=True,
)

METHOD_ARITHMETIC = MethodConfig(
    name="Arithmetic price",
    bands=(("p10", "p25", 0.15), ("p25", "p50", 0.25), ("p50", "p75", 0.25), ("p75", "p90", 0.15)),
    use_geometric_mean=False,
)

METHOD_WITH_TAILS = MethodConfig(
    name="With tails",
    bands=(("p0", "p10", 0.10), ("p10", "p25", 0.15), ("p25", "p50", 0.25),
           ("p50", "p75", 0.25), ("p75", "p90", 0.15), ("p90", "p100", 0.10)),
    use_geometric_mean=True,
)

METHOD_LOGNORMAL = MethodConfig(
    name="Lognormal dist",
    bands=(("p10", "p25", 0.15), ("p25", "p50", 0.25), ("p50", "p75", 0.25), ("p75", "p90", 0.15)),
    use_geometric_mean=True,
    assume_lognormal=True,
)

ALL_METHODS = [METHOD_BASELINE, METHOD_EQUAL_WEIGHTS, METHOD_ARITHMETIC, METHOD_WITH_TAILS, METHOD_LOGNORMAL]


def _read_percentiles(path: Path) -> dict[dt.date, dict[str, float]]:
    rows: dict[dt.date, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"date", "p10", "p25", "p50", "p75", "p90"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in r:
            d = dt.date.fromisoformat(row["date"])
            rows[d] = {k: float(row[k]) for k in ["p10", "p25", "p50", "p75", "p90"]}
    return rows


def _read_frontier_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    dates: list[float] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"date", "intelligence_index"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in r:
            d = dt.date.fromisoformat(row["date"]).toordinal()
            v = float(row["intelligence_index"])
            dates.append(float(d))
            values.append(v)
    order = np.argsort(np.array(dates, dtype=np.float64))
    t = np.array(dates, dtype=np.float64)[order]
    y = np.array(values, dtype=np.float64)[order]
    return t, y


def _cummax(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    for i in range(1, out.size):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _compress_plateaus(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return x, y
    y_round = np.round(y, 12)
    out_x: list[float] = []
    out_y: list[float] = []
    start = 0
    for i in range(1, y_round.size + 1):
        if i == y_round.size or y_round[i] != y_round[start]:
            out_x.append(float((x[start] + x[i - 1]) / 2.0))
            out_y.append(float(y_round[start]))
            start = i
    return np.array(out_x, dtype=np.float64), np.array(out_y, dtype=np.float64)


def _invert_monotone_x_of_y(x: np.ndarray, y: np.ndarray, y_query: np.ndarray) -> np.ndarray:
    x2, y2 = _compress_plateaus(x, y)
    if x2.size < 2:
        return np.full_like(y_query, float(x2[0] if x2.size else np.nan), dtype=np.float64)
    yq = np.clip(y_query, float(np.min(y2)), float(np.max(y2)))
    return np.interp(yq, y2, x2)


def _fit_normal_from_percentiles(percentiles: dict[str, float], log_space: bool = False) -> tuple[float, float]:
    """Fit μ and σ from percentiles assuming normal (or lognormal) distribution.

    Args:
        percentiles: Dict with p10, p25, p50, p75, p90
        log_space: If True, fit normal to log-transformed values (i.e., fit lognormal)

    Returns (mu, sigma) - if log_space, these are log-scale parameters.
    """
    if log_space:
        # Work in log space for lognormal fit
        log_pcts = {k: math.log(max(v, 1e-30)) for k, v in percentiles.items()}
        sigma = (log_pcts["p75"] - log_pcts["p25"]) / (2 * 0.6745)
        mu = log_pcts["p50"]
    else:
        # Normal fit
        sigma = (percentiles["p75"] - percentiles["p25"]) / (2 * 0.6745)
        mu = percentiles["p50"]
    return mu, max(sigma, 1e-10)


def _truncated_normal_mean(mu: float, sigma: float, z_lo: float, z_hi: float) -> float:
    """Expected value of normal distribution truncated to [z_lo, z_hi] in z-score space.

    E[X | z_lo < Z < z_hi] = μ + σ * (φ(z_lo) - φ(z_hi)) / (Φ(z_hi) - Φ(z_lo))
    """
    from math import erf, exp, pi, sqrt

    def phi(z: float) -> float:
        """Standard normal PDF."""
        return exp(-0.5 * z * z) / sqrt(2 * pi)

    def Phi(z: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + erf(z / sqrt(2)))

    cdf_diff = Phi(z_hi) - Phi(z_lo)
    if cdf_diff < 1e-10:
        return mu + sigma * 0.5 * (z_lo + z_hi)
    pdf_diff = phi(z_lo) - phi(z_hi)
    return mu + sigma * pdf_diff / cdf_diff


def _extrapolate_tails(percentiles: dict[str, float], log_scale: bool = False) -> dict[str, float]:
    """Extrapolate p0 and p100 from existing percentiles.

    Args:
        percentiles: Dict with p10, p25, p50, p75, p90 values
        log_scale: If True, extrapolate in log space (multiplicative); else linear
    """
    result = dict(percentiles)
    if log_scale:
        # Multiplicative extrapolation (linear in log space)
        # p0 = p10 * (p10 / p25)
        ratio_lo = percentiles["p10"] / max(percentiles["p25"], 1e-30)
        result["p0"] = max(percentiles["p10"] * ratio_lo, 1e-30)
        # p100 = p90 * (p90 / p75)
        ratio_hi = percentiles["p90"] / max(percentiles["p75"], 1e-30)
        result["p100"] = percentiles["p90"] * ratio_hi
    else:
        # Linear extrapolation
        # p0 = p10 - (p25 - p10) = 2*p10 - p25
        result["p0"] = 2 * percentiles["p10"] - percentiles["p25"]
        # p100 = p90 + (p90 - p75) = 2*p90 - p75
        result["p100"] = 2 * percentiles["p90"] - percentiles["p75"]
    return result


def compute_daily_mean_capability_age(
    fig15_intel: dict[dt.date, dict[str, float]],
    fig11_ratio: dict[dt.date, dict[str, float]],
    frontier_t: np.ndarray,
    frontier_intel: np.ndarray,
    config: MethodConfig | None = None,
) -> list[DailyPoint]:
    if config is None:
        config = METHOD_BASELINE
    frontier_intel = _cummax(frontier_intel)

    # Check if we need tail extrapolation
    needs_tails = any(lo in ("p0", "p100") or hi in ("p0", "p100") for lo, hi, _ in config.bands)

    out: list[DailyPoint] = []
    for d, iq_raw in sorted(fig15_intel.items()):
        rq_raw = fig11_ratio.get(d)
        if rq_raw is None:
            continue

        # Extrapolate tails if needed
        if needs_tails:
            iq = _extrapolate_tails(iq_raw, log_scale=False)  # Intelligence: linear
            rq = _extrapolate_tails(rq_raw, log_scale=True)   # Price ratio: log scale
        else:
            iq = iq_raw
            rq = rq_raw

        t = float(d.toordinal())

        # Price at each percentile boundary: p = intelligence * (price/intelligence).
        boundary_price = {k: iq[k] * rq[k] for k in iq.keys()}

        # Capability dates at boundary intelligence levels (inverting the frontier curve).
        pct_keys = sorted(iq.keys(), key=lambda x: int(x[1:]))  # p0, p10, p25, ...
        i_bounds = np.array([iq[k] for k in pct_keys], dtype=np.float64)
        t_capability = _invert_monotone_x_of_y(frontier_t, frontier_intel, i_bounds)
        t_capability = np.minimum(t_capability, t)
        boundary_age_days = t - t_capability
        boundary_age = {k: float(boundary_age_days[i]) for i, k in enumerate(pct_keys)}

        # Fit lognormal to prices if needed (prices are empirically lognormal)
        if config.assume_lognormal:
            # Fit lognormal to price distribution (work in log space)
            price_log_mu, price_log_sigma = _fit_normal_from_percentiles(boundary_price, log_space=True)

        weights: list[float] = []
        band_ages: list[float] = []
        for lo, hi, frac in config.bands:
            p_lo = max(boundary_price[lo], 1e-30)
            p_hi = max(boundary_price[hi], 1e-30)

            if config.assume_lognormal and lo in PERCENTILE_Z and hi in PERCENTILE_Z:
                # Use truncated lognormal expected value for price
                # E[X | a < X < b] where X ~ Lognormal: compute in log space then exp
                log_mean = _truncated_normal_mean(price_log_mu, price_log_sigma, PERCENTILE_Z[lo], PERCENTILE_Z[hi])
                avg_price = math.exp(log_mean)
            elif config.use_geometric_mean:
                avg_price = math.exp(0.5 * (math.log(p_lo) + math.log(p_hi)))
            else:
                avg_price = 0.5 * (p_lo + p_hi)
            weights.append(frac * avg_price)

            # Age: use simple midpoint (ages are derived from frontier, not distributional)
            band_age = 0.5 * (boundary_age[lo] + boundary_age[hi])
            band_ages.append(band_age)

        w = np.array(weights, dtype=np.float64)
        a = np.array(band_ages, dtype=np.float64)
        mean_age_days = float(np.sum(w * a) / np.sum(w))
        out.append(DailyPoint(date=d, mean_capability_age_days=mean_age_days))
    return out


def _monthly_mean(points: list[DailyPoint]) -> list[tuple[dt.date, float]]:
    buckets: dict[tuple[int, int], list[float]] = {}
    for p in points:
        key = (p.date.year, p.date.month)
        buckets.setdefault(key, []).append(p.mean_capability_age_days)
    out: list[tuple[dt.date, float]] = []
    for (y, m), vals in sorted(buckets.items()):
        out.append((dt.date(y, m, 1), float(np.mean(np.array(vals, dtype=np.float64)))))
    return out


def _plot_robustness(
    fig15: dict[dt.date, dict[str, float]],
    fig11: dict[dt.date, dict[str, float]],
    frontier_t: np.ndarray,
    frontier_i: np.ndarray,
    out_png: Path,
) -> None:
    """Generate robustness check plot with all method variants."""
    # Colors and styles for each variant
    styles = [
        {"color": "black", "linewidth": 2.0, "linestyle": "-"},
        {"color": "tab:blue", "linewidth": 1.2, "linestyle": "--"},
        {"color": "tab:orange", "linewidth": 1.2, "linestyle": "-."},
        {"color": "tab:green", "linewidth": 1.2, "linestyle": ":"},
        {"color": "tab:purple", "linewidth": 1.2, "linestyle": "-"},
    ]

    fig = plt.figure(figsize=(12, 6.5), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    for method, style in zip(ALL_METHODS, styles):
        points = compute_daily_mean_capability_age(fig15, fig11, frontier_t, frontier_i, config=method)
        if not points:
            continue
        monthly = _monthly_mean(points)
        mx = [d for d, _ in monthly]
        my = np.array([v for _, v in monthly], dtype=np.float64)
        ax.plot(mx, my, label=method.name, alpha=0.9, **style)

    ax.set_title("Mean Capability Age - Robustness Check")
    ax.set_ylabel("mean capability age [E[U]] (days)")
    ax.set_ylim(bottom=0)

    # Baseline for infinitely fast diffusion
    baseline_days = 365.25 / math.log(3)
    ax.axhline(baseline_days, color="red", linestyle="--", linewidth=1.0, alpha=0.5, label="1/g (instant diffusion)")

    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Plot $-weighted mean capability age over time using 4 intelligence bands"
            "(10-25, 25-50, 50-75, 75-90), with geometric-mean token price per band."
        )
    )
    ap.add_argument("--fig15", type=Path, default=Path("out/figure15_token_weighted_percentiles.csv"))
    ap.add_argument("--fig11", type=Path, default=Path("out/figure11_price_to_intelligence_ratio_percentiles.csv"))
    ap.add_argument("--frontier", type=Path, default=Path("out/figure6_black_frontier_timeseries.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("out"))
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--out_png", type=Path, default=None)
    ap.add_argument("--title", type=str, default="mean capability age [E[U]]")
    ap.add_argument("--robustness", action="store_true", help="Generate robustness check plot with all method variants")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_csv or (args.outdir / "dollar_weighted_mean_capability_age_timeseries.csv")
    out_png = args.out_png or (args.outdir / "dollar_weighted_mean_capability_age_timeseries.png")

    fig15 = _read_percentiles(args.fig15)
    fig11 = _read_percentiles(args.fig11)
    frontier_t, frontier_i = _read_frontier_series(args.frontier)

    if args.robustness:
        robustness_png = args.outdir / "capability_age_robustness.png"
        _plot_robustness(fig15, fig11, frontier_t, frontier_i, robustness_png)
        return 0

    points = compute_daily_mean_capability_age(fig15, fig11, frontier_t, frontier_i)
    if not points:
        raise SystemExit("No overlapping dates between Figure 15 and Figure 11 series.")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "mean_capability_age_days"])
        for p in points:
            w.writerow([p.date.isoformat(), f"{p.mean_capability_age_days:.6f}"])

    x = [p.date for p in points]
    y = np.array([p.mean_capability_age_days for p in points], dtype=np.float64)
    monthly = _monthly_mean(points)
    mx = [d for d, _ in monthly]
    my = np.array([v for _, v in monthly], dtype=np.float64)

    fig = plt.figure(figsize=(12, 6.5), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, color="tab:blue", linewidth=1.2, alpha=0.9, label="Daily")
    ax.plot(mx, my, color="black", linewidth=2.0, alpha=0.9, label="Monthly mean")
    ax.set_title(args.title)
    ax.set_ylabel("mean capability age [E[U]] (days)")
    ax.set_ylim(bottom=0)

    # Baseline for infinitely fast diffusion: E[U] = 1/g where g = ln(3) per year
    baseline_days = 365.25 / math.log(3)
    ax.axhline(baseline_days, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="1/g (instant diffusion)")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(loc="upper right", frameon=True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30, ha="right")

    p05 = float(np.percentile(y, 5))
    p50 = float(np.percentile(y, 50))
    p95 = float(np.percentile(y, 95))
    ax.text(
        0.01,
        0.02,
        f"p05={p05:.0f}d  median={p50:.0f}d  p95={p95:.0f}d",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        alpha=0.9,
    )
    # Key assumptions annotation
    ax.text(
        0.99,
        0.02,
        "Assumes g = ln(3)/yr (frontier 3×/yr)",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        alpha=0.7,
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
