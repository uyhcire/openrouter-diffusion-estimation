#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from extract_figure15_red_timeseries import (
    _calibrate_plot,
    _detect_gridlines,
    _extract_curve_y_by_x,
    _find_plot_rect,
    _long_gridline_mask,
    _read_bgr,
    _red_mask,
)


@dataclass(frozen=True)
class BandMetrics:
    iou: float
    miss_rate: float  # fraction of observed pixels not covered by prediction
    extra_rate: float  # fraction of predicted pixels not present in observed
    observed_n: int
    predicted_n: int


def _blue_mask(plot_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)
    # Roughly covers the blue frontier lines + labels.
    return ((sch > 60) & (vch > 80) & (hch >= 90) & (hch <= 140)).astype(np.uint8) * 255


def _kmeans_1d(values: np.ndarray, k: int, init: list[float], iters: int = 20) -> np.ndarray:
    v = values.astype(np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("kmeans input empty")
    centers = np.array(init[:k], dtype=np.float64)
    if centers.size != k:
        raise ValueError("init centers must have length k")
    for _ in range(iters):
        d = np.abs(v[:, None] - centers[None, :])
        lab = np.argmin(d, axis=1)
        for j in range(k):
            m = v[lab == j]
            if m.size:
                centers[j] = float(np.mean(m))
    return centers


def _observed_fill_masks(plot_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (observed_union, observed_dark) as boolean masks in plot pixel space.

    Uses low-saturation pixels and 3-cluster k-means on grayscale to separate:
    dark fill (p25–p75), light fill (p10–p90 only), and background (white/grid).
    """
    gray = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    h, w = gray.shape[:2]

    red = _red_mask(plot_bgr) > 0
    blue = _blue_mask(plot_bgr) > 0

    # Candidate neutral pixels (fill bands + gridlines + background whites).
    cand = (s < 40) & (v > 140)
    cand &= ~red
    cand &= ~blue

    # Remove legend and label regions (same rough masks used elsewhere).
    cand[int(0.60 * h) :, int(0.70 * w) :] = False
    cand[: int(0.18 * h), int(0.70 * w) :] = False
    cand[int(0.86 * h) :, int(0.55 * w) :] = False
    cand[:, : int(0.07 * w)] = False

    # Keep only mid-high grays; drop near-black text remnants.
    # (We intentionally do not subtract gridlines here; clustering will absorb them,
    # and they are thin enough not to dominate area-based metrics.)
    cand &= gray >= 175

    vals = gray[cand]
    if vals.size < 10_000:
        raise RuntimeError(f"Too few candidate pixels for clustering ({vals.size}).")

    # 3 clusters: [dark fill (~210), light fill (~237-ish, includes gridlines), background (~255)].
    centers = _kmeans_1d(vals, k=3, init=[210.0, 236.0, 252.0], iters=20)
    order = np.argsort(centers)
    c_dark = float(centers[order[0]])
    c_light = float(centers[order[1]])
    c_bg = float(centers[order[2]])

    # Assign each candidate pixel to nearest center.
    d_dark = np.abs(gray.astype(np.float64) - c_dark)
    d_light = np.abs(gray.astype(np.float64) - c_light)
    d_bg = np.abs(gray.astype(np.float64) - c_bg)
    lab = np.argmin(np.stack([d_dark, d_light, d_bg], axis=2), axis=2)

    observed_dark = cand & (lab == 0)
    observed_light = cand & (lab == 1)
    observed_union = observed_dark | observed_light
    return observed_union, observed_dark


def _polygon_mask(shape_hw: tuple[int, int], xs: np.ndarray, y_top: np.ndarray, y_bot: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    ok = np.isfinite(xs) & np.isfinite(y_top) & np.isfinite(y_bot)
    xs = xs[ok].astype(np.float64)
    y_top = y_top[ok].astype(np.float64)
    y_bot = y_bot[ok].astype(np.float64)
    if xs.size < 3:
        return np.zeros((h, w), dtype=np.uint8)
    xs = np.clip(xs, 0, w - 1)
    y_top = np.clip(y_top, 0, h - 1)
    y_bot = np.clip(y_bot, 0, h - 1)
    top = np.stack([xs, y_top], axis=1)
    bot = np.stack([xs[::-1], y_bot[::-1]], axis=1)
    poly = np.concatenate([top, bot], axis=0)
    poly_i = np.rint(poly).astype(np.int32).reshape((-1, 1, 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_i], color=255, lineType=cv2.LINE_AA)
    return mask


def _compute_metrics(observed: np.ndarray, predicted: np.ndarray) -> BandMetrics:
    obs = observed.astype(bool)
    pred = predicted.astype(bool)
    inter = int(np.logical_and(obs, pred).sum())
    union = int(np.logical_or(obs, pred).sum())
    obs_n = int(obs.sum())
    pred_n = int(pred.sum())
    miss = int(np.logical_and(obs, ~pred).sum())
    extra = int(np.logical_and(pred, ~obs).sum())
    return BandMetrics(
        iou=(inter / union) if union else 0.0,
        miss_rate=(miss / obs_n) if obs_n else 0.0,
        extra_rate=(extra / pred_n) if pred_n else 0.0,
        observed_n=obs_n,
        predicted_n=pred_n,
    )


def _per_column_rates(observed: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = observed.astype(bool)
    pred = predicted.astype(bool)
    h, w = obs.shape
    miss = np.zeros(w, dtype=np.float64)
    extra = np.zeros(w, dtype=np.float64)
    for x in range(w):
        o = obs[:, x]
        p = pred[:, x]
        o_n = int(o.sum())
        p_n = int(p.sum())
        miss[x] = (int(np.logical_and(o, ~p).sum()) / o_n) if o_n else np.nan
        extra[x] = (int(np.logical_and(p, ~o).sum()) / p_n) if p_n else np.nan
    return miss, extra


def _read_percentiles_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dates: list[dt.date] = []
    p10: list[float] = []
    p25: list[float] = []
    p50: list[float] = []
    p75: list[float] = []
    p90: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        if header[:1] != ["date"]:
            raise ValueError("unexpected CSV header")
        for line in f:
            line = line.strip()
            if not line:
                continue
            d, a, b, c, e, ff = line.split(",")
            dates.append(dt.date.fromisoformat(d))
            p10.append(float(a))
            p25.append(float(b))
            p50.append(float(c))
            p75.append(float(e))
            p90.append(float(ff))
    day_ord = np.array([d.toordinal() for d in dates], dtype=np.float64)
    return day_ord, np.array(p10), np.array(p25), np.array(p50), np.array(p75), np.array(p90)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify Figure 15 percentile extraction by pixel-level fill overlap metrics.")
    ap.add_argument("--pdf", type=Path, default=Path("w34608.pdf"))
    ap.add_argument("--outdir", type=Path, default=Path("out"))
    ap.add_argument("--percentiles", type=Path, default=Path("out/figure15_token_weighted_percentiles.csv"))
    ap.add_argument("--out", type=Path, default=Path("out/figure15_percentiles_verification.png"))
    ap.add_argument("--out-columns", type=Path, default=Path("out/figure15_percentiles_verification_columns.png"))
    args = ap.parse_args()

    # Load plot crop + calibrations (must match extractor).
    embedded_path = args.outdir / "figure15_embedded.png"
    if not embedded_path.exists():
        raise RuntimeError(f"Missing {embedded_path}; run extraction first.")
    bgr = _read_bgr(embedded_path)
    x0, y0, x1, y1 = _find_plot_rect(bgr)
    plot = bgr[y0 : y1 + 1, x0 : x1 + 1].copy()

    x_peaks, y_peaks = _detect_gridlines(plot)
    calib = _calibrate_plot(plot, bgr, (x0, y0, x1, y1), x_peaks, y_peaks, figure_number=15)

    # Observed fill masks (pixel space).
    obs_union, obs_dark = _observed_fill_masks(plot)

    # Predicted masks (from extracted percentiles).
    day_ord, p10, p25, p50, p75, p90 = _read_percentiles_csv(args.percentiles)
    xs = calib.x_from_days(day_ord)
    y10 = calib.y_from_val(p10)
    y25 = calib.y_from_val(p25)
    y75 = calib.y_from_val(p75)
    y90 = calib.y_from_val(p90)

    pred_union_u8 = _polygon_mask((plot.shape[0], plot.shape[1]), xs, y90, y10)
    pred_dark_u8 = _polygon_mask((plot.shape[0], plot.shape[1]), xs, y75, y25)
    pred_union = pred_union_u8 > 0
    pred_dark = pred_dark_u8 > 0

    # Exclude gridline pixels from scoring. The rasterized PDF can draw major vertical gridlines
    # as multi-pixel near-white stripes; those stripes are not part of the shaded bands but can
    # be clustered as "light fill" in the observed mask.
    h, w = plot.shape[:2]
    valid = np.ones((h, w), dtype=bool)
    # Mask legend/labels (same rough regions used in observed mask).
    valid[int(0.60 * h) :, int(0.70 * w) :] = False
    valid[: int(0.18 * h), int(0.70 * w) :] = False
    valid[int(0.86 * h) :, int(0.55 * w) :] = False
    valid[:, : int(0.07 * w)] = False

    grid = np.zeros((h, w), dtype=bool)
    for xx in x_peaks:
        x0g = max(0, int(xx) - 2)
        x1g = min(w, int(xx) + 3)
        grid[:, x0g:x1g] = True
    for yy in y_peaks:
        y0g = max(0, int(yy) - 1)
        y1g = min(h, int(yy) + 2)
        grid[y0g:y1g, :] = True
    lg = _long_gridline_mask(plot)
    lg = cv2.dilate(lg, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    grid |= (lg > 0)
    valid &= ~grid

    obs_union &= valid
    obs_dark &= valid
    pred_union &= valid
    pred_dark &= valid

    mu = _compute_metrics(obs_union, pred_union)
    md = _compute_metrics(obs_dark, pred_dark)

    # Error overlays
    miss_u = np.logical_and(obs_union, ~pred_union)
    extra_u = np.logical_and(pred_union, ~obs_union)
    miss_d = np.logical_and(obs_dark, ~pred_dark)
    extra_d = np.logical_and(pred_dark, ~obs_dark)

    def tint_overlay(base: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float = 0.65) -> np.ndarray:
        out = base.copy()
        c = np.array(color_bgr, dtype=np.float32).reshape((1, 1, 3))
        sel = mask.astype(bool)
        out_f = out.astype(np.float32)
        out_f[sel] = (1 - alpha) * out_f[sel] + alpha * c
        return np.clip(out_f, 0, 255).astype(np.uint8)

    # Pane 1: original
    pane_orig = plot
    # Pane 2: observed overlay (union=light blue tint, dark=orange tint)
    pane_obs = tint_overlay(plot, obs_union, (255, 255, 0), 0.40)
    pane_obs = tint_overlay(pane_obs, obs_dark, (0, 165, 255), 0.55)
    # Pane 3: predicted overlay
    pane_pred = tint_overlay(plot, pred_union, (255, 255, 0), 0.40)
    pane_pred = tint_overlay(pane_pred, pred_dark, (0, 165, 255), 0.55)
    # Pane 4: error overlay (miss=red, extra=blue); union dominates; dark uses stronger alpha.
    pane_err = plot.copy()
    pane_err = tint_overlay(pane_err, miss_u, (0, 0, 255), 0.65)  # missed observed = red
    pane_err = tint_overlay(pane_err, extra_u, (255, 0, 0), 0.65)  # extra predicted = blue
    pane_err = tint_overlay(pane_err, miss_d, (0, 0, 255), 0.85)
    pane_err = tint_overlay(pane_err, extra_d, (255, 0, 0), 0.85)

    fig = plt.figure(figsize=(14, 10), dpi=200)
    axs = fig.subplots(2, 2)

    def im(ax, bgr_img: np.ndarray, title: str) -> None:
        ax.imshow(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    im(axs[0, 0], pane_orig, "Original plot crop")
    im(
        axs[0, 1],
        pane_obs,
        "Observed fill masks (union=cyan tint, dark=orange tint)",
    )
    im(
        axs[1, 0],
        pane_pred,
        "Predicted fill masks (from extracted percentiles)",
    )
    im(
        axs[1, 1],
        pane_err,
        "Error map (missed observed=red, extra predicted=blue)",
    )

    fig.suptitle(
        "Figure 15 percentile extraction verification\n"
        f"UNION: IoU={mu.iou:.3f} miss={mu.miss_rate:.3f} extra={mu.extra_rate:.3f} | "
        f"DARK: IoU={md.iou:.3f} miss={md.miss_rate:.3f} extra={md.extra_rate:.3f}",
        y=0.99,
        fontsize=12,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.as_posix(), bbox_inches="tight")
    plt.close(fig)

    # Column diagnostics
    miss_u_col, extra_u_col = _per_column_rates(obs_union, pred_union)
    miss_d_col, extra_d_col = _per_column_rates(obs_dark, pred_dark)
    x = np.arange(plot.shape[1], dtype=np.float64)
    # map pixel-x to date for labeling
    x_date = np.array([dt.date.fromordinal(int(round(o))) for o in calib.days_from_x(x)], dtype=object)

    fig2 = plt.figure(figsize=(14, 6), dpi=200)
    ax1 = fig2.add_subplot(2, 1, 1)
    ax2 = fig2.add_subplot(2, 1, 2, sharex=ax1)
    ax1.plot(x_date, miss_u_col, color="red", linewidth=0.8, label="Union miss (observed not covered)")
    ax1.plot(x_date, extra_u_col, color="blue", linewidth=0.8, label="Union extra (predicted not observed)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_title("Per-column error rates (union band)")
    ax2.plot(x_date, miss_d_col, color="red", linewidth=0.8, label="Dark miss")
    ax2.plot(x_date, extra_d_col, color="blue", linewidth=0.8, label="Dark extra")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", frameon=False)
    ax2.set_title("Per-column error rates (dark band)")
    fig2.autofmt_xdate(rotation=30, ha="right")
    fig2.tight_layout()
    args.out_columns.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(args.out_columns.as_posix(), bbox_inches="tight")
    plt.close(fig2)

    print("Figure 15 percentile verification results")
    print(f"UNION  IoU={mu.iou:.4f}  miss={mu.miss_rate:.4f}  extra={mu.extra_rate:.4f}  obs={mu.observed_n}  pred={mu.predicted_n}")
    print(f"DARK   IoU={md.iou:.4f}  miss={md.miss_rate:.4f}  extra={md.extra_rate:.4f}  obs={md.observed_n}  pred={md.predicted_n}")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out_columns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
