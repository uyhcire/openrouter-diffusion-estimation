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
    _find_plot_rect,
    _long_gridline_mask,
    _read_bgr,
    _red_mask,
)


@dataclass(frozen=True)
class BandMetrics:
    iou: float
    miss_rate: float
    extra_rate: float
    observed_n: int
    predicted_n: int


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
    _h, w = obs.shape
    miss = np.full(w, np.nan, dtype=np.float64)
    extra = np.full(w, np.nan, dtype=np.float64)
    for x in range(w):
        o = obs[:, x]
        p = pred[:, x]
        o_n = int(o.sum())
        p_n = int(p.sum())
        miss[x] = (int(np.logical_and(o, ~p).sum()) / o_n) if o_n else np.nan
        extra[x] = (int(np.logical_and(p, ~o).sum()) / p_n) if p_n else np.nan
    return miss, extra


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


def _observed_fill_masks(plot_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Heuristic "observed" fill masks for Figure 11:
    - union mask = both fills (p10–p90 + p25–p75)
    - dark mask = p25–p75

    Uses low-saturation pixels and 3-cluster k-means on grayscale to separate:
    dark fill, light fill, and background.
    """
    gray = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    h, w = gray.shape[:2]

    red = _red_mask(plot_bgr) > 0
    cand = (s < 40) & (v > 120) & ~red

    # Mask legend (bottom-left for Figure 11).
    cand[int(0.60 * h) :, : int(0.35 * w)] = False
    # Mask axis label area.
    cand[:, : int(0.07 * w)] = False

    # Remove gridlines (otherwise they get clustered as "light fill" and depress IoU).
    x_peaks, y_peaks = _detect_gridlines(plot_bgr)
    for xx in x_peaks:
        x0 = max(0, int(xx) - 1)
        x1 = min(w, int(xx) + 2)
        cand[:, x0:x1] = False
    for yy in y_peaks:
        y0 = max(0, int(yy) - 0)
        y1 = min(h, int(yy) + 1)
        cand[y0:y1, :] = False

    # Also remove long gridlines directly (Figure 11's vertical gridlines can be missed by
    # `_detect_gridlines`, and show up as low-contrast stripes). Dilate slightly so anti-aliased
    # edges don't contaminate the observed fill mask.
    lg = _long_gridline_mask(plot_bgr)
    lg = cv2.dilate(lg, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    cand &= ~(lg > 0)

    vals = gray[cand].astype(np.float64)
    if vals.size < 20_000:
        raise RuntimeError(f"Too few candidate pixels for clustering ({vals.size}).")
    if vals.size > 250_000:
        vals = vals[:: int(vals.size // 250_000) + 1]

    # 3 centers roughly: dark fill (~170-210), light fill (~225-245), background (~255).
    centers = np.array([200.0, 235.0, 252.0], dtype=np.float64)
    for _ in range(18):
        d = np.abs(vals[:, None] - centers[None, :])
        lab = np.argmin(d, axis=1)
        for j in range(3):
            m = vals[lab == j]
            if m.size:
                centers[j] = float(np.mean(m))
    order = np.argsort(centers)
    c_dark, c_light, c_bg = [float(centers[i]) for i in order.tolist()]

    g = gray.astype(np.float64)
    d0 = np.abs(g - c_dark)
    d1 = np.abs(g - c_light)
    d2 = np.abs(g - c_bg)
    pix_lab = np.argmin(np.stack([d0, d1, d2], axis=2), axis=2)
    observed_dark = cand & (pix_lab == 0)
    observed_light = cand & (pix_lab == 1)
    observed_union = observed_dark | observed_light
    return observed_union, observed_dark


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify Figure 11 percentile extraction by pixel-level fill overlap.")
    ap.add_argument("--pdf", type=Path, default=Path("w34608.pdf"))
    ap.add_argument("--outdir", type=Path, default=Path("out"))
    ap.add_argument("--percentiles", type=Path, default=Path("out/figure11_price_to_intelligence_ratio_percentiles.csv"))
    ap.add_argument("--out", type=Path, default=Path("out/figure11_percentiles_verification.png"))
    ap.add_argument("--out-columns", type=Path, default=Path("out/figure11_percentiles_verification_columns.png"))
    args = ap.parse_args()

    embedded = args.outdir / "figure11_embedded.png"
    if not embedded.exists():
        raise RuntimeError(f"Missing {embedded}; run extraction first.")
    bgr = _read_bgr(embedded)
    x0, y0, x1, y1 = _find_plot_rect(bgr)
    plot = bgr[y0 : y1 + 1, x0 : x1 + 1].copy()

    x_peaks, y_peaks = _detect_gridlines(plot)
    calib = _calibrate_plot(plot, bgr, (x0, y0, x1, y1), x_peaks, y_peaks, figure_number=11)

    obs_union, obs_dark = _observed_fill_masks(plot)
    day_ord, p10, p25, _p50, p75, p90 = _read_percentiles_csv(args.percentiles)
    xs = calib.x_from_days(day_ord)
    y10 = calib.y_from_val(p10)
    y25 = calib.y_from_val(p25)
    y75 = calib.y_from_val(p75)
    y90 = calib.y_from_val(p90)

    pred_union = _polygon_mask((plot.shape[0], plot.shape[1]), xs, y90, y10) > 0
    pred_dark = _polygon_mask((plot.shape[0], plot.shape[1]), xs, y75, y25) > 0

    # Evaluate only in the valid plotting region (exclude legend/axis-label areas) so the score
    # reflects band reconstruction quality rather than “does the polygon cover the legend”.
    h, w = plot.shape[:2]
    valid = np.ones((h, w), dtype=bool)
    valid[int(0.60 * h) :, : int(0.35 * w)] = False  # legend
    valid[:, : int(0.07 * w)] = False  # y-axis label region
    # Don't score gridline pixels; they are rendered on top of the fill and can appear as
    # low-contrast stripes in Figure 11. Use both detected peaks and a dilated long-gridline mask.
    grid = np.zeros((h, w), dtype=bool)
    for xx in x_peaks:
        x0 = max(0, int(xx) - 2)
        x1 = min(w, int(xx) + 3)
        grid[:, x0:x1] = True
    for yy in y_peaks:
        y0 = max(0, int(yy) - 1)
        y1 = min(h, int(yy) + 2)
        grid[y0:y1, :] = True
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

    # Per-column error rates (helps localize "bites"/notches).
    miss_u_x, extra_u_x = _per_column_rates(obs_union, pred_union)
    miss_d_x, extra_d_x = _per_column_rates(obs_dark, pred_dark)
    x_cols = np.arange(plot.shape[1], dtype=np.float64)
    day_ord_cols = calib.days_from_x(x_cols)
    day_cols = [dt.date.fromordinal(int(round(d))) for d in day_ord_cols.tolist()]

    # Render 2x2 diagnostic image like Figure 15 verification.
    def tint_overlay(base: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
        out = base.copy()
        c = np.array(color_bgr, dtype=np.float32).reshape((1, 1, 3))
        sel = mask.astype(bool)
        out_f = out.astype(np.float32)
        out_f[sel] = (1 - alpha) * out_f[sel] + alpha * c
        return np.clip(out_f, 0, 255).astype(np.uint8)

    pane_orig = plot
    pane_obs = tint_overlay(plot, obs_union, (255, 255, 0), 0.35)
    pane_obs = tint_overlay(pane_obs, obs_dark, (0, 165, 255), 0.55)
    pane_pred = tint_overlay(plot, pred_union, (255, 255, 0), 0.35)
    pane_pred = tint_overlay(pane_pred, pred_dark, (0, 165, 255), 0.55)
    miss_u = np.logical_and(obs_union, ~pred_union)
    extra_u = np.logical_and(pred_union, ~obs_union)
    miss_d = np.logical_and(obs_dark, ~pred_dark)
    extra_d = np.logical_and(pred_dark, ~obs_dark)
    pane_err = plot.copy()
    pane_err = tint_overlay(pane_err, miss_u, (0, 0, 255), 0.65)
    pane_err = tint_overlay(pane_err, extra_u, (255, 0, 0), 0.65)
    pane_err = tint_overlay(pane_err, miss_d, (0, 0, 255), 0.85)
    pane_err = tint_overlay(pane_err, extra_d, (255, 0, 0), 0.85)

    def im(ax, bgr_img: np.ndarray, title: str) -> None:
        ax.imshow(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    fig = plt.figure(figsize=(14, 10), dpi=200)
    axs = fig.subplots(2, 2)
    im(axs[0, 0], pane_orig, "Original plot crop")
    im(axs[0, 1], pane_obs, "Observed fill masks (union=cyan tint, dark=orange tint)")
    im(axs[1, 0], pane_pred, "Predicted fill masks (from extracted percentiles)")
    im(axs[1, 1], pane_err, "Error map (missed observed=red, extra predicted=blue)")
    fig.suptitle(
        "Figure 11 percentile extraction verification\n"
        f"UNION: IoU={mu.iou:.3f} miss={mu.miss_rate:.3f} extra={mu.extra_rate:.3f} | "
        f"DARK: IoU={md.iou:.3f} miss={md.miss_rate:.3f} extra={md.extra_rate:.3f}",
        y=0.99,
        fontsize=12,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.as_posix(), bbox_inches="tight")
    plt.close(fig)

    # Per-column plots.
    fig2 = plt.figure(figsize=(14, 7), dpi=200)
    axs2 = fig2.subplots(2, 1, sharex=True)
    axs2[0].plot(day_cols, miss_u_x, color="red", lw=1.0, label="Union miss (observed not covered)")
    axs2[0].plot(day_cols, extra_u_x, color="blue", lw=1.0, label="Union extra (predicted not observed)")
    axs2[0].set_ylim(-0.02, 1.02)
    axs2[0].set_title("Per-column error rates (union band)")
    axs2[0].grid(True, alpha=0.25)
    axs2[0].legend(loc="upper right")

    axs2[1].plot(day_cols, miss_d_x, color="red", lw=1.0, label="Dark miss")
    axs2[1].plot(day_cols, extra_d_x, color="blue", lw=1.0, label="Dark extra")
    axs2[1].set_ylim(-0.02, 1.02)
    axs2[1].set_title("Per-column error rates (dark band)")
    axs2[1].grid(True, alpha=0.25)
    axs2[1].legend(loc="upper right")
    fig2.tight_layout()
    args.out_columns.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(args.out_columns.as_posix(), bbox_inches="tight")
    plt.close(fig2)

    print("Figure 11 percentile verification results")
    print(f"UNION  IoU={mu.iou:.4f}  miss={mu.miss_rate:.4f}  extra={mu.extra_rate:.4f}  obs={mu.observed_n}  pred={mu.predicted_n}")
    print(f"DARK   IoU={md.iou:.4f}  miss={md.miss_rate:.4f}  extra={md.extra_rate:.4f}  obs={md.observed_n}  pred={md.predicted_n}")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out_columns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
