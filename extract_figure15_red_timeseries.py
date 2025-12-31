#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


@dataclass(frozen=True)
class PlotCalibration:
    x_to_days_a: float
    x_to_days_b: float
    y_to_val_m: float
    y_to_val_c: float
    plot_x0: int
    plot_y0: int
    plot_x1: int
    plot_y1: int
    x_tick_dates: tuple[dt.date, ...]
    x_tick_format: str

    def days_from_x(self, x: np.ndarray) -> np.ndarray:
        return self.x_to_days_a * x + self.x_to_days_b

    def x_from_days(self, days: np.ndarray) -> np.ndarray:
        return (days - self.x_to_days_b) / self.x_to_days_a

    def val_from_y(self, y: np.ndarray) -> np.ndarray:
        return self.y_to_val_m * y + self.y_to_val_c

    def y_from_val(self, val: np.ndarray) -> np.ndarray:
        return (val - self.y_to_val_c) / self.y_to_val_m


def _select_evenly_spaced_subset(xs: list[int], k: int) -> list[int]:
    xs = sorted(xs)
    if len(xs) <= k:
        return xs
    best = None
    best_score = float("inf")
    for i in range(len(xs) - k + 1):
        cand = xs[i : i + k]
        diffs = np.diff(np.array(cand, dtype=np.float64))
        score = float(np.std(diffs) / (np.mean(diffs) + 1e-9))
        if score < best_score:
            best_score = score
            best = cand
    return best or xs[:k]


def _smooth_1d(values: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return values
    k = int(k)
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(values, kernel, mode="same")


def _find_prominent_local_maxima(values: np.ndarray, min_distance: int, keep: int) -> list[int]:
    if values.size < 3:
        return []
    v = values.astype(np.float64)
    v = _smooth_1d(v, 9)
    local = np.where((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]))[0] + 1
    if local.size == 0:
        local = np.argsort(v)[-keep:]
    local = local[np.argsort(v[local])[::-1]]  # by height desc
    chosen: list[int] = []
    for idx in local.tolist():
        if any(abs(idx - c) < min_distance for c in chosen):
            continue
        chosen.append(idx)
        if len(chosen) >= keep:
            break
    chosen.sort()
    return chosen


def _extract_embedded_figure15_image(pdf: Path, out: Path) -> Path:
    doc = fitz.open(pdf.as_posix())
    figure15_pages: list[int] = []
    for page_idx in range(doc.page_count):
        text = doc.load_page(page_idx).get_text("text")
        if "Figure 15:" in text or "FIGURE 15:" in text:
            figure15_pages.append(page_idx)
    if not figure15_pages:
        raise RuntimeError("Could not find 'Figure 15:' in PDF text.")
    page = doc.load_page(figure15_pages[0])
    imgs = page.get_images(full=True)
    if not imgs:
        raise RuntimeError("No images found on Figure 15 page.")
    # The plot is embedded as a single raster image.
    xref = imgs[0][0]
    base = doc.extract_image(xref)
    ext = base.get("ext", "png")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_path = out.with_suffix(f".{ext}")
    out_path.write_bytes(base["image"])
    return out_path


def _extract_embedded_figure_image(pdf: Path, figure_number: int, out: Path) -> Path:
    doc = fitz.open(pdf.as_posix())
    key = f"Figure {figure_number}:"
    pages: list[int] = []
    for page_idx in range(doc.page_count):
        text = doc.load_page(page_idx).get_text("text")
        if key in text or key.upper() in text:
            pages.append(page_idx)
    if not pages:
        raise RuntimeError(f"Could not find '{key}' in PDF text.")
    page = doc.load_page(pages[0])
    imgs = page.get_images(full=True)
    if not imgs:
        raise RuntimeError(f"No images found on Figure {figure_number} page.")
    xref = imgs[0][0]
    base = doc.extract_image(xref)
    ext = base.get("ext", "png")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_path = out.with_suffix(f".{ext}")
    out_path.write_bytes(base["image"])
    return out_path


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _find_plot_rect(bgr: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = gray < 90
    h, w = gray.shape[:2]

    row_counts = dark.sum(axis=1)
    col_counts = dark.sum(axis=0)

    top_region = np.arange(0, int(0.35 * h))
    bottom_region = np.arange(int(0.5 * h), h)
    top_y = int(top_region[np.argmax(row_counts[top_region])])
    bottom_y = int(bottom_region[np.argmax(row_counts[bottom_region])])

    y0 = max(0, top_y)
    y1 = min(h - 1, bottom_y)
    if y1 <= y0 + int(0.4 * h):
        raise RuntimeError("Failed to detect plot vertical bounds.")

    # Find left/right borders within the plot vertical span.
    col_counts_span = dark[y0:y1, :].sum(axis=0)
    left_region = np.arange(0, int(0.25 * w))
    right_region = np.arange(int(0.75 * w), w)
    left_x = int(left_region[np.argmax(col_counts_span[left_region])])
    right_x = int(right_region[np.argmax(col_counts_span[right_region])])

    x0 = max(0, left_x)
    x1 = min(w - 1, right_x)
    if x1 <= x0 + int(0.5 * w):
        raise RuntimeError("Failed to detect plot horizontal bounds.")

    # Shrink slightly inside the border.
    pad = 2
    return x0 + pad, y0 + pad, x1 - pad, y1 - pad


def _detect_x_tick_label_centers(embedded_bgr: np.ndarray, plot_rect: tuple[int, int, int, int]) -> list[int]:
    x0, _y0, x1, y1 = plot_rect
    strip = embedded_bgr[y1 + 1 : min(embedded_bgr.shape[0], y1 + 90), x0 : x1 + 1].copy()
    if strip.size == 0:
        return []
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    bw = (gray < 120).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (12, 4)), iterations=1)

    contours, _hier = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers: list[int] = []
    for c in contours:
        xx, yy, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < 150:
            continue
        if not (10 <= hh <= 35 and 25 <= ww <= 130):
            continue
        centers.append(int(xx + ww // 2))
    centers.sort()

    clusters: list[list[int]] = []
    for cx in centers:
        if not clusters or abs(cx - clusters[-1][-1]) > 20:
            clusters.append([cx])
        else:
            clusters[-1].append(cx)
    return [int(round(float(np.median(c)))) for c in clusters]


def _detect_gridlines(plot_bgr: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = plot_bgr.shape[:2]

    # Gridlines are very light gray (low saturation, high value) compared to the
    # plot elements and shaded percentile bands; isolate them by color.
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    grid = ((s < 25) & (v > 200) & (v < 250)).astype(np.uint8) * 255

    # Extract long horizontal lines.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(w * 0.25)), 1))
    h_lines = cv2.erode(grid, h_kernel, iterations=1)
    h_lines = cv2.dilate(h_lines, h_kernel, iterations=1)
    row_counts = (h_lines > 0).sum(axis=1)
    y_cand = np.where(row_counts > int(0.75 * w))[0]

    def cluster_centers(idxs: np.ndarray) -> list[int]:
        if idxs.size == 0:
            return []
        clusters: list[tuple[int, int]] = []
        start = int(idxs[0])
        prev = int(idxs[0])
        for val in idxs[1:].tolist():
            val = int(val)
            if val == prev + 1:
                prev = val
            else:
                clusters.append((start, prev))
                start = prev = val
        clusters.append((start, prev))
        return [int((a + b) // 2) for a, b in clusters]

    y_centers = cluster_centers(y_cand)

    # Extract long vertical lines.
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(h * 0.20))))
    v_lines = cv2.erode(grid, v_kernel, iterations=1)
    v_lines = cv2.dilate(v_lines, v_kernel, iterations=1)
    col_counts = (v_lines > 0).sum(axis=0)
    x_cand = np.where(col_counts > int(0.55 * h))[0]
    x_centers = cluster_centers(x_cand)

    return sorted(x_centers), sorted(y_centers)


def _red_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 90, 70], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 90, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _extract_curve_y_by_x(plot_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = _red_mask(plot_bgr)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        raise RuntimeError("No red components found; cannot extract curve.")

    h, w = mask.shape[:2]
    best_label = None
    best_score = -1.0
    for label in range(1, num_labels):
        x, y, ww, hh, area = stats[label]
        if ww < 0.55 * w:
            continue
        score = ww * 10 + area
        if score > best_score:
            best_score = score
            best_label = label
    if best_label is None:
        best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    curve = (labels == best_label).astype(np.uint8)

    y_by_x = np.full(w, np.nan, dtype=np.float64)
    for x in range(w):
        ys = np.where(curve[:, x] > 0)[0]
        if ys.size:
            y_by_x[x] = float(np.median(ys))
    xs = np.arange(w, dtype=np.float64)
    good = np.isfinite(y_by_x)
    if good.sum() < 50:
        raise RuntimeError("Too few red pixels along x to form a curve.")
    y_interp = np.interp(xs, xs[good], y_by_x[good])
    return xs, y_interp


def _draw_polyline_mask(shape_hw: tuple[int, int], xs: np.ndarray, ys: np.ndarray, thickness: int) -> np.ndarray:
    h, w = shape_hw
    pts = np.stack([xs, ys], axis=1)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 2:
        return np.zeros((h, w), dtype=np.uint8)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts_i = np.rint(pts).astype(np.int32).reshape((-1, 1, 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(mask, [pts_i], isClosed=False, color=255, thickness=int(thickness), lineType=cv2.LINE_AA)
    return mask


def _overlap_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    obs = observed.astype(bool)
    pred = predicted.astype(bool)
    inter = int(np.logical_and(obs, pred).sum())
    union = int(np.logical_or(obs, pred).sum())
    pred_n = int(pred.sum())
    obs_n = int(obs.sum())
    iou = (inter / union) if union else 0.0
    precision = (inter / pred_n) if pred_n else 0.0
    recall = (inter / obs_n) if obs_n else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "intersection_px": float(inter),
        "union_px": float(union),
        "observed_px": float(obs_n),
        "predicted_px": float(pred_n),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _calibrate_plot(
    plot_bgr: np.ndarray,
    embedded_bgr: np.ndarray,
    plot_rect: tuple[int, int, int, int],
    x_peaks: list[int],
    y_peaks: list[int],
    figure_number: int,
) -> PlotCalibration:
    h, w = plot_bgr.shape[:2]

    x_label_centers = _detect_x_tick_label_centers(embedded_bgr, plot_rect)
    x_label_centers = [x for x in x_label_centers if int(0.02 * w) < x < int(0.98 * w)]

    if len(x_label_centers) >= 10:
        # Figure 5 style: labels like 01/23, 04/23, ... => MM/YY.
        x_label_centers = _select_evenly_spaced_subset(x_label_centers, 12)
        x_tick_dates = (
            dt.date(2023, 1, 1),
            dt.date(2023, 4, 1),
            dt.date(2023, 7, 1),
            dt.date(2023, 10, 1),
            dt.date(2024, 1, 1),
            dt.date(2024, 4, 1),
            dt.date(2024, 7, 1),
            dt.date(2024, 10, 1),
            dt.date(2025, 1, 1),
            dt.date(2025, 4, 1),
            dt.date(2025, 7, 1),
            dt.date(2025, 10, 1),
        )
        if len(x_label_centers) != len(x_tick_dates):
            raise RuntimeError(f"Expected 12 x tick labels for MM/YY calibration; got {len(x_label_centers)}.")
        anchor_days = np.array([d.toordinal() for d in x_tick_dates], dtype=np.float64)
        xs = np.array(x_label_centers, dtype=np.float64)
        x_to_days_a, x_to_days_b = np.polyfit(xs, anchor_days, 1)
        x_tick_format = "%m/%y"
    else:
        # Figure 15 style: 5 vertical gridlines and interpret labels as MM/DD (.../25).
        if len(x_peaks) < 5:
            raise RuntimeError(f"Expected at least 5 vertical gridlines; got {len(x_peaks)}.")
        x_interior = [x for x in x_peaks if int(0.05 * w) < x < int(0.95 * w)]
        if len(x_interior) < 5:
            x_interior = x_peaks
        x_peaks_5 = _select_evenly_spaced_subset(x_interior, 5)
        year = 2025
        months = [3, 5, 7, 9, 11]
        x_tick_dates = tuple(dt.date(year, m, 25) for m in months)
        anchor_days = np.array([d.toordinal() for d in x_tick_dates], dtype=np.float64)
        xs = np.array(x_peaks_5, dtype=np.float64)
        x_to_days_a, x_to_days_b = np.polyfit(xs, anchor_days, 1)
        x_tick_format = "%m/%d"

    # Y calibration: use bottom border as 0.0 and 0.2/0.4/0.6 horizontal gridlines.
    if len(y_peaks) < 3:
        raise RuntimeError(f"Expected at least 3 horizontal gridlines; got {len(y_peaks)}.")

    y_peaks = sorted(y_peaks)
    y_candidates = [yy for yy in y_peaks if int(0.10 * h) < yy < int(0.92 * h)]
    if len(y_candidates) < 3:
        y_candidates = y_peaks

    # Choose 3 interior gridlines with near-equal spacing (0.6/0.4/0.2).
    best_triplet: tuple[int, int, int] | None = None
    best_score = float("inf")
    y_arr = np.array(sorted(set(y_candidates)), dtype=int)
    for i in range(len(y_arr) - 2):
        for j in range(i + 1, len(y_arr) - 1):
            for k in range(j + 1, len(y_arr)):
                y1, y2, y3 = int(y_arr[i]), int(y_arr[j]), int(y_arr[k])
                diffs = np.array([y2 - y1, y3 - y2], dtype=np.float64)
                if np.any(diffs <= 0):
                    continue
                score = float(np.std(diffs) / (np.mean(diffs) + 1e-9))
                # Prefer triplets spanning a reasonable portion of the plot.
                span = y3 - y1
                score += float(abs(span - 0.5 * h) / h) * 0.25
                if score < best_score:
                    best_score = score
                    best_triplet = (y1, y2, y3)

    if best_triplet is None:
        raise RuntimeError("Failed to select y gridlines for calibration.")

    y_points = np.array([best_triplet[0], best_triplet[1], best_triplet[2], h - 1], dtype=np.float64)
    y_values = np.array([0.6, 0.4, 0.2, 0.0], dtype=np.float64)
    y_to_val_m, y_to_val_c = np.polyfit(y_points, y_values, 1)

    return PlotCalibration(
        x_to_days_a=float(x_to_days_a),
        x_to_days_b=float(x_to_days_b),
        y_to_val_m=float(y_to_val_m),
        y_to_val_c=float(y_to_val_c),
        plot_x0=0,
        plot_y0=0,
        plot_x1=w - 1,
        plot_y1=h - 1,
        x_tick_dates=tuple(x_tick_dates),
        x_tick_format=x_tick_format,
    )


def _resample_daily(x_days: np.ndarray, y_vals: np.ndarray) -> tuple[list[dt.date], np.ndarray]:
    days = np.rint(x_days).astype(int)
    sel = np.isfinite(y_vals)
    days = days[sel]
    y_vals = y_vals[sel]
    if days.size == 0:
        return [], np.array([], dtype=np.float64)
    start = int(np.min(days))
    end = int(np.max(days))
    uniq = np.arange(start, end + 1)
    daily = np.full(uniq.size, np.nan, dtype=np.float64)
    for i, d in enumerate(uniq.tolist()):
        mask = days == d
        if np.any(mask):
            daily[i] = float(np.median(y_vals[mask]))
    good = np.isfinite(daily)
    if good.sum() >= 2:
        x = np.arange(daily.size)
        daily = np.interp(x, x[good], daily[good])
    dates = [dt.date.fromordinal(int(d)) for d in uniq.tolist()]
    return dates, daily


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract a figure's red median curve time series from the PDF (embedded plot image).")
    ap.add_argument("--pdf", type=Path, default=Path("w34608.pdf"))
    ap.add_argument("--outdir", type=Path, default=Path("out"))
    ap.add_argument("--figure", type=int, default=15, help="Figure number to extract (default: 15).")
    ap.add_argument("--diagnostic", action="store_true", help="Write overlay images and overlap metrics.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    embedded_path = _extract_embedded_figure_image(args.pdf, args.figure, args.outdir / f"figure{args.figure}_embedded.png")
    bgr = _read_bgr(embedded_path)

    x0, y0, x1, y1 = _find_plot_rect(bgr)
    plot = bgr[y0 : y1 + 1, x0 : x1 + 1].copy()

    x_peaks, y_peaks = _detect_gridlines(plot)
    calib = _calibrate_plot(plot, bgr, (x0, y0, x1, y1), x_peaks, y_peaks, figure_number=args.figure)

    xs, ys = _extract_curve_y_by_x(plot)
    days = calib.days_from_x(xs)
    vals = calib.val_from_y(ys)

    dates, daily_vals = _resample_daily(days, vals)

    csv_path = args.outdir / f"figure{args.figure}_red_median_timeseries.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("date,intelligence_index\n")
        for d, v in zip(dates, daily_vals, strict=True):
            f.write(f"{d.isoformat()},{v:.6f}\n")

    # Debug overlay
    overlay = plot.copy()
    for xx in x_peaks:
        cv2.line(overlay, (int(xx), 0), (int(xx), overlay.shape[0] - 1), (255, 0, 0), 1)
    for yy in y_peaks:
        cv2.line(overlay, (0, int(yy)), (overlay.shape[1] - 1, int(yy)), (0, 255, 255), 1)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1), (0, 255, 0), 1)
    overlay_path = args.outdir / f"figure{args.figure}_debug_overlay.png"
    cv2.imwrite(overlay_path.as_posix(), overlay)

    if args.diagnostic:
        # Observed vs predicted overlay in plot-pixel space.
        observed_mask = _red_mask(plot)

        date_ord = np.array([d.toordinal() for d in dates], dtype=np.float64)
        x_pred = calib.x_from_days(date_ord)
        y_pred = calib.y_from_val(daily_vals.astype(np.float64))
        predicted_mask = _draw_polyline_mask((plot.shape[0], plot.shape[1]), x_pred, y_pred, thickness=3)

        # Allow small tolerance for anti-aliasing / digitization error.
        tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        observed_tol = cv2.dilate(observed_mask, tol, iterations=1)
        predicted_tol = cv2.dilate(predicted_mask, tol, iterations=1)

        metrics = _overlap_metrics(observed_tol, predicted_tol)
        metrics_path = args.outdir / f"figure{args.figure}_overlap_metrics.txt"
        with metrics_path.open("w", encoding="utf-8") as f:
            for k, v in metrics.items():
                if k.endswith("_px"):
                    f.write(f"{k}={int(v)}\n")
                else:
                    f.write(f"{k}={v:.6f}\n")

        overlay_img = plot.copy()
        # Observed in red, predicted in cyan, overlap in yellow.
        obs = (observed_tol > 0)
        pred = (predicted_tol > 0)
        both = obs & pred
        overlay_img[obs] = (0, 0, 255)
        overlay_img[pred] = (255, 255, 0)
        overlay_img[both] = (0, 255, 255)
        curve_overlay_path = args.outdir / f"figure{args.figure}_curve_overlay.png"
        cv2.imwrite(curve_overlay_path.as_posix(), overlay_img)

    # Plot
    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dates, daily_vals, color="red", linewidth=1.5)
    ax.set_title(f"Figure {args.figure} extracted red median curve")
    ax.set_ylabel("Intelligence Index")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(list(calib.x_tick_dates))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(calib.x_tick_format))
    if dates:
        ax.set_xlim(dates[0], dates[-1])
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    plot_path = args.outdir / f"figure{args.figure}_red_median_timeseries.png"
    fig.savefig(plot_path.as_posix())
    plt.close(fig)

    meta_path = args.outdir / f"figure{args.figure}_extraction_meta.txt"
    with meta_path.open("w", encoding="utf-8") as f:
        f.write(f"embedded_image: {embedded_path.name}\n")
        f.write(f"plot_rect_in_embedded_pixels: x0={x0} y0={y0} x1={x1} y1={y1}\n")
        f.write(f"plot_size_pixels: w={plot.shape[1]} h={plot.shape[0]}\n")
        f.write(f"x_gridlines_px: {x_peaks}\n")
        f.write(f"y_gridlines_px: {y_peaks}\n")
        f.write(f"x_tick_dates: {[d.isoformat() for d in calib.x_tick_dates]}\n")
        f.write(f"x_tick_format: {calib.x_tick_format}\n")
        f.write(f"x_to_days: days = {calib.x_to_days_a:.8f} * x + {calib.x_to_days_b:.8f}\n")
        f.write(f"y_to_val: val = {calib.y_to_val_m:.10f} * y + {calib.y_to_val_c:.10f}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {overlay_path}")
    if args.diagnostic:
        print(f"Wrote {args.outdir / f'figure{args.figure}_curve_overlay.png'}")
        print(f"Wrote {args.outdir / f'figure{args.figure}_overlap_metrics.txt'}")
    if daily_vals.size:
        print(f"Sanity: min={float(np.min(daily_vals)):.3f} max={float(np.max(daily_vals)):.3f} start={float(daily_vals[0]):.3f} end={float(daily_vals[-1]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
