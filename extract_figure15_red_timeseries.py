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
    y_scale: str = "linear"  # "linear" or "log10"

    def days_from_x(self, x: np.ndarray) -> np.ndarray:
        return self.x_to_days_a * x + self.x_to_days_b

    def x_from_days(self, days: np.ndarray) -> np.ndarray:
        return (days - self.x_to_days_b) / self.x_to_days_a

    def val_from_y(self, y: np.ndarray) -> np.ndarray:
        if self.y_scale == "log10":
            return np.power(10.0, self.y_to_val_m * y + self.y_to_val_c)
        return self.y_to_val_m * y + self.y_to_val_c

    def y_from_val(self, val: np.ndarray) -> np.ndarray:
        if self.y_scale == "log10":
            v = np.clip(val.astype(np.float64), 1e-12, np.inf)
            return (np.log10(v) - self.y_to_val_c) / self.y_to_val_m
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


def _detect_y_tick_centers_from_left_strip(embedded_bgr: np.ndarray, plot_rect: tuple[int, int, int, int]) -> list[int]:
    """
    Detect y-axis tick label y-centers by scanning the strip immediately left of the plot.

    Figure 11's horizontal gridlines are very faint; this provides a more stable fallback for
    log-scale calibration by using the rendered tick labels (100/10/1/0.1).

    Returns y positions in *plot* pixel coordinates (same y-origin as the cropped plot).
    """
    x0, y0, _x1, y1 = plot_rect
    strip_x0 = max(0, x0 - 160)
    strip = embedded_bgr[y0 : y1 + 1, strip_x0:x0].copy()
    if strip.size == 0:
        return []

    h, w = strip.shape[:2]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    bw = (gray < 130).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    glyphs: list[tuple[float, float]] = []  # (cy, area)
    for lab in range(1, num):
        xx, yy, ww, hh, area = [int(v) for v in stats[lab].tolist()]
        if area < 35:
            continue
        # Tick labels sit close to the plot (right side of this strip). The axis title is
        # rotated and lives further left; bias to the right to avoid it.
        if xx < int(0.35 * w):
            continue
        if not (10 <= hh <= 30 and 6 <= ww <= 32):
            continue
        cy = yy + hh / 2.0
        glyphs.append((float(cy), float(area)))

    if not glyphs:
        return []

    glyphs.sort(key=lambda t: t[0])
    clusters: list[list[tuple[float, float]]] = []
    for cy, area in glyphs:
        if not clusters or abs(cy - clusters[-1][-1][0]) > 18:
            clusters.append([(cy, area)])
        else:
            clusters[-1].append((cy, area))

    scored: list[tuple[float, float]] = []  # (total_area, cy_weighted)
    for cl in clusters:
        total = float(sum(a for _cy, a in cl))
        cy_w = float(sum(_cy * a for _cy, a in cl) / max(1e-9, total))
        scored.append((total, cy_w))
    scored.sort(reverse=True)

    chosen: list[float] = []
    for _score, cy in scored:
        if all(abs(cy - other) > 40 for other in chosen):
            chosen.append(float(cy))
        if len(chosen) >= 4:
            break
    chosen.sort()
    return [int(round(cy)) for cy in chosen]


def _detect_gridlines(plot_bgr: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = plot_bgr.shape[:2]

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

    # Retry with progressively relaxed thresholds (needed for some figures like Figure 11).
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    attempts = [
        # (s_max, v_min, v_max, row_frac, col_frac)
        (25, 200, 250, 0.75, 0.55),
        (30, 190, 255, 0.65, 0.50),
        (35, 180, 255, 0.60, 0.45),
    ]
    best_x: list[int] = []
    best_y: list[int] = []
    for s_max, v_min, v_max, row_frac, col_frac in attempts:
        grid = ((s < s_max) & (v >= v_min) & (v <= v_max)).astype(np.uint8) * 255

        # Extract long horizontal lines.
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(w * 0.25)), 1))
        h_lines = cv2.erode(grid, h_kernel, iterations=1)
        h_lines = cv2.dilate(h_lines, h_kernel, iterations=1)
        row_counts = (h_lines > 0).sum(axis=1)
        y_cand = np.where(row_counts > int(row_frac * w))[0]
        y_centers = cluster_centers(y_cand)

        # Extract long vertical lines.
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(h * 0.20))))
        v_lines = cv2.erode(grid, v_kernel, iterations=1)
        v_lines = cv2.dilate(v_lines, v_kernel, iterations=1)
        col_counts = (v_lines > 0).sum(axis=0)
        x_cand = np.where(col_counts > int(col_frac * h))[0]
        x_centers = cluster_centers(x_cand)

        # Keep best attempt by total found (prefer more y lines).
        if (len(y_centers), len(x_centers)) > (len(best_y), len(best_x)):
            best_x, best_y = x_centers, y_centers
        if len(best_y) >= 3 and len(best_x) >= 5:
            break

    return sorted(best_x), sorted(best_y)


def _long_gridline_mask(plot_bgr: np.ndarray) -> np.ndarray:
    h, w = plot_bgr.shape[:2]
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    grid = ((s < 25) & (v > 200) & (v < 250)).astype(np.uint8) * 255

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(w * 0.25)), 1))
    h_lines = cv2.erode(grid, h_kernel, iterations=1)
    h_lines = cv2.dilate(h_lines, h_kernel, iterations=1)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(h * 0.20))))
    v_lines = cv2.erode(grid, v_kernel, iterations=1)
    v_lines = cv2.dilate(v_lines, v_kernel, iterations=1)

    return cv2.bitwise_or(h_lines, v_lines)


def _kmeans_1d_two_clusters(values: np.ndarray) -> float:
    # Returns a threshold between two clusters (midpoint of centers).
    v = values.astype(np.float64).reshape(-1)
    if v.size == 0:
        return 220.0
    c1, c2 = 200.0, 240.0
    for _ in range(12):
        d1 = np.abs(v - c1)
        d2 = np.abs(v - c2)
        m1 = v[d1 <= d2]
        m2 = v[d1 > d2]
        if m1.size:
            c1 = float(np.mean(m1))
        if m2.size:
            c2 = float(np.mean(m2))
    if c1 > c2:
        c1, c2 = c2, c1
    return (c1 + c2) / 2.0


def _kmeans_1d_k_clusters(values: np.ndarray, k: int, init: list[float], iters: int = 15) -> np.ndarray:
    v = values.astype(np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("kmeans values empty")
    if k <= 0:
        raise ValueError("k must be >= 1")
    if len(init) != k:
        raise ValueError("init must have length k")
    centers = np.array(init, dtype=np.float64)
    for _ in range(int(iters)):
        d = np.abs(v[:, None] - centers[None, :])
        lab = np.argmin(d, axis=1)
        for j in range(k):
            m = v[lab == j]
            if m.size:
                centers[j] = float(np.mean(m))
    return centers

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


def _black_curve_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    # Low value and low saturation covers black/dark gray strokes.
    mask = ((v < 80) & (s < 80)).astype(np.uint8) * 255
    # Remove thin gridlines by keeping only thicker structures.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Remove thin verticals (e.g., axis/annotation artifacts) to favor the step curve.
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
    verticals = cv2.erode(mask, v_kernel, iterations=1)
    verticals = cv2.dilate(verticals, v_kernel, iterations=1)
    mask = cv2.subtract(mask, verticals)
    return mask


def _extract_black_curve_y_by_x(plot_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = _black_curve_mask(plot_bgr)
    h, w = mask.shape[:2]
    y_by_x = np.full(w, np.nan, dtype=np.float64)
    for x in range(w):
        ys = np.where(mask[:, x] > 0)[0]
        if ys.size:
            y_by_x[x] = float(np.median(ys))
    xs = np.arange(w, dtype=np.float64)
    good = np.isfinite(y_by_x)
    if good.sum() < 100:
        raise RuntimeError("Too few black pixels along x to form a curve.")
    y_interp = np.interp(xs, xs[good], y_by_x[good])
    return xs, y_interp


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


def _extract_upper_band_y_by_x(plot_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # For Figure 5: extract the upper envelope of the 10th–90th shaded band (frontier).
    gray = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2GRAY)
    # Empirically for these figures: background is ~255, shaded bands cluster at ~229 (light) and ~181 (dark),
    # and gridlines are lighter (~235+). Excluding >=235 avoids gridlines dominating the envelope.
    band = (gray < 235) & (gray > 120)

    # Suppress legend region (top-left), which otherwise dominates the "upper envelope".
    h, w = band.shape[:2]
    band[: int(0.22 * h), : int(0.40 * w)] = False
    # Suppress bottom x-label area.
    band[int(0.96 * h) :, :] = False

    band_u8 = band.astype(np.uint8) * 255
    band_u8 = cv2.morphologyEx(
        band_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1
    )

    y_by_x = np.full(w, np.nan, dtype=np.float64)
    for x in range(w):
        ys = np.where(band_u8[:, x] > 0)[0]
        if ys.size:
            y_by_x[x] = float(np.min(ys))

    xs = np.arange(w, dtype=np.float64)
    good = np.isfinite(y_by_x)
    if good.sum() < 50:
        raise RuntimeError("Too few band pixels along x to form an upper envelope.")
    # Interpolate only across the span where band exists; don't extrapolate.
    y_interp = y_by_x.copy()
    first = int(xs[good][0])
    last = int(xs[good][-1])
    y_interp[first : last + 1] = np.interp(xs[first : last + 1], xs[good], y_by_x[good])
    return xs, y_interp


def _extract_gray_band_envelope_y_by_x(
    plot_bgr: np.ndarray, band: str, suppress_legend: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Returns (xs, y_top, y_bottom) in pixel coords for a gray shaded band.
    gray = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    red = _red_mask(plot_bgr) > 0
    # Also exclude blue highest/lowest intelligence lines/labels.
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)
    blue = (sch > 60) & (vch > 80) & (hch >= 90) & (hch <= 140)
    # We intentionally do NOT hard-remove gridlines here; instead we rely on selecting the
    # thickest per-column run, which ignores 1px gridlines.

    # Build a per-pixel candidate mask.
    if band == "dark":
        cand = (gray >= 165) & (gray <= 210)
        min_thickness = 10
    elif band == "light":
        cand = (gray >= 215) & (gray <= 245)
        min_thickness = 14
    elif band == "union":
        # Union of both gray bands (for p10/p90 envelope).
        cand = (gray >= 160) & (gray <= 245)
        min_thickness = 18
    else:
        raise ValueError("band must be 'dark', 'light', or 'union'")

    cand &= ~red
    cand &= ~blue

    if suppress_legend:
        # Figure 15 legend is bottom-right; remove to avoid boundary corruption.
        cand[int(0.60 * h) :, int(0.70 * w) :] = False

    # Remove top-right label area ("Highest intelligence") and bottom-right label ("Lowest intelligence").
    cand[: int(0.18 * h), int(0.70 * w) :] = False
    cand[int(0.86 * h) :, int(0.55 * w) :] = False

    # Remove left-axis label area to reduce stray text pixels.
    cand[:, : int(0.07 * w)] = False
    # Remove very dark text/axes (but keep shaded bands).
    cand &= gray >= 140

    cand_u8 = cand.astype(np.uint8) * 255
    cand_u8 = cv2.morphologyEx(
        cand_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
    )
    cand_u8 = cv2.morphologyEx(
        cand_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1
    )

    y_top = np.full(w, np.nan, dtype=np.float64)
    y_bot = np.full(w, np.nan, dtype=np.float64)
    for x in range(w):
        ys = np.where(cand_u8[:, x] > 0)[0]
        if ys.size:
            # If multiple segments exist, select the longest contiguous run (band thickness),
            # ignoring tiny runs from stray pixels.
            ys = ys.astype(int)
            breaks = np.where(np.diff(ys) > 1)[0]
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks, ys.size - 1]
            lengths = (ends - starts) + 1
            ok = np.where(lengths >= min_thickness)[0]
            if ok.size == 0:
                continue
            best_i = int(ok[np.argmax(lengths[ok])])
            seg = ys[starts[best_i] : ends[best_i] + 1]
            y_top[x] = float(seg.min())
            y_bot[x] = float(seg.max())

    xs = np.arange(w, dtype=np.float64)
    good = np.isfinite(y_top) & np.isfinite(y_bot)
    if good.sum() < 80:
        raise RuntimeError(f"Too few pixels detected for band='{band}'.")
    first = int(xs[good][0])
    last = int(xs[good][-1])
    y_top_i = y_top.copy()
    y_bot_i = y_bot.copy()
    # Interpolate only across the contiguous span; remaining gaps will be handled later in time domain.
    y_top_i[first : last + 1] = np.interp(xs[first : last + 1], xs[good], y_top[good])
    y_bot_i[first : last + 1] = np.interp(xs[first : last + 1], xs[good], y_bot[good])
    return xs, y_top_i, y_bot_i


def _extract_band_envelopes_anchored_to_curve(
    plot_bgr: np.ndarray, curve_y_px: np.ndarray, band: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Extract top/bottom envelopes of a filled band by selecting, per column, the contiguous
    # masked segment that contains the given curve y (median), avoiding legend/text artifacts.
    gray = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    curve_y_px = np.clip(curve_y_px.astype(np.float64), 0, h - 1)

    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)
    red = _red_mask(plot_bgr) > 0
    blue = (sch > 60) & (vch > 80) & (hch >= 90) & (hch <= 140)

    if band == "union":
        # p10–p90 filled region: very light gray fill (but distinctly below pure white).
        # Empirically for this figure the light band is ~237 and the dark band ~210.
        mask = (sch < 40) & (gray >= 205) & (gray <= 242)
        min_thickness = 14
    elif band == "dark":
        # p25–p75 filled region: darker gray fill.
        mask = (sch < 40) & (gray >= 205) & (gray <= 225)
        min_thickness = 10
    else:
        raise ValueError("band must be 'union' or 'dark'")

    mask &= ~red
    mask &= ~blue
    # Remove legend and label regions.
    mask[int(0.60 * h) :, int(0.70 * w) :] = False
    mask[: int(0.18 * h), int(0.70 * w) :] = False
    mask[int(0.86 * h) :, int(0.55 * w) :] = False
    mask[:, : int(0.07 * w)] = False

    mask_u8 = (mask.astype(np.uint8)) * 255
    mask_u8 = cv2.morphologyEx(
        mask_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
    )
    mask_u8 = cv2.morphologyEx(
        mask_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1
    )

    y_top = np.full(w, np.nan, dtype=np.float64)
    y_bot = np.full(w, np.nan, dtype=np.float64)
    for x in range(w):
        ys = np.where(mask_u8[:, x] > 0)[0]
        if ys.size == 0:
            continue
        ys = ys.astype(int)
        breaks = np.where(np.diff(ys) > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, ys.size - 1]
        lengths = (ends - starts) + 1
        y0 = int(round(curve_y_px[x]))
        # The red curve is excluded from the mask; treat the target as "near the curve"
        # rather than requiring containment.
        max_dist = 14 if band == "union" else 10
        best_min = None
        best_max = None
        best_dist = 1_000_000
        best_len = -1
        for s, e, ln in zip(starts.tolist(), ends.tolist(), lengths.tolist(), strict=True):
            if ln < min_thickness:
                continue
            seg_min = int(ys[s])
            seg_max = int(ys[e])
            if y0 < seg_min:
                dist = seg_min - y0
            elif y0 > seg_max:
                dist = y0 - seg_max
            else:
                dist = 0
            if dist > max_dist:
                continue
            if dist < best_dist or (dist == best_dist and ln > best_len):
                best_dist = dist
                best_len = ln
                best_min = seg_min
                best_max = seg_max
        if best_min is None or best_max is None:
            continue
        y_top[x] = float(best_min)
        y_bot[x] = float(best_max)

    xs = np.arange(w, dtype=np.float64)
    good = np.isfinite(y_top) & np.isfinite(y_bot)
    if good.sum() < 200:
        raise RuntimeError(f"Too few pixels detected for anchored band='{band}'.")
    first = int(xs[good][0])
    last = int(xs[good][-1])
    y_top_i = y_top.copy()
    y_bot_i = y_bot.copy()
    y_top_i[first : last + 1] = np.interp(xs[first : last + 1], xs[good], y_top[good])
    y_bot_i[first : last + 1] = np.interp(xs[first : last + 1], xs[good], y_bot[good])
    return xs, y_top_i, y_bot_i


def _extract_band_envelopes_columnwise_kmeans(
    plot_bgr: np.ndarray,
    curve_y_px: np.ndarray,
    band: str,
    legend_corner: str = "bottom_right",
    light_bg_floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust Figure 15 band envelope extraction.

    Uses a global 3-cluster grayscale model (dark fill, light fill, background) restricted to
    neutral pixels, then for each x-column chooses the contiguous segment nearest the red median.
    """
    gray = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    h, w = gray.shape[:2]
    curve_y_px = np.clip(curve_y_px.astype(np.float64), 0, h - 1)

    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)
    red = _red_mask(plot_bgr) > 0
    blue = (sch > 60) & (vch > 80) & (hch >= 90) & (hch <= 140)

    # Mask out label/legend areas (same rough regions as elsewhere).
    allowed = np.ones((h, w), dtype=bool)
    if legend_corner == "bottom_right":
        allowed[int(0.60 * h) :, int(0.70 * w) :] = False
    elif legend_corner == "bottom_left":
        allowed[int(0.60 * h) :, : int(0.35 * w)] = False
    elif legend_corner == "top_left":
        allowed[: int(0.28 * h), : int(0.35 * w)] = False
    elif legend_corner == "top_right":
        allowed[: int(0.28 * h), int(0.65 * w) :] = False
    else:
        raise ValueError("legend_corner must be bottom_right/bottom_left/top_left/top_right")
    allowed[: int(0.18 * h), int(0.70 * w) :] = False
    allowed[int(0.86 * h) :, int(0.55 * w) :] = False
    allowed[:, : int(0.07 * w)] = False
    allowed &= ~red
    allowed &= ~blue

    # Restrict to neutral-ish pixels; keep enough margin to include both bands.
    # Tighten saturation to avoid anti-aliased halos from colored lines/text.
    allowed &= (sch < 40) & (vch > 140)

    # Global 3-cluster model to separate dark fill, light fill, and background.
    vals = gray[allowed].astype(np.float64)
    if vals.size < 50_000:
        raise RuntimeError(f"Too few pixels for global band clustering ({vals.size}).")
    # Subsample for speed (deterministically).
    if vals.size > 300_000:
        vals = vals[:: int(vals.size // 300_000) + 1]
    centers = _kmeans_1d_k_clusters(vals, k=3, init=[210.0, 237.0, 252.0], iters=18)
    order = np.argsort(centers)
    c_dark = float(centers[order[0]])
    c_light = float(centers[order[1]])
    c_bg = float(centers[order[2]])

    # Threshold-based classification is more stable than nearest-center when centers drift.
    t_dark_light = float((c_dark + c_light) / 2.0)
    t_light_bg = float((c_light + c_bg) / 2.0)
    # Keep background separation strict (near-white).
    t_light_bg = float(np.clip(t_light_bg, 238.0, 252.0))
    if light_bg_floor is not None:
        t_light_bg = float(np.clip(max(t_light_bg, float(light_bg_floor)), 238.0, 252.0))
    g = gray.astype(np.float64)
    union_mask = allowed & (g <= t_light_bg)
    dark_mask = allowed & (g <= t_dark_light)

    # Figure 11 has very faint gridlines, and `_detect_gridlines` can miss many of them.
    # The long gridline mask is more reliable and prevents full-height gridline columns from
    # being misinterpreted as fill.
    if legend_corner == "bottom_left":
        long_grid = _long_gridline_mask(plot_bgr) > 0
        union_mask &= ~long_grid
        dark_mask &= ~long_grid

    # Remove gridlines using detected gridline peaks (avoid masking out the filled bands).
    # We only blank a small neighborhood around each long gridline center.
    x_peaks, y_peaks = _detect_gridlines(plot_bgr)
    if x_peaks:
        for xx in x_peaks:
            x0 = max(0, int(xx) - 1)
            x1 = min(w, int(xx) + 2)
            union_mask[:, x0:x1] = False
            dark_mask[:, x0:x1] = False
    if y_peaks:
        for yy in y_peaks:
            y = int(yy)
            if 0 <= y < h:
                union_mask[y : y + 1, :] = False
                dark_mask[y : y + 1, :] = False

    # Heal small vertical holes (gridline subtraction / AA artifacts) without bridging across big gaps.
    def _clean(mask: np.ndarray) -> np.ndarray:
        m = (mask.astype(np.uint8)) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
        return m > 0

    union_mask = _clean(union_mask)
    dark_mask = _clean(dark_mask)

    y_top = np.full(w, np.nan, dtype=np.float64)
    y_bot = np.full(w, np.nan, dtype=np.float64)
    if band == "union":
        # For p10–p90, the red curve can split the fill into two components (above/below median).
        # Using a simple per-column min/max across all pixels is too permissive: stray neutral pixels
        # far below the plot can drag p10 downward. Instead:
        # - Find contiguous fill segments per column
        # - Pick the nearest segment above the median for p90 (top envelope)
        # - Pick the nearest segment below the median for p10 (bottom envelope)
        y_med = np.clip(np.rint(curve_y_px).astype(int), 0, h - 1)
        # Figure 11 has a much taller effective band span (log y), so allow a larger search radius.
        max_dist = int(0.70 * h) if legend_corner == "bottom_left" else 120
        min_thickness = 12
        for x in range(w):
            ys = np.where(union_mask[:, x])[0]
            if ys.size == 0:
                continue
            ys = ys.astype(int)
            # Split into contiguous runs (treat small holes as gaps; masks already got vertical close/open).
            breaks = np.where(np.diff(ys) > 1)[0]
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks, ys.size - 1]
            y0 = int(y_med[x])
            best_above = None  # (dist, -thickness, seg_min, seg_max)
            # For p10 (below), Figure 15 prefers the thickest nearby segment to avoid stray pixels.
            # For Figure 11 we want the bottom-most segment below the median (the p10 tail can be thin),
            # and we mask long gridlines separately to reduce false positives.
            best_below = None
            best_cross = None
            for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
                seg_min = int(ys[s])
                seg_max = int(ys[e])
                thickness = (seg_max - seg_min) + 1
                min_th = 5 if legend_corner == "bottom_left" else min_thickness
                if thickness < min_th:
                    continue
                if seg_min <= y0 <= seg_max:
                    best_cross = (0, thickness, seg_min, seg_max)
                    break
                if seg_max < y0:
                    dist = y0 - seg_max
                    if dist <= max_dist:
                        cand = (dist, -thickness, seg_min, seg_max)
                        if best_above is None or cand[:2] < best_above[:2]:
                            best_above = cand
                elif seg_min > y0:
                    dist = seg_min - y0
                    if dist <= max_dist:
                        if legend_corner == "bottom_left":
                            cand = (-seg_max, -thickness, dist, seg_min, seg_max)
                            if best_below is None or cand[:3] < best_below[:3]:
                                best_below = cand
                        else:
                            cand = (-thickness, dist, seg_min, seg_max)
                            if best_below is None or cand[:2] < best_below[:2]:
                                best_below = cand

            if best_cross is not None:
                _, _t, seg_min, seg_max = best_cross
                y_top[x] = float(seg_min)
                y_bot[x] = float(seg_max)
                continue

            if best_above is not None:
                _d, _neg_t, seg_min, _seg_max = best_above
                y_top[x] = float(seg_min)
            if best_below is not None:
                if legend_corner == "bottom_left":
                    _neg_segmax, _neg_t, _d, _seg_min, seg_max = best_below
                    y_bot[x] = float(seg_max)
                else:
                    _neg_t, _d, _seg_min, seg_max = best_below
                    y_bot[x] = float(seg_max)

        # Figure 11's light p10 tail can be extremely close to the white background, which can
        # cause the union mask to miss the lowest part of the fill (especially late in the plot).
        # Add a gentle bottom-edge expansion: look for near-white-but-not-background pixels
        # below the median and expand p10 downward when found.
        if legend_corner == "bottom_left":
            # Use the already-detected union bottom as an anchor, and only allow a limited,
            # mostly-contiguous extension downward. This avoids "discovering" random near-white
            # background pixels as fill.
            y_med = np.clip(np.rint(curve_y_px).astype(int), 0, h - 1)
            grid = _long_gridline_mask(plot_bgr) > 0
            max_extend = int(0.12 * h)  # conservative (prevents runaway at the bottom border)
            for x in range(w):
                if not np.isfinite(y_bot[x]):
                    continue
                col_allowed = allowed[:, x] & (~grid[:, x])
                if int(col_allowed.sum()) < 50:
                    continue
                col = g[:, x]
                bg = float(np.percentile(col[col_allowed], 99.5))
                # "Near-white fill": slightly darker than bg, but not near the axes/border.
                delta = 0.8
                y0 = int(np.clip(int(round(y_bot[x])), 0, h - 1))
                y_max = min(h - 3, y0 + max_extend)
                if y_max <= max(y0 + 2, y_med[x] + 2):
                    continue
                # Only search below the median.
                y_start = max(y0, y_med[x] + 1)
                ok = col_allowed & (col >= 230.0) & (col <= (bg - delta))

                # Incrementally extend downward, tolerating a few-pixel holes.
                y_last = y0
                fails = 0
                for y in range(y_start, y_max + 1):
                    if ok[y]:
                        y_last = y
                        fails = 0
                    else:
                        fails += 1
                        if fails > 3:
                            break
                # Expand downward only (increase y), never upward.
                if y_last > y_bot[x]:
                    y_bot[x] = float(y_last)
    elif band == "dark":
        # For the dark band, per-column min/max is typically stable after global classification.
        for x in range(w):
            ys = np.where(dark_mask[:, x])[0]
            if ys.size:
                y_top[x] = float(int(ys.min()))
                y_bot[x] = float(int(ys.max()))
    else:
        raise ValueError("band must be 'union' or 'dark'")

    xs = np.arange(w, dtype=np.float64)
    # Post-cleaning: reject impossible geometry and one-column teleports, then smooth.
    ok = np.isfinite(y_top) & np.isfinite(y_bot)
    # Basic order: top must be above bottom in pixel coords.
    bad_order = ok & (y_top >= y_bot)
    y_top[bad_order] = np.nan
    y_bot[bad_order] = np.nan
    # Do not enforce “straddles median” here: the red curve is masked out, so the band can
    # appear split around the median and per-column selection already constrains proximity.
    ok = np.isfinite(y_top) & np.isfinite(y_bot)

    def _despike(a: np.ndarray, max_px: float) -> np.ndarray:
        a2 = a.copy()
        good = np.isfinite(a2)
        if good.sum() < 10:
            return a2
        idx = np.where(good)[0]
        dif = np.diff(a2[idx])
        if dif.size == 0:
            return a2
        mad = float(np.median(np.abs(dif - np.median(dif)))) + 1e-6
        thr = max(max_px, 8.0 * mad)
        spike = np.abs(dif) > thr
        if np.any(spike):
            # Kill both endpoints of spike edges.
            kill = set()
            for i, is_spike in enumerate(spike.tolist()):
                if is_spike:
                    kill.add(int(idx[i]))
                    kill.add(int(idx[i + 1]))
            for k in kill:
                a2[k] = np.nan
        return a2

    # Top/bottom can move slowly across x; large jumps are almost always gridline/label confusion.
    y_top = _despike(y_top, max_px=60.0)
    y_bot = _despike(y_bot, max_px=60.0)

    good = np.isfinite(y_top) & np.isfinite(y_bot)
    if good.sum() < 200:
        raise RuntimeError(f"Too few pixels detected for columnwise band='{band}'.")
    first = int(xs[good][0])
    last = int(xs[good][-1])

    # Median-filter in x to suppress 1-column "teleports", then interpolate across gaps.
    def medfilt_nan(a: np.ndarray, k: int) -> np.ndarray:
        k = int(k)
        if k < 3:
            return a
        r = k // 2
        out = a.copy()
        idx = np.arange(a.size)
        for i in range(a.size):
            lo = max(0, i - r)
            hi = min(a.size, i + r + 1)
            win = a[lo:hi]
            win = win[np.isfinite(win)]
            if win.size:
                out[i] = float(np.median(win))
        return out

    y_top_s = medfilt_nan(y_top, 13)
    y_bot_s = medfilt_nan(y_bot, 13)
    good_s = np.isfinite(y_top_s) & np.isfinite(y_bot_s)
    y_top_i = y_top_s.copy()
    y_bot_i = y_bot_s.copy()
    y_top_i[first : last + 1] = np.interp(xs[first : last + 1], xs[good_s], y_top_s[good_s])
    y_bot_i[first : last + 1] = np.interp(xs[first : last + 1], xs[good_s], y_bot_s[good_s])
    if band == "union" and legend_corner == "bottom_left":
        # Inflate to account for anti-aliased edges on the very light fill. Empirically, Figure 11
        # under-coverage is dominated by the lower envelope (p10), so pad the bottom more than top.
        pad_top = 1.0
        pad_bot = 6.0
        y_top_i = np.clip(y_top_i - pad_top, 0.0, float(h - 1))
        y_bot_i = np.clip(y_bot_i + pad_bot, 0.0, float(h - 1))
        bad = np.isfinite(y_top_i) & np.isfinite(y_bot_i) & (y_top_i >= y_bot_i)
        if np.any(bad):
            mid = 0.5 * (y_top_i[bad] + y_bot_i[bad])
            y_top_i[bad] = np.clip(mid - 1.0, 0.0, float(h - 1))
            y_bot_i[bad] = np.clip(mid + 1.0, 0.0, float(h - 1))
    return xs, y_top_i, y_bot_i


def _figure11_observed_like_fill_masks(plot_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Figure 11-specific fill segmentation designed to closely match the verification heuristic.

    Returns:
      - observed_union: light+dark fill pixels
      - observed_dark: dark fill pixels (p25–p75)
    """
    gray = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    h, w = gray.shape[:2]

    red = _red_mask(plot_bgr) > 0
    cand = (s < 40) & (v > 120) & ~red

    # Mask legend (bottom-left for Figure 11) and axis label area.
    cand[int(0.60 * h) :, : int(0.35 * w)] = False
    cand[:, : int(0.07 * w)] = False

    # Remove gridlines: both detected peaks and the long gridline mask.
    x_peaks, y_peaks = _detect_gridlines(plot_bgr)
    for xx in x_peaks:
        x0 = max(0, int(xx) - 1)
        x1 = min(w, int(xx) + 2)
        cand[:, x0:x1] = False
    for yy in y_peaks:
        y0 = max(0, int(yy))
        y1 = min(h, int(yy) + 1)
        cand[y0:y1, :] = False
    cand &= ~(_long_gridline_mask(plot_bgr) > 0)

    vals = gray[cand].astype(np.float64)
    if vals.size < 20_000:
        raise RuntimeError(f"Too few candidate pixels for Figure 11 clustering ({vals.size}).")
    if vals.size > 250_000:
        vals = vals[:: int(vals.size // 250_000) + 1]

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


def _extract_band_envelopes_figure11_observed_like(
    plot_bgr: np.ndarray, band: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Figure 11 band envelopes directly from observed-like masks (more stable than
    columnwise thresholding due to very faint gridlines and very light p10 tail).
    """
    obs_union, obs_dark = _figure11_observed_like_fill_masks(plot_bgr)
    mask = obs_union if band == "union" else obs_dark
    h, w = mask.shape[:2]
    xs = np.arange(w, dtype=np.float64)
    y_top = np.full(w, np.nan, dtype=np.float64)
    y_bot = np.full(w, np.nan, dtype=np.float64)

    for x in range(w):
        ys = np.where(mask[:, x])[0]
        if ys.size:
            y_top[x] = float(int(ys.min()))
            y_bot[x] = float(int(ys.max()))

    # Reuse the same post-cleaning and interpolation logic as the generic extractor by calling
    # `_extract_band_envelopes_columnwise_kmeans`'s tail section would be messy; implement a small
    # local smoother here.
    good = np.isfinite(y_top) & np.isfinite(y_bot) & (y_top < y_bot)
    if good.sum() < 200:
        raise RuntimeError(f"Too few pixels detected for Figure 11 band='{band}'.")

    first = int(xs[good][0])
    last = int(xs[good][-1])

    def _despike(a: np.ndarray, max_px: float) -> np.ndarray:
        a2 = a.copy()
        good2 = np.isfinite(a2)
        if good2.sum() < 10:
            return a2
        idx = np.where(good2)[0]
        dif = np.diff(a2[idx])
        if dif.size == 0:
            return a2
        mad = float(np.median(np.abs(dif - np.median(dif)))) + 1e-6
        thr = max(max_px, 8.0 * mad)
        spike = np.abs(dif) > thr
        if np.any(spike):
            kill = set()
            for i, is_spike in enumerate(spike.tolist()):
                if is_spike:
                    kill.add(int(idx[i]))
                    kill.add(int(idx[i + 1]))
            for k in kill:
                a2[k] = np.nan
        return a2

    y_top = _despike(y_top, max_px=80.0)
    y_bot = _despike(y_bot, max_px=80.0)

    def medfilt_nan(a: np.ndarray, k: int) -> np.ndarray:
        k = int(k)
        if k < 3:
            return a
        r = k // 2
        out = a.copy()
        for i in range(a.size):
            lo = max(0, i - r)
            hi = min(a.size, i + r + 1)
            win = a[lo:hi]
            win = win[np.isfinite(win)]
            if win.size:
                out[i] = float(np.median(win))
        return out

    y_top_s = medfilt_nan(y_top, 17 if band == "union" else 9)
    y_bot_s = medfilt_nan(y_bot, 17 if band == "union" else 9)

    good_s = np.isfinite(y_top_s) & np.isfinite(y_bot_s) & (y_top_s < y_bot_s)
    y_top_i = y_top_s.copy()
    y_bot_i = y_bot_s.copy()
    y_top_i[first : last + 1] = np.interp(xs[first : last + 1], xs[good_s], y_top_s[good_s])
    y_bot_i[first : last + 1] = np.interp(xs[first : last + 1], xs[good_s], y_bot_s[good_s])

    # Pad slightly for anti-aliased edges.
    if band == "union":
        y_top_i = np.clip(y_top_i - 1.0, 0.0, float(h - 1))
        y_bot_i = np.clip(y_bot_i + 3.0, 0.0, float(h - 1))
    else:
        y_top_i = np.clip(y_top_i - 1.0, 0.0, float(h - 1))
        y_bot_i = np.clip(y_bot_i + 1.0, 0.0, float(h - 1))

    return xs, y_top_i, y_bot_i


def _resample_daily_to_year(doys: np.ndarray, vals: np.ndarray, year: int) -> tuple[list[dt.date], np.ndarray]:
    # Resample to a full calendar year, filling gaps by interpolation.
    start = dt.date(year, 1, 1).toordinal()
    end = dt.date(year, 12, 31).toordinal()
    x = np.rint(doys).astype(int)
    sel = np.isfinite(vals) & (x >= start) & (x <= end)
    x = x[sel]
    vals = vals[sel]
    uniq = np.arange(start, end + 1)
    daily = np.full(uniq.size, np.nan, dtype=np.float64)
    for i, d in enumerate(uniq.tolist()):
        mask = x == d
        if np.any(mask):
            daily[i] = float(np.median(vals[mask]))
    good = np.isfinite(daily)
    if good.sum() >= 2:
        xi = np.arange(daily.size)
        daily = np.interp(xi, xi[good], daily[good])
    dates = [dt.date.fromordinal(int(d)) for d in uniq.tolist()]
    return dates, daily


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

    if figure_number == 11:
        # Figure 11 uses labels like 09/23, 11/23, 01/24, ... , 11/25 (every 2 months).
        if len(x_label_centers) < 8:
            raise RuntimeError(f"Expected many x tick labels for Figure 11; got {len(x_label_centers)}.")
        k = 14 if len(x_label_centers) >= 14 else len(x_label_centers)
        x_label_centers = _select_evenly_spaced_subset(x_label_centers, k)

        def add_months(d: dt.date, months: int) -> dt.date:
            y = d.year + (d.month - 1 + months) // 12
            m = (d.month - 1 + months) % 12 + 1
            return dt.date(y, m, 1)

        start = dt.date(2023, 9, 1)
        x_tick_dates = tuple(add_months(start, 2 * i) for i in range(len(x_label_centers)))
        anchor_days = np.array([d.toordinal() for d in x_tick_dates], dtype=np.float64)
        xs = np.array(x_label_centers, dtype=np.float64)
        x_to_days_a, x_to_days_b = np.polyfit(xs, anchor_days, 1)
        x_tick_format = "%m/%y"
    elif len(x_label_centers) >= 10:
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

    if figure_number == 11:
        # Figure 11 uses a log scale with major ticks at 0.1, 1, 10, 100.
        # Calibrate in log10-space: log10(val) = m*y + c.
        # Some renderings only draw 3 interior gridlines (100, 10, 1) plus the axis baseline (0.1).
        y_ticks = _detect_y_tick_centers_from_left_strip(embedded_bgr, plot_rect)
        if len(y_ticks) >= 4:
            y_points = np.array(sorted(y_ticks)[:4], dtype=np.float64)
        else:
            if len(y_peaks) < 3:
                raise RuntimeError(
                    f"Expected at least 3 horizontal gridlines for Figure 11; got {len(y_peaks)} and only {len(y_ticks)} y-ticks."
                )
            y_candidates = [yy for yy in sorted(y_peaks) if int(0.08 * h) < yy < int(0.96 * h)]
            if len(y_candidates) < 3:
                y_candidates = sorted(y_peaks)
            y3 = sorted(_select_evenly_spaced_subset([int(v) for v in y_candidates], 3))
            # Infer the 0.1 tick position by extending equal log spacing rather than using the border.
            dy = float(np.median(np.diff(np.array(y3, dtype=np.float64))))
            y_0p1 = float(np.clip(y3[-1] + dy, 0.0, float(h - 1)))
            y_points = np.array([y3[0], y3[1], y3[2], y_0p1], dtype=np.float64)
        log_values = np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float64)  # 100,10,1,0.1
        y_to_val_m, y_to_val_c = np.polyfit(y_points, log_values, 1)
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
            y_scale="log10",
        )

    # Y calibration: use bottom border as 0.0 and 0.2/0.4/0.6 horizontal gridlines.
    if len(y_peaks) < 3:
        raise RuntimeError(f"Expected at least 3 horizontal gridlines; got {len(y_peaks)}.")

    y_peaks = sorted(y_peaks)
    y_candidates = [yy for yy in y_peaks if int(0.10 * h) < yy < int(0.92 * h)]
    if len(y_candidates) < 3:
        y_candidates = y_peaks

    # Choose 3 interior gridlines with near-equal spacing.
    best_triplet: tuple[int, int, int] | None = None
    if figure_number == 6 and len(y_candidates) == 3:
        # Figure 6 has exactly three interior major gridlines (0.75, 0.50, 0.25).
        best_triplet = (int(y_candidates[0]), int(y_candidates[1]), int(y_candidates[2]))
    else:
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
                    span = y3 - y1
                    score += float(abs(span - 0.5 * h) / h) * 0.25
                    if score < best_score:
                        best_score = score
                        best_triplet = (y1, y2, y3)

    if best_triplet is None:
        raise RuntimeError("Failed to select y gridlines for calibration.")

    y_points = np.array([best_triplet[0], best_triplet[1], best_triplet[2], h - 1], dtype=np.float64)
    if figure_number == 6:
        # Figure 6 y-axis spans 0–1 with major ticks at 0.25 increments.
        y_values = np.array([1.0, 0.75, 0.50, 0.25], dtype=np.float64)
    else:
        # Figures 5/15 y-axis span ~0–0.6 with major ticks at 0.2 increments.
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
        y_scale="linear",
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


def _resample_to_grid(x_days: np.ndarray, y_vals: np.ndarray, grid_days: np.ndarray) -> np.ndarray:
    x = np.rint(x_days).astype(int)
    sel = np.isfinite(y_vals)
    x = x[sel]
    y_vals = y_vals[sel]
    out = np.full(grid_days.size, np.nan, dtype=np.float64)
    for i, d in enumerate(grid_days.tolist()):
        mask = x == d
        if np.any(mask):
            out[i] = float(np.median(y_vals[mask]))
    good = np.isfinite(out)
    if good.sum() >= 2:
        xi = np.arange(out.size, dtype=np.float64)
        out = np.interp(xi, xi[good], out[good])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract a figure's red median curve time series from the PDF (embedded plot image).")
    ap.add_argument("--pdf", type=Path, default=Path("w34608.pdf"))
    ap.add_argument("--outdir", type=Path, default=Path("out"))
    ap.add_argument("--figure", type=int, default=15, help="Figure number to extract (default: 15).")
    ap.add_argument("--diagnostic", action="store_true", help="Write overlay images and overlap metrics.")
    ap.add_argument("--curve", choices=["red_median", "p90_frontier", "black_frontier"], default="red_median")
    ap.add_argument("--percentiles", action="store_true", help="For Figure 15, extract p10/p25/p50/p75/p90 series.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    embedded_path = _extract_embedded_figure_image(args.pdf, args.figure, args.outdir / f"figure{args.figure}_embedded.png")
    bgr = _read_bgr(embedded_path)

    x0, y0, x1, y1 = _find_plot_rect(bgr)
    plot = bgr[y0 : y1 + 1, x0 : x1 + 1].copy()

    x_peaks, y_peaks = _detect_gridlines(plot)
    calib = _calibrate_plot(plot, bgr, (x0, y0, x1, y1), x_peaks, y_peaks, figure_number=args.figure)

    if args.percentiles:
        if args.figure not in (11, 15):
            raise RuntimeError("--percentiles currently supported only for --figure 11 or --figure 15.")

        # Median
        xs_m, ys_m = _extract_curve_y_by_x(plot)
        days_m = calib.days_from_x(xs_m)
        vals_m = calib.val_from_y(ys_m)
        if args.figure == 15:
            dates, p50 = _resample_daily_to_year(days_m, vals_m, year=2025)

        legend_corner = "bottom_right" if args.figure == 15 else "bottom_left"
        # Figure 11's light band can be very close to white; allow a higher fill threshold.
        light_bg_floor = None if args.figure == 15 else 251.0

        # Bands
        if args.figure == 11:
            xs_u, ytop_u, ybot_u = _extract_band_envelopes_figure11_observed_like(plot, band="union")
            xs_d, ytop_d, ybot_d = _extract_band_envelopes_figure11_observed_like(plot, band="dark")
        else:
            xs_u, ytop_u, ybot_u = _extract_band_envelopes_columnwise_kmeans(
                plot, ys_m, band="union", legend_corner=legend_corner, light_bg_floor=light_bg_floor
            )
            xs_d, ytop_d, ybot_d = _extract_band_envelopes_columnwise_kmeans(
                plot, ys_m, band="dark", legend_corner=legend_corner, light_bg_floor=light_bg_floor
            )

        days_u = calib.days_from_x(xs_u)
        days_d = calib.days_from_x(xs_d)
        p90_raw = calib.val_from_y(ytop_u)
        p10_raw = calib.val_from_y(ybot_u)
        p75_raw = calib.val_from_y(ytop_d)
        p25_raw = calib.val_from_y(ybot_d)

        if args.figure == 15:
            _d, p10 = _resample_daily_to_year(days_u, p10_raw, year=2025)
            _d, p90 = _resample_daily_to_year(days_u, p90_raw, year=2025)
            _d, p25 = _resample_daily_to_year(days_d, p25_raw, year=2025)
            _d, p75 = _resample_daily_to_year(days_d, p75_raw, year=2025)
        else:
            # Resample all series onto a common daily grid to avoid shape mismatches.
            start = int(np.floor(min(np.nanmin(days_m), np.nanmin(days_u), np.nanmin(days_d))))
            end = int(np.ceil(max(np.nanmax(days_m), np.nanmax(days_u), np.nanmax(days_d))))
            grid = np.arange(start, end + 1, dtype=int)
            dates = [dt.date.fromordinal(int(d)) for d in grid.tolist()]
            p50 = _resample_to_grid(days_m, vals_m, grid)
            p10 = _resample_to_grid(days_u, p10_raw, grid)
            p90 = _resample_to_grid(days_u, p90_raw, grid)
            p25 = _resample_to_grid(days_d, p25_raw, grid)
            p75 = _resample_to_grid(days_d, p75_raw, grid)

        # Enforce ordering (guardrail against occasional boundary glitches).
        p10 = np.minimum(p10, p90)
        p25 = np.clip(p25, p10, p90)
        p50 = np.clip(p50, p25, p90)
        p75 = np.clip(p75, p50, p90)

        if args.figure == 15:
            out_csv = args.outdir / "figure15_token_weighted_percentiles.csv"
            overlay_path = args.outdir / "figure15_percentiles_overlay.png"
            dbg_path = args.outdir / "figure15_percentiles_raw_envelopes.png"
            ts_path = args.outdir / "figure15_token_weighted_percentiles.png"
        else:
            out_csv = args.outdir / "figure11_price_to_intelligence_ratio_percentiles.csv"
            overlay_path = args.outdir / "figure11_percentiles_overlay.png"
            dbg_path = args.outdir / "figure11_percentiles_raw_envelopes.png"
            ts_path = args.outdir / "figure11_percentiles_timeseries.png"

        with out_csv.open("w", encoding="utf-8") as f:
            f.write("date,p10,p25,p50,p75,p90\n")
            for d, a, b, c, e, ff in zip(dates, p10, p25, p50, p75, p90, strict=True):
                f.write(f"{d.isoformat()},{a:.6f},{b:.6f},{c:.6f},{e:.6f},{ff:.6f}\n")

        # Diagnostics: overlay boundaries on original plot image.
        overlay = plot.copy()
        day_ord = np.array([d.toordinal() for d in dates], dtype=np.float64)
        x_plot = calib.x_from_days(day_ord)
        y_p10 = calib.y_from_val(p10)
        y_p25 = calib.y_from_val(p25)
        y_p50 = calib.y_from_val(p50)
        y_p75 = calib.y_from_val(p75)
        y_p90 = calib.y_from_val(p90)
        for yarr, color, thick in [
            (y_p10, (255, 0, 255), 2),  # magenta
            (y_p25, (0, 165, 255), 2),  # orange
            (y_p50, (0, 0, 255), 2),  # red
            (y_p75, (255, 255, 0), 2),  # cyan
            (y_p90, (0, 255, 0), 2),  # green
        ]:
            mask_line = _draw_polyline_mask((overlay.shape[0], overlay.shape[1]), x_plot, yarr, thickness=thick)
            overlay[mask_line > 0] = color

        cv2.imwrite(overlay_path.as_posix(), overlay)

        # Additional debug overlays for raw envelopes (plot-pixel space).
        dbg = plot.copy()
        for yarr, color in [
            (ytop_u, (0, 255, 0)),   # p90 (top of union)
            (ybot_u, (255, 0, 255)), # p10 (bottom of union)
            (ytop_d, (255, 255, 0)), # p75
            (ybot_d, (0, 165, 255)), # p25
        ]:
            mline = _draw_polyline_mask((dbg.shape[0], dbg.shape[1]), xs_u if yarr is ytop_u or yarr is ybot_u else xs_d, yarr, thickness=2)
            dbg[mline > 0] = color
        cv2.imwrite(dbg_path.as_posix(), dbg)

        # Plot percentiles
        fig = plt.figure(figsize=(10, 4), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        ax.fill_between(dates, p10, p90, color="lightgray", alpha=0.6, label="p10–p90")
        ax.fill_between(dates, p25, p75, color="gray", alpha=0.6, label="p25–p75")
        ax.plot(dates, p50, color="red", linewidth=1.2, label="p50")
        ax.set_title(f"Figure {args.figure} extracted percentiles")
        ax.set_ylabel("Intelligence Index" if args.figure == 15 else "Price-to-Intelligence Ratio")
        if calib.y_scale == "log10":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", frameon=False)
        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(ts_path.as_posix())
        plt.close(fig)

        print(f"Wrote {out_csv}")
        print(f"Wrote {overlay_path}")
        print(f"Wrote {dbg_path}")
        print(f"Wrote {ts_path}")
        return 0

    if args.curve == "red_median":
        xs, ys = _extract_curve_y_by_x(plot)
        curve_slug = "red_median"
    elif args.curve == "p90_frontier":
        xs, ys = _extract_upper_band_y_by_x(plot)
        curve_slug = "p90_frontier"
    else:
        xs, ys = _extract_black_curve_y_by_x(plot)
        curve_slug = "black_frontier"
    days = calib.days_from_x(xs)
    vals = calib.val_from_y(ys)

    dates, daily_vals = _resample_daily(days, vals)

    csv_path = args.outdir / f"figure{args.figure}_red_median_timeseries.csv"
    if curve_slug != "red_median":
        csv_path = args.outdir / f"figure{args.figure}_{curve_slug}_timeseries.csv"
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
        if curve_slug == "red_median":
            observed_mask = _red_mask(plot)
        elif curve_slug == "black_frontier":
            observed_mask = _black_curve_mask(plot)
        else:
            observed_mask = _draw_polyline_mask((plot.shape[0], plot.shape[1]), xs, ys, thickness=3)

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
        curve_overlay_path = args.outdir / f"figure{args.figure}_{curve_slug}_curve_overlay.png"
        cv2.imwrite(curve_overlay_path.as_posix(), overlay_img)

    # Plot
    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dates, daily_vals, color="red", linewidth=1.5)
    ax.set_title(f"Figure {args.figure} extracted {curve_slug}")
    ax.set_ylabel("Intelligence Index")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(list(calib.x_tick_dates))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(calib.x_tick_format))
    if dates:
        ax.set_xlim(dates[0], dates[-1])
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    plot_path = args.outdir / f"figure{args.figure}_{curve_slug}_timeseries.png"
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
        print(f"Wrote {args.outdir / f'figure{args.figure}_{curve_slug}_curve_overlay.png'}")
        print(f"Wrote {args.outdir / f'figure{args.figure}_overlap_metrics.txt'}")
    if daily_vals.size:
        print(f"Sanity: min={float(np.min(daily_vals)):.3f} max={float(np.max(daily_vals)):.3f} start={float(daily_vals[0]):.3f} end={float(daily_vals[-1]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
