#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def _add_image(ax, path: Path, title: str) -> None:
    img = mpimg.imread(path.as_posix())
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create side-by-side diagnostic grids comparing original figures vs extracted series/overlays."
    )
    ap.add_argument("--outdir", type=Path, default=Path("out"))
    ap.add_argument(
        "--mode",
        choices=["basic", "percentiles15", "figure6", "figure11", "overlay11_15", "all"],
        default="all",
        help="Which diagnostic grid(s) to generate.",
    )
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path (only for --mode=basic).")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    def make_basic() -> None:
        out = args.out or (outdir / "figure5_figure15_comparison.png")
        fig = plt.figure(figsize=(14, 10), dpi=200)
        axs = fig.subplots(2, 2)
        _add_image(axs[0, 0], outdir / "figure5_embedded.png", "Figure 5 (original)")
        _add_image(axs[0, 1], outdir / "figure5_red_median_timeseries.png", "Figure 5 (extracted red median)")
        _add_image(axs[1, 0], outdir / "figure15_embedded.png", "Figure 15 (original)")
        _add_image(axs[1, 1], outdir / "figure15_red_median_timeseries.png", "Figure 15 (extracted red median)")
        fig.tight_layout()
        fig.savefig(out.as_posix(), bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")

    def make_percentiles15() -> None:
        out = outdir / "figure15_percentiles_comparison.png"
        fig = plt.figure(figsize=(14, 10), dpi=200)
        axs = fig.subplots(2, 2)
        _add_image(axs[0, 0], outdir / "figure15_embedded.png", "Figure 15 (original)")
        _add_image(axs[0, 1], outdir / "figure15_percentiles_overlay.png", "Figure 15 (percentiles overlay on plot)")
        _add_image(axs[1, 0], outdir / "figure15_percentiles_raw_envelopes.png", "Figure 15 (raw envelopes in pixel-space)")
        _add_image(axs[1, 1], outdir / "figure15_token_weighted_percentiles.png", "Figure 15 (extracted percentiles time series)")
        fig.tight_layout()
        fig.savefig(out.as_posix(), bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")

    def make_figure6() -> None:
        out = outdir / "figure6_comparison.png"
        fig = plt.figure(figsize=(14, 7), dpi=200)
        axs = fig.subplots(1, 3)
        _add_image(axs[0], outdir / "figure6_embedded.png", "Figure 6 (original)")
        _add_image(axs[1], outdir / "figure6_black_frontier_curve_overlay.png", "Figure 6 (curve overlay on plot)")
        _add_image(axs[2], outdir / "figure6_black_frontier_timeseries.png", "Figure 6 (extracted frontier series)")
        fig.tight_layout()
        fig.savefig(out.as_posix(), bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")

    def make_figure11() -> None:
        out = outdir / "figure11_percentiles_comparison.png"
        fig = plt.figure(figsize=(14, 10), dpi=200)
        axs = fig.subplots(2, 2)
        _add_image(axs[0, 0], outdir / "figure11_embedded.png", "Figure 11 (original)")
        _add_image(axs[0, 1], outdir / "figure11_percentiles_overlay.png", "Figure 11 (percentiles overlay on plot)")
        _add_image(axs[1, 0], outdir / "figure11_percentiles_raw_envelopes.png", "Figure 11 (raw envelopes in pixel-space)")
        _add_image(axs[1, 1], outdir / "figure11_percentiles_timeseries.png", "Figure 11 (extracted percentiles time series)")
        fig.tight_layout()
        fig.savefig(out.as_posix(), bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")

    def make_overlay11_15() -> None:
        out = outdir / "figure11_figure15_overlay_comparison.png"
        fig = plt.figure(figsize=(14, 10), dpi=200)
        axs = fig.subplots(2, 2)
        _add_image(axs[0, 0], outdir / "figure11_embedded.png", "Figure 11 (original)")
        _add_image(axs[0, 1], outdir / "figure11_percentiles_overlay.png", "Figure 11 (overlay)")
        _add_image(axs[1, 0], outdir / "figure15_embedded.png", "Figure 15 (original)")
        _add_image(axs[1, 1], outdir / "figure15_percentiles_overlay.png", "Figure 15 (overlay)")
        fig.tight_layout()
        fig.savefig(out.as_posix(), bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")

    if args.mode in ("basic", "all"):
        make_basic()
    if args.mode in ("percentiles15", "all"):
        make_percentiles15()
    if args.mode in ("figure6", "all"):
        make_figure6()
    if args.mode in ("figure11", "all"):
        make_figure11()
    if args.mode in ("overlay11_15", "all"):
        make_overlay11_15()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
