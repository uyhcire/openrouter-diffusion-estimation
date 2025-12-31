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
    ap = argparse.ArgumentParser(description="Create a 2x2 grid comparing original figures vs extracted red-curve plots.")
    ap.add_argument("--outdir", type=Path, default=Path("out"))
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path (default: out/figure5_figure15_comparison.png)")
    args = ap.parse_args()

    outdir = args.outdir
    out = args.out or (outdir / "figure5_figure15_comparison.png")

    fig = plt.figure(figsize=(14, 10), dpi=200)
    axs = fig.subplots(2, 2)

    _add_image(axs[0, 0], outdir / "figure5_embedded.png", "Figure 5 (original)")
    _add_image(axs[0, 1], outdir / "figure5_red_median_timeseries.png", "Figure 5 (extracted red median)")
    _add_image(axs[1, 0], outdir / "figure15_embedded.png", "Figure 15 (original)")
    _add_image(axs[1, 1], outdir / "figure15_red_median_timeseries.png", "Figure 15 (extracted red median)")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.as_posix(), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

