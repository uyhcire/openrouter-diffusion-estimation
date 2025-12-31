#!/usr/bin/env python3
"""
Create visualization comparing scraped percentiles with paper's Figure 15.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / "data"
PAPER_FILE = Path(__file__).parent.parent / "out" / "figure15_token_weighted_percentiles.csv"
SCRAPED_FILE = DATA_DIR / "historical_percentiles.csv"
OUTPUT_FILE = DATA_DIR / "figure15_comparison.png"


def main():
    # Load data
    paper = pd.read_csv(PAPER_FILE, parse_dates=["date"])
    scraped = pd.read_csv(SCRAPED_FILE, parse_dates=["date"])

    print(f"Paper data: {len(paper)} days ({paper['date'].min().date()} to {paper['date'].max().date()})")
    print(f"Scraped data: {len(scraped)} days ({scraped['date'].min().date()} to {scraped['date'].max().date()})")

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Colors for percentile bands
    colors = {
        "p90": "#1f77b4",
        "p75": "#2ca02c",
        "p50": "#d62728",
        "p25": "#9467bd",
        "p10": "#8c564b",
    }

    # Plot 1: Paper's data
    ax1 = axes[0]
    for pct in ["p10", "p25", "p50", "p75", "p90"]:
        ax1.plot(paper["date"], paper[pct], label=pct, color=colors[pct], linewidth=1.5)

    ax1.set_ylabel("Intelligence Index", fontsize=12)
    ax1.set_title("Paper's Figure 15: Token-Weighted Intelligence Percentiles (Jan-Sep 2025)", fontsize=14)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot 2: Scraped data
    ax2 = axes[1]
    for pct in ["p10", "p25", "p50", "p75", "p90"]:
        ax2.plot(scraped["date"], scraped[pct], label=pct, color=colors[pct], linewidth=1.5)

    ax2.set_ylabel("Intelligence Index", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_title("Scraped Data: Token-Weighted Intelligence Percentiles (Oct-Dec 2025)", fontsize=14)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_FILE}")

    # Create overlay comparison
    fig2, ax = plt.subplots(figsize=(14, 7))

    # Plot paper data (solid lines)
    for pct in ["p10", "p25", "p50", "p75", "p90"]:
        ax.plot(paper["date"], paper[pct], color=colors[pct], linewidth=1.5,
                label=f"{pct} (paper)" if pct == "p50" else None)

    # Plot scraped data (dashed lines)
    for pct in ["p10", "p25", "p50", "p75", "p90"]:
        ax.plot(scraped["date"], scraped[pct], color=colors[pct], linewidth=2,
                linestyle="--", label=f"{pct} (scraped)" if pct == "p50" else None)

    ax.set_ylabel("Intelligence Index", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("Figure 15 Comparison: Paper (solid) vs Scraped (dashed)", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    overlay_file = DATA_DIR / "figure15_overlay.png"
    plt.tight_layout()
    plt.savefig(overlay_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {overlay_file}")

    # Print comparison at overlap point
    paper_end = paper.iloc[-1]
    scraped_start = scraped.iloc[0]

    print("\n" + "=" * 60)
    print("Comparison at transition point:")
    print("=" * 60)
    print(f"Paper end ({paper_end['date'].date()}):")
    for pct in ["p10", "p25", "p50", "p75", "p90"]:
        print(f"  {pct}: {paper_end[pct]:.3f}")

    print(f"\nScraped start ({scraped_start['date'].date()}):")
    for pct in ["p10", "p25", "p50", "p75", "p90"]:
        print(f"  {pct}: {scraped_start[pct]:.3f}")

    print("\nDifference (scraped - paper):")
    for pct in ["p10", "p25", "p50", "p75", "p90"]:
        diff = scraped_start[pct] - paper_end[pct]
        print(f"  {pct}: {diff:+.3f} ({100*diff/paper_end[pct]:+.1f}%)")


if __name__ == "__main__":
    main()
