#!/usr/bin/env python3
"""
Create two-pane comparison plots for Figures 5, 11, and 15.

Left pane: Original paper data
Right pane: Paper data (before Oct 2) + Scraped data (Oct 2+)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
PAPER_DATA = REPO_ROOT / "out"
SCRAPING_DATA = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

# Splice date - where we switch from paper to scraped data
SPLICE_DATE = "2025-10-02"

# Color scheme for percentiles
COLORS = {
    "p10": "#8c564b",  # brown
    "p25": "#9467bd",  # purple
    "p50": "#d62728",  # red
    "p75": "#2ca02c",  # green
    "p90": "#1f77b4",  # blue
}


def load_figure15_data():
    """Load Figure 15 data (intelligence percentiles)."""
    paper = pd.read_csv(PAPER_DATA / "figure15_token_weighted_percentiles.csv", parse_dates=["date"])
    scraped = pd.read_csv(SCRAPING_DATA / "historical_percentiles.csv", parse_dates=["date"])
    # Apply 7-day rolling average to smooth out daily volatility
    cols_to_smooth = ["p10", "p25", "p50", "p75", "p90"]
    scraped[cols_to_smooth] = scraped[cols_to_smooth].rolling(window=7, min_periods=1).mean()
    return paper, scraped


def load_figure11_data():
    """Load Figure 11 data (price-to-intelligence ratios)."""
    paper = pd.read_csv(PAPER_DATA / "figure11_price_to_intelligence_ratio_percentiles.csv", parse_dates=["date"])
    scraped = pd.read_csv(OUTPUT_DIR / "price_ratio_percentiles.csv", parse_dates=["date"])
    return paper, scraped


def load_figure5_data():
    """Load Figure 5 data (P90 frontier)."""
    paper = pd.read_csv(PAPER_DATA / "figure5_p90_frontier_timeseries.csv", parse_dates=["date"])
    scraped = pd.read_csv(SCRAPING_DATA / "historical_percentiles.csv", parse_dates=["date"])
    # Apply 7-day rolling average to smooth out daily volatility
    scraped["p90"] = scraped["p90"].rolling(window=7, min_periods=1).mean()
    # Extract just p90 from scraped and rename column
    scraped = scraped[["date", "p90"]].rename(columns={"p90": "intelligence_index"})
    return paper, scraped


def splice_data(paper: pd.DataFrame, scraped: pd.DataFrame, splice_date: str, columns: list[str]) -> pd.DataFrame:
    """
    Create spliced dataset: paper data before splice_date, scraped data after.
    """
    splice_dt = pd.to_datetime(splice_date)

    # Paper data before splice date
    paper_before = paper[paper["date"] < splice_dt].copy()

    # Scraped data from splice date onwards
    scraped_after = scraped[scraped["date"] >= splice_dt].copy()

    # Ensure same columns
    paper_before = paper_before[["date"] + columns]
    scraped_after = scraped_after[["date"] + columns]

    # Concatenate
    spliced = pd.concat([paper_before, scraped_after], ignore_index=True)
    spliced = spliced.sort_values("date")

    return spliced


def plot_two_pane_percentiles(paper: pd.DataFrame, spliced: pd.DataFrame, title: str,
                               output_file: Path, percentile_cols: list[str],
                               log_scale: bool = False, y_label: str = "Value"):
    """Create two-pane comparison plot for percentile data."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    splice_dt = pd.to_datetime(SPLICE_DATE)

    # Left pane: Paper data only
    ax1 = axes[0]
    for col in percentile_cols:
        ax1.plot(paper["date"], paper[col], color=COLORS[col], linewidth=1.5, label=col)

    ax1.set_title("Paper Data (Original)", fontsize=14)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel(y_label, fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Right pane: Spliced data (paper before Oct 2 + scraped after)
    ax2 = axes[1]
    for col in percentile_cols:
        ax2.plot(spliced["date"], spliced[col], color=COLORS[col], linewidth=1.5, label=col)

    # Add vertical line at splice point
    ax2.axvline(x=splice_dt, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.text(splice_dt, ax2.get_ylim()[1] * 0.95, " Oct 2\n (scraped→)",
             fontsize=9, ha="left", va="top")

    ax2.set_title("Paper + Scraped (Oct 2 onwards)", fontsize=14)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    if log_scale:
        ax1.set_yscale("log")
        ax2.set_yscale("log")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


def plot_two_pane_single_line(paper: pd.DataFrame, spliced: pd.DataFrame, title: str,
                               output_file: Path, value_col: str, y_label: str = "Value"):
    """Create two-pane comparison plot for single-line data (like Figure 5)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    splice_dt = pd.to_datetime(SPLICE_DATE)

    # Left pane: Paper data only
    ax1 = axes[0]
    ax1.plot(paper["date"], paper[value_col], color=COLORS["p90"], linewidth=2)
    ax1.set_title("Paper Data (Original)", fontsize=14)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel(y_label, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Right pane: Spliced data
    ax2 = axes[1]
    ax2.plot(spliced["date"], spliced[value_col], color=COLORS["p90"], linewidth=2)

    # Add vertical line at splice point
    ax2.axvline(x=splice_dt, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.text(splice_dt, ax2.get_ylim()[1] * 0.95, " Oct 2\n (scraped→)",
             fontsize=9, ha="left", va="top")

    ax2.set_title("Paper + Scraped (Oct 2 onwards)", fontsize=14)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


def compute_consistency_metrics(paper: pd.DataFrame, scraped: pd.DataFrame,
                                 columns: list[str], splice_date: str) -> dict:
    """Compute consistency metrics for the overlap period."""
    splice_dt = pd.to_datetime(splice_date)

    # Get overlap period from paper
    paper_overlap = paper[paper["date"] >= splice_dt].copy()

    # Merge on date
    merged = paper_overlap.merge(scraped, on="date", suffixes=("_paper", "_scraped"))

    metrics = {}
    for col in columns:
        paper_col = f"{col}_paper"
        scraped_col = f"{col}_scraped"

        if paper_col not in merged.columns or scraped_col not in merged.columns:
            continue

        diff = merged[scraped_col] - merged[paper_col]
        pct_diff = (diff / merged[paper_col]) * 100

        metrics[col] = {
            "mean_abs_diff": abs(diff).mean(),
            "mean_pct_diff": pct_diff.mean(),
            "max_abs_diff": abs(diff).max(),
            "correlation": merged[paper_col].corr(merged[scraped_col]),
            "n_days": len(merged),
        }

    return metrics


def main():
    print("=" * 60)
    print("Creating Two-Pane Comparison Plots")
    print("=" * 60)

    percentile_cols = ["p10", "p25", "p50", "p75", "p90"]

    # Figure 15: Token-weighted intelligence percentiles
    print("\n1. Figure 15: Token-Weighted Intelligence Percentiles")
    paper15, scraped15 = load_figure15_data()
    spliced15 = splice_data(paper15, scraped15, SPLICE_DATE, percentile_cols)
    plot_two_pane_percentiles(
        paper15, spliced15,
        "Figure 15: Token-Weighted Intelligence Percentiles",
        OUTPUT_DIR / "figure15_comparison.png",
        percentile_cols,
        y_label="Intelligence Index"
    )

    # Figure 11: Price-to-Intelligence ratios
    print("\n2. Figure 11: Price-to-Intelligence Ratio")
    paper11, scraped11 = load_figure11_data()
    spliced11 = splice_data(paper11, scraped11, SPLICE_DATE, percentile_cols)
    plot_two_pane_percentiles(
        paper11, spliced11,
        "Figure 11: Price-to-Intelligence Ratio Percentiles",
        OUTPUT_DIR / "figure11_comparison.png",
        percentile_cols,
        log_scale=True,
        y_label="$/M-tokens per Intelligence Point"
    )

    # Figure 5: P90 Frontier
    print("\n3. Figure 5: P90 Frontier")
    paper5, scraped5 = load_figure5_data()
    spliced5 = splice_data(paper5, scraped5, SPLICE_DATE, ["intelligence_index"])
    plot_two_pane_single_line(
        paper5, spliced5,
        "Figure 5: P90 Frontier Intelligence",
        OUTPUT_DIR / "figure5_comparison.png",
        "intelligence_index",
        y_label="Intelligence Index"
    )

    # Compute and save consistency metrics
    print("\n4. Computing consistency metrics...")
    report_lines = ["# Consistency Report: Paper vs Scraped Data", ""]

    # Figure 15 metrics
    metrics15 = compute_consistency_metrics(paper15, scraped15, percentile_cols, SPLICE_DATE)
    report_lines.append("## Figure 15: Token-Weighted Intelligence Percentiles")
    report_lines.append("")
    report_lines.append("| Percentile | Mean Abs Diff | Mean % Diff | Max Abs Diff | Correlation | Days |")
    report_lines.append("|------------|---------------|-------------|--------------|-------------|------|")
    for col in percentile_cols:
        if col in metrics15:
            m = metrics15[col]
            report_lines.append(f"| {col} | {m['mean_abs_diff']:.4f} | {m['mean_pct_diff']:+.1f}% | {m['max_abs_diff']:.4f} | {m['correlation']:.3f} | {m['n_days']} |")
    report_lines.append("")

    # Figure 11 metrics
    metrics11 = compute_consistency_metrics(paper11, scraped11, percentile_cols, SPLICE_DATE)
    report_lines.append("## Figure 11: Price-to-Intelligence Ratio")
    report_lines.append("")
    report_lines.append("| Percentile | Mean Abs Diff | Mean % Diff | Max Abs Diff | Correlation | Days |")
    report_lines.append("|------------|---------------|-------------|--------------|-------------|------|")
    for col in percentile_cols:
        if col in metrics11:
            m = metrics11[col]
            report_lines.append(f"| {col} | {m['mean_abs_diff']:.4f} | {m['mean_pct_diff']:+.1f}% | {m['max_abs_diff']:.4f} | {m['correlation']:.3f} | {m['n_days']} |")
    report_lines.append("")

    # Figure 5 metrics
    metrics5 = compute_consistency_metrics(paper5, scraped5, ["intelligence_index"], SPLICE_DATE)
    report_lines.append("## Figure 5: P90 Frontier")
    report_lines.append("")
    if "intelligence_index" in metrics5:
        m = metrics5["intelligence_index"]
        report_lines.append(f"- Mean absolute difference: {m['mean_abs_diff']:.4f}")
        report_lines.append(f"- Mean % difference: {m['mean_pct_diff']:+.1f}%")
        report_lines.append(f"- Max absolute difference: {m['max_abs_diff']:.4f}")
        report_lines.append(f"- Correlation: {m['correlation']:.3f}")
        report_lines.append(f"- Overlap days: {m['n_days']}")

    # Save report
    report_file = OUTPUT_DIR / "consistency_report.txt"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"   Saved: {report_file}")

    print("\n" + "=" * 60)
    print("Done! Generated:")
    print("  - figure5_comparison.png")
    print("  - figure11_comparison.png")
    print("  - figure15_comparison.png")
    print("  - consistency_report.txt")


if __name__ == "__main__":
    main()
