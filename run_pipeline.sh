#!/usr/bin/env bash
# Full pipeline to regenerate all outputs from the PDF.
# Usage: ./run_pipeline.sh [--pdf PATH] [--outdir PATH]

set -euo pipefail

PDF="${PDF:-LLM_Demand.pdf}"
OUTDIR="${OUTDIR:-out}"
PYTHON="${PYTHON:-.venv/bin/python}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pdf) PDF="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Pipeline configuration ==="
echo "PDF:    $PDF"
echo "OUTDIR: $OUTDIR"
echo "PYTHON: $PYTHON"
echo

mkdir -p "$OUTDIR"

echo "=== Step 1: Base extractions from PDF ==="

echo "[1/6] Figure 5 red median..."
$PYTHON extract_figure15_red_timeseries.py --pdf "$PDF" --figure 5 --curve red_median --outdir "$OUTDIR" --diagnostic

echo "[2/6] Figure 5 p90 frontier..."
$PYTHON extract_figure15_red_timeseries.py --pdf "$PDF" --figure 5 --curve p90_frontier --outdir "$OUTDIR" --diagnostic

echo "[3/6] Figure 6 black frontier..."
$PYTHON extract_figure15_red_timeseries.py --pdf "$PDF" --figure 6 --curve black_frontier --outdir "$OUTDIR" --diagnostic

echo "[4/6] Figure 15 red median..."
$PYTHON extract_figure15_red_timeseries.py --pdf "$PDF" --figure 15 --curve red_median --outdir "$OUTDIR" --diagnostic

echo "[5/6] Figure 15 percentiles..."
$PYTHON extract_figure15_red_timeseries.py --pdf "$PDF" --figure 15 --percentiles --outdir "$OUTDIR"

echo "[6/6] Figure 11 percentiles..."
$PYTHON extract_figure15_red_timeseries.py --pdf "$PDF" --figure 11 --percentiles --outdir "$OUTDIR"

echo
echo "=== Step 2: Verification ==="

echo "[1/2] Verify Figure 15 percentiles..."
$PYTHON verify_figure15_percentiles.py --pdf "$PDF" --outdir "$OUTDIR"

echo "[2/2] Verify Figure 11 percentiles..."
$PYTHON verify_figure11_percentiles.py --pdf "$PDF" --outdir "$OUTDIR"

echo
echo "=== Step 3: Derived outputs ==="

echo "[1/6] Token-weighted model age (fig5 red median)..."
$PYTHON estimate_token_weighted_model_age.py \
    --fig5 "$OUTDIR/figure5_red_median_timeseries.csv" \
    --fig15 "$OUTDIR/figure15_red_median_timeseries.csv" \
    --outdir "$OUTDIR"

echo "[2/6] Token-weighted model age (fig5 p90 frontier)..."
$PYTHON estimate_token_weighted_model_age.py \
    --fig5 "$OUTDIR/figure5_p90_frontier_timeseries.csv" \
    --fig15 "$OUTDIR/figure15_red_median_timeseries.csv" \
    --outdir "$OUTDIR"

echo "[3/6] Token-weighted model age (fig6 black frontier)..."
$PYTHON estimate_token_weighted_model_age.py \
    --fig5 "$OUTDIR/figure6_black_frontier_timeseries.csv" \
    --fig15 "$OUTDIR/figure15_red_median_timeseries.csv" \
    --outdir "$OUTDIR"

echo "[4/6] Dollar-weighted mean capability age..."
$PYTHON plot_dollar_weighted_mean_capability_age.py \
    --fig15 "$OUTDIR/figure15_token_weighted_percentiles.csv" \
    --fig11 "$OUTDIR/figure11_price_to_intelligence_ratio_percentiles.csv" \
    --frontier "$OUTDIR/figure6_black_frontier_timeseries.csv" \
    --outdir "$OUTDIR"

echo "[5/6] Capability age robustness checks..."
$PYTHON plot_dollar_weighted_mean_capability_age.py \
    --fig15 "$OUTDIR/figure15_token_weighted_percentiles.csv" \
    --fig11 "$OUTDIR/figure11_price_to_intelligence_ratio_percentiles.csv" \
    --frontier "$OUTDIR/figure6_black_frontier_timeseries.csv" \
    --outdir "$OUTDIR" \
    --robustness

echo "[6/6] Model age scatterplot..."
$PYTHON make_model_age_scatterplot.py \
    --figure5 "$OUTDIR/figure5_red_median_timeseries.csv" \
    --figure15 "$OUTDIR/figure15_red_median_timeseries.csv" \
    --outdir "$OUTDIR"

echo
echo "=== Step 4: Comparison grids ==="
$PYTHON make_figure_comparison_grid.py --outdir "$OUTDIR" --mode all

echo
echo "=== Pipeline complete ==="
echo "Outputs written to: $OUTDIR/"
