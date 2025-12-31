# Figure 15 red-line time series extraction

This repo contains a small script to extract the **red median curve** in **Figure 15** from `w34608.pdf` into a calibrated daily time series.

Note: in this PDF, the plot itself is embedded as a raster image object on the page (not vector strokes), so the script extracts that embedded image stream from the PDF and then digitizes the red curve.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install pymupdf opencv-python-headless numpy matplotlib

python extract_figure15_red_timeseries.py --pdf w34608.pdf --outdir out
```

## Outputs

- `out/figure15_red_median_timeseries.csv` (365 daily points for 2025)
- `out/figure15_red_median_timeseries.png` (quick plot sanity check)
- `out/figure15_debug_overlay.png` (detected gridlines overlayed on the extracted figure image)
- `out/figure15_extraction_meta.txt` (calibration parameters)
