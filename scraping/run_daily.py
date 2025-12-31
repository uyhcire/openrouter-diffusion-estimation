#!/usr/bin/env python3
"""Run all daily scraping tasks and compute percentiles.

Usage:
    python run_daily.py

This script:
1. Scrapes OpenRouter models API (pricing data)
2. Scrapes OpenRouter rankings page (token usage)
3. Scrapes Artificial Analysis leaderboard (intelligence scores)
4. Computes token-weighted intelligence percentiles
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_script(script_name: str) -> bool:
    """Run a Python script and return success status."""
    script_path = Path(__file__).parent / script_name
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"ERROR: {script_name} failed with code {result.returncode}")
        return False
    return True


def main():
    start_time = datetime.now(timezone.utc)
    print(f"Starting daily scrape at {start_time.isoformat()}")

    scripts = [
        "scrape_openrouter_models.py",
        "scrape_openrouter_rankings.py",
        "scrape_artificial_analysis.py",
        "compute_percentiles.py",
    ]

    results = {}
    for script in scripts:
        results[script] = run_script(script)

    # Summary
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Duration: {duration:.1f} seconds")

    all_success = True
    for script, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {script}: {status}")
        if not success:
            all_success = False

    if all_success:
        print("\nAll tasks completed successfully!")
    else:
        print("\nSome tasks failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
