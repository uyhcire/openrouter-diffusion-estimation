#!/usr/bin/env python3
"""
Sync data directory to Google Cloud Storage bucket.

Usage:
    python sync.py          # Sync data/ to GCS
    python sync.py --dry-run  # Preview what would be synced
"""

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
BUCKET = "gs://openrouter-data-scraping"
GSUTIL = "/opt/homebrew/share/google-cloud-sdk/bin/gsutil"


def sync(dry_run: bool = False):
    """Sync data directory to GCS bucket."""
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        sys.exit(1)

    # Build gsutil rsync command
    cmd = [GSUTIL, "-m", "rsync", "-r"]

    if dry_run:
        cmd.append("-n")  # Dry run

    cmd.extend([str(DATA_DIR), BUCKET])

    print(f"{'[DRY RUN] ' if dry_run else ''}Syncing {DATA_DIR} -> {BUCKET}")
    print(f"Command: {' '.join(cmd)}\n")

    # Run sync
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Sync complete!")
    else:
        print(f"\nSync failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    sync(dry_run=dry_run)


if __name__ == "__main__":
    main()
