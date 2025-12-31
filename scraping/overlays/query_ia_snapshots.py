#!/usr/bin/env python3
"""
Query Internet Archive CDX API for OpenRouter API snapshots in our date range.
"""

import requests
from datetime import datetime

def main():
    print("Querying Internet Archive for OpenRouter /api/v1/models snapshots...")
    print("Date range: Oct 2, 2025 - Dec 31, 2025\n")

    cdx_url = "https://web.archive.org/cdx/search/cdx"
    params = {
        "url": "openrouter.ai/api/v1/models",
        "output": "json",
        "from": "20251002",  # Oct 2, 2025
        "to": "20251231",    # Dec 31, 2025
        "filter": "statuscode:200",  # Only successful responses
    }

    resp = requests.get(cdx_url, params=params, timeout=60)
    print(f"Status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"Error: {resp.text[:500]}")
        return

    data = resp.json()
    print(f"Total rows: {len(data)}")

    if len(data) <= 1:
        print("No snapshots found in date range.")
        # Let's check what date range IS available
        print("\nChecking all available snapshots...")
        params_all = {
            "url": "openrouter.ai/api/v1/models",
            "output": "json",
            "filter": "statuscode:200",
        }
        resp_all = requests.get(cdx_url, params=params_all, timeout=60)
        if resp_all.status_code == 200:
            all_data = resp_all.json()
            print(f"Total snapshots available: {len(all_data) - 1}")  # -1 for header
            if len(all_data) > 1:
                # Get date range
                timestamps = [row[1] for row in all_data[1:]]
                dates = [datetime.strptime(ts[:8], "%Y%m%d") for ts in timestamps]
                print(f"Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")

                # Count by month
                from collections import Counter
                months = Counter(d.strftime("%Y-%m") for d in dates)
                print("\nSnapshots by month:")
                for month in sorted(months.keys()):
                    print(f"  {month}: {months[month]} snapshots")
        return

    # Header is first row
    header = data[0]
    snapshots = data[1:]

    print(f"Snapshots found: {len(snapshots)}")

    # Parse and display
    for row in snapshots[:20]:  # Show first 20
        timestamp = row[1]
        dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        size = row[6]
        print(f"  {dt.strftime('%Y-%m-%d %H:%M:%S')} - {size} bytes")

    if len(snapshots) > 20:
        print(f"  ... and {len(snapshots) - 20} more")

    # Summary by date
    from collections import Counter
    dates = [datetime.strptime(row[1][:8], "%Y%m%d").strftime("%Y-%m-%d") for row in snapshots]
    date_counts = Counter(dates)
    print(f"\nUnique dates with snapshots: {len(date_counts)}")


if __name__ == "__main__":
    main()
