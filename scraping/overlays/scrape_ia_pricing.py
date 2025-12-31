#!/usr/bin/env python3
"""
Scrape historical OpenRouter pricing from Internet Archive.
Downloads snapshots and extracts pricing data.
"""

import json
import requests
import time
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "ia_snapshots"


def get_snapshot_list(from_date: str, to_date: str) -> list[dict]:
    """Query CDX API for available snapshots."""
    print(f"Querying CDX API for snapshots from {from_date} to {to_date}...")

    cdx_url = "https://web.archive.org/cdx/search/cdx"
    params = {
        "url": "openrouter.ai/api/v1/models",
        "output": "json",
        "from": from_date,
        "to": to_date,
        "filter": "statuscode:200",
    }

    resp = requests.get(cdx_url, params=params, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    if len(data) <= 1:
        return []

    # Parse into list of dicts
    header = data[0]
    snapshots = []
    for row in data[1:]:
        snap = dict(zip(header, row))
        snap["datetime"] = datetime.strptime(snap["timestamp"], "%Y%m%d%H%M%S")
        snap["date"] = snap["datetime"].strftime("%Y-%m-%d")
        snap["wayback_url"] = f"https://web.archive.org/web/{snap['timestamp']}id_/https://openrouter.ai/api/v1/models"
        snapshots.append(snap)

    return snapshots


def download_snapshot(snapshot: dict) -> dict | None:
    """Download a single snapshot and parse JSON."""
    url = snapshot["wayback_url"]
    print(f"  Downloading {snapshot['date']} ({snapshot['timestamp']})...")

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        # The response should be JSON
        data = resp.json()
        return data
    except requests.RequestException as e:
        print(f"    Error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"    JSON decode error: {e}")
        return None


def extract_pricing(api_response: dict, snapshot_date: str) -> list[dict]:
    """Extract pricing data from OpenRouter API response."""
    models = api_response.get("data", [])
    pricing_list = []

    for model in models:
        model_id = model.get("id")
        if not model_id:
            continue

        pricing = model.get("pricing", {})
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")

        if prompt_price is None or completion_price is None:
            continue

        try:
            pricing_list.append({
                "date": snapshot_date,
                "model_id": model_id,
                "prompt_price": float(prompt_price),
                "completion_price": float(completion_price),
                "created": model.get("created"),
            })
        except (ValueError, TypeError):
            continue

    return pricing_list


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Get snapshots for our date range
    snapshots = get_snapshot_list("20251002", "20251231")
    print(f"Found {len(snapshots)} snapshots\n")

    if not snapshots:
        print("No snapshots found!")
        return

    # Deduplicate by date (keep first per day)
    seen_dates = set()
    unique_snapshots = []
    for snap in snapshots:
        if snap["date"] not in seen_dates:
            seen_dates.add(snap["date"])
            unique_snapshots.append(snap)

    print(f"Unique dates: {len(unique_snapshots)}\n")

    # Download each snapshot
    all_pricing = []
    for snap in unique_snapshots:
        data = download_snapshot(snap)
        if data:
            pricing = extract_pricing(data, snap["date"])
            print(f"    Extracted pricing for {len(pricing)} models")
            all_pricing.extend(pricing)

            # Save raw snapshot
            snapshot_file = OUTPUT_DIR / f"openrouter_{snap['date']}.json"
            with open(snapshot_file, "w") as f:
                json.dump(data, f)

        # Be nice to Internet Archive
        time.sleep(1)

    # Save combined pricing data
    output_file = OUTPUT_DIR / "historical_pricing.json"
    with open(output_file, "w") as f:
        json.dump(all_pricing, f, indent=2)

    print(f"\nSaved {len(all_pricing)} pricing records to {output_file}")

    # Summary
    dates = sorted(set(p["date"] for p in all_pricing))
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Unique dates: {len(dates)}")


if __name__ == "__main__":
    main()
