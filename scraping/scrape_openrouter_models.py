#!/usr/bin/env python3
"""Scrape OpenRouter models API and save daily snapshot."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import requests


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"


def fetch_models(api_key: str | None = None) -> dict:
    """Fetch all models from OpenRouter API."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(OPENROUTER_API_URL, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_pricing_records(data: dict, scraped_at: str) -> list[dict]:
    """Extract normalized pricing records from API response."""
    records = []
    for model in data.get("data", []):
        pricing = model.get("pricing", {})
        records.append({
            "scraped_at": scraped_at,
            "model_id": model.get("id"),
            "name": model.get("name"),
            "created": model.get("created"),
            "prompt_price_usd": float(pricing.get("prompt", 0)),
            "completion_price_usd": float(pricing.get("completion", 0)),
            "context_length": model.get("context_length"),
            "modality": model.get("architecture", {}).get("modality"),
        })
    return records


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    scraped_at = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Fetching OpenRouter models at {scraped_at}...")
    data = fetch_models(api_key)

    # Save raw snapshot
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    snapshot_path = snapshots_dir / f"openrouter_models_{today}.json"
    with open(snapshot_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved raw snapshot to {snapshot_path}")

    # Extract and save pricing records
    records = extract_pricing_records(data, scraped_at)
    records_path = output_dir / f"openrouter_pricing_{today}.json"
    with open(records_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Extracted {len(records)} models, saved to {records_path}")


if __name__ == "__main__":
    main()
