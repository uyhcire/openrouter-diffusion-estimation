#!/usr/bin/env python3
"""
Scrape daily token usage from OpenRouter model activity pages.

Resumable: saves each model immediately, skips already-scraped models.
"""

import json
import re
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright

# Output directory for individual model files
OUTPUT_DIR = Path(__file__).parent / "data" / "model_activity"
MODELS_FILE = Path(__file__).parent / "data" / "openrouter_pricing_2025-12-31.json"


def get_scraped_models() -> set[str]:
    """Return set of model IDs that have already been scraped today."""
    if not OUTPUT_DIR.exists():
        return set()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    scraped = set()

    for f in OUTPUT_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                # Only count as scraped if it was scraped today
                if data.get("scraped_date") == today:
                    scraped.add(data.get("model_id"))
        except (json.JSONDecodeError, KeyError):
            pass

    return scraped


def model_id_to_filename(model_id: str) -> str:
    """Convert model ID to safe filename."""
    return model_id.replace("/", "__").replace(":", "_") + ".json"


def scrape_model_activity(page, model_id: str) -> dict | None:
    """Scrape activity data for a single model."""
    url = f"https://openrouter.ai/{model_id}/activity"

    try:
        page.goto(url, wait_until="networkidle", timeout=30000)

        # Scroll to trigger lazy loading of historical data
        for _ in range(5):
            page.evaluate("window.scrollBy(0, 1000)")
            page.wait_for_timeout(300)

        html = page.content()

        # Extract daily records from embedded JSON
        # Format: "date":"2025-12-31 00:00:00","model_permaslug":"...","variant":"standard","total_completion_tokens":N,"total_prompt_tokens":N
        pattern = r'date\\":\\"(2025-\d{2}-\d{2})[^"]*\\",\\"model_permaslug\\":\\"([^\\]+)\\",\\"variant\\":\\"standard\\",\\"total_completion_tokens\\":(\d+),\\"total_prompt_tokens\\":(\d+)'

        matches = re.findall(pattern, html)

        if not matches:
            return None

        # Aggregate by date (there may be multiple entries per date for different variants)
        daily_data = {}
        for date, _, completion, prompt in matches:
            if date not in daily_data:
                daily_data[date] = {"prompt_tokens": 0, "completion_tokens": 0}
            daily_data[date]["prompt_tokens"] += int(prompt)
            daily_data[date]["completion_tokens"] += int(completion)

        # Convert to sorted list
        daily_list = [
            {
                "date": date,
                "prompt_tokens": data["prompt_tokens"],
                "completion_tokens": data["completion_tokens"],
                "total_tokens": data["prompt_tokens"] + data["completion_tokens"],
            }
            for date, data in sorted(daily_data.items())
        ]

        return {
            "model_id": model_id,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "days_of_data": len(daily_list),
            "date_range": {
                "start": daily_list[0]["date"] if daily_list else None,
                "end": daily_list[-1]["date"] if daily_list else None,
            },
            "daily_usage": daily_list,
        }

    except Exception as e:
        print(f"    Error: {e}")
        return None


def save_model_data(data: dict) -> None:
    """Save model data to individual JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = model_id_to_filename(data["model_id"])
    filepath = OUTPUT_DIR / filename

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_all_models() -> list[str]:
    """Load list of all model IDs from the models API data."""
    if not MODELS_FILE.exists():
        raise FileNotFoundError(f"Models file not found: {MODELS_FILE}")

    with open(MODELS_FILE) as f:
        data = json.load(f)

    # Handle both list format and dict format
    if isinstance(data, list):
        return [model["model_id"] for model in data]
    else:
        return [model["id"] for model in data.get("models", [])]


def main():
    """Main scraping loop with resume support."""
    print("=" * 60)
    print("OpenRouter Model Activity Scraper (Resumable)")
    print("=" * 60)

    # Load all models
    all_models = load_all_models()
    print(f"\nTotal models in API: {len(all_models)}")

    # Check which are already scraped
    already_scraped = get_scraped_models()
    print(f"Already scraped today: {len(already_scraped)}")

    # Models to scrape
    to_scrape = [m for m in all_models if m not in already_scraped]
    print(f"Remaining to scrape: {len(to_scrape)}")

    if not to_scrape:
        print("\nAll models already scraped today!")
        return

    print(f"\nStarting scrape of {len(to_scrape)} models...")
    print("Progress will be saved after each model.\n")

    successful = 0
    failed = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for i, model_id in enumerate(to_scrape, 1):
            print(f"[{i}/{len(to_scrape)}] {model_id}...", end=" ", flush=True)

            start_time = time.time()
            data = scrape_model_activity(page, model_id)
            elapsed = time.time() - start_time

            if data:
                save_model_data(data)
                print(f"{data['days_of_data']} days ({elapsed:.1f}s)")
                successful += 1
            else:
                # Save a marker so we don't retry failed models repeatedly
                save_model_data({
                    "model_id": model_id,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "days_of_data": 0,
                    "error": "No activity data found",
                    "daily_usage": [],
                })
                print(f"no data ({elapsed:.1f}s)")
                failed += 1

            # Small delay to be nice to the server
            time.sleep(0.5)

        browser.close()

    print("\n" + "=" * 60)
    print(f"Scraping complete!")
    print(f"  Successful: {successful}")
    print(f"  No data: {failed}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
