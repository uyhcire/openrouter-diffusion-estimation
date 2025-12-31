#!/usr/bin/env python3
"""Scrape Artificial Analysis leaderboard for intelligence scores.

Fetches all 8 benchmarks used in the Intelligence Index:
- MMLU-Pro, HLE, GPQA Diamond, AIME 2025, SciCode, LiveCodeBench, IFBench, AA-LCR
Plus the composite intelligence_index itself.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright


LEADERBOARD_URL = "https://artificialanalysis.ai/leaderboards/models"

# All benchmark fields to extract (matching paper's Intelligence Index components)
BENCHMARK_FIELDS = [
    "intelligence_index",  # The composite index itself
    "mmlu_pro",            # MMLU-Pro
    "hle",                 # Humanity's Last Exam
    "gpqa",                # GPQA Diamond
    "aime",                # AIME 2025
    "scicode",             # SciCode
    "livecodebench",       # LiveCodeBench
    "ifbench",             # IFBench
    "lcr",                 # AA-LCR (Long Context Reasoning)
    "math_index",          # Math index (for backward compatibility)
    "coding_index",        # Coding index (for backward compatibility)
]


def scrape_intelligence_scores() -> list[dict]:
    """Scrape intelligence scores from Artificial Analysis leaderboard."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(LEADERBOARD_URL, timeout=60000)
        page.wait_for_load_state("networkidle")

        html = page.content()
        browser.close()

    # Extract model data from embedded JSON in HTML
    # The data is escaped with backslashes: \"field\":value
    results = []

    # Find all model entries by looking for name field
    for match in re.finditer(r'\\"name\\":\\"([^"\\]{3,60})\\"', html):
        name = match.group(1)

        # Filter out HTML metadata names
        if any(
            x in name.lower()
            for x in ["viewport", "description", "twitter", "size-adjust", "theme", "og:"]
        ):
            continue

        # Get a window around this name to extract benchmark values
        start = max(0, match.start() - 500)
        end = min(len(html), match.end() + 2000)
        window = html[start:end]

        # Extract all benchmark fields
        model_data = {"name": name}
        has_any_benchmark = False

        for field in BENCHMARK_FIELDS:
            field_match = re.search(rf'\\"{field}\\":([0-9.]+)', window)
            if field_match:
                model_data[field] = float(field_match.group(1))
                has_any_benchmark = True
            else:
                model_data[field] = None

        # Only include models with at least one benchmark
        if has_any_benchmark:
            results.append(model_data)

    # Deduplicate by name, keeping entry with most benchmark data
    seen = {}
    for r in results:
        name = r["name"]
        if name not in seen:
            seen[name] = r
        else:
            # Keep the one with more non-null values
            existing_count = sum(1 for k, v in seen[name].items() if v is not None and k != "name")
            new_count = sum(1 for k, v in r.items() if v is not None and k != "name")
            if new_count > existing_count:
                seen[name] = r

    return list(seen.values())


def main():
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    scraped_at = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Scraping Artificial Analysis at {scraped_at}...")
    scores = scrape_intelligence_scores()

    # Add metadata
    data = {
        "scraped_at": scraped_at,
        "model_count": len(scores),
        "scores": scores,
    }

    output_path = output_dir / f"aa_intelligence_{today}.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Scraped {len(scores)} models with benchmark scores")
    print(f"Saved to {output_path}")

    # Count models with each benchmark
    print("\nBenchmark coverage:")
    for field in BENCHMARK_FIELDS:
        count = sum(1 for m in scores if m.get(field) is not None)
        print(f"  {field:20}: {count:3} models")

    # Show top 10 by intelligence_index
    print("\nTop 10 by intelligence_index:")
    sorted_by_ii = sorted(scores, key=lambda x: x.get("intelligence_index") or 0, reverse=True)
    for m in sorted_by_ii[:10]:
        ii = m.get("intelligence_index")
        ii_str = f"{ii:.2f}" if ii else "N/A"
        print(f"  {m['name'][:45]:45} intelligence_index={ii_str}")


if __name__ == "__main__":
    main()
