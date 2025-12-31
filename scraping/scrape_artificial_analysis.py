#!/usr/bin/env python3
"""Scrape Artificial Analysis leaderboard for intelligence scores."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright


LEADERBOARD_URL = "https://artificialanalysis.ai/leaderboards/models"


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
    for match in re.finditer(r'\\"math_index\\":([0-9.]+)', html):
        math_val = float(match.group(1))
        start = max(0, match.start() - 1000)
        end = min(len(html), match.end() + 500)
        window = html[start:end]

        # Find the name in this window
        name_match = re.search(r'\\"name\\":\\"([^"\\]{3,60})\\"', window)
        if not name_match:
            continue

        name = name_match.group(1)
        # Filter out HTML metadata names
        if any(
            x in name.lower()
            for x in ["viewport", "description", "twitter", "size-adjust", "theme"]
        ):
            continue

        # Get mmlu_pro too
        mmlu_match = re.search(r'\\"mmlu_pro\\":([0-9.]+)', window)
        mmlu_val = float(mmlu_match.group(1)) if mmlu_match else None

        # Get coding index
        coding_match = re.search(r'\\"coding_index\\":([0-9.]+)', window)
        coding_val = float(coding_match.group(1)) if coding_match else None

        results.append(
            {
                "name": name,
                "math_index": math_val,
                "mmlu_pro": mmlu_val,
                "coding_index": coding_val,
            }
        )

    # Deduplicate by name
    seen = set()
    unique = []
    for r in results:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique.append(r)

    return unique


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

    print(f"Scraped {len(scores)} models with intelligence scores")
    print(f"Saved to {output_path}")

    # Show top 10 by math_index
    print("\nTop 10 by math_index:")
    for m in sorted(scores, key=lambda x: x.get("math_index", 0), reverse=True)[:10]:
        print(f"  {m['name'][:45]:45} math={m['math_index']:6.2f}")


if __name__ == "__main__":
    main()
