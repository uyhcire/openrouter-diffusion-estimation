#!/usr/bin/env python3
"""Scrape OpenRouter rankings page for token usage data."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright


RANKINGS_URL = "https://openrouter.ai/rankings"


def scrape_rankings() -> list[dict]:
    """Scrape token usage rankings from OpenRouter."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(RANKINGS_URL, timeout=30000)
        page.wait_for_load_state("networkidle")

        # Click "Show more" buttons to expand list
        for _ in range(10):
            try:
                buttons = page.locator("button:has-text('Show more')").all()
                for btn in buttons:
                    if btn.is_visible():
                        btn.click()
                        page.wait_for_timeout(300)
            except Exception:
                pass

        html = page.content()
        browser.close()

    # Parse rankings from HTML
    pattern = r'href="(/[a-z0-9-]+/[a-z0-9-:\.]+)"[^>]*>[^<]*</a>.*?([\d.]+)([BMT])\s*tokens'

    rankings = []
    for match in re.finditer(pattern, html, re.DOTALL | re.IGNORECASE):
        href = match.group(1)
        value = float(match.group(2))
        unit = match.group(3).upper()

        if "/apps" in href:
            continue

        if unit == "B":
            tokens = value
        elif unit == "T":
            tokens = value * 1000
        elif unit == "M":
            tokens = value / 1000
        else:
            continue

        model_id = href.lstrip("/")
        if model_id and "/" in model_id:
            rankings.append((model_id, tokens))

    # Deduplicate (keep first occurrence = highest rank)
    seen = set()
    unique = []
    for model_id, tokens in rankings:
        if model_id not in seen:
            seen.add(model_id)
            unique.append({"model_id": model_id, "tokens_billions": tokens})

    return unique


def main():
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    scraped_at = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Scraping OpenRouter rankings at {scraped_at}...")
    rankings = scrape_rankings()

    # Add metadata
    data = {
        "scraped_at": scraped_at,
        "model_count": len(rankings),
        "rankings": rankings,
    }

    output_path = output_dir / f"openrouter_rankings_{today}.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Scraped {len(rankings)} models with token usage")
    print(f"Saved to {output_path}")

    # Show top 10
    print("\nTop 10 by token usage:")
    for r in rankings[:10]:
        print(f"  {r['model_id']:45} {r['tokens_billions']:8.1f}B tokens")


if __name__ == "__main__":
    main()
