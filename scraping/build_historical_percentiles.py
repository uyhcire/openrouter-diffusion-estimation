#!/usr/bin/env python3
"""
Build historical token-weighted intelligence percentiles from scraped model activity data.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Paths
DATA_DIR = Path(__file__).parent / "data"
ACTIVITY_DIR = DATA_DIR / "model_activity"
AA_FILE = DATA_DIR / "aa_intelligence_2025-12-31.json"
OUTPUT_FILE = DATA_DIR / "historical_percentiles.csv"
OUTPUT_JSON = DATA_DIR / "historical_percentiles.json"


def load_activity_data() -> dict[str, dict[str, int]]:
    """Load all model activity data. Returns {model_id: {date: total_tokens}}"""
    activity = {}

    for f in ACTIVITY_DIR.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)

        model_id = data.get("model_id")
        if not model_id or not data.get("daily_usage"):
            continue

        activity[model_id] = {
            day["date"]: day["total_tokens"]
            for day in data["daily_usage"]
        }

    return activity


def load_intelligence_scores() -> dict[str, float]:
    """Load intelligence scores from Artificial Analysis. Returns {aa_name: math_index}"""
    with open(AA_FILE) as f:
        data = json.load(f)

    # Build mapping from AA name to score
    aa_scores = {}
    for model in data.get("scores", []):
        name = model.get("name", "").lower()
        score = model.get("math_index")
        if name and score is not None:
            aa_scores[name] = score / 100  # Normalize to 0-1

    return aa_scores


def fuzzy_match_model(model_id: str, aa_scores: dict[str, float]) -> tuple[str, float] | None:
    """Try to match OpenRouter model_id to AA score."""
    # Extract model name parts
    parts = model_id.lower().replace("/", " ").replace("-", " ").replace(":", " ").replace("_", " ").split()

    best_match = None
    best_score = 0

    for aa_name, score in aa_scores.items():
        aa_parts = set(aa_name.replace("-", " ").replace("_", " ").split())

        # Count matching words
        matches = sum(1 for p in parts if p in aa_parts or any(p in ap for ap in aa_parts))

        if matches > best_score:
            best_score = matches
            best_match = (aa_name, score)

    # Require at least 2 matching parts
    if best_score >= 2:
        return best_match
    return None


def compute_percentiles(tokens: list[float], scores: list[float], percentiles: list[int]) -> dict[str, float]:
    """Compute token-weighted percentiles."""
    if not tokens or not scores:
        return {f"p{p}": None for p in percentiles}

    # Sort by score
    sorted_pairs = sorted(zip(scores, tokens))
    sorted_scores = [s for s, t in sorted_pairs]
    sorted_tokens = [t for s, t in sorted_pairs]

    total_tokens = sum(sorted_tokens)
    if total_tokens == 0:
        return {f"p{p}": None for p in percentiles}

    # Compute cumulative token share
    cumsum = np.cumsum(sorted_tokens) / total_tokens

    result = {}
    for p in percentiles:
        threshold = p / 100
        # Find first index where cumsum >= threshold
        idx = np.searchsorted(cumsum, threshold)
        idx = min(idx, len(sorted_scores) - 1)
        result[f"p{p}"] = sorted_scores[idx]

    return result


def main():
    print("Building historical token-weighted percentiles...")

    # Load data
    print("\n1. Loading activity data...")
    activity = load_activity_data()
    print(f"   Loaded {len(activity)} models with activity data")

    print("\n2. Loading intelligence scores...")
    aa_scores = load_intelligence_scores()
    print(f"   Loaded {len(aa_scores)} models with intelligence scores")

    # Match models
    print("\n3. Matching models to intelligence scores...")
    matched = {}
    for model_id in activity:
        match = fuzzy_match_model(model_id, aa_scores)
        if match:
            matched[model_id] = match[1]  # Store the score

    print(f"   Matched {len(matched)} of {len(activity)} models ({100*len(matched)/len(activity):.1f}%)")

    # Get all unique dates
    all_dates = set()
    for model_usage in activity.values():
        all_dates.update(model_usage.keys())
    all_dates = sorted(all_dates)

    print(f"\n4. Computing percentiles for {len(all_dates)} dates...")
    print(f"   Date range: {all_dates[0]} to {all_dates[-1]}")

    # Compute percentiles for each date
    percentile_list = [10, 25, 50, 75, 90]
    results = []

    for date in all_dates:
        # Get tokens and scores for this date
        tokens = []
        scores = []

        for model_id, score in matched.items():
            daily_tokens = activity[model_id].get(date, 0)
            if daily_tokens > 0:
                tokens.append(daily_tokens)
                scores.append(score)

        if tokens:
            pcts = compute_percentiles(tokens, scores, percentile_list)
            pcts["date"] = date
            pcts["models_with_data"] = len(tokens)
            pcts["total_tokens"] = sum(tokens)
            results.append(pcts)

    # Save as CSV
    print(f"\n5. Saving results...")
    with open(OUTPUT_FILE, "w") as f:
        headers = ["date", "p10", "p25", "p50", "p75", "p90", "models_with_data", "total_tokens"]
        f.write(",".join(headers) + "\n")
        for row in results:
            values = [str(row.get(h, "")) for h in headers]
            f.write(",".join(values) + "\n")

    print(f"   Saved CSV: {OUTPUT_FILE}")

    # Save as JSON with metadata
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "models_matched": len(matched),
            "models_with_activity": len(activity),
            "date_range": {"start": all_dates[0], "end": all_dates[-1]},
            "percentiles": results
        }, f, indent=2)

    print(f"   Saved JSON: {OUTPUT_JSON}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary of latest percentiles:")
    print("=" * 60)
    latest = results[-1]
    print(f"Date: {latest['date']}")
    print(f"Models with data: {latest['models_with_data']}")
    print(f"Total tokens: {latest['total_tokens']/1e9:.1f}B")
    for p in percentile_list:
        print(f"  p{p}: {latest[f'p{p}']:.3f}")


if __name__ == "__main__":
    main()
