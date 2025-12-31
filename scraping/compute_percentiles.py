#!/usr/bin/env python3
"""Compute token-weighted intelligence percentiles from scraped data.

This replicates Figure 15 from the NBER paper: "Token-Weighted Average Intelligence"
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_name(name: str) -> str:
    """Normalize model name for fuzzy matching."""
    return (
        name.lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace(".", " ")
        .replace(":", " ")
    )


def build_model_mapping(
    rankings: list[dict], scores: list[dict]
) -> list[dict]:
    """Match OpenRouter rankings to Artificial Analysis scores."""
    # Build score lookup by normalized name
    score_lookup = {}
    for s in scores:
        norm = normalize_name(s["name"])
        score_lookup[norm] = s
        # Also index by key words
        for word in norm.split():
            if len(word) > 4:
                if word not in score_lookup:
                    score_lookup[word] = s

    matched = []
    for r in rankings:
        model_id = r["model_id"]
        parts = model_id.split("/")
        model_part = parts[1] if len(parts) >= 2 else model_id
        normalized_id = normalize_name(model_part)

        # Try exact match
        if normalized_id in score_lookup:
            score_data = score_lookup[normalized_id]
            matched.append({
                "model_id": model_id,
                "aa_name": score_data["name"],
                "tokens_billions": r["tokens_billions"],
                "math_index": score_data["math_index"],
                "mmlu_pro": score_data.get("mmlu_pro"),
            })
            continue

        # Try word-based partial match
        words = normalized_id.split()
        for word in words:
            if len(word) > 4 and word in score_lookup:
                score_data = score_lookup[word]
                matched.append({
                    "model_id": model_id,
                    "aa_name": score_data["name"],
                    "tokens_billions": r["tokens_billions"],
                    "math_index": score_data["math_index"],
                    "mmlu_pro": score_data.get("mmlu_pro"),
                })
                break

    return matched


def compute_token_weighted_percentiles(
    matched_data: list[dict],
) -> dict[str, float]:
    """Compute token-weighted percentiles of intelligence.

    Returns p10, p25, p50, p75, p90 of the token-weighted distribution.
    """
    if not matched_data:
        return {}

    # Sort by intelligence (math_index)
    sorted_data = sorted(matched_data, key=lambda x: x["math_index"])

    # Compute cumulative token share
    total_tokens = sum(d["tokens_billions"] for d in sorted_data)
    if total_tokens == 0:
        return {}

    cumulative = 0
    percentiles = {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    thresholds = {"p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90}

    for d in sorted_data:
        cumulative += d["tokens_billions"]
        share = cumulative / total_tokens

        for pname, threshold in thresholds.items():
            if percentiles[pname] is None and share >= threshold:
                # Normalize math_index to 0-1 scale (divide by 100)
                percentiles[pname] = d["math_index"] / 100.0

    return percentiles


def main():
    data_dir = Path(__file__).parent / "data"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Load rankings
    rankings_path = data_dir / f"openrouter_rankings_{today}.json"
    if not rankings_path.exists():
        print(f"Rankings file not found: {rankings_path}")
        return

    with open(rankings_path) as f:
        rankings_data = json.load(f)
    rankings = rankings_data["rankings"]

    # Load intelligence scores
    scores_path = data_dir / f"aa_intelligence_{today}.json"
    if not scores_path.exists():
        print(f"Intelligence scores file not found: {scores_path}")
        return

    with open(scores_path) as f:
        scores_data = json.load(f)
    scores = scores_data["scores"]

    print(f"Loaded {len(rankings)} rankings, {len(scores)} intelligence scores")

    # Match models
    matched = build_model_mapping(rankings, scores)
    print(f"Matched {len(matched)}/{len(rankings)} models")

    if not matched:
        print("No matches found!")
        return

    # Compute percentiles
    percentiles = compute_token_weighted_percentiles(matched)

    # Output
    result = {
        "date": today,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "models_matched": len(matched),
        "total_tokens_billions": sum(d["tokens_billions"] for d in matched),
        "percentiles": percentiles,
        "matched_models": matched,
    }

    output_path = data_dir / f"token_weighted_percentiles_{today}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nToken-weighted intelligence percentiles for {today}:")
    for p, val in percentiles.items():
        if val is not None:
            print(f"  {p}: {val:.6f}")

    print(f"\nSaved to {output_path}")

    # Also append to CSV for time series
    csv_path = data_dir / "percentiles_timeseries.csv"
    row = {"date": today, **percentiles}
    df = pd.DataFrame([row])

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        # Remove existing row for today if present
        existing = existing[existing["date"] != today]
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"Appended to {csv_path}")


if __name__ == "__main__":
    main()
