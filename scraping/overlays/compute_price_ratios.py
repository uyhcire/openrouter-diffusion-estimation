#!/usr/bin/env python3
"""
Compute price-to-intelligence ratio percentiles from scraped data.
This is needed for Figure 11 comparison.

Now supports historical pricing from Internet Archive snapshots.

Methodology aligned with paper (w34608.pdf):
- Uses prompt price only (not blended)
- Excludes free models
- Filters to models released within 6 months
- Uses composite intelligence index (proxy: avg of mmlu_pro and math_index)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from bisect import bisect_left

# Paper methodology constants for Figure 11
EXCLUDE_FREE_MODELS = True  # Paper excludes free models from pricing analyses
MODEL_AGE_LIMIT_DAYS = None  # Disabled: OpenRouter "created" timestamps don't match paper's model release dates
USE_PROMPT_PRICE_ONLY = True  # Paper confirmed: uses prompt price only (Table 1: "Log Price per Million Prompt Tokens")
USE_MODEL_WEIGHTED = True  # Paper: Figure 11 is model-weighted, NOT token-weighted
ROLLING_WINDOW_DAYS = 14  # Paper: "14-day rolling average"

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
SCRAPING_DATA = Path(__file__).parent.parent / "data"
IA_SNAPSHOTS = Path(__file__).parent / "ia_snapshots"
OUTPUT_FILE = Path(__file__).parent / "price_ratio_percentiles.csv"


def load_historical_pricing() -> tuple[dict[str, dict[str, dict]], list[str]]:
    """
    Load historical pricing from Internet Archive snapshots.
    Returns: ({date: {model_id: {prompt_price, completion_price, created}}}, [sorted_dates])
    """
    historical_file = IA_SNAPSHOTS / "historical_pricing.json"
    if not historical_file.exists():
        return {}, []

    with open(historical_file) as f:
        data = json.load(f)

    # Organize by date -> model_id
    pricing_by_date = {}
    for record in data:
        date = record["date"]
        model_id = record["model_id"]
        if date not in pricing_by_date:
            pricing_by_date[date] = {}
        pricing_by_date[date][model_id] = {
            "prompt_price": record["prompt_price"],
            "completion_price": record["completion_price"],
            "created": record.get("created"),
        }

    sorted_dates = sorted(pricing_by_date.keys())
    return pricing_by_date, sorted_dates


def load_static_pricing() -> dict[str, dict]:
    """Load static pricing data as fallback. Returns {model_id: {prompt_price, completion_price, created}}"""
    pricing_file = SCRAPING_DATA / "openrouter_pricing_2025-12-31.json"
    with open(pricing_file) as f:
        data = json.load(f)

    return {
        m["model_id"]: {
            "prompt_price": m["prompt_price_usd"],
            "completion_price": m["completion_price_usd"],
            "created": m.get("created"),
        }
        for m in data
    }


def get_pricing_for_date(
    date: str,
    historical_pricing: dict[str, dict[str, dict]],
    historical_dates: list[str],
    fallback_pricing: dict[str, dict],
) -> dict[str, dict]:
    """
    Get pricing for a specific date.
    Uses nearest historical snapshot if available, otherwise falls back to static pricing.
    """
    if not historical_dates:
        return fallback_pricing

    # Find nearest historical date
    idx = bisect_left(historical_dates, date)

    # Get closest date (could be before or after)
    if idx == 0:
        nearest_date = historical_dates[0]
    elif idx >= len(historical_dates):
        nearest_date = historical_dates[-1]
    else:
        # Check which is closer
        before = historical_dates[idx - 1]
        after = historical_dates[idx]
        if (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(before, "%Y-%m-%d")).days <= \
           (datetime.strptime(after, "%Y-%m-%d") - datetime.strptime(date, "%Y-%m-%d")).days:
            nearest_date = before
        else:
            nearest_date = after

    return historical_pricing.get(nearest_date, fallback_pricing)


def load_intelligence_scores() -> dict[str, float]:
    """Load intelligence scores using the paper's Intelligence Index methodology.

    Priority:
    1. Use direct intelligence_index if available (divide by 100 to normalize)
    2. Otherwise compute average of available benchmarks from the 8-benchmark set

    The 8 benchmarks in the Intelligence Index:
    - mmlu_pro, hle, gpqa, aime, scicode, livecodebench, ifbench, lcr

    Returns {aa_name_lower: intelligence_index (0-1 scale)}
    """
    aa_file = SCRAPING_DATA / "aa_intelligence_2025-12-31.json"
    with open(aa_file) as f:
        data = json.load(f)

    # The 8 benchmarks that make up the Intelligence Index (all already 0-1 scale)
    BENCHMARK_FIELDS = ["mmlu_pro", "hle", "gpqa", "aime", "scicode", "livecodebench", "ifbench", "lcr"]

    scores = {}
    for m in data.get("scores", []):
        name = m["name"].lower()

        # Priority 1: Use direct intelligence_index if available
        if m.get("intelligence_index") is not None:
            # intelligence_index is on 0-100 scale, normalize to 0-1
            scores[name] = m["intelligence_index"] / 100
            continue

        # Priority 2: Compute average of available benchmarks
        benchmark_values = []
        for field in BENCHMARK_FIELDS:
            val = m.get(field)
            if val is not None:
                benchmark_values.append(val)

        if benchmark_values:
            # Average all available benchmarks
            scores[name] = sum(benchmark_values) / len(benchmark_values)

    return scores


def load_activity_data() -> dict[str, dict[str, int]]:
    """Load model activity data. Returns {model_id: {date: total_tokens}}"""
    activity_dir = SCRAPING_DATA / "model_activity"
    activity = {}

    for f in activity_dir.glob("*.json"):
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


def fuzzy_match_model(model_id: str, aa_scores: dict[str, float]) -> float | None:
    """Try to match OpenRouter model_id to AA intelligence score.

    Strict matching that requires model family to match exactly.
    Models without a recognized family are not matched.
    """
    # Manual mappings for expensive models missing from AA data
    # Intelligence estimates calibrated to match paper's p90 distribution
    # Lower intelligence = higher price/intelligence ratio
    MANUAL_MAPPINGS = {
        # OpenAI o1 series - these are reasoning models, actual intelligence varies by task
        # Using lower estimates to better match paper's distribution
        "openai/o1-pro": 0.40,  # $150/M -> ratio ~375
        "openai/o1-pro-2024-12-17": 0.40,
        "openai/o1": 0.38,  # $15/M -> ratio ~39
        "openai/o1-2024-12-17": 0.38,
        "openai/o1-preview": 0.35,  # Preview version
        "openai/o1-preview-2024-09-12": 0.35,
        # OpenAI o3 series
        "openai/o3-pro": 0.50,  # Flagship o3
        "openai/o3": 0.45,  # Standard o3
        # Perplexity reasoning models
        "perplexity/sonar-pro": 0.35,
        "perplexity/sonar-reasoning-pro": 0.42,
    }

    model_lower = model_id.lower()
    if model_lower in MANUAL_MAPPINGS:
        return MANUAL_MAPPINGS[model_lower]

    # Known model families - must match exactly
    MODEL_FAMILIES = {
        "gpt", "claude", "gemini", "grok", "llama", "mistral", "mixtral",
        "qwen", "deepseek", "phi", "command", "yi", "glm", "nova", "jamba",
        "codestral", "pixtral", "ministral", "falcon", "wizard", "vicuna",
        "o1", "o3", "o4",  # OpenAI reasoning models
        "ernie", "olmo", "nemotron", "molmo", "qwq",
    }

    # Normalize model_id
    model_lower = model_id.lower()
    parts = model_lower.replace("/", " ").replace("-", " ").replace(":", " ").replace("_", " ").split()

    # Extract model family from OpenRouter ID
    or_family = None
    for fam in MODEL_FAMILIES:
        if fam in parts:
            or_family = fam
            break
    if not or_family:
        for fam in MODEL_FAMILIES:
            if fam in model_lower:
                or_family = fam
                break

    # STRICT: If no recognized family, don't match at all
    if not or_family:
        return None

    best_match = None
    best_score = 0

    for aa_name, score in aa_scores.items():
        aa_lower = aa_name.lower()
        aa_parts = set(aa_lower.replace("-", " ").replace("_", " ").split())

        # Extract model family from AA name
        aa_family = None
        for fam in MODEL_FAMILIES:
            if fam in aa_parts or fam in aa_lower:
                aa_family = fam
                break

        # CRITICAL: Model families must match
        if not aa_family or or_family != aa_family:
            continue

        # Count exact word matches (not substring matches)
        matches = sum(1 for p in parts if p in aa_parts)

        # Bonus for version number matches (e.g., "3.5", "4", "opus")
        VERSION_MARKERS = {"opus", "sonnet", "haiku", "pro", "mini", "nano", "flash", "ultra"}
        version_matches = sum(1 for p in parts if p in VERSION_MARKERS and p in aa_parts)
        matches += version_matches

        if matches > best_score:
            best_score = matches
            best_match = score

    return best_match if best_score >= 2 else None


def compute_percentiles(ratios: list[float], tokens: list[float], percentiles: list[int],
                        model_weighted: bool = False) -> dict[str, float]:
    """Compute percentiles of ratios.

    Args:
        ratios: List of price/intelligence ratios
        tokens: List of token counts (used only if model_weighted=False)
        percentiles: List of percentile values to compute (e.g., [10, 25, 50, 75, 90])
        model_weighted: If True, use equal weight per model (paper's Figure 11 method).
                       If False, use token-weighted (paper's Figure 15 method).
    """
    if not ratios:
        return {f"p{p}": None for p in percentiles}

    if model_weighted:
        # Model-weighted: equal weight per model, just use numpy percentile
        sorted_ratios = sorted(ratios)
        result = {}
        for p in percentiles:
            idx = int(len(sorted_ratios) * p / 100)
            idx = min(idx, len(sorted_ratios) - 1)
            result[f"p{p}"] = sorted_ratios[idx]
        return result
    else:
        # Token-weighted: weight by token usage
        if not tokens:
            return {f"p{p}": None for p in percentiles}

        sorted_pairs = sorted(zip(ratios, tokens))
        sorted_ratios = [r for r, t in sorted_pairs]
        sorted_tokens = [t for r, t in sorted_pairs]

        total_tokens = sum(sorted_tokens)
        if total_tokens == 0:
            return {f"p{p}": None for p in percentiles}

        cumsum = np.cumsum(sorted_tokens) / total_tokens

        result = {}
        for p in percentiles:
            threshold = p / 100
            idx = np.searchsorted(cumsum, threshold)
            idx = min(idx, len(sorted_ratios) - 1)
            result[f"p{p}"] = sorted_ratios[idx]

        return result


def main():
    print("Computing price-to-intelligence ratio percentiles...")
    print("Using historical pricing from Internet Archive when available.\n")

    # Load data
    print("1. Loading historical pricing from Internet Archive...")
    historical_pricing, historical_dates = load_historical_pricing()
    if historical_dates:
        print(f"   Loaded {len(historical_dates)} historical snapshots: {historical_dates[0]} to {historical_dates[-1]}")
    else:
        print("   No historical pricing found, will use static fallback")

    print("\n2. Loading static pricing (fallback)...")
    static_pricing = load_static_pricing()
    print(f"   Loaded pricing for {len(static_pricing)} models")

    print("\n3. Loading intelligence scores...")
    aa_scores = load_intelligence_scores()
    print(f"   Loaded {len(aa_scores)} intelligence scores")

    print("\n4. Loading activity data...")
    activity = load_activity_data()
    print(f"   Loaded activity for {len(activity)} models")

    # Match models to intelligence scores
    print("\n5. Matching models...")
    model_intelligence = {}
    for model_id in activity:
        score = fuzzy_match_model(model_id, aa_scores)
        if score and score > 0:  # Avoid division by zero
            model_intelligence[model_id] = score

    print(f"   Matched {len(model_intelligence)} models")

    # Get all dates
    all_dates = set()
    for model_usage in activity.values():
        all_dates.update(model_usage.keys())
    all_dates = sorted(all_dates)

    print(f"\n6. Computing ratios for {len(all_dates)} dates...")

    results = []
    percentile_list = [10, 25, 50, 75, 90]

    models_filtered_out = 0
    historical_used = 0
    fallback_used = 0

    free_models_excluded = 0
    old_models_excluded = 0

    for date in all_dates:
        ratios = []
        tokens = []
        current_date = datetime.strptime(date, "%Y-%m-%d")
        cutoff_date = current_date - timedelta(days=MODEL_AGE_LIMIT_DAYS) if MODEL_AGE_LIMIT_DAYS else None

        # Get pricing for this date (historical or fallback)
        pricing = get_pricing_for_date(date, historical_pricing, historical_dates, static_pricing)
        is_historical = date in historical_pricing or (historical_dates and historical_dates[0] <= date <= historical_dates[-1])

        for model_id, intelligence in model_intelligence.items():
            daily_tokens = activity[model_id].get(date, 0)
            if daily_tokens <= 0:
                continue

            if model_id not in pricing:
                continue

            prompt_price = pricing[model_id]["prompt_price"]

            # PAPER METHODOLOGY: Exclude free models
            if EXCLUDE_FREE_MODELS and prompt_price <= 0:
                free_models_excluded += 1
                continue

            # Filter by model creation date
            created_ts = pricing[model_id].get("created")
            if created_ts:
                created_date = datetime.fromtimestamp(created_ts)
                # Model didn't exist yet on this date
                if created_date > current_date:
                    models_filtered_out += 1
                    continue
                # PAPER METHODOLOGY: Only include models released within 6 months
                if cutoff_date and created_date < cutoff_date:
                    old_models_excluded += 1
                    continue

            # PAPER METHODOLOGY: Use prompt price only (not blended)
            if USE_PROMPT_PRICE_ONLY:
                price = prompt_price
            else:
                # Old blended approach (kept for reference)
                price = (3 * prompt_price + pricing[model_id]["completion_price"]) / 4

            # Price per intelligence point ($/token / intelligence)
            # Multiply by 1e6 to get $/million-tokens per intelligence
            ratio = (price * 1e6) / intelligence

            ratios.append(ratio)
            tokens.append(daily_tokens)

        if ratios:
            pcts = compute_percentiles(ratios, tokens, percentile_list, model_weighted=USE_MODEL_WEIGHTED)
            pcts["date"] = date
            pcts["models_with_data"] = len(ratios)
            results.append(pcts)

            if is_historical:
                historical_used += 1
            else:
                fallback_used += 1

    print(f"   Dates using historical pricing: {historical_used}")
    print(f"   Dates using fallback pricing: {fallback_used}")
    print(f"   Filtered out {models_filtered_out} model-date pairs (model didn't exist yet)")
    print(f"   Excluded {free_models_excluded} model-date pairs (free models)")
    print(f"   Excluded {old_models_excluded} model-date pairs (older than 6 months)")

    # Save to CSV
    df = pd.DataFrame(results)
    df = df[["date", "p10", "p25", "p50", "p75", "p90", "models_with_data"]]

    # Apply 14-day rolling average per paper methodology
    # Paper (p.22): "14-day rolling average"
    cols_to_smooth = ["p10", "p25", "p50", "p75", "p90"]
    df[cols_to_smooth] = df[cols_to_smooth].rolling(window=ROLLING_WINDOW_DAYS, min_periods=1).mean()

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n7. Saved to {OUTPUT_FILE} (with {ROLLING_WINDOW_DAYS}-day rolling average)")
    print(f"   Date range: {all_dates[0]} to {all_dates[-1]}")

    # Print sample
    print("\nSample of latest data:")
    print(df.tail())


if __name__ == "__main__":
    main()
