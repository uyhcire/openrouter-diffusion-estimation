#!/usr/bin/env python3
"""
Analyze price-to-intelligence ratio data to understand why p75/p90 are too low.
Paper's p90 is ~10.3, ours is ~2.9.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Load data
pricing_path = Path("data/openrouter_pricing_2025-12-31.json")
intelligence_path = Path("data/aa_intelligence_2025-12-31.json")

with open(pricing_path) as f:
    pricing_data = json.load(f)  # This is a list of models

with open(intelligence_path) as f:
    intelligence_data = json.load(f)  # Dict with "scores" key

print(f"Loaded {len(pricing_data)} pricing entries")

# Get intelligence scores
scores = intelligence_data.get("scores", [])
print(f"Loaded {len(scores)} intelligence score entries")

# Build intelligence lookup by model name (normalized)
def normalize_model_name(name):
    """Normalize model name for matching"""
    return name.lower().replace("-", " ").replace("_", " ").replace(".", " ").replace(":", " ")

# Build intelligence lookup - two approaches
intel_lookup = {}
intel_by_keyword = {}

BENCHMARK_FIELDS = ["mmlu_pro", "hle", "gpqa", "aime", "scicode", "livecodebench", "ifbench", "lcr"]

for model in scores:
    model_name = model.get("name", "")
    intel_idx = model.get("intelligence_index")

    # If no direct intelligence_index, compute from benchmarks
    if intel_idx is None:
        benchmark_values = []
        for field in BENCHMARK_FIELDS:
            val = model.get(field)
            if val is not None:
                benchmark_values.append(val)
        if benchmark_values:
            # Average benchmarks and scale to 0-100 like intelligence_index
            intel_idx = sum(benchmark_values) / len(benchmark_values) * 100

    if model_name and intel_idx is not None:
        norm_name = normalize_model_name(model_name)
        intel_lookup[norm_name] = {
            "model": model_name,
            "intelligence_index": intel_idx
        }
        # Also index by keywords
        for word in norm_name.split():
            if len(word) > 3:
                if word not in intel_by_keyword:
                    intel_by_keyword[word] = []
                intel_by_keyword[word].append({
                    "model": model_name,
                    "intelligence_index": intel_idx
                })

print(f"\nTotal models with intelligence scores: {len(intel_lookup)}")

# Use the fuzzy matching logic from compute_price_ratios.py
MODEL_FAMILIES = {
    "gpt", "claude", "gemini", "grok", "llama", "mistral", "mixtral",
    "qwen", "deepseek", "phi", "command", "yi", "glm", "nova", "jamba",
    "codestral", "pixtral", "ministral", "falcon", "wizard", "vicuna",
    "o1", "o3", "o4",  # OpenAI reasoning models
    "ernie", "olmo", "nemotron", "molmo", "qwq", "seed",
}

def fuzzy_match_model(model_id: str) -> dict | None:
    """Try to match OpenRouter model_id to AA intelligence score."""
    model_lower = model_id.lower()
    parts = model_lower.replace("/", " ").replace("-", " ").replace(":", " ").replace("_", " ").split()

    # Extract model family
    or_family = None
    for fam in MODEL_FAMILIES:
        if fam in parts or fam in model_lower:
            or_family = fam
            break

    if not or_family:
        return None

    best_match = None
    best_score = 0

    for aa_name, info in intel_lookup.items():
        aa_parts = set(aa_name.split())

        # Extract family from AA name
        aa_family = None
        for fam in MODEL_FAMILIES:
            if fam in aa_parts or fam in aa_name:
                aa_family = fam
                break

        if not aa_family or or_family != aa_family:
            continue

        # Count matches
        matches = sum(1 for p in parts if p in aa_parts)

        VERSION_MARKERS = {"opus", "sonnet", "haiku", "pro", "mini", "nano", "flash", "ultra", "3", "4", "2", "1"}
        version_matches = sum(1 for p in parts if p in VERSION_MARKERS and p in aa_parts)
        matches += version_matches

        if matches > best_score:
            best_score = matches
            best_match = info

    return best_match if best_score >= 2 else None

# Calculate price/intelligence ratios
results = []
unmatched_expensive = []

for model in pricing_data:
    model_id = model.get("model_id", "")
    model_name = model.get("name", "")
    # Prices are per-token in USD, convert to per-million-tokens
    prompt_price = model.get("prompt_price_usd", 0) * 1e6  # Now $/M tokens

    # Try to find matching intelligence score
    intel_match = fuzzy_match_model(model_id)

    if intel_match and prompt_price > 0:
        intel_idx = intel_match["intelligence_index"]
        # Ratio: $/M-tokens per intelligence-point (normalized to 0-1)
        # This matches compute_price_ratios.py methodology
        intel_normalized = intel_idx / 100  # 0-1 scale
        ratio = prompt_price / intel_normalized
        results.append({
            "model_id": model_id,
            "model_name": model_name,
            "prompt_price": prompt_price,
            "intelligence": intel_idx,
            "intel_norm": intel_normalized,
            "ratio": ratio,
            "matched_to": intel_match["model"]
        })
    elif prompt_price > 5:  # Track expensive unmatched models
        unmatched_expensive.append({
            "model_id": model_id,
            "model_name": model_name,
            "prompt_price": prompt_price
        })

print(f"Models with both price and intelligence: {len(results)}")

# Sort by ratio descending
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("ratio", ascending=False)

print("\n" + "="*80)
print("TOP 30 HIGHEST PRICE/INTELLIGENCE RATIOS (most expensive per unit intelligence)")
print("="*80)
pd.set_option('display.max_colwidth', 40)
pd.set_option('display.width', 200)
print(results_df.head(30).to_string(index=False))

print("\n" + "="*80)
print("PERCENTILE ANALYSIS")
print("="*80)
ratios = results_df["ratio"].values
print(f"p10: {np.percentile(ratios, 10):.4f}")
print(f"p25: {np.percentile(ratios, 25):.4f}")
print(f"p50: {np.percentile(ratios, 50):.4f}")
print(f"p75: {np.percentile(ratios, 75):.4f}")
print(f"p90: {np.percentile(ratios, 90):.4f}")
print(f"max: {np.max(ratios):.4f}")

print("\n" + "="*80)
print("WHAT RATIO NEEDED FOR PAPER'S P90 (~10.3)")
print("="*80)
paper_p90 = 10.3
print(f"Paper's p90: {paper_p90}")
print(f"Our p90: {np.percentile(ratios, 90):.4f}")
print(f"\nTo achieve p90 of 10.3, we need ~10% of models with ratio >= 10.3")
models_above_target = len(results_df[results_df["ratio"] >= paper_p90])
print(f"Models with ratio >= 10.3: {models_above_target}")
pct_needed = 10  # 10% for p90
models_needed = int(len(results_df) * pct_needed / 100)
print(f"Models needed (10% of {len(results_df)}): {models_needed}")

print("\n" + "="*80)
print("EXPENSIVE MODELS (prompt_price > $10/M tokens) - ALL")
print("="*80)
expensive_models = []
for model in pricing_data:
    model_id = model.get("model_id", "")
    model_name = model.get("name", "")
    prompt_price = model.get("prompt_price_usd", 0) * 1e6  # $/M tokens

    if prompt_price > 10:
        intel_match = fuzzy_match_model(model_id)
        has_intel = intel_match is not None
        intel_idx = intel_match["intelligence_index"] if intel_match else None
        ratio = prompt_price / intel_idx if intel_idx else None

        expensive_models.append({
            "model_id": model_id,
            "model_name": model_name,
            "prompt_price": prompt_price,
            "has_intelligence": has_intel,
            "intelligence": intel_idx,
            "ratio": ratio
        })

expensive_df = pd.DataFrame(expensive_models)
expensive_df = expensive_df.sort_values("prompt_price", ascending=False)
print(f"Total expensive models (>$10/M): {len(expensive_df)}")
print(expensive_df.to_string(index=False))

print("\n" + "="*80)
print("EXPENSIVE MODELS MISSING INTELLIGENCE SCORES")
print("="*80)
if len(expensive_df) > 0:
    missing_intel = expensive_df[expensive_df["has_intelligence"] == False]
    print(f"Expensive models missing intelligence: {len(missing_intel)}")
    if len(missing_intel) > 0:
        print(missing_intel.to_string(index=False))
else:
    missing_intel = pd.DataFrame()
    print("No expensive models found")

print("\n" + "="*80)
print("MODELS THAT COULD RAISE P90 IF INCLUDED")
print("="*80)
# What if missing models had average or low intelligence?
avg_intel = results_df["intel_norm"].mean()
print(f"Average intelligence of matched models: {avg_intel:.2f}")

if len(missing_intel) > 0:
    for _, row in missing_intel.iterrows():
        hypothetical_ratio = row["prompt_price"] / avg_intel
        print(f"  {row['model_name']}: ${row['prompt_price']:.2f}/M -> ratio {hypothetical_ratio:.4f} if intel={avg_intel:.0f}")

print("\n" + "="*80)
print("DISTRIBUTION OF RATIOS")
print("="*80)
bins = [0, 0.5, 1, 2, 3, 5, 10, 20, 50, 100]
for i in range(len(bins)-1):
    count = len(results_df[(results_df["ratio"] >= bins[i]) & (results_df["ratio"] < bins[i+1])])
    print(f"  {bins[i]:.1f} - {bins[i+1]:.1f}: {count} models")
count = len(results_df[results_df["ratio"] >= bins[-1]])
print(f"  >= {bins[-1]:.1f}: {count} models")

# Look at what creates the paper's distribution
print("\n" + "="*80)
print("SIMULATING PAPER'S DISTRIBUTION")
print("="*80)
# If paper has p90=10.3 and p75=~3.5, what kind of models are they including?
print("Paper approximate percentiles: p10=0.23, p25=0.3, p50=0.45, p75=3.5, p90=10.3")
print("\nPossible explanations:")
print("1. Paper includes models with very high prices we don't have")
print("2. Paper includes fine-tuned/custom models with lower ELOs")
print("3. Paper uses different intelligence metric")
print("4. Paper includes older snapshots where prices were higher")
print("5. Paper includes providers beyond OpenRouter")

print("\n" + "="*80)
print("ALL UNMATCHED EXPENSIVE MODELS (>$5/M tokens)")
print("="*80)
unmatched_df = pd.DataFrame(unmatched_expensive)
if len(unmatched_df) > 0:
    unmatched_df = unmatched_df.sort_values("prompt_price", ascending=False)
    print(f"Total unmatched expensive models: {len(unmatched_df)}")
    print(unmatched_df.to_string(index=False))
else:
    print("None")

print("\n" + "="*80)
print("ALL INTELLIGENCE SCORES AVAILABLE")
print("="*80)
for name, info in sorted(intel_lookup.items(), key=lambda x: -x[1]["intelligence_index"])[:50]:
    print(f"  {info['model']:40s} -> intel={info['intelligence_index']:.2f}")

print("\n" + "="*80)
print("LOWEST INTELLIGENCE SCORES (could create high ratios)")
print("="*80)
for name, info in sorted(intel_lookup.items(), key=lambda x: x[1]["intelligence_index"])[:20]:
    print(f"  {info['model']:40s} -> intel={info['intelligence_index']:.2f}")

print("\n" + "="*80)
print("SUMMARY: WHY P90 IS TOO LOW")
print("="*80)
print("""
Our computed p90: {:.2f}
Paper's p90: ~10.3
Gap: Paper's p90 is about {:.1f}x higher than ours

KEY FINDINGS:

1. MISSING EXPENSIVE MODELS: 4 expensive OpenAI models are missing intelligence scores:
   - o1-pro ($150/M) - would have ratio ~250-375 depending on intelligence
   - o3-pro ($20/M) - would have ratio ~33-50
   - o1 ($15/M) - would have ratio ~25-37
   - GPT-5.2 Pro ($21/M) - would have ratio ~35-52

2. OUR DISTRIBUTION:
   - p10: {:.2f} (paper: ~0.23)
   - p25: {:.2f} (paper: ~0.3)
   - p50: {:.2f} (paper: ~0.45)
   - p75: {:.2f} (paper: ~3.5)
   - p90: {:.2f} (paper: ~10.3)

3. MODELS WITH RATIO >= 10.3: {} out of {} ({:.1f}%)
   We need ~10% of models with ratio >= 10.3 to hit paper's p90

4. HIGHEST RATIOS WE HAVE:
   - GPT-4: 53.24 (price=$30, intel=56.35)
   - Claude 3 Opus: 37.25 (price=$15, intel=40.27)
   - Claude Opus 4: 27.64 (price=$15, intel=54.28)

5. TO ACHIEVE PAPER'S P90:
   - Include o1-pro, o1, o3-pro with appropriate intelligence scores
   - Add more expensive/inefficient models to the dataset
   - Or the paper uses different data sources/methodology
""".format(
    np.percentile(ratios, 90),
    10.3 / np.percentile(ratios, 90),
    np.percentile(ratios, 10),
    np.percentile(ratios, 25),
    np.percentile(ratios, 50),
    np.percentile(ratios, 75),
    np.percentile(ratios, 90),
    len(results_df[results_df["ratio"] >= 10.3]),
    len(results_df),
    100 * len(results_df[results_df["ratio"] >= 10.3]) / len(results_df)
))
