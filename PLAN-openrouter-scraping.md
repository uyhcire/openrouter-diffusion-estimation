# Plan: Continuous OpenRouter Data Collection

This document outlines a plan for continuously scraping OpenRouter data to replicate and extend the analysis from NBER Working Paper w34608 ("The Emerging Market for Intelligence: Pricing, Supply, and Demand for LLMs").

## Background

The paper analyzes LLM market dynamics using OpenRouter API usage data. Key metrics tracked include:
- **Price-to-intelligence ratio**: Model pricing normalized by capability (Figure 11)
- **Token-weighted intelligence**: Usage-weighted aggregate intelligence scores (Figure 15)
- **Percentile distributions**: p10/p25/p50/p75/p90 bands showing market spread

## Data Sources

### 1. OpenRouter Models API (Primary)

**Endpoint**: `GET https://openrouter.ai/api/v1/models`

**Authentication**: Bearer token (API key)

**Response schema** (per model):
```json
{
  "id": "provider/model-name",
  "canonical_slug": "provider/model-name-version",
  "name": "Human Readable Name",
  "created": 1766505011,           // Unix timestamp
  "description": "...",
  "context_length": 262144,
  "hugging_face_id": "",

  "architecture": {
    "modality": "text+image->text",
    "input_modalities": ["image", "text"],
    "output_modalities": ["text"],
    "tokenizer": "Other"
  },

  "pricing": {
    "prompt": "0.000000075",        // USD per input token
    "completion": "0.0000003",      // USD per output token
    "request": "0",
    "image": "0",
    "web_search": "0",
    "internal_reasoning": "0"
  },

  "top_provider": {
    "context_length": 262144,
    "max_completion_tokens": 16384,
    "is_moderated": false
  }
}
```

**Key fields to capture**:
- `id`: Model identifier
- `created`: Model addition timestamp
- `pricing.prompt` + `pricing.completion`: Token costs
- `context_length`: Capability indicator
- `architecture.modality`: Input/output types

### 2. Intelligence Scores (Supplementary)

The paper uses "intelligence tiers" to normalize prices. Likely sources:

#### Option A: Artificial Analysis Intelligence Index
- **URL**: https://artificialanalysis.ai/leaderboards/models
- **Methodology**: Composite of MMLU-Pro, GPQA Diamond, AIME 2025, LiveCodeBench, etc.
- **Challenge**: No public API; requires web scraping

#### Option B: Chatbot Arena ELO (LMSYS)
- **URL**: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
- **Data**: CSV export available from the leaderboard
- **Methodology**: Crowdsourced pairwise comparisons, Bradley-Terry model

#### Option C: OpenRouter Rankings
- **URL**: https://openrouter.ai/rankings
- **Data**: May include usage-based rankings

### 3. Historical Model Data

OpenRouter doesn't provide historical pricing snapshots. To track changes:
- Scrape daily and diff against previous day
- Log all price changes with timestamps
- Track model additions/removals

## Data Schema Design

### `models.jsonl` (append-only log)
```json
{
  "scraped_at": "2025-01-15T00:00:00Z",
  "model_id": "anthropic/claude-3-opus",
  "name": "Claude 3 Opus",
  "created": 1709596800,
  "prompt_price_usd": 0.000015,
  "completion_price_usd": 0.000075,
  "context_length": 200000,
  "modality": "text+image->text"
}
```

### `intelligence_scores.csv`
```csv
date,model_id,source,score
2025-01-15,anthropic/claude-3-opus,artificial_analysis,87.3
2025-01-15,anthropic/claude-3-opus,chatbot_arena,1254
```

### `daily_snapshots/YYYY-MM-DD.json`
Full API response archive for reproducibility.

## Implementation Components

### 1. Scraper: `scrape_openrouter.py`

```python
#!/usr/bin/env python3
"""Scrape OpenRouter models API and save daily snapshots."""

import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"

def fetch_models(api_key: str | None = None) -> dict:
    """Fetch all models from OpenRouter API."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(OPENROUTER_API_URL, headers=headers)
    response.raise_for_status()
    return response.json()

def save_snapshot(data: dict, output_dir: Path) -> Path:
    """Save raw API response as daily snapshot."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snapshot_dir = output_dir / "daily_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_dir / f"{today}.json"
    with open(snapshot_path, "w") as f:
        json.dump(data, f, indent=2)

    return snapshot_path

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

def append_to_log(records: list[dict], output_dir: Path):
    """Append pricing records to JSONL log."""
    log_path = output_dir / "models.jsonl"
    with open(log_path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    output_dir = Path("data/openrouter")

    scraped_at = datetime.now(timezone.utc).isoformat()
    data = fetch_models(api_key)

    save_snapshot(data, output_dir)
    records = extract_pricing_records(data, scraped_at)
    append_to_log(records, output_dir)

    print(f"Scraped {len(records)} models at {scraped_at}")

if __name__ == "__main__":
    main()
```

### 2. Scraper: `scrape_intelligence_scores.py`

```python
#!/usr/bin/env python3
"""Scrape intelligence scores from various leaderboards."""

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# Chatbot Arena leaderboard (public CSV)
ARENA_LEADERBOARD_URL = (
    "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard"
    "/resolve/main/elo_results.csv"  # May need adjustment
)

def fetch_chatbot_arena_scores() -> pd.DataFrame:
    """Fetch latest Chatbot Arena ELO scores."""
    # Note: Actual URL may differ; may need HF API or web scraping
    # This is a placeholder - actual implementation needs verification
    pass

def scrape_artificial_analysis() -> pd.DataFrame:
    """Scrape Artificial Analysis Intelligence Index."""
    # Requires Selenium or similar for JS-rendered content
    # Or find if they have a public data export
    pass
```

### 3. Scheduler: `cron` or GitHub Actions

**Option A: Local cron job**
```bash
# Run daily at midnight UTC
0 0 * * * cd /path/to/repo && /path/to/venv/bin/python scrape_openrouter.py
```

**Option B: GitHub Actions workflow**
```yaml
# .github/workflows/scrape.yml
name: Daily OpenRouter Scrape

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight UTC
  workflow_dispatch:      # Manual trigger

jobs:
  scrape:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install requests pandas

      - name: Run scraper
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: python scrape_openrouter.py

      - name: Commit data
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add data/
          git diff --staged --quiet || git commit -m "Daily scrape $(date -u +%Y-%m-%d)"
          git push
```

### 4. Analysis: Compute Paper Metrics

To replicate Figure 11 (price-to-intelligence ratio) and Figure 15 (token-weighted intelligence):

#### Price-to-Intelligence Ratio

```python
def compute_price_to_intelligence(
    pricing_df: pd.DataFrame,
    intelligence_df: pd.DataFrame,
    date: str
) -> pd.DataFrame:
    """
    Compute price-to-intelligence ratio for each model.

    Price = (prompt_price + completion_price) / 2  # or weighted by typical I/O ratio
    Ratio = Price / Intelligence_Score
    """
    merged = pricing_df.merge(intelligence_df, on="model_id")

    # Blended price (adjust weights based on typical usage patterns)
    merged["blended_price"] = (
        merged["prompt_price_usd"] * 0.3 +
        merged["completion_price_usd"] * 0.7  # Completion typically dominates cost
    ) * 1_000_000  # Per million tokens

    merged["price_to_intelligence"] = merged["blended_price"] / merged["score"]

    return merged
```

#### Token-Weighted Intelligence

```python
def compute_token_weighted_intelligence(
    intelligence_df: pd.DataFrame,
    usage_weights: pd.DataFrame  # If available from OpenRouter
) -> float:
    """
    Compute usage-weighted average intelligence.

    Note: OpenRouter doesn't publicly expose per-model usage.
    The paper likely used internal data access.

    Approximation: Weight by market share estimates or equal weights.
    """
    merged = intelligence_df.merge(usage_weights, on="model_id")

    weighted_sum = (merged["score"] * merged["token_share"]).sum()
    total_weight = merged["token_share"].sum()

    return weighted_sum / total_weight
```

#### Percentile Computation

```python
def compute_percentiles(values: pd.Series) -> dict:
    """Compute p10/p25/p50/p75/p90 for a series of values."""
    return {
        "p10": values.quantile(0.10),
        "p25": values.quantile(0.25),
        "p50": values.quantile(0.50),
        "p75": values.quantile(0.75),
        "p90": values.quantile(0.90),
    }
```

## Known Limitations

### 1. Token Usage Data Not Public
OpenRouter doesn't expose per-model token usage. The paper likely had internal data access from OpenRouter. Alternatives:
- Use equal weights across models
- Estimate weights from market share proxies
- Contact OpenRouter for research data access

### 2. Historical Pricing
No historical API - must scrape daily and build history over time. The paper likely had access to historical data from OpenRouter's internal databases.

### 3. Intelligence Score Mapping
Model IDs may not match exactly between OpenRouter and leaderboards. Need fuzzy matching:
```python
def match_model_to_leaderboard(openrouter_id: str, leaderboard_names: list[str]) -> str | None:
    """Fuzzy match OpenRouter model ID to leaderboard entry."""
    # e.g., "anthropic/claude-3-opus" -> "claude-3-opus-20240229"
    pass
```

### 4. Rate Limits
OpenRouter API has rate limits. Daily scraping should be well within limits, but monitor response codes.

## Directory Structure

```
data/
├── openrouter/
│   ├── daily_snapshots/
│   │   ├── 2025-01-01.json
│   │   ├── 2025-01-02.json
│   │   └── ...
│   ├── models.jsonl           # Append-only pricing log
│   └── price_changes.csv      # Detected price changes
├── intelligence/
│   ├── chatbot_arena/
│   │   └── elo_scores.csv
│   └── artificial_analysis/
│       └── index_scores.csv
└── derived/
    ├── price_to_intelligence_daily.csv
    └── token_weighted_intelligence_daily.csv
```

## Next Steps

1. **Implement basic scraper** (`scrape_openrouter.py`)
2. **Set up GitHub Actions** for daily automation
3. **Build intelligence score pipeline** (Chatbot Arena is easiest)
4. **Create model ID matching** between data sources
5. **Implement analysis scripts** for paper metrics
6. **Add monitoring/alerting** for scraper failures

## References

- [NBER w34608](https://www.nber.org/papers/w34608) - "The Emerging Market for Intelligence"
- [OpenRouter API Docs](https://openrouter.ai/docs/api/api-reference/models/get-models)
- [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
- [Artificial Analysis](https://artificialanalysis.ai/methodology/intelligence-benchmarking)
- [OpenRouter State of AI](https://openrouter.ai/state-of-ai)
