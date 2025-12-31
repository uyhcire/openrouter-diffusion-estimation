# Limitations: Paper Replication

This document describes known limitations and discrepancies when comparing our scraped data to the NBER paper (w34608.pdf) figures.

## Figure 15: Token-Weighted Intelligence Percentiles

**Status**: Good match for most percentiles

| Percentile | Mean % Diff | Notes |
|------------|-------------|-------|
| p10 | +31% | Consistently higher |
| p25 | +7% | Good match |
| p50 | +16% | Reasonable |
| p75 | +17% | Reasonable |
| p90 | +3% | Good match |

**Likely causes of p10 discrepancy**:
- Token weighting differences at the low end
- Model set differences for cheapest models

## Figure 11: Price-to-Intelligence Ratio

**Status**: Lower percentiles match well, upper percentiles have significant gap

| Percentile | Mean % Diff | Notes |
|------------|-------------|-------|
| p10 | -22% | Acceptable |
| p25 | -6% | Good match |
| p50 | -13% | Decent |
| p75 | -15% | Decent |
| p90 | **-36%** | Significant gap |

### Root Cause Analysis for p90 Gap

The p90 is ~36% lower than the paper's value. Investigation found:

1. **Model count threshold**: We have 12 models with ratio ≥13 (7% of ~170), but need ~17 (10%) for p90=13

2. **High-ratio models we include**:
   - openai/o1-pro ($150/M, ratio ~375)
   - openai/gpt-4 variants ($30/M, ratio ~69)
   - openai/o3-pro, o1 ($15-20/M, ratio ~40)
   - anthropic/claude-3-opus, claude-opus-4 ($15/M, ratio ~27-37)
   - openai/gpt-4-turbo variants ($10/M, ratio ~23)

3. **Possible causes**:
   - **Smaller paper model set**: If paper uses N=120 vs our N=170, same 12 models would give p90=13
   - **Delisted models**: Some expensive models may have been removed between paper's data and our Dec 31 snapshot
   - **Different intelligence source**: Paper may use intelligence scores from a different source with lower values for expensive models
   - **Missing enterprise models**: Paper may include expensive enterprise/custom models not available on OpenRouter

### Methodology Verified Correct

The following methodology aspects were verified against paper quotes:
- Model-weighted percentiles (not token-weighted) for Figure 11
- Prompt price only (confirmed: "Log Price per Million Prompt Tokens")
- 14-day rolling average
- Excludes free models

## Figure 5: P90 Frontier

**Status**: Moderate discrepancy

| Metric | Value |
|--------|-------|
| Mean % Diff | -17% |
| Correlation | -0.04 |

**Likely causes**:
- Different model set defining the frontier
- Timing differences in intelligence score updates

## Data Source Limitations

### OpenRouter Pricing
- Single snapshot from Dec 31, 2025
- Historical pricing from Internet Archive (14 snapshots, Oct 2 - Dec 6)
- Some models may have been delisted or repriced

### Artificial Analysis Intelligence Scores
- 392 models with scores
- Only 44 have direct `intelligence_index`; rest computed from 8 benchmarks
- Manual mappings added for expensive reasoning models (o1, o3 series) missing from AA

### Model Activity Data
- 91 days of token usage (Oct 2 - Dec 31, 2025)
- Only includes models with activity; excludes inactive expensive models
- Activity scraped from OpenRouter model pages

## Recommendations for Improvement

1. **More historical snapshots**: Scrape OpenRouter API daily to capture model additions/removals and price changes

2. **Alternative intelligence sources**: Cross-reference with other benchmark aggregators

3. **Manual model mapping**: Expand manual intelligence score mappings for models missing from AA

4. **Model set analysis**: Investigate exact model filtering criteria used in paper (may exclude certain model types)

5. **Contact authors**: The paper authors may be able to clarify exact methodology for Figure 11
