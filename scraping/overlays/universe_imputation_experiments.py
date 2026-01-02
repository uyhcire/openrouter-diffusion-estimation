#!/usr/bin/env python3
"""
Experiment with different model-universe imputation strategies for Figure 11.

Goal: find which strategy best matches the paper's Figure 11 percentiles.

This script is intentionally self-contained and reads only local artifacts:
- Paper digitized percentiles: out/figure11_price_to_intelligence_ratio_percentiles.csv
- IA pricing snapshots: scraping/overlays/ia_snapshots/historical_pricing.json
- OpenRouter static snapshot: scraping/data/openrouter_pricing_2025-12-31.json
- Artificial Analysis scores: scraping/data/aa_intelligence_2025-12-31.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_CSV = REPO_ROOT / "out" / "figure11_price_to_intelligence_ratio_percentiles.csv"
DEFAULT_AA_FILE = REPO_ROOT / "scraping" / "data" / "aa_intelligence_2025-12-31.json"
STATIC_PRICING_FILE = REPO_ROOT / "scraping" / "data" / "openrouter_pricing_2025-12-31.json"
IA_HISTORICAL_PRICING_FILE = REPO_ROOT / "scraping" / "overlays" / "ia_snapshots" / "historical_pricing.json"
ACTIVITY_DIR = REPO_ROOT / "scraping" / "data" / "model_activity"


MODEL_FAMILIES = {
    "gpt",
    "claude",
    "gemini",
    "grok",
    "llama",
    "mistral",
    "mixtral",
    "qwen",
    "deepseek",
    "phi",
    "command",
    "yi",
    "glm",
    "nova",
    "jamba",
    "codestral",
    "pixtral",
    "ministral",
    "falcon",
    "wizard",
    "vicuna",
    "o1",
    "o3",
    "o4",
    "ernie",
    "olmo",
    "nemotron",
    "molmo",
    "qwq",
}


MANUAL_INTELLIGENCE = {
    "openai/o1-pro": 0.40,
    "openai/o1-pro-2024-12-17": 0.40,
    "openai/o1": 0.38,
    "openai/o1-2024-12-17": 0.38,
    "openai/o1-preview": 0.35,
    "openai/o1-preview-2024-09-12": 0.35,
    "openai/o3-pro": 0.50,
    "openai/o3": 0.45,
    "perplexity/sonar-pro": 0.35,
    "perplexity/sonar-reasoning-pro": 0.42,
}


BENCHMARK_FIELDS = [
    "mmlu_pro",
    "hle",
    "gpqa",
    "aime",
    "scicode",
    "livecodebench",
    "ifbench",
    "lcr",
]


def _tokenize(s: str) -> list[str]:
    s = s.lower()
    # Treat "+" as a meaningful modifier.
    s = s.replace("+", " plus ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if t]
    # Normalize some frequent variants.
    norm = []
    for t in toks:
        if t in {"r+", "rplus"}:
            norm.extend(["r", "plus"])
        else:
            norm.append(t)
    return norm


@dataclass(frozen=True)
class Pricing:
    prompt_price_usd_per_token: float
    completion_price_usd_per_token: float
    created: int | None


@dataclass(frozen=True)
class AaEntry:
    name: str
    score: float
    tokens: frozenset[str]


def _parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")


def load_paper() -> list[dict[str, str]]:
    if not PAPER_CSV.exists():
        raise FileNotFoundError(f"Missing paper CSV: {PAPER_CSV}")

    rows: list[dict[str, str]] = []
    with PAPER_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        expected = {"date", "p10", "p25", "p50", "p75", "p90"}
        if not reader.fieldnames:
            raise ValueError(f"Paper CSV has no header: {PAPER_CSV}")
        missing = expected - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Paper CSV missing columns: {sorted(missing)}")
        for r in reader:
            rows.append(r)

    # Sort by date string (YYYY-MM-DD).
    rows.sort(key=lambda r: r["date"])
    return rows


def load_static_pricing() -> dict[str, Pricing]:
    data = json.loads(STATIC_PRICING_FILE.read_text())
    out: dict[str, Pricing] = {}
    for m in data:
        model_id = m.get("model_id")
        if not model_id:
            continue
        out[model_id] = Pricing(
            prompt_price_usd_per_token=float(m.get("prompt_price_usd", 0.0)),
            completion_price_usd_per_token=float(m.get("completion_price_usd", 0.0)),
            created=m.get("created"),
        )
    return out


def load_ia_pricing_by_snapshot_date() -> tuple[dict[str, dict[str, Pricing]], list[str]]:
    if not IA_HISTORICAL_PRICING_FILE.exists():
        return {}, []

    records = json.loads(IA_HISTORICAL_PRICING_FILE.read_text())
    by_date: dict[str, dict[str, Pricing]] = defaultdict(dict)
    for r in records:
        d = r["date"]
        mid = r["model_id"]
        by_date[d][mid] = Pricing(
            prompt_price_usd_per_token=float(r.get("prompt_price", 0.0)),
            completion_price_usd_per_token=float(r.get("completion_price", 0.0)),
            created=r.get("created"),
        )
    dates = sorted(by_date.keys())
    return dict(by_date), dates


def load_activity_presence_by_date() -> dict[str, set[str]]:
    """
    Returns {date: {model_id, ...}} for models with total_tokens > 0 that day.

    This is a lossy proxy for "models active on OpenRouter that day".
    """
    by_date: dict[str, set[str]] = defaultdict(set)
    if not ACTIVITY_DIR.exists():
        return {}
    for f in ACTIVITY_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        model_id = data.get("model_id")
        if not model_id:
            continue
        for day in data.get("daily_usage", []) or []:
            d = day.get("date")
            tok = day.get("total_tokens", 0)
            if d and tok and tok > 0:
                by_date[str(d)].add(model_id)
    return dict(by_date)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _transpose(mat: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*mat, strict=True)]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    bt = _transpose(b)
    out: list[list[float]] = []
    for row in a:
        out_row: list[float] = []
        for col in bt:
            out_row.append(sum(x * y for x, y in zip(row, col, strict=True)))
        out.append(out_row)
    return out


def _matvec(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(x * y for x, y in zip(row, v, strict=True)) for row in a]


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float] | None:
    """
    Solve A x = b with Gaussian elimination + partial pivoting.
    Returns None if singular.
    """
    n = len(a)
    if n == 0 or any(len(row) != n for row in a) or len(b) != n:
        raise ValueError("Invalid linear system dimensions")

    # Build augmented matrix.
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        # Pivot.
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            return None
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # Normalize pivot row.
        pivot = aug[col][col]
        inv = 1.0 / pivot
        for j in range(col, n + 1):
            aug[col][j] *= inv

        # Eliminate below.
        for r in range(col + 1, n):
            factor = aug[r][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    # Back substitution.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))
    return x


def _ols_fit(xs: list[list[float]], ys: list[float]) -> tuple[list[float], float] | None:
    """
    Fit y ~= w·x + b by OLS (with intercept).
    Returns (w, b).
    """
    if not xs or not ys or len(xs) != len(ys):
        return None
    n = len(xs)
    k = len(xs[0])
    if any(len(r) != k for r in xs):
        raise ValueError("Inconsistent feature widths")

    # Augment with intercept.
    x_aug = [r[:] + [1.0] for r in xs]
    xt = _transpose(x_aug)
    xtx = _matmul(xt, x_aug)  # (k+1)x(k+1)
    xty = _matvec(xt, ys)  # (k+1,)
    beta = _solve_linear_system(xtx, xty)
    if beta is None:
        return None
    w = beta[:k]
    b = beta[k]
    return w, b


def load_intelligence_scores(aa_file: Path, mode: str) -> dict[str, float]:
    data = json.loads(aa_file.read_text())
    rows = []
    for m in data.get("scores", []):
        name = str(m.get("name", "")).strip().lower()
        if not name:
            continue
        row = {"name": name}
        row["intelligence_index"] = m.get("intelligence_index")
        row["estimated_intelligence_index"] = m.get("estimated_intelligence_index")
        for k in BENCHMARK_FIELDS:
            row[k] = m.get(k)
        # Back-compat features.
        row["math_index"] = m.get("math_index")
        row["coding_index"] = m.get("coding_index")
        rows.append(row)

    # Means for imputing missing benchmark features.
    bench_means: dict[str, float] = {}
    for k in BENCHMARK_FIELDS:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        if vals:
            bench_means[k] = _mean(vals)

    def compute_from_benchmarks(r: dict[str, object], *, require_all: bool) -> float | None:
        vals = []
        for k in BENCHMARK_FIELDS:
            v = r.get(k)
            if v is None:
                if require_all:
                    return None
                continue
            vals.append(float(v))
        return _mean(vals) if vals else None

    if mode in {"full", "avg8_available"}:
        scores: dict[str, float] = {}
        for r in rows:
            name = r["name"]
            ii = r.get("intelligence_index")
            if ii is not None:
                scores[name] = float(ii) / 100.0
                continue
            est = r.get("estimated_intelligence_index")
            if est is not None:
                scores[name] = float(est) / 100.0
                continue
            v = compute_from_benchmarks(r, require_all=False)
            if v is not None:
                scores[name] = v
        return scores

    if mode == "direct_only":
        scores = {}
        for r in rows:
            ii = r.get("intelligence_index")
            if ii is None:
                continue
            scores[r["name"]] = float(ii) / 100.0
        return scores

    if mode == "avg8_require_all":
        scores = {}
        for r in rows:
            ii = r.get("intelligence_index")
            if ii is not None:
                scores[r["name"]] = float(ii) / 100.0
                continue
            est = r.get("estimated_intelligence_index")
            if est is not None:
                scores[r["name"]] = float(est) / 100.0
                continue
            v = compute_from_benchmarks(r, require_all=True)
            if v is not None:
                scores[r["name"]] = v
        return scores

    if mode == "regress_8bench":
        # Train on models with direct Intelligence Index and at least 1 benchmark.
        train_x: list[list[float]] = []
        train_y: list[float] = []
        for r in rows:
            ii = r.get("intelligence_index")
            if ii is None:
                continue
            feats = []
            any_feat = False
            for k in BENCHMARK_FIELDS:
                v = r.get(k)
                if v is None:
                    feats.append(bench_means.get(k, 0.0))
                else:
                    feats.append(float(v))
                    any_feat = True
            if not any_feat:
                continue
            train_x.append(feats)
            train_y.append(float(ii) / 100.0)

        fit = _ols_fit(train_x, train_y)
        if fit is None:
            return load_intelligence_scores("avg8_available")
        w, b = fit

        scores = {}
        for r in rows:
            name = r["name"]
            ii = r.get("intelligence_index")
            if ii is not None:
                scores[name] = float(ii) / 100.0
                continue
            feats = []
            any_feat = False
            for k in BENCHMARK_FIELDS:
                v = r.get(k)
                if v is None:
                    feats.append(bench_means.get(k, 0.0))
                else:
                    feats.append(float(v))
                    any_feat = True
            if not any_feat:
                continue
            pred = sum(wi * xi for wi, xi in zip(w, feats, strict=True)) + b
            # Clamp to [0, 1] (index is on 0-1 scale in the paper).
            if pred < 0:
                pred = 0.0
            if pred > 1:
                pred = 1.0
            scores[name] = float(pred)
        return scores

    if mode == "regress_math_coding":
        # Train y (direct ii) from (math_index, coding_index) where available.
        train_x = []
        train_y = []
        for r in rows:
            ii = r.get("intelligence_index")
            if ii is None:
                continue
            mi = r.get("math_index")
            ci = r.get("coding_index")
            if mi is None and ci is None:
                continue
            x0 = float(mi) / 100.0 if mi is not None else 0.0
            x1 = float(ci) / 100.0 if ci is not None else 0.0
            train_x.append([x0, x1])
            train_y.append(float(ii) / 100.0)
        fit = _ols_fit(train_x, train_y)
        if fit is None:
            return load_intelligence_scores("avg8_available")
        w, b = fit

        scores = {}
        for r in rows:
            name = r["name"]
            ii = r.get("intelligence_index")
            if ii is not None:
                scores[name] = float(ii) / 100.0
                continue
            mi = r.get("math_index")
            ci = r.get("coding_index")
            if mi is None and ci is None:
                continue
            x0 = float(mi) / 100.0 if mi is not None else 0.0
            x1 = float(ci) / 100.0 if ci is not None else 0.0
            pred = w[0] * x0 + w[1] * x1 + b
            if pred < 0:
                pred = 0.0
            if pred > 1:
                pred = 1.0
            scores[name] = float(pred)
        return scores

    raise ValueError(f"Unknown intelligence mode: {mode}")


def _infer_family(tokens: set[str], raw: str) -> str | None:
    for fam in MODEL_FAMILIES:
        if fam in tokens:
            return fam
    for fam in MODEL_FAMILIES:
        if fam in raw:
            return fam
    return None


def prepare_aa_by_family(aa_scores: dict[str, float]) -> dict[str, list[AaEntry]]:
    by_family: dict[str, list[AaEntry]] = defaultdict(list)
    for name, score in aa_scores.items():
        raw = name.lower()
        toks = set(_tokenize(raw))
        fam = _infer_family(toks, raw)
        if not fam:
            continue
        by_family[fam].append(AaEntry(name=name, score=float(score), tokens=frozenset(toks)))
    return dict(by_family)


def fuzzy_match_model(model_id: str, aa_by_family: dict[str, list[AaEntry]]) -> float | None:
    ml = model_id.lower()
    if ml in MANUAL_INTELLIGENCE:
        return MANUAL_INTELLIGENCE[ml]

    parts = _tokenize(ml.replace("/", " "))

    or_tokens = set(parts)
    or_family = _infer_family(or_tokens, ml)
    if not or_family:
        return None

    salient = {
        "plus",
        "pro",
        "max",
        "mini",
        "nano",
        "ultra",
        "flash",
        "turbo",
        "preview",
        "reasoning",
        "instruct",
        "chat",
        "vision",
        "vl",
        "coder",
        "code",
        "r",
    }
    or_salient = {t for t in or_tokens if t in salient}
    or_nums = {t for t in or_tokens if any(c.isdigit() for c in t)}

    best_key: tuple[float, float, int, int, int] | None = None
    best_match: float | None = None

    candidates = aa_by_family.get(or_family, [])
    for entry in candidates:
        aa_parts = entry.tokens

        common = len(or_tokens & aa_parts)
        if common == 0:
            continue

        missing_salient = len(or_salient - aa_parts)
        num_bonus = len(or_nums & aa_parts)
        union = len(or_tokens | aa_parts)
        jacc = common / union if union else 0.0

        # Prefer matches that (a) share more tokens, (b) match versions, and (c) don't drop key modifiers.
        key = (
            float(common * 3 + num_bonus * 2 - missing_salient * 2),
            float(jacc),
            int(common),
            int(-missing_salient),
            int(-abs(len(aa_parts) - len(or_tokens))),
        )

        if best_key is None or key > best_key:
            best_key = key
            best_match = entry.score

    # Require a minimum overlap strength to avoid accidental matches in large AA universes.
    return best_match if best_key is not None and best_key[0] >= 4 else None


def percentile_values_model_weighted(values: list[float], percentiles: list[int]) -> dict[str, float | None]:
    if not values:
        return {f"p{p}": None for p in percentiles}
    sorted_vals = sorted(values)
    out: dict[str, float] = {}
    for p in percentiles:
        idx = int(len(sorted_vals) * p / 100)
        idx = min(idx, len(sorted_vals) - 1)
        out[f"p{p}"] = float(sorted_vals[idx])
    return out


def pick_snapshot_date(date: str, snapshot_dates: list[str], strategy: str) -> str | None:
    if not snapshot_dates:
        return None

    # Keep `snapshot_dates` sorted, compare lexicographically (YYYY-MM-DD).
    if strategy == "nearest":
        target = _parse_date(date)
        best = None
        best_dist = None
        for d in snapshot_dates:
            dist = abs((_parse_date(d) - target).days)
            if best is None or dist < best_dist:
                best = d
                best_dist = dist
        return best

    if strategy == "asof_prev":
        prev = [d for d in snapshot_dates if d <= date]
        return prev[-1] if prev else snapshot_dates[0]

    if strategy == "asof_next":
        nxt = [d for d in snapshot_dates if d >= date]
        return nxt[0] if nxt else snapshot_dates[-1]

    raise ValueError(f"Unknown snapshot-date strategy: {strategy}")


def universe_and_pricing_for_date(
    date: str,
    pricing_static: dict[str, Pricing],
    ia_pricing_by_date: dict[str, dict[str, Pricing]],
    ia_dates: list[str],
    activity_presence_by_date: dict[str, set[str]] | None,
    universe_strategy: str,
    snapshot_pick_strategy: str,
) -> dict[str, Pricing]:
    """
    Returns {model_id: Pricing} for the model universe on `date`.
    """
    if universe_strategy == "static_2025_12_31":
        return pricing_static

    if universe_strategy == "activity_nonzero_tokens":
        if not activity_presence_by_date:
            return {}
        mids = activity_presence_by_date.get(date, set())
        if not mids:
            return {}
        # Pricing source: IA snapshot (selected) with per-model fallback to static.
        snap = pick_snapshot_date(date, ia_dates, snapshot_pick_strategy)
        snap_pricing = ia_pricing_by_date.get(snap, {}) if snap else {}
        out: dict[str, Pricing] = {}
        for mid in mids:
            out[mid] = snap_pricing.get(mid) or pricing_static.get(mid) or Pricing(0.0, 0.0, None)
        return out

    if universe_strategy == "ia_snapshot":
        snap = pick_snapshot_date(date, ia_dates, snapshot_pick_strategy)
        if snap is None:
            return pricing_static
        return ia_pricing_by_date.get(snap, pricing_static)

    if universe_strategy == "ia_snapshot_then_static":
        if not ia_dates:
            return pricing_static
        last = ia_dates[-1]
        if date > last:
            return pricing_static
        snap = pick_snapshot_date(date, ia_dates, snapshot_pick_strategy)
        if snap is None:
            return pricing_static
        return ia_pricing_by_date.get(snap, pricing_static)

    if universe_strategy == "ia_union_upto":
        if not ia_dates:
            return pricing_static
        eligible = [d for d in ia_dates if d <= date]
        if not eligible:
            eligible = [ia_dates[0]]
        # For each model_id, use last-seen pricing <= date (or earliest snapshot).
        out: dict[str, Pricing] = {}
        for d in eligible:
            for mid, pr in ia_pricing_by_date[d].items():
                out[mid] = pr
        return out

    if universe_strategy == "ia_union_upto_then_static":
        if not ia_dates:
            return pricing_static
        last = ia_dates[-1]
        if date > last:
            return pricing_static
        eligible = [d for d in ia_dates if d <= date]
        if not eligible:
            eligible = [ia_dates[0]]
        out: dict[str, Pricing] = {}
        for d in eligible:
            for mid, pr in ia_pricing_by_date[d].items():
                out[mid] = pr
        return out

    if universe_strategy == "ia_intersection":
        if not ia_dates:
            return pricing_static
        eligible = [d for d in ia_dates if d <= date]
        if not eligible:
            eligible = [ia_dates[0]]
        common = None
        for d in eligible:
            mids = set(ia_pricing_by_date[d].keys())
            common = mids if common is None else (common & mids)
        if not common:
            return {}
        # Price from last eligible snapshot.
        last = eligible[-1]
        snap = ia_pricing_by_date.get(last, {})
        return {mid: snap[mid] for mid in common if mid in snap}

    raise ValueError(f"Unknown universe strategy: {universe_strategy}")


def compute_series(
    dates: list[str],
    pricing_static: dict[str, Pricing],
    ia_pricing_by_date: dict[str, dict[str, Pricing]],
    ia_dates: list[str],
    aa_scores: dict[str, float],
    activity_presence_by_date: dict[str, set[str]] | None,
    *,
    universe_strategy: str,
    snapshot_pick_strategy: str,
    price_mode: str,
    exclude_free: bool,
    min_prompt_price_usd_per_m: float,
    percentiles: list[int],
) -> list[dict[str, object]]:
    aa_by_family = prepare_aa_by_family(aa_scores)
    rows: list[dict[str, object]] = []
    for date in dates:
        pricing_universe = universe_and_pricing_for_date(
            date=date,
            pricing_static=pricing_static,
            ia_pricing_by_date=ia_pricing_by_date,
            ia_dates=ia_dates,
            activity_presence_by_date=activity_presence_by_date,
            universe_strategy=universe_strategy,
            snapshot_pick_strategy=snapshot_pick_strategy,
        )

        ratios: list[float] = []
        for mid, pr in pricing_universe.items():
            if exclude_free and pr.prompt_price_usd_per_token <= 0:
                continue
            if (pr.prompt_price_usd_per_token * 1e6) < min_prompt_price_usd_per_m:
                continue

            intel = fuzzy_match_model(mid, aa_by_family)
            if not intel or intel <= 0:
                continue

            if price_mode == "prompt":
                price = pr.prompt_price_usd_per_token
            elif price_mode == "blended_3p1c":
                price = (3.0 * pr.prompt_price_usd_per_token + pr.completion_price_usd_per_token) / 4.0
            else:
                raise ValueError(f"Unknown price_mode: {price_mode}")

            ratios.append((price * 1e6) / intel)

        pcts = percentile_values_model_weighted(ratios, percentiles)
        rows.append({"date": date, **pcts, "n_models": len(ratios)})

    return rows


def _rolling_mean(series: list[float | None], window: int) -> list[float | None]:
    out: list[float | None] = []
    buf: list[float] = []
    for v in series:
        if v is None or not math.isfinite(v):
            buf.append(float("nan"))
        else:
            buf.append(float(v))
        if len(buf) > window:
            buf.pop(0)
        vals = [x for x in buf if math.isfinite(x)]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def score_fit(paper: list[dict[str, str]], candidate: list[dict[str, object]], percentiles: list[int]) -> dict[str, float]:
    cand_by_date: dict[str, dict[str, object]] = {str(r["date"]): r for r in candidate}
    paper_dates = [str(r["date"]) for r in paper]

    out: dict[str, float] = {}
    out["n_days"] = float(len(paper_dates))

    for p in percentiles:
        abs_errs: list[float] = []
        pct_errs: list[float] = []
        for r in paper:
            d = str(r["date"])
            c = cand_by_date.get(d)
            if not c:
                continue
            paper_val = float(r[f"p{p}"])
            cand_val_obj = c.get(f"p{p}")
            if cand_val_obj is None:
                continue
            cand_val = float(cand_val_obj)
            diff = cand_val - paper_val
            abs_errs.append(abs(diff))
            if paper_val != 0:
                pct_errs.append(abs(diff / paper_val) * 100.0)

        out[f"mae_p{p}"] = float(sum(abs_errs) / len(abs_errs)) if abs_errs else float("nan")
        out[f"mape_p{p}"] = float(sum(pct_errs) / len(pct_errs)) if pct_errs else float("nan")

    maes = [out[f"mae_p{p}"] for p in percentiles if math.isfinite(out[f"mae_p{p}"])]
    out["mae_mean"] = float(sum(maes) / len(maes)) if maes else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-02")
    ap.add_argument("--end", default="2025-12-21")
    ap.add_argument("--rolling", type=int, default=14, help="Rolling window to apply to candidate series")
    ap.add_argument("--aa-file", type=Path, default=DEFAULT_AA_FILE)
    ap.add_argument(
        "--intelligence",
        choices=["full", "avg8_available", "avg8_require_all", "direct_only", "regress_8bench", "regress_math_coding"],
        default="full",
    )
    ap.add_argument("--price-mode", choices=["prompt", "blended_3p1c"], default="prompt")
    ap.add_argument("--exclude-free", action="store_true", default=True)
    ap.add_argument("--include-free", dest="exclude_free", action="store_false")
    ap.add_argument("--min-prompt-price-usd-per-m", type=float, default=0.0)
    args = ap.parse_args()

    paper = load_paper()
    paper = [r for r in paper if args.start <= str(r["date"]) <= args.end]
    if not paper:
        raise ValueError("No paper rows in requested date window")

    pricing_static = load_static_pricing()
    ia_pricing_by_date, ia_dates = load_ia_pricing_by_snapshot_date()
    aa_scores = load_intelligence_scores(args.aa_file, args.intelligence)
    activity_presence_by_date = load_activity_presence_by_date()

    percentiles = [10, 25, 50, 75, 90]
    candidate_dates = [str(r["date"]) for r in paper]

    variants = []
    universe_strategies = [
        ("static_2025_12_31", None),
        ("activity_nonzero_tokens", "asof_prev"),
        ("ia_snapshot", "nearest"),
        ("ia_snapshot", "asof_prev"),
        ("ia_snapshot", "asof_next"),
        ("ia_union_upto", "asof_prev"),
        ("ia_snapshot_then_static", "asof_prev"),
        ("ia_union_upto_then_static", "asof_prev"),
        ("ia_intersection", "asof_prev"),
    ]

    for universe, snap_pick in universe_strategies:
        snap_pick = snap_pick or "asof_prev"
        cand = compute_series(
            candidate_dates,
            pricing_static,
            ia_pricing_by_date,
            ia_dates,
            aa_scores,
            activity_presence_by_date,
            universe_strategy=universe,
            snapshot_pick_strategy=snap_pick,
            price_mode=args.price_mode,
            exclude_free=args.exclude_free,
            min_prompt_price_usd_per_m=args.min_prompt_price_usd_per_m,
            percentiles=percentiles,
        )

        # Apply rolling average to candidate percentiles.
        for p in percentiles:
            key = f"p{p}"
            series = [r.get(key) for r in cand]
            rolled = _rolling_mean([None if v is None else float(v) for v in series], args.rolling)
            for r, v in zip(cand, rolled, strict=True):
                r[key] = v

        metrics = score_fit(paper, cand, percentiles)
        variants.append(
            {
                "universe": universe,
                "snapshot_pick": snap_pick,
                "rolling": args.rolling,
                "intelligence": args.intelligence,
                "price_mode": args.price_mode,
                "exclude_free": args.exclude_free,
                "min_prompt_price_usd_per_m": args.min_prompt_price_usd_per_m,
                **metrics,
            }
        )

    variants.sort(key=lambda r: (float("inf") if not math.isfinite(r["mae_mean"]) else r["mae_mean"]))

    show_cols = [
        "universe",
        "snapshot_pick",
        "intelligence",
        "price_mode",
        "min_prompt_price_usd_per_m",
        "mae_mean",
        "mae_p10",
        "mae_p50",
        "mae_p90",
        "mape_p90",
        "n_days",
    ]

    # Simple fixed-width printing.
    def fmt(v: object) -> str:
        if isinstance(v, float):
            if not math.isfinite(v):
                return "nan"
            return f"{v:.6g}"
        return str(v)

    widths = {c: max(len(c), max(len(fmt(r.get(c, ""))) for r in variants)) for c in show_cols}
    header = " ".join(c.ljust(widths[c]) for c in show_cols)
    print(header)
    print("-" * len(header))
    for r in variants:
        print(" ".join(fmt(r.get(c, "")).ljust(widths[c]) for c in show_cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
