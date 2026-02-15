# PREFLOP Street Plan (Feasibility + Implementation Path)

## Short Answer

Yes, adding a `PREFLOP` street is feasible and can improve the project if you treat it as a **prior-risk signal** (not a replacement for RIVER).  
The strongest use case is improving **early-street risk calibration** (FLOP/TURN) and reducing false-safe decisions.

## Why It Can Improve Signal

Current findings show:
- RIVER has strong path-risk signal.
- FLOP has weak but non-zero signal.
- TURN is mostly flat.

`PREFLOP` can add **orthogonal information** unavailable in price bars:
- issuance dynamics (price revisions, amendment frequency),
- attention/hype intensity before listing,
- market context at listing time.

This should help most where the model is currently weakest: **before enough trading history exists**.

## Critical Pushback (What Can Go Wrong)

- Tiny sample size (94 symbols) makes rich NLP easy to overfit.
- Timestamp leakage is the biggest risk for web data.
- Source drift (site layout/API changes) can break reproducibility.
- Unstructured text can increase noise more than signal if not constrained.

Design rule: start with low-dimensional, timestamp-audited features first; add NLP only after baseline lift is proven.

## Data Scope (V1 -> V2)

### V1: Structured Prelisting Features (recommended first)

Use fields that are deterministic and easier to validate:
- SEC filing cadence features:
  - `days_from_first_s1_to_ipo`
  - `s1_amendment_count_30d`
  - `days_since_last_amendment`
- Offering revision features:
  - `offer_price_revision_pct`
  - `offer_size_revision_pct`
- Deal/context features:
  - `lead_underwriter_tier` (coarse bucket)
  - `sector_ipo_heat_90d` (from your own IPO universe)
  - `preipo_market_vol` (SPY/sector ETF realized vol before IPO)

### V2: Lightweight NLP Hype Features

Only after V1 shows lift:
- `headline_count_7d` (attention intensity)
- `hype_lexicon_score_7d` (simple dictionary score, no LLM dependency)
- `sentiment_dispersion_7d` (std of headline scores)
- `hype_acceleration_3d_vs_14d`

Keep NLP small, explainable, and reproducible.

## Time-Correctness Requirements (Non-Negotiable)

For every preflop record, store:
- `published_ts_utc`
- `ingested_ts_utc`
- `source_id`
- `symbol`

Feature cutoff for each symbol must be:
- `published_ts_utc < listing_day 09:30 ET`

No exceptions. If timestamp is missing/unreliable, drop the record.

## Proposed Storage + Schema

- Raw events: `data/raw_preflop/{SYMBOL}/events.parquet`
- Curated features: `data/features/preflop_features.parquet`
- Schema module: `src/ipo_risk_engine/data/preflop_schema.py`
- Builder module: `src/ipo_risk_engine/features/preflop_features.py`

Suggested event schema:
- `symbol: str`
- `event_type: str` (`filing`, `news`, `pricing_update`)
- `published_ts_utc: datetime`
- `source: str`
- `title: str | null`
- `body_text: str | null`
- `numeric_payload: struct | null`

## Integration Into Existing Streets

1. Add `PREFLOP` handling in `src/ipo_risk_engine/features/streets.py` as a no-bars street.
2. Build one `PREFLOP` snapshot per symbol in `src/ipo_risk_engine/snapshots/builder.py`.
3. Treat preflop features as prior context that can be appended to FLOP/TURN/RIVER feature sets.
4. Update `src/ipo_risk_engine/dataset/assemble.py` to include preflop columns in dataset output.

## Modeling Strategy (Defensible with Small N)

1. Baseline first:
   - Preflop-only classifier for `tail_event` (regularized logistic or ridge-style proxy).
2. Incremental lift tests:
   - FLOP vs FLOP+PREFLOP
   - TURN vs TURN+PREFLOP
   - RIVER vs RIVER+PREFLOP
3. Primary success metrics:
   - Brier Skill Score vs climatology
   - false-safe rate change at matched coverage
   - calibration reliability, not just correlation

## Milestone Plan

### M-P1: Data Contract + Ingestion
- Define schemas, source adapters, and timestamp rules.
- Output: validated `events.parquet` per symbol.

### M-P2: V1 Structured Features
- Implement deterministic numeric preflop features only.
- Output: `preflop_features.parquet`.

### M-P3: Snapshot Integration
- Add PREFLOP snapshot + carry-forward into later streets.
- Output: rebuilt train/val/test datasets.

### M-P4: Lift Evaluation
- Run ablations and calibration checks by street.
- Output: table of metric deltas and confidence intervals.

### M-P5: V2 NLP (Optional, gated)
- Add minimal lexicon hype module only if M-P4 is positive.
- Output: incremental lift report vs V1.

## Go/No-Go Criteria

Proceed with full preflop rollout only if both hold:
- `FLOP+PREFLOP` improves Brier Skill Score over FLOP baseline by >= 5%.
- Policy false-safe rate decreases by >= 10% at comparable coverage.

If not, keep preflop as an exploratory module and do not wire it into policy.

## Suggested Official Docs to Read Before Implementation

1. Polars docs: joins/groupby window patterns for time-constrained feature assembly.
2. scikit-learn docs: probability calibration and Brier score interpretation.

## Practical Recommendation

Implement V1 structured preflop features first.  
Do not start with full scraped NLP pipelines. With current sample size, a small, auditable feature set is far more likely to produce defensible improvements.
