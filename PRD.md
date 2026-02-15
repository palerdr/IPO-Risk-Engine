# IPO Hand Engine - Product Requirements Document

## 1. Vision

Build a **local, learning-focused risk research engine** that treats IPOs as poker hands with discrete streets. At each street, quantify tail risk and produce calibrated assessments.

**Core insight:** IPO price paths exhibit regime-like behavior - chaos, discovery, trend, stabilization. Like poker hands, even "good" IPOs (strong fundamentals) can suffer disasters. The goal is calibrated disaster avoidance.

**Non-goal:** This is NOT a return predictor or alpha generator. It's a **risk quantification engine**.

---

## 2. The Poker Isomorphism

| Poker | IPO Trading | Window | Bar Granularity |
|-------|-------------|--------|-----------------|
| PREFLOP | Pre-listing context | Before Day 0 | N/A (future: SEC filings) |
| FLOP | Day 0 price discovery | Market hours Day 0 | 5-minute bars |
| TURN | Early trend establishment | Days 1-5 | Hourly bars |
| RIVER | Mature trading regime | Days 6-20 | Daily bars |

At each street:
- **Board texture** = feature vector (volatility, liquidity, path risk)
- **Hand strength** = estimated tail risk probability
- **Hand type** = regime label for interpretability (future)
- **Action** = FOLD / SMALL_BET / SIZE_UP / HEDGE / EXIT

**Why poker?** Even pocket aces lose ~15% of the time. The framework forces probabilistic thinking about risk, not deterministic predictions.

---

## 3. Primary Target (Labels)

**Two-tail regression targets** — continuous values, not binary classification:

| Label | Definition | Purpose |
|-------|-----------|---------|
| `forward_mdd_{h}d` | Worst peak-to-trough drawdown over next h trading days (high→low) | Left tail (disaster risk) |
| `forward_mfr_{h}d` | Best trough-to-peak runup over next h trading days (low→high) | Right tail (upside potential) |

**Horizons:** h ∈ {7, 20} trading days (configurable)

**Design rationale:**
- Regression preserves full signal — policy applies thresholds at decision time
- Both tails captured: MDD always ≤ 0, MFR always ≥ 0
- Path-dependent extrema (not simple returns) — captures actual experienced risk
- Using high/low (not close) for conservative intraday-aware measurement
- Tails are more predictable than means (volatility clustering)

---

## 4. Evaluation Objective

Policy chooses actions to reduce realized tail events subject to:
- **Avoid-rate ≤ 30%** (don't trivially avoid everything)
- **Turnover constraint** (one decision per street)
- **CVaR budget constraint** (hard gate, not soft penalty)

**Primary metrics:**
- Tail-event recall (catch disasters)
- Regression accuracy (MAE/RMSE on MDD and MFR predictions)
- Calibration of tail proxies (monotone mapping quality)
- Policy PnL under sizing rules

---

## 5. Feature Hypotheses (Working Theory)

### Hypothesis 1: First-Hour Volatility
High realized volatility in the first hour of Day 0 signals chaotic price discovery where participants lack information → elevated MDD risk.

**Features:** `first_hour_realized_vol`, `first_hour_range`, `first_30m_vol`

### Hypothesis 2: Retail-Dominated Flow
Large volume with small average trade size suggests retail-dominated flow without institutional anchoring → elevated MDD risk (weak hands).

**Features:** `volume_per_trade` (proxy), `volume_profile_skew`

### Hypothesis 3: Hype Fade
Price closing well below intraday high on Day 0 suggests hype exhaustion → elevated MDD risk in subsequent days.

**Features:** `close_vs_high_day0`, `intraday_reversal`

### Hypothesis 4: Gap Behavior
Overnight gaps that don't fill suggest directional conviction; gaps that fill suggest noise → gap patterns predict trend persistence.

**Features:** `gap_fill_ratio`, `overnight_gap_mean`

---

## 6. System Architecture

### 6.1 Data Layer

| Source | Granularity | Purpose |
|--------|-------------|---------|
| Alpaca API (SIP) | 5-minute | FLOP features (Day 0 intraday) |
| Alpaca API (SIP) | Hourly | TURN features (Days 1-5) |
| Alpaca API (SIP) | Daily | RIVER features, MDD labels |

**Storage:**
```
data/raw/{SYMBOL}/
  ├── bars_5m.parquet
  ├── bars_1h.parquet
  └── bars_1d.parquet
```

**Feed:** SIP for historical (15-min delay on free tier is acceptable for backtesting). Architecture supports swapping to real-time SIP later.

### 6.2 Street Layer

Street boundaries defined by trading day index from IPO date:
- FLOP: Day 0 (index 0)
- TURN: Days 1-5 (indices 1-5)
- RIVER: Days 6-20 (indices 6-20)

HandState at each street boundary contains:
- symbol, street, asof_ts
- features dict
- diagnostics (row counts, missing rates)

### 6.3 Feature Layer

**All features consolidated in `features/street_features.py`**

**FLOP features** (7 features from 5-min bars):
1. first_hour_return - Price change in first trading hour
2. rest_of_day_return - Price change after first hour
3. volume_first_hour_pct - Volume concentration in first hour
4. avg_trade_size - volume / trade_count (retail vs institutional proxy)
5. close_vs_vwap - Close deviation from VWAP
6. time_to_high_minutes - When intraday high occurred
7. intraday_mdd - Maximum drawdown on Day 0

**TURN features** (6 features from hourly bars):
1. overnight_gap_mean - Average overnight gap
2. gap_fill_ratio - Fraction of gaps that get filled
3. volume_decay_ratio - Volume trend (last/first day)
4. hourly_realized_vol - Realized volatility from hourly returns
5. cum_return - Cumulative return over TURN period
6. max_drawdown_turn - Max drawdown during TURN

**RIVER features** (9 features from daily bars):
1. realized_vol - Daily realized volatility
2. range_mean - Average normalized range (ATR proxy)
3. dollar_volume_mean - Average dollar volume
4. amihud_illiquidity - |return| / dollar_volume
5. cum_return - Cumulative return over RIVER
6. trend_strength - Return / volatility (Sharpe-like)
7. max_drawdown_river - Max drawdown during RIVER
8. worst_day_return - Most negative daily return
9. best_day_return - Most positive daily return

**Total: 22 features**

**All features must pass:**
1. Schema validation (correct dtypes)
2. Leakage tests (shuffle test, future-shift test)

### 6.4 Inference Layer

**Dual approach — KNN baseline + learned model:**

**KNN Retrieval (non-parametric baseline):**
- Standardize features per street
- Find K nearest historical street snapshots (same street)
- Return neighbor MDD/MFR values as empirical distribution
- No training required — interpretable "comps"

**MLP Regression (learned model):**
- Input: feature vector (7/13/22 dims depending on street)
- Output: (predicted_mdd, predicted_mfr) — two continuous values
- Manual PyTorch training loop, no abstractions
- Street indicator included as input (unified model)
- Loss: MSE or Huber on both targets

**Calibration layer (Option B with calibration):**
- f(predicted_mdd) → calibrated downside proxy via monotone map
- g(predicted_mfr) → calibrated upside proxy via monotone map
- Maps learned on validation data (piecewise-linear or isotonic)
- Allows speaking CVaR language without full quantile regression

### 6.5 Policy Layer

**Two-layer game-theoretic policy:**

**Layer 1 — Eligibility Gate (binary):**
- CVaR proxy `f(predicted_mdd) > risk_budget[street]` → FOLD (dominant_risk=TAIL)
- Liquidity proxy below threshold → FOLD (dominant_risk=LIQUIDITY)
- Model confidence below minimum → FOLD (dominant_risk=UNCERTAINTY)
- Purpose: no amount of small sizing fixes structurally untradeable situations

**Layer 2 — Sizing (conditional on eligibility):**
```
score = g(predicted_mfr) - λ[street] × f(predicted_mdd)
bucket = map_score_to_bucket(score, street)
```
- Output: discrete buckets {0%, 25%, 50%, 100%} of max allocation
- Discrete sizing is more stable than continuous under noisy tail regressions

**Street-dependent parameters:**
| Street | Max Size | λ (downside weight) | CVaR Budget | Min Confidence |
|--------|----------|--------------------|----|----------------|
| FLOP   | 50%      | 2.0                | 0.15 | 0.6          |
| TURN   | 100%     | 1.5                | 0.20 | 0.4          |
| RIVER  | 100%     | 1.0                | 0.25 | 0.3          |

FLOP is about survival and optionality. TURN allows scaling. RIVER allows full conviction.

**Policy output schema:**
```python
{
    eligible: bool,
    size_bucket: 0 | 0.25 | 0.5 | 1.0,
    confidence: float,
    dominant_risk: TAIL | LIQUIDITY | VOLATILITY | UNCERTAINTY
}
```

**Key constraints:**
- CVaR is a gate, not a sizing metric
- Never allow continuous sizing on FLOP
- Never reuse thresholds across streets
- Never let sizing override CVaR violations

---

## 7. Key Invariants

1. **Time correctness:** Features at time t use only data from times ≤ t
2. **Reproducibility:** Same seed + config = identical results
3. **Schema validation:** All data passes Polars schema checks before processing
4. **No leakage:** Shuffle test drops model accuracy to chance level
5. **Feed agnostic:** Code works with any Alpaca feed (IEX or SIP)

---

## 8. Milestones (Learning-Focused Build)

### Phase 0: Foundation Reset
**M0: Project Cleanup**
- Archive old PRD files
- Verify package structure and imports
- Confirm Alpaca connectivity with SIP feed
- Establish config for multi-timeframe ingestion

### Phase 1: Data Infrastructure
**M1: Multi-Timeframe Ingestion**
- Extend alpaca_client.py for 5m, 1h, 1d bars
- Schema validation for each timeframe
- Caching and incremental updates
- Unit tests for schema compliance

**M2: Street Windows (Multi-Granularity)**
- Extend street definitions to specify bar granularity
- Map street boundaries to correct bar timeframes
- Tests for boundary correctness

### Phase 2: Feature Engineering
**M3: FLOP Features (5-min bars)**
- Opening dynamics features
- First-hour volatility features
- Volume profile features
- Leakage tests

**M4: TURN Features (Hourly bars)**
- Overnight gap features
- Volume decay features
- Hourly volatility features
- Leakage tests

**M5: RIVER Features (Daily bars)**
- Standard daily features (vol, liquidity, path)
- Rebuild with schema validation
- Leakage tests

**M6: Forward MDD Labels**
- Strict forward-only slicing
- MDD computation at multiple horizons
- Label alignment with street snapshots

### Phase 3: Inference
**M7: Snapshot Builder**
- Assemble (features, mdd, mfr) tuples per street per symbol
- Handle accumulating feature vectors (FLOP=7, TURN=13, RIVER=22)
- Store as parquet for training and retrieval

**M8: Regression Model**
- KNN baseline (non-parametric)
- MLP regressor (PyTorch, manual loop)
- Train to predict (mdd, mfr) from features
- Leakage tests: shuffle should destroy performance

**M9: Calibration Layer**
- Fit monotone maps f(mdd), g(mfr) on validation data
- Isotonic or piecewise-linear calibration
- Validate calibration quality (coverage, monotonicity)

### Phase 4: Policy and Evaluation
**M10: Two-Layer Policy**
- Eligibility gate (CVaR budget per street)
- Sizing function (asymmetry score → discrete buckets)
- Street-dependent parameters
- Policy output schema

**M11: Backtest Harness**
- Replay street decisions across symbols
- Metrics: avoid-rate, recall, PnL, sizing stability
- Summary artifacts

---

## 9. Repository Structure

```
ipo-risk-engine/
├── CLAUDE.md              # Learning mode instructions
├── PRD.md                 # This document
├── progress.txt           # Build progress tracking
├── pyproject.toml
├── data/
│   ├── raw/{SYMBOL}/      # Cached bars by symbol
│   └── snapshots/         # Assembled training data
├── artifacts/
│   └── runs/              # Model checkpoints, calibration maps
├── scripts/
│   ├── test_*.py          # Validation scripts per milestone
│   └── smoke_*.py         # API connectivity checks
├── src/ipo_risk_engine/
│   ├── config/
│   │   └── settings.py    # Env vars, paths, seeds
│   ├── data/
│   │   ├── alpaca_client.py
│   │   ├── ingest.py
│   │   ├── store.py
│   │   └── schemas.py     # Bar schemas for all timeframes
│   ├── features/
│   │   ├── streets.py         # Street definitions + timeframe mapping
│   │   └── street_features.py # ALL street features (FLOP, TURN, RIVER)
│   ├── labels/
│   │   └── mdd.py         # Forward MDD + MFR computation
│   ├── snapshots/
│   │   └── builder.py     # Snapshot assembly (features + labels)
│   ├── inference/
│   │   ├── knn.py         # KNN retrieval baseline
│   │   ├── model.py       # MLP regressor (PyTorch)
│   │   └── calibration.py # Monotone calibration maps
│   ├── policy/
│   │   └── rules.py       # Two-layer policy (gate + sizing)
│   ├── report/
│   │   └── schema.py      # PolicyOutput dataclass
│   └── backtest/
│       └── harness.py     # Replay + evaluation metrics
├── tests/
│   └── ...
└── archive/               # Old code for reference
    ├── v0_state_inference/ # Synthetic simulation patterns
    ├── v0_daily_pipeline/  # Old daily-only pipeline
    ├── v0_scripts/         # Dead scripts
    └── old_prd/            # Previous PRD versions
```

---

## 10. Data Conventions

- **Canonical time column:** `ts` (UTC, timezone-aware)
- **Bar schemas:** Defined per timeframe in `data/schema.py`
- **Secrets:** Never logged, `.env` stays local
- **Cached data:** Under `data/`, gitignored
- **Seeds:** All randomness controlled via config for reproducibility

---

## 11. Success Criteria

**Technical:**
- Can explain every function and why it exists
- Pipeline runs end-to-end with reproducible results
- All leakage tests pass
- Feature hypotheses are testable and tested

**Learning:**
- Understand Polars execution model (LazyFrame vs DataFrame)
- Understand time-correct feature engineering patterns
- Understand calibration vs accuracy tradeoffs
- Can articulate why each feature might predict MDD

**Portfolio:**
- Clean, documented codebase
- Clear PRD showing product thinking
- Demonstrates: Polars, time-series ML, financial domain knowledge
