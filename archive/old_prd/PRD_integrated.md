# IPO Hand Engine - Integrated PRD

## 1. Vision

Build a **local, event-driven research engine** that treats IPOs as poker hands with discrete streets. At each street, estimate tail risk and produce actionable assessments.

**Core insight:** IPO price paths exhibit regime-like behavior that can be characterized and compared to historical patterns.

**Non-goal:** This is a **disaster-avoidance engine**, not a return predictor or alpha generator.

---

## 2. The Poker Isomorphism

| Poker | IPO Trading | Window |
|-------|-------------|--------|
| PREFLOP | Pre-listing / Day 0 | First day |
| FLOP | Early price discovery | Days 1-5 |
| TURN | Trend establishment | Days 6-20 |
| RIVER | Mature trading | Days 21+ |

At each street:
- **Board texture** = feature vector describing volatility, liquidity, gaps, path risk
- **Hand strength** = estimated tail risk from historical comps
- **Hand type** = regime/cluster label for interpretability
- **Action** = FOLD / SMALL_BET / SIZE_UP / HEDGE / EXIT

---

## 3. Primary Target (Labels)

For each street-as-of time `t` and horizon `h` trading days:

```
y_h(t) = 1 if forward_MDD(t, t+h) <= -25% else 0
```

Where forward max drawdown measures the worst peak-to-trough decline over the horizon.

**Horizons:** h = 1, 3, 5, 10 trading days

---

## 4. Evaluation Objective

Policy chooses actions to reduce tail events subject to:
- Avoid-rate <= 30% (don't trivially avoid everything)
- Turnover constraint (one decision per street)

**Primary metrics:**
- Tail-event recall (catch disasters)
- Calibration error (Brier / ECE)

---

## 5. System Architecture

### 5.1 Data Layer

| Source | Purpose |
|--------|---------|
| Alpaca API | Real daily bars for IPOs |
| Synthetic Simulation | Testing pipelines on known ground truth |

### 5.2 Feature Layer

**Daily Features (core_daily.py):**
- Returns: ret_1d, logret_1d, gap_oc, intraday_ret
- Volatility: realized_vol, range_mean
- Liquidity: dollar_volume_mean, amihud_mean
- Path: cum_return, max_drawdown, worst_day, best_day

**Street Features (pipeline.py):**
- Rolling aggregations over street windows
- Time-correct: only use data with ts <= current

### 5.3 Inference Layer

**Retrieval (KNN):**
- Standardize features
- Find K nearest historical street snapshots
- Estimate p_hat = mean(y_h among neighbors)

**Learned (optional):**
- MLP/GRU classifiers for regime inference
- Trained on synthetic data, validated on real
- Provides "hand type" for interpretability

### 5.4 Policy Layer

- Map risk estimates to actions
- Simple threshold rules initially
- Produce risk reports with drivers and assumptions

---

## 6. Repository Structure

```
ipo-risk-engine/
├── CLAUDE.md                 # Learning mode instructions
├── PRD_integrated.md         # This file
├── data/
│   ├── raw/                  # Alpaca bars + synthetic events
│   ├── processed/            # Features, supervised datasets
│   └── cache/                # API response cache
├── artifacts/
│   └── runs/                 # Model checkpoints, reports
├── scripts/
│   ├── ingest.py             # Ingest Alpaca data
│   ├── simulate.py           # Generate synthetic data
│   ├── build_features.py     # Feature pipeline
│   ├── build_snapshots.py    # Street snapshot builder
│   ├── train.py              # Train models
│   └── evaluate.py           # Evaluation and reports
├── src/ipo_risk_engine/
│   ├── config/
│   │   └── settings.py       # Env vars, paths
│   ├── data/
│   │   ├── alpaca_client.py  # Alpaca API wrapper
│   │   ├── store.py          # Data I/O
│   │   └── schema.py         # Polars schemas
│   ├── sim/                  # Synthetic simulation
│   │   ├── regime.py         # Latent regime process
│   │   ├── events.py         # Event generator
│   │   └── generator.py      # Orchestrator
│   ├── features/
│   │   ├── calendar.py       # Trading day calendar
│   │   ├── streets.py        # Street window definitions
│   │   ├── core_daily.py     # Daily feature primitives
│   │   ├── spec.py           # Feature specifications
│   │   └── pipeline.py       # Feature pipeline
│   ├── torch/
│   │   ├── data.py           # Dataset classes
│   │   ├── models.py         # MLP, GRU
│   │   └── train.py          # Training loop
│   ├── policy/
│   │   ├── retrieval.py      # KNN comps engine
│   │   ├── regimes.py        # Regime clustering
│   │   └── rules.py          # Action decisions
│   ├── report/
│   │   ├── schema.py         # HandState, RiskReport
│   │   └── render.py         # Report rendering
│   └── eval/
│       └── metrics.py        # Evaluation metrics
└── tests/
```

---

## 7. Milestones (Rebuild with Ownership)

### Phase 1: Foundation (Rebuild from First Principles)

**M0: Project Setup**
- Clean repo structure
- Config + seeds + reproducibility
- Verify Alpaca connection

**M1: Data Ingestion**
- Alpaca daily bar ingestion
- Parquet storage with schema validation
- Trading day calendar

**M2: Street Windows**
- Define street boundaries
- Slice bars into street windows
- Unit tests for boundary correctness

### Phase 2: Feature Engineering

**M3: Core Daily Features**
- Basic returns (ret_1d, logret_1d, gap_oc)
- Volatility proxies
- Liquidity measures
- **Leakage tests**

**M4: Forward MDD Labels**
- Strict forward-only slicing
- MDD computation
- Binary threshold labels

**M5: Street Snapshots**
- Build HandState at each street boundary
- Feature + label alignment
- Snapshot parquet dataset

### Phase 3: Inference

**M6: Retrieval Engine (KNN)**
- Feature standardization
- KNN within same street
- p_hat estimation + diagnostics

**M7: Regime Clustering**
- Unsupervised hand types
- Cluster interpretability
- Stability checks

**M8: Learned Models (Optional)**
- MLP classifier baseline
- GRU sequence model
- Compare to KNN retrieval

### Phase 4: Policy and Evaluation

**M9: Baseline Policy**
- Threshold rules
- Action mapping
- Risk report rendering

**M10: Backtest Harness**
- Replay street decisions
- Avoid-rate, turnover, recall
- Summary artifacts

### Phase 5: Synthetic Validation

**M11: Synthetic Simulation**
- Latent regime process
- Event generation
- Ground truth labels

**M12: Pipeline Validation**
- Run full pipeline on synthetic data
- Verify time correctness
- Compare learned vs retrieval

---

## 8. Key Invariants

1. **Time correctness:** Features at time t use only data from times <= t
2. **Reproducibility:** Same seed + config = identical results
3. **Schema validation:** All data passes Polars schema checks
4. **No leakage:** Shuffle test drops accuracy to chance

---

## 9. Data Conventions

- Canonical time column: `ts` (UTC)
- Daily bars schema: `symbol, ts, open, high, low, close, volume, vwap, trade_count`
- Never log secrets (.env stays local)
- All cached data under `data/`, gitignored

---

## 10. Success Criteria

**Technical:**
- Can explain every function and why it exists
- Pipeline runs end-to-end with reproducible results
- Leakage tests pass

**Portfolio:**
- Clean, documented codebase
- Clear PRD showing product thinking
- Demonstrates: Polars, PyTorch, time-series ML, financial domain knowledge
