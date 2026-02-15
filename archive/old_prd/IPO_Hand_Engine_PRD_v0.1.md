# IPO Hand Engine (Alpaca MVP) — PRD + Codex Task Pack (v0.1)

## 0. Purpose

Build a **local, event-driven research engine** that treats a newly observed instrument (IPO/new listing) like a **poker hand** with discrete “streets” (PREFLOP/FLOP/TURN/RIVER).  
At each street \(s\), compute:

- A feature vector \(I_s\) describing **volatility, liquidity, gaps, and path risk**
- A **tail-risk label/estimate** focused on **forward max drawdown** events
- Optional **unsupervised regime** assignment (“hand type”) for interpretability
- A **policy action**: `FOLD | SMALL_BET | SIZE_UP | HEDGE | EXIT`
- A **risk report** explaining the decision and key drivers

**Non-goal (MVP):** do not claim alpha or return prediction. The primary objective is **calibrated disaster avoidance**.

---

## 1. Objective and Evaluation

### 1.1 Primary Target (Supervised Labels)
For each street-as-of time \(t\) and horizon \(h\in\{1,3,5,10\}\) trading days:

\[
y_h(t) = \mathbf{1}\{\mathrm{MDD}_{t\rightarrow t+h} \le -25\%\}
\]

Where forward max drawdown is computed on the price path over the horizon.

### 1.2 Policy Objective (Research)
Given estimated risk \(\hat p_h(t)\), choose actions to reduce realized tail events subject to constraints:

- Avoid-rate \(\le 30\%\) (do not trivially avoid everything)
- Turnover \(\le X\) (define later; start with “trade at most once per street”)

**Primary metric:** tail-event recall (catch disasters), plus calibration error (Brier/ECE) for \(\hat p_h\).

---

## 2. Current Status (Completed)

### 2.1 Environment / Repo
- Windows + PowerShell + VS Code
- `.venv` created and working
- `src/` layout with isolated modules
- `.env` / `.env.example` and `.gitignore` set up

### 2.2 Data Ingestion
- Alpaca market data integration via `alpaca-py`
- Canonical daily bar schema stored to Parquet:
  - `data/raw/{SYMBOL}/bars_1d.parquet`
  - Columns: `symbol, ts, open, high, low, close, volume, vwap, trade_count`
- Caching verified (`from_cache=True` on second run)
- Sorting verified by `ts`

### 2.3 Streets
- Street window generator using trading days inferred from cached bars
- Option A:
  - FLOP: day index 0 (1 day)
  - TURN: indices 1–5 (5 days)
  - RIVER: indices 6–20 (15 days)

---

## 3. System Architecture

### 3.1 Modules

**Config**
- `ipo_risk_engine/config/settings.py`
  - Loads env vars (`ALPACA_API_KEY`, `ALPACA_API_SECRET`, etc.)
  - Fail-fast validation

**Data**
- `ipo_risk_engine/data/alpaca_client.py` (thin wrapper)
- `ipo_risk_engine/data/store.py` (paths + read/write)
- `ipo_risk_engine/data/ingest.py`
  - `ingest_daily_bars(...)`
  - `normalize_daily_bars(...)`

**Features**
- `ipo_risk_engine/features/calendar.py`
  - `trading_days_from_bars_1d(...)`
- `ipo_risk_engine/features/streets.py`
  - `compute_street_windows(...)`
- `ipo_risk_engine/features/core_daily.py`
  - (next) `slice_window`, `add_basic_returns`, `compute_core_features`, `compute_forward_mdd_label`

**Report / Schema**
- `ipo_risk_engine/report/schema.py`
  - `HandState`, `RiskReport`, enums for `Action`, `Street`

---

## 4. MVP Scope (Next Milestones)

### Milestone M1 — Core Daily Feature Pipeline
**Goal:** Produce a deterministic feature dict for any street window.

Deliver:
- `slice_window(bars_1d, start, end)`
- `add_basic_returns(bars_window)`
- `compute_core_features(window)` returning 10–15 floats

Acceptance criteria:
- Works on cached `bars_1d.parquet` for at least one symbol
- Outputs are JSON-serializable floats
- No lookahead leakage (features only use data within window)

---

### Milestone M2 — Forward MDD Labeling (Primary Target)
**Goal:** Compute \(y_h\) labels per horizon with strict forward-only slicing.

Deliver:
- `compute_forward_mdd_label(bars_1d, asof, horizon_days, threshold=-0.25) -> float`
- Unit tests for:
  - horizon slicing correctness
  - MDD math on small synthetic series

Acceptance criteria:
- Deterministic label results
- No overlap mistakes at boundaries ([asof, asof+h] vs (asof, asof+h])

---

### Milestone M3 — HandState Builder (Street Snapshot)
**Goal:** At each street boundary, build a `HandState`.

Deliver:
- `build_hand_state(symbol, asof, street, bars_1d) -> HandState`
  - slices bars for the street window
  - computes features
  - computes diagnostics (row counts, missing rates)

Acceptance criteria:
- Produces `HandState` for FLOP/TURN/RIVER from cached bars

---

### Milestone M4 — Unsupervised Regimes (“Hand Types”)
**Goal:** Cluster hand snapshots into regimes for interpretability.

Deliver:
- Dataset builder: rows = (symbol, street, asof), cols = features
- Standardization (z-score) and clustering (start with KMeans, later HDBSCAN)
- Outputs:
  - `regime_id`
  - cluster summaries (mean feature profiles)
  - stability checks (optional)

Acceptance criteria:
- Can assign regime to a new HandState
- Generates a small textual interpretation per regime (based on top feature z-scores)

---

### Milestone M5 — Baseline Policy + Report
**Goal:** Map risk estimates and/or regimes to actions.

Deliver:
- Simple threshold policy using:
  - regime (optional)
  - core features (and later \(\hat p_h\))
- `render_report(HandState) -> RiskReport`

Acceptance criteria:
- Report includes:
  - action + confidence
  - top drivers and key metrics
  - assumptions list

---

### Milestone M6 — Backtest Harness (Research)
**Goal:** Replay street decisions across symbols.

Deliver:
- Event loop over (symbol, street boundary)
- Transaction cost proxy (simple)
- Metrics: avoid-rate, turnover, tail-event recall

Acceptance criteria:
- Single command produces summary metrics and saves results artifacts

---

## 5. Data Conventions and Safety

- Never print or log Alpaca secrets.
- `.env` must remain local; `.env.example` is committed.
- All cached data saved under `data/` and ignored by git.
- Canonical time column is `ts` (UTC).

---

## 6. Codex Task List + Prompts

Use the following as **atomic tasks**. Each task should be completed with:
- minimal code additions
- docstrings
- 1–2 tests when relevant
- no overengineering

### Task 1 — Implement `slice_window` (core_daily.py)
**Prompt for Codex:**
> Implement `slice_window(bars_1d, start, end)` in `src/ipo_risk_engine/features/core_daily.py`.  
> Requirements: validate required columns, filter `[start, end)`, preserve/ensure sort by `ts`, return Polars DataFrame. Add a small test using a tiny synthetic dataframe.

---

### Task 2 — Implement `add_basic_returns` (core_daily.py)
**Prompt for Codex:**
> Implement `add_basic_returns(bars)` in `core_daily.py`.  
> Add columns: `ret_1d`, `logret_1d`, `gap_oc`, `intraday_ret` using Polars expressions and `shift(1)`.  
> Do not drop nulls; first row should have null lag-based values. Add unit test on synthetic 3-row series.

---

### Task 3 — Implement `compute_forward_mdd_label` (core_daily.py)
**Prompt for Codex:**
> Implement `compute_forward_mdd_label(bars_1d, asof, horizon_days, threshold=-0.25) -> float`.  
> Slice forward window strictly after `asof` up to `horizon_days` trading rows.  
> Compute running peak of `close`, drawdown series `close/peak - 1`, take min drawdown.  
> Return 1.0 if min_drawdown <= threshold else 0.0.  
> Add tests on synthetic paths with known MDD.

---

### Task 4 — Implement `compute_core_features` (core_daily.py)
**Prompt for Codex:**
> Implement `compute_core_features(window)` returning dict[str, float] of 10–15 features.  
> Use `add_basic_returns` first.  
> Include: realized_vol (std of logret), range_mean, cum_return, worst_day_return, best_day_return, trend_strength, dollar_volume_mean, amihud_mean, trade_count_mean (if present), max_drawdown_in_window.  
> Ensure JSON-serializable floats; handle small windows gracefully.

---

### Task 5 — Build `build_hand_state` function (new module `features/hand.py`)
**Prompt for Codex:**
> Create `src/ipo_risk_engine/features/hand.py` with `build_hand_state(symbol, asof, street, bars_1d, street_window) -> HandState`.  
> It should slice the bars for the window, compute features, add diagnostics (rows, date range), and return HandState.

---

### Task 6 — Create snapshot dataset builder for clustering
**Prompt for Codex:**
> Create `src/ipo_risk_engine/policy/regimes.py` to:  
> (1) build a tabular dataset from multiple HandStates;  
> (2) standardize features;  
> (3) run KMeans with configurable k;  
> (4) attach regime_id;  
> (5) provide regime summaries (mean feature z-scores).  
> Keep it simple and deterministic.

---

### Task 7 — Baseline policy (rules)
**Prompt for Codex:**
> Create `src/ipo_risk_engine/policy/rules.py` with `decide_action(hand: HandState) -> (action, confidence, drivers)`.  
> Use a few core features (vol, amihud, worst_day_return, max_drawdown_in_window) to map to actions.  
> Confidence should be a deterministic function of margin-to-threshold.

---

### Task 8 — Report rendering
**Prompt for Codex:**
> Create `src/ipo_risk_engine/report/render.py` with `render_report(hand: HandState, decision) -> RiskReport`.  
> Include drivers, metrics subset, assumptions list. No external formatting dependencies required.

---

### Task 9 — CLI wiring (Typer)
**Prompt for Codex:**
> Create `src/ipo_risk_engine/cli.py` using Typer with commands:  
> `ingest SYMBOL --days 30`  
> `run SYMBOL` (build HandState at each street boundary and print RiskReport)  
> Store artifacts under `data/reports/`.

---

## 7. Runbook (Developer)

### Activating environment (Windows PowerShell)
- From repo root:
  - `.\.venv\Scripts\Activate.ps1`
- Confirm prompt shows `(.venv)`

### Installing deps
- `python -m pip install -r requirements.txt` (optional later if you add one)
- Or install via pip as done during setup.

### Smoke scripts
- `python scripts/ingest_daily_pltr.py`
- `python scripts/test_streets_pltr.py`

---

## 8. Risks / Notes

- IPO calendars and listing dates are not handled by Alpaca alone; MVP can treat “first observed date” as listing_day.
- Intraday FLOP (5m/1m bars) is a Phase-2 extension:
  - Add `ingest_bars_1m`, normalized schema, and adjust FLOP windows to market hours.

---
