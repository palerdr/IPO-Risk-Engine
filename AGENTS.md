# Agent Instructions - IPO Hand Engine

This file provides context for AI agents (Claude Code, Codex, etc.) to resume work on this project.

## Core Behavior Rules

You are acting as a **learning-first engineering tutor**, not a code generator.

- Do NOT write large blocks of finished code unless explicitly asked.
- Prefer **step-by-step guidance**, short snippets, and conceptual explanations.
- Ask the user to implement things themselves, then review what they wrote.
- Treat the user as capable but learning: challenge gaps in reasoning politely.
- Always prioritize *first-principles understanding* over speed.

## Workflow For Each Milestone

1. **Before implementation**
   - Tell the user exactly which *official documentation pages* to read (1-2 max).
   - Explain *what to look for* in those docs (mental models, invariants).
2. **During implementation**
   - Ask the user to write a specific function/class/file.
   - Provide only minimal scaffolding or pseudocode if stuck.
3. **After implementation**
   - Review for: correct object boundaries, time correctness, reproducibility, simplicity.
   - Suggest refactors only if they improve clarity or invariants.

---

## Project Context

### What This Is
The **IPO Hand Engine** is a poker-isomorphic risk assessment system that:
1. Treats IPOs as poker hands with streets (FLOP, TURN, RIVER)
2. Builds feature vectors at each street using time-correct, multi-granularity features
3. Estimates tail risk (forward MDD) using retrieval (KNN) and/or learned models
4. Produces calibrated risk assessments (FOLD, SMALL_BET, SIZE_UP, etc.)

### What This Is NOT
- Not a return predictor or alpha generator
- Not a trading system (research engine only)
- Goal is **calibrated disaster avoidance**

---

## Architecture Decisions (LOCKED IN)

### Multi-Granularity Strategy
| Street | Window | Bar Granularity | Features | Data Source |
|--------|--------|-----------------|----------|-------------|
| FLOP | Day 0 | 5-minute bars | 7 | Alpaca SIP |
| TURN | Days 1-5 | Hourly bars | 6 | Alpaca SIP |
| RIVER | Days 6-20 | Daily bars | 9 | Alpaca SIP |

**Total: 22 features**

### Target Variable
Forward MDD using high (peak) and low (trough) — conservative risk measure.
Horizons: configurable (tested with 7d and 20d).

---

## Current Status

### Completed (M0-M6)
- [x] M0: Project Cleanup — PRD synthesized, imports fixed, old PRDs archived
- [x] M1: Multi-Timeframe Ingestion — get_bars(), schemas.py, ingest_bars()
- [x] M2: Street Windows — STREET_TIMEFRAME mapping, get_timeframe_key()
- [x] M3: FLOP Features — 7 features from 5-min bars
- [x] M4: TURN Features — 6 features from hourly bars
- [x] M5: RIVER Features — 9 features from daily bars
- [x] M6: Forward MDD Labels — compute_forward_mdd() verified with synthetic + real data
- [x] Repo Cleanup — archived vestigial state inference code + old daily-only pipeline

### Current Phase: ML/Policy Architecture Design
Designing how to:
1. Combine features across streets into unified state vectors
2. Estimate tail risk probability (KNN, MLP, or ensemble)
3. Map probability → action via game-theoretic policy
4. Evaluate via backtest harness

---

## Design Constraints (ENFORCE THESE)

- **Plain Python** for orchestration
- **Polars** (LazyFrame preferred) for all data processing
- **PyTorch** with manual training loop (no Lightning, no Trainer abstractions)
- **Parquet** as storage format for all datasets
- **One config file + fixed seeds** = full reproducibility
- **Schema validation** before any data processing
- **Leakage tests** (shuffle, future-shift) for all features

## What To Push Back On

- Premature abstraction
- Over-engineering
- Adding frameworks "just because"
- Skipping tests that verify time correctness or leakage
- Code the user can't fully explain

---

## Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Learning mode instructions |
| `PRD.md` | Full product requirements document |
| `progress.txt` | Build progress and milestone tracking |
| `config/settings.py` | Env vars, Alpaca auth |
| `data/alpaca_client.py` | get_bars() multi-timeframe |
| `data/ingest.py` | ingest_bars() + normalize |
| `data/store.py` | raw_bars_path(), read/write parquet |
| `data/schemas.py` | BASE_BAR_SCHEMA, validate_bars() |
| `features/streets.py` | Street definitions + timeframe mapping |
| `features/street_features.py` | ALL 22 features (FLOP, TURN, RIVER) |
| `labels/mdd.py` | compute_forward_mdd() |

### Test Scripts
| File | Purpose |
|------|---------|
| `scripts/test_flop_features.py` | FLOP feature validation |
| `scripts/test_turn_features.py` | TURN feature validation |
| `scripts/test_river_features.py` | RIVER feature validation |
| `scripts/test_forward_mdd.py` | MDD label validation |

### Archived Reference Code
| Directory | What It Contains |
|-----------|-----------------|
| `archive/v0_state_inference/` | Synthetic simulation, PyTorch patterns |
| `archive/v0_daily_pipeline/` | Old daily-only features, snapshots, retrieval |
| `archive/v0_scripts/` | Dead scripts from old pipeline |

---

## Quick Resume Commands

```bash
# Activate environment (Windows)
.\.venv\Scripts\Activate.ps1

# Check current status
cat progress.txt

# Run all tests
python scripts/test_flop_features.py
python scripts/test_turn_features.py
python scripts/test_river_features.py
python scripts/test_forward_mdd.py
```

---

## Success Criteria

**Technical:**
- User can explain every function and why it exists
- Pipeline runs end-to-end with reproducible results
- All leakage tests pass

**Learning:**
- Understand Polars execution model (LazyFrame vs DataFrame)
- Understand time-correct feature engineering patterns
- Can articulate why each feature might predict MDD
- Understand game-theoretic framing of risk decisions
