# Model Card: IPO Risk Engine v1.1

## Overview

Predicts tail risk (severe drawdown probability) for recent IPOs using RIVER-street daily bar features + PREFLOP structured metadata. Outputs a 3-action policy: FOLD (stay out), SMALL_BET (reduced position), SIZE_UP (full position).

## Model Architecture

```
Input: 63 features (49 RIVER daily-bar + 14 PREFLOP V1 metadata)
  → RiverCommittee (Ridge α=1000 + KNN k=7 ensemble, StandardScaler)
  → Isotonic calibration (OOF, 3 temporal splits)
  → Constrained threshold optimization (FSR ≤ 3% tuning margin)
  → ActionThresholds → {FOLD, SMALL_BET, SIZE_UP}
```

## Training Data

- **656 RIVER snapshots** from 656 unique IPO symbols (2016-2025)
- **Temporal range:** IPO dates from 2016 through early 2025
- **Street:** RIVER only (trading days 21-60 post-IPO)
- **Source:** Alpaca daily bars + EDGAR EFTS S-1/A filings

## Event Labels

| Label | Definition | Base Rate |
|-------|-----------|-----------|
| `adverse_20` | \|MDD_20d\| >= 20% | ~47% |
| `severe_30` | \|MDD_20d\| >= 30% | ~30% |

- `adverse_20` is the calibration target (isotonic maps score → P(adverse))
- `severe_30` is the safety constraint (FSR computed against this)

## Hard KPIs (must hold for deployment)

| KPI | Threshold | v1.1 Actual | Status |
|-----|-----------|-------------|--------|
| Worst-fold FSR | <= 10% | 7.1% | PASS |
| Fold-3 (2025) FSR | <= 10% | 5.0% | PASS |
| Tech sector FSR | <= baseline (11.8%) | 11.8% | PASS |

**FSR** (False Safe Rate): Among true severe events (|MDD_20d| >= 30%), fraction assigned SIZE_UP. This is the primary safety metric — a high FSR means the model is sending users into severe drawdowns with full positions.

## Soft KPIs (track, improve in v2)

| KPI | Target | v1.1 Actual | Status |
|-----|--------|-------------|--------|
| Fold-3 BSS lift vs baseline | >= +0.05 | +0.106 | PASS |
| Median BSS (all folds) | > 0 | 0.232 | PASS |
| Median SMALL_BET | >= 8% | 9.8% | PASS |
| Fold-3 Detection | (track) | 87.0% | — |

**BSS** (Brier Skill Score): 1 - BS_model / BS_climatology. Measures calibration quality vs always-predict-base-rate.
**SMALL_BET %**: Fraction of IPOs assigned intermediate position. Ensures 3-action policy is alive (not collapsed to binary FOLD/SIZE_UP).

## Stability Report (4 expanding temporal folds)

| Fold | Train | Test | FSR   | Det   | BSS   | SB%   |
|------|-------|------|-------|-------|-------|-------|
| 0    | ≤2021 | 2022 | 0.0%  | 94.5% | 0.094 | 16.9% |
| 1    | ≤2022 | 2023 | 0.0%  | 96.7% | 0.230 | 7.9%  |
| 2    | ≤2023 | 2024 | 7.1%  | 83.9% | 0.252 | 9.3%  |
| 3    | ≤2024 | 2025 | 5.0%  | 87.0% | 0.233 | 10.3% |

## Limitations

1. **Within-sector discrimination erosion:** Calibration slope declines over folds (1.22→0.76). The model's ability to rank risk within a sector degrades as distribution drifts.
2. **Tech sector FSR at ceiling:** 11.8% tech FSR is flat vs baseline — preflop features didn't improve tech safety, only maintained it. A sector guardrail could help (v1.2).
3. **Small dataset:** 656 RIVER snapshots across 4 folds. Individual fold metrics have high variance.
4. **SMALL_BET near threshold:** Median 9.8% is barely above the 8% relaxed gate.

## Frozen Configuration

```json
{
  "model": "RiverCommittee (Ridge α=1000, KNN k=7)",
  "calibrator": "isotonic",
  "max_fsr_tune": 0.03,
  "min_size_up_pct": 0.10,
  "min_small_bet_pct": 0.12,
  "max_fold_pct": 0.80,
  "oof_splits": 3,
  "seed": 42,
  "features": 63,
  "street": "RIVER"
}
```

## Reproduce

```bash
python -m scripts.run_preflop_v1_eval
```

Full config: `artifacts/v1.1/config.json`
