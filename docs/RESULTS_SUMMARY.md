# Results Summary: v1.0 → v1.1

Locked: 2026-02-15

## What Changed

**v1.0** (river_only): 49 features from RIVER daily bars only, MAX_FSR_TUNE=0.05
**v1.1** (A_full): 63 features (+14 PREFLOP V1), MAX_FSR_TUNE=0.03

PREFLOP V1 adds structured EDGAR metadata + sector one-hot encoding:
- `preflop_s1_lead_days` — first bar date minus earliest S-1 filing date
- `preflop_exchange_nyse`, `preflop_exchange_nasdaq` — exchange one-hot
- `preflop_s1a_count` — number of S-1/A amendment filings (from EDGAR EFTS)
- `preflop_days_since_last_amendment` — first bar date minus last amendment date
- 9 sector one-hot columns (health_care, technology, industrials, financials, consumer_goods, consumer_services, energy, oil_gas, real_estate)

## Falsification (v1.0 Diagnosis Correction)

v1.0 attributed fold-3's 43.5% detection to "compositional shift" (sector mix change). Three falsification checks **disproved** this:

1. **Reweighting test:** Reweighting fold 0-2 detection to fold-3's sector mix barely moved numbers (97.4%→97.4%, 81.9%→81.9%, 77.5%→77.5%). Sector mix does NOT explain the drop.
2. **Within-sector breakdown:** Detection dropped *within* nearly every sector. Health care: 88.4%→45.5%. Technology: 76.2%→42.9%. The drop is universal, not compositional.
3. **Calibration slope:** OLS slope of p_tail vs adverse_20 declined monotonically: 1.223→0.942→0.867→0.759. Model discrimination is eroding over time, not just recalibrating.

**Corrected diagnosis:** Within-sector discrimination erosion, not compositional shift. The model's ability to rank risk within a sector degrades as the data distribution drifts.

## v1.1 Metrics (4 expanding folds, MAX_FSR_TUNE=0.03)

| Fold | Test | FSR   | Det   | BSS   | SB%   | SU%   | FOLD% |
|------|------|-------|-------|-------|-------|-------|-------|
| 0    | 2022 | 0.0%  | 94.5% | 0.094 | 16.9% | 4.6%  | 78.5% |
| 1    | 2023 | 0.0%  | 96.7% | 0.230 | 7.9%  | 12.2% | 79.9% |
| 2    | 2024 | 7.1%  | 83.9% | 0.252 | 9.3%  | 17.9% | 72.9% |
| 3    | 2025 | 5.0%  | 87.0% | 0.233 | 10.3% | 19.8% | 69.8% |

## Before/After Comparison (fold-3, 2025 holdout)

| Metric       | v1.0 (river_only) | v1.1 (A_full) | Change    |
|-------------|-------------------|---------------|-----------|
| FSR         | 5.0%              | 5.0%          | unchanged |
| Detection   | 85.5%             | 87.0%         | +1.5pp    |
| BSS         | 0.127             | 0.233         | **+0.106** |
| Tech FSR    | 11.8%             | 11.8%         | unchanged |
| SMALL_BET   | 14.7%             | 10.3%         | -4.4pp    |
| SIZE_UP     | 12.1%             | 19.8%         | +7.7pp    |
| FOLD        | 73.3%             | 69.8%         | -3.5pp    |

Key takeaway: At matched safety (FSR=5.0%), v1.1 achieves **+0.106 BSS lift** — the model is substantially better calibrated. The detection gain is modest (+1.5pp) because the tight FSR constraint compresses both models to similar operating points.

## Ablation Proof

Three variants tested at identical MAX_FSR_TUNE=0.03:

| Variant | Features | Fold-3 FSR | Fold-3 BSS | dBSS   | Tech FSR | Med SB% | DoD    |
|---------|----------|-----------|------------|--------|----------|---------|--------|
| baseline | 49 (river only) | 5.0% | 0.127 | — | 11.8% | 14.7% | — |
| **A_full** | **63 (+14 all)** | **5.0%** | **0.233** | **+0.106** | **11.8%** | **9.8%** | **ALL PASS** |
| B_no_amend_time | 62 (+13) | 5.0% | 0.243 | +0.116 | 11.8% | 7.3% | FAIL (SB) |
| C_no_amendments | 61 (+12) | 10.0% | 0.246 | +0.119 | 23.5% | 6.8% | FAIL (Tech, SB) |

### What the ablation proves

1. **`preflop_s1a_count` is the critical safety feature.** Removing it (variant C) causes tech FSR to double from 11.8% → 23.5% and fold-3 FSR to hit the 10% ceiling. The amendment count lets the model distinguish heavily-amended filings (higher risk) from clean ones.

2. **`preflop_days_since_last_amendment` contributes to SMALL_BET coverage.** A_full (with timing) achieves 9.8% median SMALL_BET vs B's 7.3%. The timing signal helps the model place more IPOs in the intermediate risk tier.

3. **Sector one-hot alone improves BSS but destroys safety.** C has the highest raw BSS lift (+0.119) but fails both Tech FSR and SMALL_BET gates. Sector encoding learns base rates but can't discriminate within-sector — exactly the problem the falsification checks diagnosed.

4. **A_full is the only variant passing all 4 revised DoD gates.**

## Reproduce

```bash
python -m scripts.run_preflop_v1_eval
```
