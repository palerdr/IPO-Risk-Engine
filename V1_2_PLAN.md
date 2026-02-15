# v1.2 Plan: NLP + Applicability Proof

Locked date: 2026-02-15  
Baseline to beat: v1.1 (`docs/V1_LOCK.md`, `docs/MODEL_CARD.md`)

## Goal

Demonstrate that PREFLOP NLP features improve IPO risk decision quality and remain safe under both:

1. Historical walk-forward testing (out-of-time).
2. Prospective forward paper run (future IPOs).

This is a risk-engine objective (drawdown avoidance), not a return-maximization objective.

## Point 1: Walk-Forward Backtest (Historical)

### Scope

Add a minimal, point-in-time NLP block to PREFLOP and evaluate with the existing v1.1 stack:

- RiverCommittee (Ridge + KNN)
- OOF isotonic calibration
- constrained 3-action policy thresholds

### Implementation Steps

1. Build point-in-time text snapshot dataset per IPO.
2. Extract minimal NLP v1 features:
   - `preflop_nlp_risk_density`
   - `preflop_nlp_uncertainty_density`
   - `preflop_nlp_promo_density`
   - `preflop_nlp_length_log`
   - optional low-dim text embedding components (max 10 dims)
3. Merge NLP features into existing PREFLOP feature path.
4. Run the same expanding-fold stability harness as v1.1.
5. Compare v1.1 vs v1.2 at matched safety tuning.

### Historical Acceptance Gates (pre-registered)

- Worst-fold FSR <= 10% (hard gate)
- Median BSS > 0
- Fold-3 BSS lift >= +0.03 vs v1.1 at matched FSR tuning
- Median SMALL_BET >= 8%
- Paired bootstrap for pooled delta BSS: CI excludes 0 (preferred), or one-sided p < 0.10

## Point 2: Forward Paper Run (Future IPOs)

### Scope

Run v1.2 prospectively on new IPOs with frozen config and track matured outcomes.

### Implementation Steps

1. Freeze deployment artifact:
   - feature list
   - model weights
   - calibrator
   - thresholds
   - config hash
2. Run weekly scoring:
   - ingest new IPOs
   - compute features
   - store timestamped `p_tail` + action predictions
3. Join matured outcomes when label horizon completes.
4. Produce rolling KPI dashboard by overall + sector slice.

### Forward Acceptance Gates

- Prospective FSR <= 10%
- Prospective BSS > 0
- SMALL_BET in [8%, 40%]
- No persistent sector safety leak (especially technology)

## Deliverables

1. `docs/V1_2_RESULTS.md`:
   - v1.1 vs v1.2 table
   - fold-wise KPIs
   - prospective KPI summary
2. `artifacts/v1.2/config.json` (frozen run config)
3. `artifacts/v1.2/predictions.csv` (timestamped forward predictions)
4. 2-3 case studies where model action diverged from hype narrative

## Risks and Controls

- Risk: text leakage from post-decision filings
  - Control: strict filing timestamp cutoff at decision time
- Risk: overfitting from high-dimensional text
  - Control: low-dimensional NLP v1 + expanding-fold evaluation
- Risk: policy collapse to binary actions
  - Control: SMALL_BET structural floor in threshold optimization

## Definition of Done (v1.2)

- All four forward acceptance gates pass.
- Historical gates pass on expanding folds.
- v1.2 results reproducible from a single command and fixed config.

## Suggested Execution Order

1. Implement NLP v1 feature extractor.
2. Integrate into PREFLOP merge path.
3. Run historical expanding-fold evaluation.
4. Freeze v1.2 artifact and begin prospective paper run.
5. Publish `docs/V1_2_RESULTS.md`.
