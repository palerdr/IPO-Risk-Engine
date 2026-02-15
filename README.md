# IPO Risk Engine

Safety-first IPO risk engine with poker-style actions:
`FOLD`, `SMALL_BET`, `SIZE_UP`.

## Repo Layout

- `src/ipo_risk_engine/`: core package
  - `config/`: environment settings
  - `data/`: ingestion, schema validation, parquet I/O
  - `features/`: street and preflop feature builders
  - `snapshots/`: symbol/street snapshot construction
  - `dataset/`: snapshot flattening and temporal splits
  - `models/`: committee model, calibration, evaluation helpers
  - `policy/`: action assignment and constrained threshold tuning
- `scripts/`: runnable entry points (dataset build, pipeline runs, diagnostics)
  - `scripts/maintenance/`: one-off maintenance utilities
- `tests/`: test suite
- `docs/`: model card, lock report, and experiment summaries
- `archive/`: deprecated/legacy code kept for reference only

## Main Commands

- Build dataset:
  - `python scripts/build_dataset.py`
- Run full pipeline:
  - `python scripts/run_full_pipeline.py`
- Run stability report:
  - `python scripts/run_stability_report.py`
- Run preflop ablation:
  - `python scripts/run_preflop_v1_eval.py`

## Notes

- Project uses temporal splits only (no random split).
- Generated artifacts and local data outputs are ignored via `.gitignore`.
