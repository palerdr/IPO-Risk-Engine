# Development guide

You maintain a research benchmark and a static dashboard. Preserve the reported target and the cohort when you make a code cleanup. Treat a model, feature, or target change as a new experiment.

## Set up and check

Use Python 3.13 and Node 22.13 or newer. Run from the repository root:

```sh
uv sync --locked --extra challengers
npm --prefix web ci
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
npm --prefix web run build
```

Use `uv run ruff check . --fix` and `uv run ruff format .` to fix Python imports and formatting. Ruff excludes the archive and notebooks. TypeScript checks unused declarations and type errors. Keep financial fixtures and existing assertions intact.

## Read the code

| Location | Responsibility |
|---|---|
| `src/ipo_research/data.py`, `dataset.py` | Fetch the MVP snapshot and construct dated features and outcomes |
| `src/ipo_research/models.py`, `evaluate.py` | Fit the fixed MVP models and export forecasts |
| `src/ipo_research/expanded/sources.py`, `storage.py` | Collect registry sources; read compressed snapshots and verify content hashes |
| `src/ipo_research/expanded/dataset.py`, `models.py` | Build the expanded feature views and train challengers |
| `src/ipo_research/expanded/evaluate.py`, `verify.py` | Run chronological comparisons and reload saved models for checks |
| `src/ipo_research/expanded/decision_audit.py` | Compare one-feature rankings and audit entry-return proxies |
| `src/ipo_research/expanded/report.py`, `report_text.py` | Assemble the report and maintain its narrative |
| `web/App.tsx`, `web/components/ResearchCharts.tsx` | Manage replay state and render research charts |
| `web/Valuation.tsx`, `web/lib/valuation/` | Present and calculate the separate DCF scenario |
| `web/lib/download.ts`, `web/lib/research.ts` | Export snapshots and expose the frozen report types |

## Preserve research evidence

You hash decompressed JSON content for the compressed research snapshots. You hash source files as stored. Formatting a hashed Python file changes provenance even if predictions stay equal.

After changing MVP source, run `uv run ipo-research evaluate` and the Python suite. Compare forecasts and metrics with the previous committed report. Inspect provenance changes before accepting generated data.

After changing expanded dataset or model code, run `ipo-challengers build`, `train`, and `verify` with uv. The trainer invalidates cached models when the training source hashes change. Then run `ipo-challengers report` and regenerate HTML if you change the rendered report. Compare predictions with the previous snapshot for a cleanup; document a new experiment if you change model behavior.

You edit report prose in `report_text.py`. You generate `docs/CHALLENGER_STUDY.md` and `research/expanded/report-artifact.json` from code. Avoid editing those outputs by hand. You can choose a Markdown destination with `uv run ipo-challengers report --report-directory path/to/reports`.

You keep the allocator protocol as a draft until its data and execution checks pass. You do not relabel the existing drawdown forecasts as allocation returns.

## Update dependencies and source data

You install exact web versions from `web/package-lock.json` and Python versions from `uv.lock`. Inspect release notes before changing a major version. Use `npm --prefix web outdated` and `npm --prefix web audit` for the dashboard. Keep the research environment stable unless you plan to refit and compare the saved forecasts.

As of September 7, 2026, the active scientific packages match the releases checked on PyPI. The dashboard uses React 19.2.8, TypeScript 7.0.2, and Vitest 5.0.0. You retain Node 22 type definitions to match the minimum supported Node line. You can review [TypeScript migration notes](https://devblogs.microsoft.com/typescript/announcing-typescript-7-0-rc/) and [Vitest migration notes](https://main.vitest.dev/guide/migration/) before the next major update.

You preserve source snapshots when refreshing upstream data. Copy `research/` to a separate working directory before running fetch commands. Keep raw response caches and trained model files outside Git; retain their hashes and verification receipts in the study. See [research/expanded/README.md](research/expanded/README.md) for source-refresh commands and storage details.

## Archive boundary

You treat `legacy/` as a record of the previous prototype. You do not import its code into the active package or include it in active test discovery. Its dependency files and performance claims describe that prototype. Use the active study guides for current project claims.
