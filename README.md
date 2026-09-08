# IPO Risk Research

You can replay real listing-risk predictions and inspect the model evaluation. The MVP estimates the probability of a 20% adjusted-close drawdown over the next 20 sessions after observing a listing's first 20 sessions.

You have a frozen public-data sample with 84 eligible listings and 42 held-out predictions. The primary model uses regularized logistic regression. You can compare it with a training-event-rate baseline and shallow gradient-boosted trees. You make no position-sizing or investment-return claim.

Read the [study and interview guide](docs/STUDY.md) for measured results and limits.

You can inspect the expanded CatBoost and TabNet experiment in the [challenger report](docs/CHALLENGER_REPORT.html) or the [methods and reproduction guide](docs/CHALLENGER_STUDY.md). The frozen cohort covers 2010–2025 and contains 1,358 pre-IPO observations and 1,343 day-20 observations. The study compares models on 980 and 964 chronological test cases. Public price coverage remains incomplete; inspect the annual coverage audit before using population-level claims.

**Run the expanded study**

```sh
uv sync --locked --extra challengers
uv run --extra challengers ipo-challengers build
uv run --extra challengers ipo-challengers train
uv run --extra challengers ipo-challengers verify
uv run --extra challengers ipo-challengers report
```

You can run these commands without network access after installing the locked dependencies. The trainer saves models and their preprocessing under `artifacts/challengers`. It reloads each saved model and checks its predictions. You can inspect fold metrics and individual forecasts in `research/expanded/results.json.gz`.

You store the large snapshots as deterministic gzip files in Git and calculate their provenance hashes over decompressed content. The training and verification commands include the 2018-onward coverage sensitivity. Read the [research storage and reproduction notes](research/expanded/README.md) for details.

You regenerate Markdown and report data with the report command above. HTML regeneration requires the external Data Analytics plugin path passed to `research/expanded/render_report.mjs`; the repository alone does not contain its renderer. You can open the committed HTML without the plugin.

You preserve the MVP report below as a benchmark. The existing dashboard reads that report; the expanded experiment has its own report.

**Run the dashboard**

Use Node 22.13 or a later supported release:

```sh
npm --prefix web ci
npm --prefix web run dev
```

Open the local address from Vite. Use Risk research for the historical replay. Use Anthropic valuation for the separate DCF case study. You can export either analysis from the interface.

**Reproduce the risk study**

Use Python 3.11 or a later supported release and uv:

```sh
uv sync --locked
uv run ipo-research evaluate
```

You can reproduce the report without credentials or network access. The command reads `research/input.json` and writes `web/data/research.json`. Vite loads the saved report; the browser does not train or serve models.

You can fetch a new price snapshot with `uv run ipo-research fetch --refresh`. This command uses Yahoo Finance's public chart endpoint and the fixed registry in `research/universe.json`. The endpoint has no stability guarantee. Preserve the bundled input if you need to reproduce the reported results. Cached responses remain under `research/raw/`.

**Verify the MVP**

```sh
uv run --extra challengers python -m unittest discover -s tests -v
npm --prefix web run build
```

The Python suite checks feature cutoffs and chronological splits. It also reproduces the report and verifies frozen model predictions. The web build runs type checks and tests before producing `web/dist/`.

**Read the active code**

| Location | Purpose |
|---|---|
| `src/ipo_research/data.py` and `dataset.py` | Freeze source data, check coverage, and construct dated observations |
| `src/ipo_research/models.py` and `evaluate.py` | Fit the fixed models and save held-out predictions with provenance |
| `web/App.tsx` and `web/Valuation.tsx` | Present the historical replay and separate valuation case study |
| `research/` and `web/data/research.json` | Preserve the cohort input and generated research report |

You can inspect the prior prototype under [legacy/](legacy/README.md). Its poker policy and historical performance claims have no role in the active MVP. You retain its source and tests for reference.
