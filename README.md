# IPO Risk Research

You can reproduce IPO forecast experiments and inspect their limits. You have no demonstrated allocation-policy benefit. The [decision audit](research/decision/README.md) explains the single-feature benchmarks and the mismatch between drawdown events and allocation losses.

## Run the dashboard

Use Node 22.13 or newer. Run these commands from the repository root:

```sh
npm --prefix web ci
npm --prefix web run dev
```

Open the URL that Vite prints. You need no backend or credentials. **Risk research** replays the frozen MVP. **Anthropic valuation** presents a separate DCF scenario calculator.

## Choose the research workflow

| Workflow | Evidence and status | Read more |
|---|---|---|
| MVP replay | 84 eligible listings; 42 held-out predictions from fixed logistic and boosted-tree models | [Study guide](docs/STUDY.md) |
| Expanded experiment | 1,358 pre-IPO and 1,343 day-20 observations; CatBoost and TabNet comparisons | [Challenger report](docs/CHALLENGER_REPORT.html), [methods](docs/CHALLENGER_STUDY.md) |
| Allocator design | Draft accept/decline policy; no fitted return model or execution backtest | [Decision protocol](research/decision/protocol.json) |
| Anthropic valuation | Assumption-based valuation with a dated evidence snapshot | [Dashboard guide](web/README.md) |

The dashboard reads the MVP results. You open the expanded report as a separate HTML file. Neither experiment validates an allocation recommendation for Anthropic.

## Reproduce the studies

Use Python 3.13 and uv for the recorded research environment. The `.python-version` file selects that Python line. Run from the repository root:

```sh
uv sync --locked --extra challengers
uv run ipo-research evaluate
uv run ipo-challengers audit
```

The first command installs the locked dependencies. The evaluation command rebuilds `web/data/research.json`; the audit command writes `research/decision/audit.json`. Both use bundled data after installation.

To refit the expanded study, including the 2018-onward sensitivity:

```sh
uv run ipo-challengers build
uv run ipo-challengers train
uv run ipo-challengers verify
uv run ipo-challengers report
```

You retain model files under the ignored `artifacts/challengers/` directory. A fresh clone needs `train` before `verify`. The report command uses committed forecasts and generates Markdown, report data, and a SQL audit. HTML regeneration needs an external renderer; see the [research guide](research/expanded/README.md).

## Maintain the project

```sh
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
npm --prefix web run build
```

Run `uv sync --locked --extra challengers` before the full Python suite. The web build includes type checks and tests. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the code map and the rules for updating research artifacts.

You retain the prior prototype in [legacy/](legacy/README.md). It has separate dependencies and historical claims. You exclude it from active checks.
