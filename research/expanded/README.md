# Expanded IPO experiment

You have no demonstrated allocation-policy benefit from this experiment. Read the [decision audit](../decision/README.md) before using the scores. You can inspect the [HTML report](../../docs/CHALLENGER_REPORT.html) or its [methods and results](../../docs/CHALLENGER_STUDY.md).

You retain 1,393 price matches from 3,031 in-scope Field–Ritter records dated 2010–2025. After feature and calendar checks, you have 1,358 pre-IPO and 1,343 day-20 observations. Missing histories create survivorship risk, with lower coverage in earlier years.

## Reproduce from a clone

Run from the repository root with Python 3.13:

```sh
uv sync --locked --extra challengers
uv run ipo-challengers build
uv run ipo-challengers train
uv run ipo-challengers verify
uv run ipo-challengers report
```

You train both the primary experiment and the sensitivity restricted to listings from 2018 onward. You retain the original candidate grids and compare the refits on matching 2020–2025 test issuers. The sensitivity changes training size and period composition along with coverage.

You save model files under the ignored `artifacts/challengers/` directory. The verifier needs those files, so run `train` first in a fresh clone. You can generate the report from committed forecasts without refitting. The report command also recomputes the decision audit.

## Inspect stored evidence

| File | Purpose |
|---|---|
| `protocol.json`, `coverage-protocol.json` | Preserve the main experiment and post-hoc sensitivity definitions |
| `universe.json.gz` | Preserve registry records, source hashes, and date cross-references |
| `input.json.gz` | Preserve accepted histories and the exclusion ledger |
| `dataset.json.gz` | Preserve dated features, outcomes, and coverage counts |
| `results.json.gz`, `coverage-results.json.gz` | Preserve fold scores and issuer forecasts |
| `verification.json`, `coverage-verification.json` | Record input reconciliation and saved-model checks |
| `report-artifact.json`, `report-notes.json` | Preserve report content and source metadata |
| `report-verification.json` | Record the HTML renderer checks |
| `audit.sqlite3` | Recompute Brier errors and coverage with SQL; generated and ignored |

You store the large snapshots as deterministic gzip files in Git. You calculate their provenance hashes over decompressed JSON bytes. You can inspect a snapshot with `gzip -dc research/expanded/input.json.gz`. You need no data service or Git LFS to reproduce the frozen study.

## Refresh source data

Copy `research/` to a separate working directory to preserve the reported snapshot. Run the `registry` and `fetch` commands with `--directory` pointing to that copy's expanded directory. You cache responses under `raw/` and fetch progress in `download-ledger.json`. Use `fetch --retry-failures` to retry recorded exclusions.

You use historical offer terms and issuer traits. You omit financial statements because the pipeline has no dated filing extraction. You use the preceding SPY close as the market-data cutoff. The source compilations describe IPO-time facts but do not supply per-field publication timestamps.

You credit the Field–Ritter dataset of company founding dates, as used by Field and Karpoff (2002) and Loughran and Ritter (2004). Read the [source definitions](https://site.warrington.ufl.edu/ritter/files/founding-dates.pdf). You omit the discontinued internet flag and the post-issue share count, whose definition and coverage vary with share class.

## Maintain the report

You edit narrative templates in `src/ipo_research/expanded/report_text.py` and assemble tables in `report.py`. The report command writes Markdown to `docs/`; use `--report-directory` to select another destination.

HTML regeneration requires an external Data Analytics plugin installation. Run from the repository root:

```sh
node research/expanded/render_report.mjs /path/to/data-analytics-plugin
```

You pass the installed plugin directory. The wrapper imports its `skills/build-report/scripts/build_portable_artifact.mjs` and `deliver_portable_artifact.mjs` scripts, then runs the packaged reader checks. The repository does not include this renderer. You can open the committed HTML without it, and you can rebuild the dashboard and Markdown from the repository alone.
