# Expanded IPO experiment

You have no demonstrated allocation-policy benefit from this experiment. Read the [decision audit](../decision/README.md) for the one-feature comparisons and entry-price checks. The report command recomputes that audit from the frozen inputs.

You use one row per issuer at each observation stage. You have 1,358 pre-IPO rows and 1,343 day-20 rows from a Field–Ritter registry of 4,497 records dated 2010–2025.

You should read the [measured results](../../docs/CHALLENGER_STUDY.md) before interpreting the model scores. You retain 1,393 price matches from 3,031 in-scope records. Missing histories create survivorship risk, with lower coverage in the earlier years.

| File | Purpose |
|---|---|
| `protocol.json` | Preserve the candidate models and chronological validation rules from before training |
| `universe.json.gz` | Preserve registry records, source hashes, and cross-source date references |
| `input.json.gz` | Preserve accepted price histories and the exclusion ledger |
| `dataset.json.gz` | Preserve dated features, outcome windows, and the coverage audit |
| `results.json.gz` | Preserve fold scores, selected configurations, and individual forecasts |
| `coverage-protocol.json` | Preserve the post-hoc training restriction to listings from 2018 onward |
| `coverage-results.json.gz` | Preserve refitted forecasts on matching 2020–2025 test issuers |
| `coverage-verification.json` | Record saved-model checks for the restricted training cohort |
| `verification.json` | Record input reconciliation and saved-model verification |
| `audit.sqlite3` | Recompute Brier errors and source coverage with SQL |
| `report-artifact.json` | Preserve the portable report's narrative and aggregate tables |

You can reproduce features with `uv run --extra challengers ipo-challengers build`. You can refit models with `uv run --extra challengers ipo-challengers train`. You can verify the saved models with `uv run --extra challengers ipo-challengers verify`.

You can create a new source snapshot with the `registry` and `fetch` commands. Use `--directory` with a copy of this directory to preserve the reported study. The fetch command caches upstream responses under `raw/` and saves progress in `download-ledger.json`. You can resume a stopped fetch. Use `--retry-failures` to retry recorded exclusions. You should inspect coverage changes before replacing a published snapshot.

You use historical offer terms and issuer traits for pre-IPO predictions. You omit financial statements because the current pipeline has no dated filing extraction. You use the preceding SPY close as the market-data cutoff. The source compilations describe IPO-time facts but do not supply per-field publication timestamps.

You credit the Field–Ritter dataset of company founding dates, as used by Field and Karpoff (2002) and Loughran and Ritter (2004). You can inspect the [source definitions](https://site.warrington.ufl.edu/ritter/files/founding-dates.pdf). You omit the source's internet flag because the author stopped updating it. You omit its post-issue share count because coverage depends on share class and the author warns about ADR conversion errors.

You store the large research snapshots as deterministic gzip files in Git. You retain the full JSON content and reproduce the study from a clone without a separate data service or Git LFS. You calculate provenance SHA-256 hashes over decompressed JSON bytes. You can inspect a snapshot with `gzip -dc research/expanded/input.json.gz`. You keep the small protocol and verification records as text. You exclude the rebuildable SQL audit database and source-response caches from Git.

You run both the primary study and the 2018-onward training sensitivity with `ipo-challengers train`. You verify both runs with `ipo-challengers verify`. The sensitivity preserves the original candidate grids and tests on the same 2020–2025 issuers. You can inspect its separate protocol before refitting. You should treat it as a post-hoc diagnostic that changes training size and period composition along with coverage.

You can regenerate Markdown and `report-artifact.json` with `ipo-challengers report`. You cannot regenerate `docs/CHALLENGER_REPORT.html` from the repository alone. You need the external Data Analytics plugin, then pass its installed directory to `node research/expanded/render_report.mjs`. The wrapper imports that plugin's build-report scripts and packaged reader. You can open the committed HTML without that dependency.
