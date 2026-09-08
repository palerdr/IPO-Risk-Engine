# Prior prototype

You retain the v1 source and tests here for reference. The active application uses `src/ipo_research` at the repository root and has no imports from this directory.

You can inspect the prior Ridge/KNN model and poker-style policy under `src/ipo_risk_engine`. Historical reports under `docs/` contain claims that the current checkout cannot reproduce. The `web/` snapshot contains the prior valuation interface and synthetic risk walkthrough.

You can run the archived Python checks from this directory with `uv sync` and `uv run python -m unittest discover -s tests -v`. You can run the archived web tests from `web/` after `npm ci`. You need separate credentials and historical datasets for the v1 market-data scripts. The replacement MVP does not need them.

You should use the root [study guide](../docs/STUDY.md) for current resume claims and demonstration steps.
