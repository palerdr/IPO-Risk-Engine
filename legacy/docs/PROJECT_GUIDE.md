# Project and interview guide

You have a local valuation application and a Python risk-research pipeline. Use the valuation application as your first interview demonstration. Use the synthetic Python walkthrough to explain the research code without claiming historical predictive performance.

**The shortest explanation**

For valuation, you choose operating assumptions and discount the resulting cash flows. You can reverse the calculation to find the growth required by an assumed entry valuation.

For risk research, you extract features from post-IPO price bars, estimate drawdown severity with a Ridge/KNN ensemble, calibrate adverse-event probabilities, and assign risk tiers. This model has no role in the Anthropic DCF calculation.

**What you can demonstrate now**

| Capability | Evidence from this checkout | Boundary |
|---|---|---|
| Ten-year DCF and enterprise-to-equity bridge | Financial fixtures and the interactive application | Inputs are scenario assumptions; unknown net cash blocks equity value |
| Reverse valuation and sensitivity analysis | Solver fixtures and UI tests that apply the result | The solver finds a mathematical growth rate, without estimating its probability |
| Dated evidence and reproducible JSON exports | Source ledger and an export/recalculation test through the UI | Reviewed disclosures remain fixed; there is no live filing feed |
| Python feature/model/calibration/policy flow | Offline synthetic walkthrough and core tests | This run establishes code execution, not predictive accuracy |

The web suite has 41 passing tests: 33 financial and data tests, plus eight interaction tests in a simulated DOM. The Python suite has seven passing tests, including the existing drawdown fixture, a snapshot-builder check with mocked data access, and a check that the saved dashboard walkthrough matches a fresh Python run. The local server returned HTTP 200, and the production build passed. A browser was unavailable for visual inspection.

Sites reports zero saved versions and no live URL for this project. You have a local demo, not a published application.

**Five-minute demonstration**

1. Start the web app and leave net cash blank. The illustrative inputs produce about $224.4B of enterprise value, while equity value remains unavailable. Explain the distinction between enterprise and equity value.
2. Choose Load example · $0 net cash to set a zero-net-cash assumption and a $965B entry valuation. Select Required growth. The solver returns about 54.44% annual growth in years 1–5 under the other default assumptions. Apply that growth to reproduce about $965B in the forward view.
3. Change WACC and expand Valuation sensitivity to inspect the sensitivity table. Explain that reinvestment consumes cash as revenue grows. The initial scenario has a terminal contribution above 100% because forecast-period cash flows have negative present value.
4. Export the scenario and expand Sources and missing disclosures to open a source link. Explain that the $47B starting revenue is an illustrative assumption referenced to a dated run-rate disclosure; it is not annual recognized revenue. Open Risk engine to show the saved Python walkthrough and its synthetic-data label. This view does not run Python in the browser or estimate Anthropic risk.

These numbers describe model behavior under the supplied assumptions. They do not establish Anthropic's investment value or an IPO price.

**Resume wording you can use**

For the application:

- Built a React/TypeScript IPO valuation workbench with ten-year discounted cash flow analysis and a reverse-valuation solver for required revenue growth.
- Added source-linked financial inputs and reproducible scenario exports; verified financial calculations and application interactions with 41 automated tests.

For the Python research work:

- Developed a Python IPO risk pipeline using Polars and scikit-learn to transform price bars into drawdown features and calibrated risk tiers.
- Implemented chronological evaluation and a Ridge/KNN ensemble; added a reproducible offline walkthrough from synthetic prices to policy outputs.

Use the application bullets if you need a short project entry. Add the Python bullets if you want to discuss ML research. Avoid claims of AI filing extraction or deployed production use: this version has neither capability. Avoid return, Sharpe-ratio, or predictive-accuracy claims until you reproduce them with audited historical data.

**Interview topics and limits**

Explain the financial model through `web/lib/valuation/engine.ts`. The engine uses unlevered cash flow, includes terminal reinvestment, and requires WACC to exceed terminal growth. You can discuss the omission of tax loss carryforwards and future financing dilution as explicit model limits.

Explain the application through `web/App.tsx`. React holds form state, converts percentages into decimal rates, and passes inputs to pure calculation functions. Native controls and plain CSS keep the application independent of a server and component library. The browser code has two runtime dependencies: React and React DOM.

Explain the Python data flow through the package directories: `data` ingests and normalizes bars; `features` and `labels` create model inputs and targets; `models` fits and calibrates estimates; `policy` assigns action labels. The risk labels describe drawdowns over a future window. They do not represent a portfolio return.

You should discuss the research limitations with precision:

- Historical model-card claims refer to 656 IPO snapshots and results such as a +0.106 Brier skill improvement. This checkout lacks the underlying datasets and run artifacts. The current verification does not reproduce those results.
- The amendment builder aggregates filings without a decision-date cutoff, and augmentation uses a full-dataset imputation median. Chronological splits alone do not remove those leakage paths.
- The threshold optimizer can retain an infeasible default. An inspected example requested a 10% false-safe cap and returned 100%. The offline walkthrough uses fixed thresholds and makes no safety-cap claim.
- The daily high/low drawdown calculation assumes an intraday ordering that daily bars do not establish. The `backtest` and `report` packages contain no implementation. The project has no portfolio execution simulator.

Define false-safe rate as the fraction of actual severe events assigned `SIZE_UP`. That denominator differs from the fraction of `SIZE_UP` decisions that become severe events. Describe `FOLD`, `SMALL_BET`, and `SIZE_UP` as research labels; this project places no orders.

**What the simplification changed**

The web app now uses Vite with plain React and CSS. It preserves valuation controls and outputs while removing the server-rendering scaffold and unused UI components. Runtime dependencies fell from 19 to two. The optional browser-agent hook was outside the valuation workflow and has been removed.

Python dependencies now have one definition in `pyproject.toml` and a reproducible lockfile. The cleanup removed unused dependency declarations and commented code. Existing Python function and class definitions match their pre-cleanup versions. Historical scripts and tests remain available.

**Dashboard presentation**

You start with four valuation metrics and four primary inputs. You can load an example with an explicit zero-net-cash assumption. You expand the forecast table and model equations when you need calculation detail. The Risk engine view explains the original post-listing model and shows its saved synthetic output. The RIVER example needs price and volume history for sessions 0–60. You have no Anthropic post-listing bars in this dataset.

The presentation follows the dark dashboard example in the [Ryze guide](https://www.get-ryze.ai/blog/build-ad-dashboards-with-claude-ai-guide), with compact metric cards above a pair of charts. You retain the valuation functions and financial assumptions from the prior version.
