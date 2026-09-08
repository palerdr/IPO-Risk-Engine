# Study and interview guide

You can demonstrate forecast evaluation and its limits. You have no validated allocation policy. The [decision audit](../research/decision/README.md) explains the single-feature comparisons and the mismatch between peak-to-trough events and entry-based losses.

You can demonstrate a completed retrospective risk study on a curated sample of real listings. You can reproduce the predictions from the bundled input, then inspect the model settings and held-out outcomes in the dashboard.

**Scope and target**

You observe the first 20 exchange sessions after listing. You predict whether the adjusted closing price suffers a peak-to-trough decline of at least 20% over sessions 21–40. You include the session-20 close as the initial running peak. You do not estimate the return from an executable entry price. You do not measure intraday drawdowns.

You calculate eight features from the observation window. These cover price returns and volatility, with volume change and SPY market conditions. You use no filing text or accounting data in this MVP. Sector metadata appears in the replay but does not enter the models.

**Data and provenance**

You start from 94 entries in the prior project's curated 2024–2025 listing registry. You fetch Yahoo Finance daily adjusted-close prices and volume, plus SPY observations for the exchange calendar and market features. You freeze the normalized result in `research/input.json`. You keep request URLs and retrieval timestamps with the data, along with hashes of the upstream responses. The local `research/raw/` cache holds those responses when you fetch them.

You retain 84 listings. You exclude ten entries for unavailable histories or a disagreement between the registry date and the vendor's first-trade date. You can inspect the complete exclusion ledger in the dashboard. You do not change dates to force records into the cohort.

You have a convenience sample. You do not have a complete historical IPO universe or an independent audit of security types. Missing or delisted histories can create survivorship bias. Yahoo's adjusted history can reflect revisions made after the original prediction date. Price-ratio features remove dependence on uniform rescaling, but they do not establish point-in-time vendor accuracy. You cannot extend this sample's metrics to the IPO population.

**Models and evaluation**

You fix these model families and settings before evaluating their test predictions:

| Model | Configuration | Role |
|---|---|---|
| Training event rate | Fraction of drawdown events in each fold's eligible training data | Minimum probability benchmark |
| Logistic regression | StandardScaler fitted on training data; L2 regularization with C = 0.1 | Prespecified primary model |
| Gradient-boosted trees | 75 trees; depth 2; learning rate 0.03; minimum 10 observations per leaf | Prespecified nonlinear challenger |

You use shallow trees with numerical inputs, so the MVP needs no categorical encoding library. You apply no post-hoc calibrator. You assess calibration with held-out probability bins. You do not claim calibrated probabilities from the choice of model family.

You use the earlier half of distinct observation dates to seed training. You evaluate the later dates in three expanding blocks. You keep equal observation dates together and one observation per issuer. You include a training row only if its entire outcome window ends before the first test observation date. You fit the scaler within each training fold.

| Fold | Training rows | Held-out rows | Unmatured training labels excluded | Last training outcome | First test observation |
|---|---:|---:|---:|---|---|
| 1 | 36 | 15 | 6 | 2024-10-08 | 2024-10-29 |
| 2 | 57 | 13 | 0 | 2025-02-12 | 2025-04-25 |
| 3 | 64 | 14 | 6 | 2025-09-18 | 2025-09-23 |

You do not tune hyperparameters or choose a model from these test results. You retain logistic regression as the primary model. A subsequent change needs a new study version and a fresh evaluation plan.

**Measured results in this sample**

You have 42 held-out listings, including 17 drawdown events. You calculate the baseline probability from each fold's training outcomes. You do not use the test event rate to generate baseline predictions.

| Model | Brier score ↓ | Log loss ↓ | Average precision ↑ | Brier skill ↑ |
|---|---:|---:|---:|---:|
| Training event rate | 0.269 | 0.746 | 0.378 | 0.000 |
| Logistic regression | 0.154 | 0.474 | 0.858 | 0.429 |
| Shallow boosted trees | 0.217 | 0.647 | 0.741 | 0.194 |

You calculate Brier skill as `1 - model Brier / baseline Brier` on the same held-out rows. The logistic model has 42.9% lower squared probability error than the training-rate baseline in this sample. This is not classification accuracy or an investment return.

You obtain an exploratory 95% Brier-skill interval of about 0.281–0.611 for logistic regression. You resample paired model and baseline predictions in observation-month blocks with seed 42 and 500 draws. This interval describes uncertainty in the saved held-out sample. It excludes model-fitting and sample-selection uncertainty. Overlapping target windows can cross month boundaries. Treat it as a sensitivity estimate for a small cohort.

You can inspect fixed probability cutoffs from 10% to 50%. Precision uses flagged listings as its denominator; recall uses actual ≥20% drawdown events. The severe-event miss rate uses actual ≥30% drawdown events. You do not optimize a risk cap or assign a position size.

**Demonstration**

1. Open Risk research. Explain the 20-session observation window and the next-20-session target. State that the study uses a curated sample with 42 held-out cases.
2. Replay the default case, FVR. Inspect the prediction date and training cutoff before revealing the outcome. Show the observed feature values and source link.
3. Select KLC to discuss a miss. The logistic model assigns about 9.4%, and the target drawdown event occurs. Explain that an event can occur at a low predicted probability; you assess calibration across cases.
4. Compare the model table and inspect the temporal folds. Export the study to show its input hash and frozen logistic coefficients. Use Anthropic valuation to explain the separate pre-listing DCF workflow.

The default case follows observation-date order. You can inspect all held-out cases through the selector, including misses. KLC is a post-evaluation example of a failure, not an independently selected validation case.

**Resume wording**

- Built an IPO drawdown-risk research system with regularized logistic regression and a gradient-boosted-tree challenger, using dated features and chronological evaluation.
- Reproduced 42 held-out predictions from a frozen 84-listing sample; measured a 0.154 Brier score versus 0.269 for a training-event-rate baseline and exposed model errors through historical replay.

You should retain the sample qualifier beside the metrics. You can discuss why you chose a fixed target and simple models, then explain the exclusion log and label-maturity checks. You have no validated portfolio-performance claim or population-wide risk guarantee.

**Reproduce and inspect**

Run `uv run ipo-research evaluate` from the repository root. The command regenerates `web/data/research.json` from `research/input.json`. You can compare the full report with a fresh run through the Python tests. The report includes input hashes and package versions, with the fitted logistic parameters for each fold.

You can use `predict_frozen` from `ipo_research.models` to reproduce a saved model's probability from its feature values. The function rejects prediction dates on or before the model's last training outcome. It does not provide a live-price service.

The old 656-IPO and +0.106 Brier-skill claims remain under `legacy/`. They describe another dataset and model definition. You cannot combine them with the v2 results.

**Verification status**

You have 27 passing Python tests: 13 for the MVP and 14 for the expanded challenger study. You also have 38 passing web tests. The web suite includes the 33 preserved financial fixtures and five tests for the new interface. The production build passes, and its HTML and referenced assets return HTTP 200. You can reproduce the saved report from its frozen input and reproduce each logistic probability from its exported fold model.

The local audit matched all 85 upstream response hashes and adjusted-close arrays against the frozen input, including SPY. No browser was available for visual inspection. The hosting source endpoint timed out, so this version has no published URL.
