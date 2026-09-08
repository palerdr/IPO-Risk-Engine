# IPO challenger study

## Decision relevance audit

You have no demonstrated decision value from this forecast benchmark. On 953 pre-IPO cases with an offer price, I measured AUC of 0.705 for negative log offer price and 0.702 for logistic regression on those same cases. On 964 day-20 cases, I measured 0.727 for realized volatility; the best model AUC was 0.718. I added these one-feature references after the original evaluation, so you should treat the comparison as a diagnostic. AUC describes ranking for the existing drawdown label. It does not establish the expected loss of an allocation policy.

## Entry-based outcomes

I reproduced a nonnegative terminal return proxy for 342 of 496 pre-IPO events (69.0%), and 87 of 446 day-20 events (19.5%). You must keep the price bases visible: the pre-IPO proxy divides vendor-adjusted closes by nominal offer prices. You need split and distribution accounting on the original offered-share basis before calling that proxy allocator P&L. You should also distinguish a terminal recovery from a path that stayed above entry. I measured a median first-close/offer proxy of 27.0% and a correlation of 0.064 with the pre-IPO event label. The current label measures a decline from a running peak and does not encode loss from the allocation price.

## Coverage association

Across the sixteen IPO years from 2010 to 2025, I measured correlations of 0.810 before the IPO and 0.892 at day 20 between price coverage and annual event rate. You cannot use these correlations to separate missing-history bias from changes in issuer mix or market conditions. The 2018-onward refit below tests dependence on the older cohort; it does not remove survivorship bias.

## Historical forecast comparison

I trained CatBoost and TabNet for pre-IPO and day-20 drawdown forecasts. Across 980 pre-IPO test cases, I measured CatBoost Brier error of 0.2437, versus 0.2572 for logistic regression. Across 964 day-20 test cases, I measured 0.2212 for CatBoost with issuer traits and 0.2265 for the original logistic benchmark. You should treat these as point-estimate gains: both paired confidence intervals include zero. I measured higher aggregate error for TabNet in both stages. After restricting training to listings from 2018 onward, I measured pre-IPO Brier error of 0.2352 for logistic regression and 0.2501 for CatBoost on the same 795 test cases from 2020–2025.

## Targets

You predict a maximum adjusted-close peak-to-trough decline of at least 20%. For pre-IPO forecasts, you score after final pricing and before the first trade, then measure the first 20 closing prices. For day-20 forecasts, you score at the twentieth close and measure the next 20 sessions, including that close as the initial peak. You compare models within each stage because the two targets cover different price windows. You read Brier error as mean squared probability error, with lower values indicating better forecasts. You read ROC-AUC as ranking quality, with higher values indicating better separation. You calculate Brier skill as 1 minus model Brier divided by reference Brier on the same test rows. You compare against each fold's training event rate and against a constant 50% forecast (Brier 0.2500). Positive skill means lower error than that reference; negative skill means higher error.

## Held-out results

| Stage | Model | Brier | Skill vs train rate | Skill vs 50% | ROC-AUC | Test cases |
|---|---|---|---|---|---|---|
| Pre-IPO | Training rate | 0.3015 | 0.0% | -20.6% | 0.6027 | 980 |
| Pre-IPO | Logistic | 0.2572 | 14.7% | -2.9% | 0.7048 | 980 |
| Pre-IPO | Boosted trees | 0.2644 | 12.3% | -5.8% | 0.6941 | 980 |
| Pre-IPO | CatBoost + traits | 0.2437 | 19.2% | 2.5% | 0.6678 | 980 |
| Pre-IPO | TabNet + traits | 0.2683 | 11.0% | -7.3% | 0.6603 | 980 |
| Day 20 | Training rate | 0.2847 | 0.0% | -13.9% | 0.5783 | 964 |
| Day 20 | Logistic | 0.2265 | 20.4% | 9.4% | 0.7016 | 964 |
| Day 20 | Boosted trees | 0.2228 | 21.7% | 10.9% | 0.7180 | 964 |
| Day 20 | Logistic + traits | 0.2238 | 21.4% | 10.5% | 0.7182 | 964 |
| Day 20 | CatBoost prices | 0.2206 | 22.5% | 11.7% | 0.6932 | 964 |
| Day 20 | CatBoost + traits | 0.2212 | 22.3% | 11.5% | 0.7083 | 964 |
| Day 20 | TabNet + traits | 0.2343 | 17.7% | 6.3% | 0.6961 | 964 |

## Training coverage sensitivity

I matched prices for 388 of 1,364 in-scope listings from 2010–2017 (28.4%), and 1,005 of 1,667 from 2018–2025 (60.3%). At your request, I added a post-hoc sensitivity that trains on listings from 2018 onward. I retained the candidate grids and inner validation rules, then refitted the models for each test block. You compare both training cohorts on identical 2020–2025 held-out issuers. You read a negative Brier change as lower error for the restricted training cohort. You can assess dependence on the earlier cohort, but this restriction also changes sample size and market regimes. You cannot isolate the effect of missing histories or remove survivorship bias with this comparison.

| Stage | Test period | Train from 2010 | Train from 2018 | Test cases |
|---|---|---|---|---|
| Pre-IPO | 2020–2021 | 557 | 179 | 345 |
| Pre-IPO | 2022–2023 | 899 | 521 | 139 |
| Pre-IPO | 2024–2025 | 1045 | 667 | 311 |
| Day 20 | 2020–2021 | 548 | 172 | 339 |
| Day 20 | 2022–2023 | 875 | 499 | 143 |
| Day 20 | 2024–2025 | 1026 | 650 | 302 |

You read both skill columns below for the models trained from 2018 onward. You compare each model with that run's training-rate forecasts and with constant 50% forecasts.

| Stage | Model | Brier: train from 2010 | Brier: train from 2018 | Change | Skill vs train rate | Skill vs 50% |
|---|---|---|---|---|---|---|
| Pre-IPO | Training rate | 0.3194 | 0.2771 | -0.0423 | 0.0% | -10.8% |
| Pre-IPO | Logistic | 0.2643 | 0.2352 | -0.0291 | 15.1% | 5.9% |
| Pre-IPO | Boosted trees | 0.2729 | 0.2453 | -0.0276 | 11.5% | 1.9% |
| Pre-IPO | CatBoost + traits | 0.2519 | 0.2501 | -0.0018 | 9.7% | -0.05% |
| Pre-IPO | TabNet + traits | 0.2791 | 0.2446 | -0.0345 | 11.7% | 2.2% |
| Day 20 | Training rate | 0.2919 | 0.2592 | -0.0327 | 0.0% | -3.7% |
| Day 20 | Logistic | 0.2215 | 0.2167 | -0.0047 | 16.4% | 13.3% |
| Day 20 | Boosted trees | 0.2202 | 0.2259 | +0.0057 | 12.9% | 9.6% |
| Day 20 | Logistic + traits | 0.2182 | 0.2138 | -0.0044 | 17.5% | 14.5% |
| Day 20 | CatBoost prices | 0.2229 | 0.2226 | -0.0003 | 14.1% | 11.0% |
| Day 20 | CatBoost + traits | 0.2216 | 0.2170 | -0.0046 | 16.3% | 13.2% |
| Day 20 | TabNet + traits | 0.2321 | 0.2234 | -0.0088 | 13.8% | 10.7% |

## Uncertainty

I measured a CatBoost-minus-logistic Brier difference of -0.0135 before the IPO, with a 95% interval of [-0.0328, +0.0040]. For day 20, I measured -0.0053, with an interval of [-0.0158, +0.0050]. I used 2,000 paired resamples of calendar-quarter blocks. You should read these as conditional intervals for the retained cohort; they omit uncertainty from missing issuers and model selection. A fixed 50% forecast has Brier error of 0.2500. That reference limits the strength of the pre-IPO result. For pre-IPO CatBoost, I measured mean probability of 42.3% against an event rate of 51.3%. You should inspect calibration before using a probability threshold.

## Data coverage

I started from 4,497 Field–Ritter records dated 2010–2025. After scope exclusions, I requested prices for 3,031 records and matched 1,393 (46.0%). I retained 1,358 pre-IPO observations and 1,343 day-20 observations after calendar and price checks. I verified listing dates against an extra dated registry source for 1,315 pre-IPO observations. You have broader historical coverage than the MVP, but missing histories can cause survivorship bias. You cannot extend these scores to the complete IPO population.

## Features

For pre-IPO models, you use seven numeric features and three categorical features: log offer price, age capped at 80 years, the preceding 90-day IPO count, preceding market returns over 20 and 60 sessions, 20-session market volatility, 60-session market drawdown, ADR status, venture backing, and dual-class status. For day-20 challengers, you add the original eight price/volume and market features. You can compare price-only CatBoost with enriched logistic regression to separate feature changes from model changes. I omitted underwriters from the common feature view because the public workbook ends in September 2020. I excluded filing financial statements and the stale internet flag. You have an offer-terms and issuer-traits model.

## Validation

I used four expanding test blocks: 2018–2019, 2020–2021, 2022–2023, and 2024–2025. For each block, I excluded training cases whose outcome window reached the test period. I kept one observation per issuer within each stage and fitted preprocessing on training rows. For each challenger, I selected one of three configurations using an inner chronological validation split. I used validation Brier error for configuration selection and early stopping, then refitted on the mature outer training set. I averaged predictions from seeds 42, 137, and 2026. I used no class reweighting or post-hoc calibrator.

## Period sensitivity

You get different rankings across periods. In 2022–2025, I measured day-20 Brier error of 0.1921 for enriched logistic regression and 0.2085 for enriched CatBoost. For the final two blocks in the pre-IPO study, validation selected a one-tree CatBoost fit. You should interpret that result as a near-constant forecast, with little evidence of useful nonlinear structure in that period. You can inspect the seed scores and selected training lengths in the saved fold records.

## Limits

You use current research compilations of IPO-time facts. The sources do not provide an archived publication timestamp for each feature. You therefore have a retrospective reconstruction, with code-enforced market-data cutoffs. I screened acquisition companies with name and unit-ticker rules; those rules leave security-type classification risk. I excluded histories with calendar gaps or nonpositive prices/volume. Those exclusions can omit distressed issuers. You should treat all observed drawdowns as close-based vendor-adjusted outcomes. You have no estimate of intraday loss or trading returns.

## Next use

You have no demonstrated allocation-policy benefit from these classifiers. For the next study, use an allocator who can accept or decline a fixed allocation at the offer price and exit at close 20. You need continuous entry-based returns on a common share basis and dated offering terms before fitting a return model. You should compare expected policy loss with a single-feature policy and with accepting or declining all eligible allocations. For Anthropic, build a table of comparable deal terms and offer-based price paths. You must define the comparable cohort before counting it or inspecting returns. The current study does not validate a model recommendation for Anthropic.

## Reproduce

You can rebuild features and refit the study from the frozen input without network access.

```sh
uv sync --locked --extra challengers
uv run --extra challengers ipo-challengers build
uv run --extra challengers ipo-challengers train
uv run --extra challengers ipo-challengers verify
uv run --extra challengers ipo-challengers report
```

You can regenerate Markdown and report-artifact.json with the repository's report command. To regenerate CHALLENGER_REPORT.html, you need an installed external Data Analytics plugin. You pass its directory to `node research/expanded/render_report.mjs`. The wrapper imports build-report scripts and the packaged reader from that plugin path; the repository does not include those dependencies. You can open the committed HTML without the plugin.

You can refresh the registry with `ipo-challengers registry`, then fetch prices with `ipo-challengers fetch`. Use a copied research directory to preserve the published input and protocol. You can retry failed downloads with `fetch --retry-failures`. The trainer caches models under `artifacts/challengers` and invalidates them when training inputs or model code change.

You retain the original 84-listing benchmark in `research/input.json` and `web/data/research.json`. The dashboard still presents that MVP; use the expanded report for this experiment.

## Sources

You use the [Field–Ritter dataset](https://site.warrington.ufl.edu/ritter/ipo-data/), [IPOScoop historical offer records](https://www.iposcoop.com/scoop-track-record-from-2000-to-present/), and [Stock Analysis IPO lists](https://stockanalysis.com/ipos/). You use Yahoo Finance chart responses for adjusted closes and volume. You can inspect exact URLs and SHA-256 hashes in `research/expanded/universe.json.gz` and `input.json.gz`. Credit the Field–Ritter dataset of company founding dates, as used by Field and Karpoff (2002) and Loughran and Ritter (2004).
