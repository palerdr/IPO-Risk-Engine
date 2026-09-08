"""Keep the study narrative separate from report layout and SQL audits."""


def pct(value: float) -> str:
    return f"{value * 100:.2f}%" if 0 < abs(value) < 0.0005 else f"{value * 100:.1f}%"


def interval(values: list[float]) -> str:
    return f"[{values[0]:+.4f}, {values[1]:+.4f}]"


def narrative(result: dict, sensitivity: dict, decision: dict) -> dict[str, str]:
    pre, day = result["stages"]["0"], result["stages"]["20"]
    quality = result["quality"]
    in_scope = sum(row["in_scope"] for row in quality["years"])
    matched = sum(row["price_matched"] for row in quality["years"])
    pc, pl = pre["models"]["catboost"], pre["models"]["logistic"]
    dc, dl = day["models"]["catboost"], day["models"]["logistic"]
    audit_pre, audit_day = decision["stages"]["0"], decision["stages"]["20"]
    older = [row for row in quality["years"] if row["year"] < 2018]
    newer = [row for row in quality["years"] if row["year"] >= 2018]
    summary = (
        f"I trained CatBoost and TabNet for pre-IPO and day-20 drawdown forecasts. "
        f"Across {pc['n']:,} pre-IPO test cases, I measured CatBoost Brier error of {pc['brier']:.4f}, "
        f"versus {pl['brier']:.4f} for logistic regression. Across {dc['n']:,} day-20 test cases, "
        f"I measured {dc['brier']:.4f} for CatBoost with issuer traits and {dl['brier']:.4f} for the original logistic benchmark. "
        "You should treat these as point-estimate gains: both paired confidence intervals include zero. "
        "I measured higher aggregate error for TabNet in both stages. "
        f"After restricting training to listings from 2018 onward, I measured pre-IPO Brier error of "
        f"{sensitivity['stages']['0']['models']['logistic']['brier']:.4f} for logistic regression and "
        f"{sensitivity['stages']['0']['models']['catboost']['brier']:.4f} for CatBoost on the same "
        f"{sensitivity['stages']['0']['models']['catboost']['n']} test cases from 2020–2025."
    )
    definitions = (
        "You predict a maximum adjusted-close peak-to-trough decline of at least 20%. "
        "For pre-IPO forecasts, you score after final pricing and before the first trade, then measure the first 20 closing prices. "
        "For day-20 forecasts, you score at the twentieth close and measure the next 20 sessions, including that close as the initial peak. "
        "You compare models within each stage because the two targets cover different price windows. "
        "You read Brier error as mean squared probability error, with lower values indicating better forecasts. "
        "You read ROC-AUC as ranking quality, with higher values indicating better separation. "
        "You calculate Brier skill as 1 minus model Brier divided by reference Brier on the same test rows. "
        "You compare against each fold's training event rate and against a constant 50% forecast (Brier 0.2500). "
        "Positive skill means lower error than that reference; negative skill means higher error."
    )
    coverage = (
        f"I started from {quality['registry_count']:,} Field–Ritter records dated 2010–2025. "
        f"After scope exclusions, I requested prices for {in_scope:,} records and matched {matched:,} "
        f"({pct(matched / in_scope)}). I retained {pre['eligible']:,} pre-IPO observations and {day['eligible']:,} day-20 observations after calendar and price checks. "
        f"I verified listing dates against an extra dated registry source for {quality['dated_cross_references']:,} pre-IPO observations. "
        "You have broader historical coverage than the MVP, but missing histories can cause survivorship bias. "
        "You cannot extend these scores to the complete IPO population."
    )
    methods = (
        "I used four expanding test blocks: 2018–2019, 2020–2021, 2022–2023, and 2024–2025. "
        "For each block, I excluded training cases whose outcome window reached the test period. "
        "I kept one observation per issuer within each stage and fitted preprocessing on training rows. "
        "For each challenger, I selected one of three configurations using an inner chronological validation split. "
        "I used validation Brier error for configuration selection and early stopping, then refitted on the mature outer training set. "
        "I averaged predictions from seeds 42, 137, and 2026. I used no class reweighting or post-hoc calibrator."
    )
    features = (
        "For pre-IPO models, you use seven numeric features and three categorical features: log offer price, age capped at 80 years, "
        "the preceding 90-day IPO count, preceding market returns over 20 and 60 sessions, 20-session market volatility, "
        "60-session market drawdown, ADR status, venture backing, and dual-class status. "
        "For day-20 challengers, you add the original eight price/volume and market features. "
        "You can compare price-only CatBoost with enriched logistic regression to separate feature changes from model changes. "
        "I omitted underwriters from the common feature view because the public workbook ends in September 2020. "
        "I excluded filing financial statements and the stale internet flag. You have an offer-terms and issuer-traits model."
    )
    uncertainty = (
        f"I measured a CatBoost-minus-logistic Brier difference of {pc['brier'] - pl['brier']:+.4f} before the IPO, "
        f"with a 95% interval of {interval(pc['brier_difference_vs_logistic_ci'])}. For day 20, I measured "
        f"{dc['brier'] - dl['brier']:+.4f}, with an interval of {interval(dc['brier_difference_vs_logistic_ci'])}. "
        "I used 2,000 paired resamples of calendar-quarter blocks. You should read these as conditional intervals for the retained cohort; "
        "they omit uncertainty from missing issuers and model selection. "
        "A fixed 50% forecast has Brier error of 0.2500. That reference limits the strength of the pre-IPO result. "
        f"For pre-IPO CatBoost, I measured mean probability of {pct(pc['mean_probability'])} against an event rate of {pct(pc['event_rate'])}. "
        "You should inspect calibration before using a probability threshold."
    )
    recency = (
        "You get different rankings across periods. In 2022–2025, I measured day-20 Brier error of "
        f"{day['recent_2022_2025']['logistic_enriched']['brier']:.4f} for enriched logistic regression and "
        f"{day['recent_2022_2025']['catboost']['brier']:.4f} for enriched CatBoost. "
        "For the final two blocks in the pre-IPO study, validation selected a one-tree CatBoost fit. "
        "You should interpret that result as a near-constant forecast, with little evidence of useful nonlinear structure in that period. "
        "You can inspect the seed scores and selected training lengths in the saved fold records."
    )
    limitations = (
        "You use current research compilations of IPO-time facts. The sources do not provide an archived publication timestamp for each feature. "
        "You therefore have a retrospective reconstruction, with code-enforced market-data cutoffs. "
        "I screened acquisition companies with name and unit-ticker rules; those rules leave security-type classification risk. "
        "I excluded histories with calendar gaps or nonpositive prices/volume. Those exclusions can omit distressed issuers. "
        "You should treat all observed drawdowns as close-based vendor-adjusted outcomes. "
        "You have no estimate of intraday loss or trading returns."
    )
    next_steps = (
        "You have no demonstrated allocation-policy benefit from these classifiers. "
        "For the next study, use an allocator who can accept or decline a fixed allocation at the offer price and exit at close 20. "
        "You need continuous entry-based returns on a common share basis and dated offering terms before fitting a return model. "
        "You should compare expected policy loss with a single-feature policy and with accepting or declining all eligible allocations. "
        "For Anthropic, build a table of comparable deal terms and offer-based price paths. "
        "You must define the comparable cohort before counting it or inspecting returns. "
        "The current study does not validate a model recommendation for Anthropic."
    )
    decision_summary = (
        "You have no demonstrated decision value from this forecast benchmark. "
        f"On {audit_pre['ranking_cases']} pre-IPO cases with an offer price, I measured AUC of "
        f"{audit_pre['single_feature_auc']:.3f} for negative log offer price and "
        f"{audit_pre['model_aucs_same_cases']['logistic']:.3f} for logistic regression on those same cases. "
        f"On {audit_day['ranking_cases']} day-20 cases, I measured {audit_day['single_feature_auc']:.3f} for realized volatility; "
        f"the best model AUC was {max(audit_day['model_aucs_same_cases'].values()):.3f}. "
        "I added these one-feature references after the original evaluation, so you should treat the comparison as a diagnostic. "
        "AUC describes ranking for the existing drawdown label. It does not establish the expected loss of an allocation policy."
    )
    target_audit = (
        f"I reproduced a nonnegative terminal return proxy for {audit_pre['events_with_nonnegative_terminal_proxy']} "
        f"of {audit_pre['event_cases_with_return_proxy']} pre-IPO events ({pct(audit_pre['share_events_with_nonnegative_terminal_proxy'])}), "
        f"and {audit_day['events_with_nonnegative_terminal_proxy']} of {audit_day['event_cases_with_return_proxy']} "
        f"day-20 events ({pct(audit_day['share_events_with_nonnegative_terminal_proxy'])}). "
        "You must keep the price bases visible: the pre-IPO proxy divides vendor-adjusted closes by nominal offer prices. "
        "You need split and distribution accounting on the original offered-share basis before calling that proxy allocator P&L. "
        "You should also distinguish a terminal recovery from a path that stayed above entry. "
        f"I measured a median first-close/offer proxy of {pct(audit_pre['median_first_close_offer_proxy'])} "
        f"and a correlation of {audit_pre['first_close_offer_proxy_event_correlation']:.3f} with the pre-IPO event label. "
        "The current label measures a decline from a running peak and does not encode loss from the allocation price."
    )
    coverage_audit = (
        f"Across the sixteen IPO years from 2010 to 2025, I measured correlations of "
        f"{audit_pre['yearly_coverage_event_correlation']:.3f} before the IPO and "
        f"{audit_day['yearly_coverage_event_correlation']:.3f} at day 20 between price coverage and annual event rate. "
        "You cannot use these correlations to separate missing-history bias from changes in issuer mix or market conditions. "
        "The 2018-onward refit below tests dependence on the older cohort; it does not remove survivorship bias."
    )
    coverage_sensitivity = (
        f"I matched prices for {sum(r['price_matched'] for r in older):,} of {sum(r['in_scope'] for r in older):,} "
        f"in-scope listings from 2010–2017 ({pct(sum(r['price_matched'] for r in older) / sum(r['in_scope'] for r in older))}), "
        f"and {sum(r['price_matched'] for r in newer):,} of {sum(r['in_scope'] for r in newer):,} "
        f"from 2018–2025 ({pct(sum(r['price_matched'] for r in newer) / sum(r['in_scope'] for r in newer))}). "
        "At your request, I added a post-hoc sensitivity that trains on listings from 2018 onward. "
        "I retained the candidate grids and inner validation rules, then refitted the models for each test block. "
        "You compare both training cohorts on identical 2020–2025 held-out issuers. "
        "You read a negative Brier change as lower error for the restricted training cohort. "
        "You can assess dependence on the earlier cohort, but this restriction also changes sample size and market regimes. "
        "You cannot isolate the effect of missing histories or remove survivorship bias with this comparison."
    )
    rendering = (
        "You can regenerate Markdown and report-artifact.json with the repository's report command. "
        "To regenerate CHALLENGER_REPORT.html, you need an installed external Data Analytics plugin. "
        "You pass its directory to `node research/expanded/render_report.mjs`. "
        "The wrapper imports build-report scripts and the packaged reader from that plugin path; "
        "the repository does not include those dependencies. You can open the committed HTML without the plugin."
    )
    return {
        "summary": summary,
        "definitions": definitions,
        "coverage": coverage,
        "methods": methods,
        "features": features,
        "uncertainty": uncertainty,
        "recency": recency,
        "limitations": limitations,
        "next_steps": next_steps,
        "decision_summary": decision_summary,
        "target_audit": target_audit,
        "coverage_audit": coverage_audit,
        "coverage_sensitivity": coverage_sensitivity,
        "rendering": rendering,
    }
