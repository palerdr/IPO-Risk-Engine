"""Generate the study readout from saved forecasts, with source references."""
import json
from pathlib import Path
import sqlite3

import numpy as np

from .decision_audit import audit
from .evaluate import stats
from .sources import load_json, save_json, sha

NAMES = {"base_rate": "Training rate", "logistic": "Logistic",
         "boosted_trees": "Boosted trees", "logistic_enriched": "Logistic + traits",
         "catboost_price": "CatBoost prices", "catboost": "CatBoost + traits", "tabnet": "TabNet + traits"}


def pct(value):
    return f"{value * 100:.2f}%" if 0 < abs(value) < .0005 else f"{value * 100:.1f}%"


def interval(values):
    return f"[{values[0]:+.4f}, {values[1]:+.4f}]"


def report(directory: Path):
    result = load_json(directory / "results.json.gz")
    data = load_json(directory / "dataset.json.gz")
    sensitivity = load_json(directory / "coverage-results.json.gz")
    decision = audit(directory, directory.parent / "decision/audit.json")
    pre, day = result["stages"]["0"], result["stages"]["20"]
    quality = result["quality"]
    # Analysts can reproduce Brier scores with a second calculation engine.
    audit_path = directory / "audit.sqlite3"
    with sqlite3.connect(audit_path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS forecasts (stage TEXT, model TEXT, issuer_id TEXT, event INTEGER, probability REAL)")
        db.execute("DELETE FROM forecasts")
        db.executemany("INSERT INTO forecasts VALUES (?, ?, ?, ?, ?)",
                       [(stage, name, row["issuer_id"], row["event"], p)
                        for stage, study in result["stages"].items() for row in study["predictions"]
                        for name, p in row["probabilities"].items()])
        scoring_sql = Path(__file__).with_name("audit.sql").read_text()
        audit_rows = db.execute(scoring_sql).fetchall()
        for stage, name, count, events, brier, mean in audit_rows:
            metric = result["stages"][stage]["models"][name]
            assert count == metric["n"] and events == metric["events"]
            np.testing.assert_allclose([brier, mean], [metric["brier"], metric["mean_probability"]], atol=1e-12)
        coverage_sql = "SELECT year, SUM(in_scope) AS in_scope, SUM(matched) AS matched, 1.0 * SUM(matched) / SUM(in_scope) AS coverage FROM registry_coverage GROUP BY year ORDER BY year"
        db.execute("CREATE TABLE IF NOT EXISTS registry_coverage (year INTEGER, in_scope INTEGER, matched INTEGER)")
        db.execute("DELETE FROM registry_coverage")
        accepted = {row["id"] for row in load_json(directory / "input.json.gz")["listings"]}
        registry = load_json(directory / "universe.json.gz")["registry"]
        db.executemany("INSERT INTO registry_coverage VALUES (?, ?, ?)",
                       [(int(row["offer_date"][:4]), int(not row["scope_exclusion"]), int(row["id"] in accepted)) for row in registry])
        for year, scope, matched_count, share in db.execute(coverage_sql):
            expected = next(row for row in quality["years"] if row["year"] == year)
            assert scope == expected["in_scope"] and matched_count == expected["price_matched"]
            assert share == expected["price_coverage"]
    in_scope = sum(r["in_scope"] for r in quality["years"])
    matched = sum(r["price_matched"] for r in quality["years"])
    pc, pl = pre["models"]["catboost"], pre["models"]["logistic"]
    dc, dl = day["models"]["catboost"], day["models"]["logistic"]
    title = "IPO challenger study"
    summary = (f"I trained CatBoost and TabNet for pre-IPO and day-20 drawdown forecasts. "
               f"Across {pc['n']:,} pre-IPO test cases, I measured CatBoost Brier error of {pc['brier']:.4f}, "
               f"versus {pl['brier']:.4f} for logistic regression. Across {dc['n']:,} day-20 test cases, "
               f"I measured {dc['brier']:.4f} for CatBoost with issuer traits and {dl['brier']:.4f} for the original logistic benchmark. "
               "You should treat these as point-estimate gains: both paired confidence intervals include zero. "
               "I measured higher aggregate error for TabNet in both stages. "
               f"After restricting training to listings from 2018 onward, I measured pre-IPO Brier error of "
               f"{sensitivity['stages']['0']['models']['logistic']['brier']:.4f} for logistic regression and "
               f"{sensitivity['stages']['0']['models']['catboost']['brier']:.4f} for CatBoost on the same "
               f"{sensitivity['stages']['0']['models']['catboost']['n']} test cases from 2020–2025.")
    definitions = ("You predict a maximum adjusted-close peak-to-trough decline of at least 20%. "
                   "For pre-IPO forecasts, you score after final pricing and before the first trade, then measure the first 20 closing prices. "
                   "For day-20 forecasts, you score at the twentieth close and measure the next 20 sessions, including that close as the initial peak. "
                   "You compare models within each stage because the two targets cover different price windows. "
                   "You read Brier error as mean squared probability error, with lower values indicating better forecasts. "
                   "You read ROC-AUC as ranking quality, with higher values indicating better separation. "
                   "You calculate Brier skill as 1 minus model Brier divided by reference Brier on the same test rows. "
                   "You compare against each fold's training event rate and against a constant 50% forecast (Brier 0.2500). "
                   "Positive skill means lower error than that reference; negative skill means higher error.")
    coverage = (f"I started from {quality['registry_count']:,} Field–Ritter records dated 2010–2025. "
                f"After scope exclusions, I requested prices for {in_scope:,} records and matched {matched:,} "
                f"({pct(matched / in_scope)}). I retained {pre['eligible']:,} pre-IPO observations and {day['eligible']:,} day-20 observations after calendar and price checks. "
                f"I verified listing dates against an extra dated registry source for {quality['dated_cross_references']:,} pre-IPO observations. "
                "You have broader historical coverage than the MVP, but missing histories can cause survivorship bias. "
                "You cannot extend these scores to the complete IPO population.")
    methods = ("I used four expanding test blocks: 2018–2019, 2020–2021, 2022–2023, and 2024–2025. "
               "For each block, I excluded training cases whose outcome window reached the test period. "
               "I kept one observation per issuer within each stage and fitted preprocessing on training rows. "
               "For each challenger, I selected one of three configurations using an inner chronological validation split. "
               "I used validation Brier error for configuration selection and early stopping, then refitted on the mature outer training set. "
               "I averaged predictions from seeds 42, 137, and 2026. I used no class reweighting or post-hoc calibrator.")
    features = ("For pre-IPO models, you use seven numeric features and three categorical features: log offer price, age capped at 80 years, "
                "the preceding 90-day IPO count, preceding market returns over 20 and 60 sessions, 20-session market volatility, "
                "60-session market drawdown, ADR status, venture backing, and dual-class status. "
                "For day-20 challengers, you add the original eight price/volume and market features. "
                "You can compare price-only CatBoost with enriched logistic regression to separate feature changes from model changes. "
                "I omitted underwriters from the common feature view because the public workbook ends in September 2020. "
                "I excluded filing financial statements and the stale internet flag. You have an offer-terms and issuer-traits model.")
    uncertainty = (f"I measured a CatBoost-minus-logistic Brier difference of {pc['brier']-pl['brier']:+.4f} before the IPO, "
                   f"with a 95% interval of {interval(pc['brier_difference_vs_logistic_ci'])}. For day 20, I measured "
                   f"{dc['brier']-dl['brier']:+.4f}, with an interval of {interval(dc['brier_difference_vs_logistic_ci'])}. "
                   "I used 2,000 paired resamples of calendar-quarter blocks. You should read these as conditional intervals for the retained cohort; "
                   "they omit uncertainty from missing issuers and model selection. "
                   "A fixed 50% forecast has Brier error of 0.2500. That reference limits the strength of the pre-IPO result. "
                   f"For pre-IPO CatBoost, I measured mean probability of {pct(pc['mean_probability'])} against an event rate of {pct(pc['event_rate'])}. "
                   "You should inspect calibration before using a probability threshold.")
    recency = ("You get different rankings across periods. In 2022–2025, I measured day-20 Brier error of "
               f"{day['recent_2022_2025']['logistic_enriched']['brier']:.4f} for enriched logistic regression and "
               f"{day['recent_2022_2025']['catboost']['brier']:.4f} for enriched CatBoost. "
               "For the final two blocks in the pre-IPO study, validation selected a one-tree CatBoost fit. "
               "You should interpret that result as a near-constant forecast, with little evidence of useful nonlinear structure in that period. "
               "You can inspect the seed scores and selected training lengths in the saved fold records.")
    limitations = ("You use current research compilations of IPO-time facts. The sources do not provide an archived publication timestamp for each feature. "
                   "You therefore have a retrospective reconstruction, with code-enforced market-data cutoffs. "
                   "I screened acquisition companies with name and unit-ticker rules; those rules leave security-type classification risk. "
                   "I excluded histories with calendar gaps or nonpositive prices/volume. Those exclusions can omit distressed issuers. "
                   "You should treat all observed drawdowns as close-based vendor-adjusted outcomes. "
                   "You have no estimate of intraday loss or trading returns.")
    next_steps = ("You have no demonstrated allocation-policy benefit from these classifiers. "
                  "For the next study, use an allocator who can accept or decline a fixed allocation at the offer price and exit at close 20. "
                  "You need continuous entry-based returns on a common share basis and dated offering terms before fitting a return model. "
                  "You should compare expected policy loss with a single-feature policy and with accepting or declining all eligible allocations. "
                  "For Anthropic, build a table of comparable deal terms and offer-based price paths. "
                  "You must define the comparable cohort before counting it or inspecting returns. "
                  "The current study does not validate a model recommendation for Anthropic.")
    audit_pre, audit_day = decision["stages"]["0"], decision["stages"]["20"]
    decision_summary = (
        "You have no demonstrated decision value from this forecast benchmark. "
        f"On {audit_pre['ranking_cases']} pre-IPO cases with an offer price, I measured AUC of "
        f"{audit_pre['single_feature_auc']:.3f} for negative log offer price and "
        f"{audit_pre['model_aucs_same_cases']['logistic']:.3f} for logistic regression on those same cases. "
        f"On {audit_day['ranking_cases']} day-20 cases, I measured {audit_day['single_feature_auc']:.3f} for realized volatility; "
        f"the best model AUC was {max(audit_day['model_aucs_same_cases'].values()):.3f}. "
        "I added these one-feature references after the original evaluation, so you should treat the comparison as a diagnostic. "
        "AUC describes ranking for the existing drawdown label. It does not establish the expected loss of an allocation policy.")
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
        "The current label measures a decline from a running peak and does not encode loss from the allocation price.")
    coverage_audit = (
        f"Across the sixteen IPO years from 2010 to 2025, I measured correlations of "
        f"{audit_pre['yearly_coverage_event_correlation']:.3f} before the IPO and "
        f"{audit_day['yearly_coverage_event_correlation']:.3f} at day 20 between price coverage and annual event rate. "
        "You cannot use these correlations to separate missing-history bias from changes in issuer mix or market conditions. "
        "The 2018-onward refit below tests dependence on the older cohort; it does not remove survivorship bias.")


    metric_rows, fold_rows, screening_rows, robustness_rows = [], [], [], []
    for stage, value in result["stages"].items():
        label = "Pre-IPO" if stage == "0" else "Day 20"
        for name, metric in value["models"].items():
            metric_rows.append({"stage": label, "model": NAMES[name], "brier": metric["brier"],
                                "brier_display": f"{metric['brier']:.4f}", "auc": f"{metric['roc_auc']:.4f}",
                                "ap": f"{metric['average_precision']:.4f}", "log_loss": f"{metric['log_loss']:.4f}",
                                "brier_skill": pct(metric["brier_skill"]), "skill_50": pct(metric["brier_skill_vs_constant_50"]), "n": metric["n"], "events": metric["events"],
                                "difference_ci": interval(metric["brier_difference_vs_logistic_ci"])})
        for fold in value["folds"]:
            record = {"stage": label, "period": f"{fold['test_start'][:4]}–{fold['test_end'][:4]}",
                      "train": fold["train_count"], "test": fold["test_count"], "purged": fold["purged_count"],
                      "train_event_rate": pct(fold["train_event_rate"]),
                      "test_event_rate": pct(fold["models"]["logistic"]["event_rate"])}
            record.update({name: f"{fold['models'][name]['brier']:.4f}" for name in ("logistic", "catboost", "tabnet")})
            fold_rows.append(record)
        for name in ("logistic", "catboost", "tabnet"):
            screen = value["models"][name]["screening"][1]
            screening_rows.append({"stage": label, "model": NAMES[name], "flagged": screen["flagged"],
                                   "precision": pct(screen["precision"]), "recall": pct(screen["recall"]),
                                   "severe_missed": f"{screen['missed_severe']}/{screen['severe_events']}"})
        lookup = {row["id"]: row for row in data["stages"][stage]}
        segments = {
            "Exclude prior MVP listings": [row for row in value["predictions"] if not row["prior_mvp_case"]],
            "IPO offer price >= $10": [row for row in value["predictions"]
                                       if lookup[row["id"]]["features"]["log_offer_price"] is not None
                                       and lookup[row["id"]]["features"]["log_offer_price"] >= np.log(10)],
        }
        for segment, rows in segments.items():
            record = {"stage": label, "segment": segment, "n": len(rows)}
            record.update({name: f"{stats(rows, name)['brier']:.4f}" for name in ("logistic", "catboost", "tabnet")})
            robustness_rows.append(record)
    sensitivity_rows, training_rows = [], []
    for stage, restricted in sensitivity["stages"].items():
        label = "Pre-IPO" if stage == "0" else "Day 20"
        held_out = {row["id"]: row for row in restricted["predictions"]}
        matched_rows = [row for row in result["stages"][stage]["predictions"] if row["id"] in held_out]
        assert len(matched_rows) == len(held_out)
        assert all(row["event"] == held_out[row["id"]]["event"] for row in matched_rows)
        for name, metric in restricted["models"].items():
            main = stats(matched_rows, name)
            sensitivity_rows.append({"stage": label, "model": NAMES[name], "n": metric["n"],
                                     "main": f"{main['brier']:.4f}", "restricted": f"{metric['brier']:.4f}",
                                     "change": f"{metric['brier'] - main['brier']:+.4f}",
                                     "brier_skill": pct(metric["brier_skill"]),
                                     "skill_50": pct(metric["brier_skill_vs_constant_50"])})
        for fold in restricted["folds"]:
            main_fold = next(f for f in result["stages"][stage]["folds"] if f["test_start"] == fold["test_start"])
            training_rows.append({"stage": label, "period": f"{fold['test_start'][:4]}–{fold['test_end'][:4]}",
                                  "main": main_fold["train_count"], "restricted": fold["train_count"],
                                  "n": fold["test_count"]})
    sensitivity_sql = "SELECT cohort, stage, model, COUNT(*) AS cases, AVG((probability-event)*(probability-event)) AS brier FROM coverage_forecasts GROUP BY cohort, stage, model"
    with sqlite3.connect(audit_path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS coverage_forecasts (cohort TEXT, stage TEXT, model TEXT, event INTEGER, probability REAL)")
        db.execute("DELETE FROM coverage_forecasts")
        for stage, restricted in sensitivity["stages"].items():
            ids = {row["id"] for row in restricted["predictions"]}
            for cohort, predictions in (("2010", result["stages"][stage]["predictions"]), ("2018", restricted["predictions"])):
                db.executemany("INSERT INTO coverage_forecasts VALUES (?, ?, ?, ?, ?)",
                               [(cohort, stage, name, row["event"], probability) for row in predictions if row["id"] in ids
                                for name, probability in row["probabilities"].items()])
        for cohort, stage, name, count, brier in db.execute(sensitivity_sql):
            selected = [row for row in (result if cohort == "2010" else sensitivity)["stages"][stage]["predictions"]
                        if row["as_of"] >= "2020-01-01"]
            metric = stats(selected, name)
            assert count == metric["n"]
            np.testing.assert_allclose(brier, metric["brier"], atol=1e-12)
    older = [r for r in quality["years"] if r["year"] < 2018]
    newer = [r for r in quality["years"] if r["year"] >= 2018]
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
        "You cannot isolate the effect of missing histories or remove survivorship bias with this comparison.")
    rendering = (
        "You can regenerate Markdown and report-artifact.json with the repository's report command. "
        "To regenerate CHALLENGER_REPORT.html, you need an installed external Data Analytics plugin. "
        "You pass its directory to `node research/expanded/render_report.mjs`. "
        "The wrapper imports build-report scripts and the packaged reader from that plugin path; "
        "the repository does not include those dependencies. You can open the committed HTML without the plugin.")
    coverage_rows = [{"year": r["year"], "registry": r["registry"], "in_scope": r["in_scope"],
                      "matched": r["price_matched"], "coverage": r["price_coverage"],
                      "pre": r["stage_counts"]["0"], "day": r["stage_counts"]["20"]} for r in quality["years"]]
    sources = [{"id": "results", "label": "Frozen challenger forecasts and evaluation", "path": "research/expanded/results.json.gz",
                "query": {"engine": "SQLite", "language": "sql", "description": "Recompute Brier error and event counts from the frozen model forecasts in research/expanded/audit.sqlite3. Compare against Python evaluation; retain ranking metrics and uncertainty from results.json.gz.",
                          "sql": scoring_sql,
                          "tables_used": ["forecasts"],
                          "executed_at": result["generated_at"]}},
               {"id": "coverage_source", "label": "Registry and price-coverage reconciliation", "path": "research/expanded/audit.sqlite3",
                "query": {"engine": "SQLite", "language": "sql", "sql": coverage_sql,
                          "description": "Count in-scope Field–Ritter registry rows and accepted Yahoo price matches by year.",
                          "tables_used": ["registry_coverage"]}},
               {"id": "registry", "label": "Field–Ritter issuer registry and source audit", "path": "research/expanded/universe.json.gz",
                "href": "https://site.warrington.ufl.edu/ritter/ipo-data/"},
               {"id": "protocol", "label": "Frozen training and evaluation protocol", "path": "research/expanded/protocol.json"}]
    sources.append({"id": "coverage_sensitivity", "label": "Training coverage sensitivity on shared test issuers",
                    "path": "research/expanded/coverage-results.json.gz",
                    "query": {"engine": "SQLite", "language": "sql", "sql": sensitivity_sql,
                              "description": "Recompute both training cohorts' Brier errors on matching 2020–2025 test issuers. Read training counts from the saved folds and skill from Python evaluation.",
                              "tables_used": ["coverage_forecasts"]}})
    sources.append({"id": "decision_audit", "label": "Frozen-data decision relevance audit",
                    "path": "research/decision/audit.json"})
    blocks, charts, tables = [], [], []

    def prose(key, heading, body, source="results"):
        blocks.append({"id": key, "type": "markdown", "body": f"## {heading}\n\n{body}", "sourceId": source})

    def table(key, title, dataset, columns, sort):
        tables.append({"id": key, "title": title, "dataset": dataset, "sourceId": "coverage_sensitivity" if dataset in ("coverage_training", "coverage_sensitivity") else "results",
                       "defaultSort": {"field": sort, "direction": "asc"},
                       "columns": [{"field": field, "label": label} for field, label in columns]})
        blocks.append({"id": key + "-block", "type": "table", "tableId": key, "layout": "full"})

    blocks.append({"id": "title", "type": "markdown", "body": f"# {title}"})
    prose("decision-summary", "You have no demonstrated allocation-policy benefit", decision_summary, "decision_audit")
    prose("entry-target", "You need entry-based outcomes for an allocator", target_audit, "decision_audit")
    prose("coverage-audit", "Coverage and event rates change together", coverage_audit, "decision_audit")
    prose("summary", "You can inspect the historical forecast comparison", summary)
    prose("definitions", "You compare forecasts within each stage", definitions, "protocol")
    for stage, dataset in (("Pre-IPO", "pre_metrics"), ("Day 20", "day_metrics")):
        prose(dataset + "-intro", f"{stage} model comparison",
              "You can compare probability error in the bars and inspect ranking quality in the table. "
              "You should prefer a model only after reviewing its period results and uncertainty.")
        charts.append({"id": dataset, "type": "bar", "title": f"{stage} Brier error", "dataset": dataset,
                       "sourceId": "results", "encodings": {"x": {"field": "model", "type": "nominal", "label": "Model"},
                                                             "y": {"field": "brier", "type": "quantitative", "label": "Brier error"}},
                       "layout": "full"})
        blocks.append({"id": dataset + "-chart", "type": "chart", "chartId": dataset, "layout": "full"})
        table(dataset + "-table", f"{stage} metrics", dataset,
              [("model", "Model"), ("brier_display", "Brier"), ("brier_skill", "Skill vs train rate"), ("skill_50", "Skill vs 50%"),
               ("auc", "ROC-AUC"), ("n", "Cases")], "brier_display")
    prose("uncertainty", "You have no clear improvement over logistic regression", uncertainty)
    prose("coverage", "Missing price histories limit the cohort", coverage, "results")
    charts.append({"id": "coverage", "type": "line", "title": "Price-history coverage by IPO year", "dataset": "coverage",
                   "sourceId": "coverage_source", "valueFormat": "percent", "encodings": {
                       "x": {"field": "year", "type": "ordinal", "label": "IPO year"},
                       "y": {"field": "coverage", "type": "quantitative", "label": "Share of in-scope registry"}}})
    blocks.append({"id": "coverage-chart", "type": "chart", "chartId": "coverage", "layout": "full"})
    table("coverage-table", "Cohort coverage", "coverage", [("year", "IPO year"), ("in_scope", "In scope"),
          ("matched", "Price matched"), ("pre", "Pre-IPO usable"), ("day", "Day-20 usable")], "year")
    prose("features", "You use dated market data and historical issuer traits", features, "protocol")
    prose("methods", "You select challengers before the outer test period", methods, "protocol")
    prose("periods", "You get different rankings across periods", recency)
    table("fold-table", "Brier error by test period", "folds", [("stage", "Stage"), ("period", "Test period"),
          ("test", "Cases"),
          ("logistic", "Logistic"), ("catboost", "CatBoost"), ("tabnet", "TabNet")], "period")
    prose("screening", "High recall requires broad flagging",
          "At the fixed 30% threshold, you flag most listings with CatBoost. You should compare the fraction flagged with precision and missed severe events. "
          "Here, a severe event means a drawdown of at least 30%. You have a risk-screening diagnostic, with no trading-policy validation.")
    table("screening-table", "Screening at a 30% probability cutoff", "screening", [("stage", "Stage"), ("model", "Model"),
          ("flagged", "Flagged"), ("precision", "Precision"), ("recall", "Recall"), ("severe_missed", "Severe events missed")], "stage")
    prose("robustness", "You can inspect cohort sensitivity",
          "I excluded prior MVP listings from one sensitivity cut because you have seen some of their outcomes. "
          "I added the offer-price cut after viewing the aggregate results to inspect cohort mix. "
          "You should treat that cut as a diagnostic; I used no subgroup score for model selection.")
    table("robustness-table", "Brier error on sensitivity subsets", "robustness", [("stage", "Stage"), ("segment", "Subset"),
          ("n", "Cases"), ("logistic", "Logistic"), ("catboost", "CatBoost"), ("tabnet", "TabNet")], "stage")
    prose("coverage-sensitivity", "You can compare training coverage", coverage_sensitivity, "coverage_sensitivity")
    table("coverage-training-table", "Training cohort sizes", "coverage_training",
          [("stage", "Stage"), ("period", "Test period"), ("main", "Train from 2010"),
           ("restricted", "Train from 2018"), ("n", "Test cases")], "period")
    table("coverage-sensitivity-table", "Brier error on matched 2020–2025 test cases", "coverage_sensitivity",
          [("stage", "Stage"), ("model", "Model"), ("main", "Train from 2010"),
           ("restricted", "Train from 2018"), ("change", "Change")], "stage")
    table("coverage-skill-table", "Brier skill after training from 2018", "coverage_sensitivity",
          [("stage", "Stage"), ("model", "Model"), ("brier_skill", "Skill vs train rate"),
           ("skill_50", "Skill vs 50%"), ("n", "Cases")], "stage")
    prose("rendering", "You need the external plugin to rebuild HTML", rendering)
    prose("limits", "You need stronger evidence for population claims", limitations)
    prose("next", "You need an allocator study before recommending an action", next_steps)
    artifact = {"surface": "report", "manifest": {"version": 1, "surface": "report", "title": title,
                  "generatedAt": result["generated_at"], "blocks": blocks, "charts": charts, "tables": tables, "sources": sources},
                "snapshot": {"version": 1, "status": "ready", "generatedAt": result["generated_at"], "datasets": {
                    "pre_metrics": [r for r in metric_rows if r["stage"] == "Pre-IPO"],
                    "day_metrics": [r for r in metric_rows if r["stage"] == "Day 20"],
                    "coverage": coverage_rows, "folds": fold_rows, "screening": screening_rows, "robustness": robustness_rows, "coverage_training": training_rows, "coverage_sensitivity": sensitivity_rows}},
                "sources": sources}
    save_json(directory / "report-artifact.json", artifact)
    save_json(directory / "report-notes.json", {"results_sha256": sha(directory / "results.json.gz"),
              "coverage_results_sha256": sha(directory / "coverage-results.json.gz"),
              "decision_audit": "research/decision/audit.json; methods in expanded/decision_audit.py; executed companion research/decision/audit.ipynb",
              "decision_audit_visual": "Exact comparisons remain in prose because the two stages have different complete-case denominators. The existing full model tables remain below.",
              "html_regeneration": rendering, "audience": "technical", "delivery": "html", "structure": "Definitions precede comparisons. Next steps include transfer questions.",
              "charts": "Use separate Brier bars for the two targets and a coverage line across sixteen IPO years. Tables preserve exact values.",
              "subgroup_selection": "Offer price >= $10 is a post-hoc cohort diagnostic; no refitting or parameter changes.",
              "source_credit": "Field–Ritter dataset of company founding dates; Field and Karpoff (2002), Loughran and Ritter (2004)."})
    def markdown_table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
                          *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]])

    lines = [f"# {title}", "## Decision relevance audit", decision_summary, "## Entry-based outcomes", target_audit, "## Coverage association", coverage_audit, "## Historical forecast comparison", summary, "## Targets", definitions, "## Held-out results",
             markdown_table(["Stage", "Model", "Brier", "Skill vs train rate", "Skill vs 50%", "ROC-AUC", "Test cases"],
                            [[r[k] for k in ("stage", "model", "brier_display", "brier_skill", "skill_50", "auc", "n")]
                             for r in metric_rows]),
             "## Training coverage sensitivity", coverage_sensitivity,
             markdown_table(["Stage", "Test period", "Train from 2010", "Train from 2018", "Test cases"],
                            [[r[k] for k in ("stage", "period", "main", "restricted", "n")] for r in training_rows]),
             "You read both skill columns below for the models trained from 2018 onward. You compare each model with that run's training-rate forecasts and with constant 50% forecasts.",
             markdown_table(["Stage", "Model", "Brier: train from 2010", "Brier: train from 2018", "Change", "Skill vs train rate", "Skill vs 50%"],
                            [[r[k] for k in ("stage", "model", "main", "restricted", "change", "brier_skill", "skill_50")]
                             for r in sensitivity_rows])]
    for heading, body in (("Uncertainty", uncertainty), ("Data coverage", coverage), ("Features", features),
                          ("Validation", methods), ("Period sensitivity", recency), ("Limits", limitations), ("Next use", next_steps)):
        lines.extend([f"\n## {heading}\n", body])
    lines.extend(["\n## Reproduce\n", "You can rebuild features and refit the study from the frozen input without network access.",
                  "\n```sh\nuv sync --locked --extra challengers\nuv run --extra challengers ipo-challengers build\nuv run --extra challengers ipo-challengers train\nuv run --extra challengers ipo-challengers verify\nuv run --extra challengers ipo-challengers report\n```\n",
                  rendering,
                  "You can refresh the registry with `ipo-challengers registry`, then fetch prices with `ipo-challengers fetch`. "
                  "Use a copied research directory to preserve the published input and protocol. "
                  "You can retry failed downloads with `fetch --retry-failures`. "
                  "The trainer caches models under `artifacts/challengers` and invalidates them when training inputs or model code change.",
                  "\nYou retain the original 84-listing benchmark in `research/input.json` and `web/data/research.json`. "
                  "The dashboard still presents that MVP; use the expanded report for this experiment.",
                  "\n## Sources\n",
                  "You use the [Field–Ritter dataset](https://site.warrington.ufl.edu/ritter/ipo-data/), "
                  "[IPOScoop historical offer records](https://www.iposcoop.com/scoop-track-record-from-2000-to-present/), "
                  "and [Stock Analysis IPO lists](https://stockanalysis.com/ipos/). You use Yahoo Finance chart responses for adjusted closes and volume. "
                  "You can inspect exact URLs and SHA-256 hashes in `research/expanded/universe.json.gz` and `input.json.gz`. "
                  "Credit the Field–Ritter dataset of company founding dates, as used by Field and Karpoff (2002) and Loughran and Ritter (2004)."])
    docs = directory.resolve().parents[1] / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "CHALLENGER_STUDY.md").write_text("\n\n".join(line.strip() for line in lines) + "\n")
    print(f"Report input: {directory / 'report-artifact.json'}", flush=True)
    return artifact
