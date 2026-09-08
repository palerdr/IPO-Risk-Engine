"""Generate the study readout from saved forecasts, with source references."""

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from .decision_audit import audit
from .evaluate import stats
from .report_text import interval, narrative, pct
from .storage import load_json, save_json, sha

NAMES = {
    "base_rate": "Training rate",
    "logistic": "Logistic",
    "boosted_trees": "Boosted trees",
    "logistic_enriched": "Logistic + traits",
    "catboost_price": "CatBoost prices",
    "catboost": "CatBoost + traits",
    "tabnet": "TabNet + traits",
}


def report(directory: Path, docs: Path = Path("docs")):
    result = load_json(directory / "results.json.gz")
    data = load_json(directory / "dataset.json.gz")
    sensitivity = load_json(directory / "coverage-results.json.gz")
    decision = audit(directory, directory.parent / "decision/audit.json")
    quality = result["quality"]
    text = narrative(result, sensitivity, decision)
    # Analysts can reproduce Brier scores with a second calculation engine.
    audit_path = directory / "audit.sqlite3"
    with closing(sqlite3.connect(audit_path)) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS forecasts (stage TEXT, model TEXT, issuer_id TEXT, event INTEGER, probability REAL)"
        )
        db.execute("DELETE FROM forecasts")
        db.executemany(
            "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?)",
            [
                (stage, name, row["issuer_id"], row["event"], p)
                for stage, study in result["stages"].items()
                for row in study["predictions"]
                for name, p in row["probabilities"].items()
            ],
        )
        scoring_sql = Path(__file__).with_name("audit.sql").read_text()
        audit_rows = db.execute(scoring_sql).fetchall()
        for stage, name, count, events, brier, mean in audit_rows:
            metric = result["stages"][stage]["models"][name]
            if not (count == metric["n"] and events == metric["events"]):
                raise ValueError(
                    'Audit mismatch: count == metric["n"] and events == metric["events"]'
                )
            np.testing.assert_allclose(
                [brier, mean], [metric["brier"], metric["mean_probability"]], atol=1e-12
            )
        coverage_sql = "SELECT year, SUM(in_scope) AS in_scope, SUM(matched) AS matched, 1.0 * SUM(matched) / SUM(in_scope) AS coverage FROM registry_coverage GROUP BY year ORDER BY year"
        db.execute(
            "CREATE TABLE IF NOT EXISTS registry_coverage (year INTEGER, in_scope INTEGER, matched INTEGER)"
        )
        db.execute("DELETE FROM registry_coverage")
        accepted = {row["id"] for row in load_json(directory / "input.json.gz")["listings"]}
        registry = load_json(directory / "universe.json.gz")["registry"]
        db.executemany(
            "INSERT INTO registry_coverage VALUES (?, ?, ?)",
            [
                (
                    int(row["offer_date"][:4]),
                    int(not row["scope_exclusion"]),
                    int(row["id"] in accepted),
                )
                for row in registry
            ],
        )
        for year, scope, matched_count, share in db.execute(coverage_sql):
            expected = next(row for row in quality["years"] if row["year"] == year)
            if not (scope == expected["in_scope"] and matched_count == expected["price_matched"]):
                raise ValueError(
                    'Audit mismatch: scope == expected["in_scope"] and matched_count == expected["price_matched"]'
                )
            if not (share == expected["price_coverage"]):
                raise ValueError('Audit mismatch: share == expected["price_coverage"]')
    title = "IPO challenger study"

    metric_rows, fold_rows, screening_rows, robustness_rows = [], [], [], []
    for stage, value in result["stages"].items():
        label = "Pre-IPO" if stage == "0" else "Day 20"
        for name, metric in value["models"].items():
            metric_rows.append(
                {
                    "stage": label,
                    "model": NAMES[name],
                    "brier": metric["brier"],
                    "brier_display": f"{metric['brier']:.4f}",
                    "auc": f"{metric['roc_auc']:.4f}",
                    "ap": f"{metric['average_precision']:.4f}",
                    "log_loss": f"{metric['log_loss']:.4f}",
                    "brier_skill": pct(metric["brier_skill"]),
                    "skill_50": pct(metric["brier_skill_vs_constant_50"]),
                    "n": metric["n"],
                    "events": metric["events"],
                    "difference_ci": interval(metric["brier_difference_vs_logistic_ci"]),
                }
            )
        for fold in value["folds"]:
            record = {
                "stage": label,
                "period": f"{fold['test_start'][:4]}–{fold['test_end'][:4]}",
                "train": fold["train_count"],
                "test": fold["test_count"],
                "purged": fold["purged_count"],
                "train_event_rate": pct(fold["train_event_rate"]),
                "test_event_rate": pct(fold["models"]["logistic"]["event_rate"]),
            }
            record.update(
                {
                    name: f"{fold['models'][name]['brier']:.4f}"
                    for name in ("logistic", "catboost", "tabnet")
                }
            )
            fold_rows.append(record)
        for name in ("logistic", "catboost", "tabnet"):
            screen = value["models"][name]["screening"][1]
            screening_rows.append(
                {
                    "stage": label,
                    "model": NAMES[name],
                    "flagged": screen["flagged"],
                    "precision": pct(screen["precision"]),
                    "recall": pct(screen["recall"]),
                    "severe_missed": f"{screen['missed_severe']}/{screen['severe_events']}",
                }
            )
        lookup = {row["id"]: row for row in data["stages"][stage]}
        segments = {
            "Exclude prior MVP listings": [
                row for row in value["predictions"] if not row["prior_mvp_case"]
            ],
            "IPO offer price >= $10": [
                row
                for row in value["predictions"]
                if lookup[row["id"]]["features"]["log_offer_price"] is not None
                and lookup[row["id"]]["features"]["log_offer_price"] >= np.log(10)
            ],
        }
        for segment, rows in segments.items():
            record = {"stage": label, "segment": segment, "n": len(rows)}
            record.update(
                {
                    name: f"{stats(rows, name)['brier']:.4f}"
                    for name in ("logistic", "catboost", "tabnet")
                }
            )
            robustness_rows.append(record)
    sensitivity_rows, training_rows = [], []
    for stage, restricted in sensitivity["stages"].items():
        label = "Pre-IPO" if stage == "0" else "Day 20"
        held_out = {row["id"]: row for row in restricted["predictions"]}
        matched_rows = [
            row for row in result["stages"][stage]["predictions"] if row["id"] in held_out
        ]
        if not (len(matched_rows) == len(held_out)):
            raise ValueError("Audit mismatch: len(matched_rows) == len(held_out)")
        if not (all(row["event"] == held_out[row["id"]]["event"] for row in matched_rows)):
            raise ValueError(
                'Audit mismatch: all(row["event"] == held_out[row["id"]]["event"] for row in matched_rows)'
            )
        for name, metric in restricted["models"].items():
            main = stats(matched_rows, name)
            sensitivity_rows.append(
                {
                    "stage": label,
                    "model": NAMES[name],
                    "n": metric["n"],
                    "main": f"{main['brier']:.4f}",
                    "restricted": f"{metric['brier']:.4f}",
                    "change": f"{metric['brier'] - main['brier']:+.4f}",
                    "brier_skill": pct(metric["brier_skill"]),
                    "skill_50": pct(metric["brier_skill_vs_constant_50"]),
                }
            )
        for fold in restricted["folds"]:
            main_fold = next(
                f for f in result["stages"][stage]["folds"] if f["test_start"] == fold["test_start"]
            )
            training_rows.append(
                {
                    "stage": label,
                    "period": f"{fold['test_start'][:4]}–{fold['test_end'][:4]}",
                    "main": main_fold["train_count"],
                    "restricted": fold["train_count"],
                    "n": fold["test_count"],
                }
            )
    sensitivity_sql = "SELECT cohort, stage, model, COUNT(*) AS cases, AVG((probability-event)*(probability-event)) AS brier FROM coverage_forecasts GROUP BY cohort, stage, model"
    with closing(sqlite3.connect(audit_path)) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS coverage_forecasts (cohort TEXT, stage TEXT, model TEXT, event INTEGER, probability REAL)"
        )
        db.execute("DELETE FROM coverage_forecasts")
        for stage, restricted in sensitivity["stages"].items():
            ids = {row["id"] for row in restricted["predictions"]}
            for cohort, predictions in (
                ("2010", result["stages"][stage]["predictions"]),
                ("2018", restricted["predictions"]),
            ):
                db.executemany(
                    "INSERT INTO coverage_forecasts VALUES (?, ?, ?, ?, ?)",
                    [
                        (cohort, stage, name, row["event"], probability)
                        for row in predictions
                        if row["id"] in ids
                        for name, probability in row["probabilities"].items()
                    ],
                )
        for cohort, stage, name, count, brier in db.execute(sensitivity_sql):
            selected = [
                row
                for row in (result if cohort == "2010" else sensitivity)["stages"][stage][
                    "predictions"
                ]
                if row["as_of"] >= "2020-01-01"
            ]
            metric = stats(selected, name)
            if not (count == metric["n"]):
                raise ValueError('Audit mismatch: count == metric["n"]')
            np.testing.assert_allclose(brier, metric["brier"], atol=1e-12)
    coverage_rows = [
        {
            "year": r["year"],
            "registry": r["registry"],
            "in_scope": r["in_scope"],
            "matched": r["price_matched"],
            "coverage": r["price_coverage"],
            "pre": r["stage_counts"]["0"],
            "day": r["stage_counts"]["20"],
        }
        for r in quality["years"]
    ]
    sources = [
        {
            "id": "results",
            "label": "Frozen challenger forecasts and evaluation",
            "path": "research/expanded/results.json.gz",
            "query": {
                "engine": "SQLite",
                "language": "sql",
                "description": "Recompute Brier error and event counts from the frozen model forecasts in research/expanded/audit.sqlite3. Compare against Python evaluation; retain ranking metrics and uncertainty from results.json.gz.",
                "sql": scoring_sql,
                "tables_used": ["forecasts"],
                "executed_at": result["generated_at"],
            },
        },
        {
            "id": "coverage_source",
            "label": "Registry and price-coverage reconciliation",
            "path": "research/expanded/audit.sqlite3",
            "query": {
                "engine": "SQLite",
                "language": "sql",
                "sql": coverage_sql,
                "description": "Count in-scope Field–Ritter registry rows and accepted Yahoo price matches by year.",
                "tables_used": ["registry_coverage"],
            },
        },
        {
            "id": "registry",
            "label": "Field–Ritter issuer registry and source audit",
            "path": "research/expanded/universe.json.gz",
            "href": "https://site.warrington.ufl.edu/ritter/ipo-data/",
        },
        {
            "id": "protocol",
            "label": "Frozen training and evaluation protocol",
            "path": "research/expanded/protocol.json",
        },
    ]
    sources.append(
        {
            "id": "coverage_sensitivity",
            "label": "Training coverage sensitivity on shared test issuers",
            "path": "research/expanded/coverage-results.json.gz",
            "query": {
                "engine": "SQLite",
                "language": "sql",
                "sql": sensitivity_sql,
                "description": "Recompute both training cohorts' Brier errors on matching 2020–2025 test issuers. Read training counts from the saved folds and skill from Python evaluation.",
                "tables_used": ["coverage_forecasts"],
            },
        }
    )
    sources.append(
        {
            "id": "decision_audit",
            "label": "Frozen-data decision relevance audit",
            "path": "research/decision/audit.json",
        }
    )
    blocks, charts, tables = [], [], []

    def prose(key, heading, body, source="results"):
        blocks.append(
            {"id": key, "type": "markdown", "body": f"## {heading}\n\n{body}", "sourceId": source}
        )

    def table(key, title, dataset, columns, sort):
        tables.append(
            {
                "id": key,
                "title": title,
                "dataset": dataset,
                "sourceId": "coverage_sensitivity"
                if dataset in ("coverage_training", "coverage_sensitivity")
                else "results",
                "defaultSort": {"field": sort, "direction": "asc"},
                "columns": [{"field": field, "label": label} for field, label in columns],
            }
        )
        blocks.append({"id": key + "-block", "type": "table", "tableId": key, "layout": "full"})

    blocks.append({"id": "title", "type": "markdown", "body": f"# {title}"})
    prose(
        "decision-summary",
        "You have no demonstrated allocation-policy benefit",
        text["decision_summary"],
        "decision_audit",
    )
    prose(
        "entry-target",
        "You need entry-based outcomes for an allocator",
        text["target_audit"],
        "decision_audit",
    )
    prose(
        "coverage-audit",
        "Coverage and event rates change together",
        text["coverage_audit"],
        "decision_audit",
    )
    prose("summary", "You can inspect the historical forecast comparison", text["summary"])
    prose("definitions", "You compare forecasts within each stage", text["definitions"], "protocol")
    for stage, dataset in (("Pre-IPO", "pre_metrics"), ("Day 20", "day_metrics")):
        prose(
            dataset + "-intro",
            f"{stage} model comparison",
            "You can compare probability error in the bars and inspect ranking quality in the table. "
            "You should prefer a model only after reviewing its period results and uncertainty.",
        )
        charts.append(
            {
                "id": dataset,
                "type": "bar",
                "title": f"{stage} Brier error",
                "dataset": dataset,
                "sourceId": "results",
                "encodings": {
                    "x": {"field": "model", "type": "nominal", "label": "Model"},
                    "y": {"field": "brier", "type": "quantitative", "label": "Brier error"},
                },
                "layout": "full",
            }
        )
        blocks.append(
            {"id": dataset + "-chart", "type": "chart", "chartId": dataset, "layout": "full"}
        )
        table(
            dataset + "-table",
            f"{stage} metrics",
            dataset,
            [
                ("model", "Model"),
                ("brier_display", "Brier"),
                ("brier_skill", "Skill vs train rate"),
                ("skill_50", "Skill vs 50%"),
                ("auc", "ROC-AUC"),
                ("n", "Cases"),
            ],
            "brier_display",
        )
    prose(
        "uncertainty", "You have no clear improvement over logistic regression", text["uncertainty"]
    )
    prose("coverage", "Missing price histories limit the cohort", text["coverage"], "results")
    charts.append(
        {
            "id": "coverage",
            "type": "line",
            "title": "Price-history coverage by IPO year",
            "dataset": "coverage",
            "sourceId": "coverage_source",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "year", "type": "ordinal", "label": "IPO year"},
                "y": {
                    "field": "coverage",
                    "type": "quantitative",
                    "label": "Share of in-scope registry",
                },
            },
        }
    )
    blocks.append(
        {"id": "coverage-chart", "type": "chart", "chartId": "coverage", "layout": "full"}
    )
    table(
        "coverage-table",
        "Cohort coverage",
        "coverage",
        [
            ("year", "IPO year"),
            ("in_scope", "In scope"),
            ("matched", "Price matched"),
            ("pre", "Pre-IPO usable"),
            ("day", "Day-20 usable"),
        ],
        "year",
    )
    prose(
        "features",
        "You use dated market data and historical issuer traits",
        text["features"],
        "protocol",
    )
    prose(
        "methods",
        "You select challengers before the outer test period",
        text["methods"],
        "protocol",
    )
    prose("periods", "You get different rankings across periods", text["recency"])
    table(
        "fold-table",
        "Brier error by test period",
        "folds",
        [
            ("stage", "Stage"),
            ("period", "Test period"),
            ("test", "Cases"),
            ("logistic", "Logistic"),
            ("catboost", "CatBoost"),
            ("tabnet", "TabNet"),
        ],
        "period",
    )
    prose(
        "screening",
        "High recall requires broad flagging",
        "At the fixed 30% threshold, you flag most listings with CatBoost. You should compare the fraction flagged with precision and missed severe events. "
        "Here, a severe event means a drawdown of at least 30%. You have a risk-screening diagnostic, with no trading-policy validation.",
    )
    table(
        "screening-table",
        "Screening at a 30% probability cutoff",
        "screening",
        [
            ("stage", "Stage"),
            ("model", "Model"),
            ("flagged", "Flagged"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("severe_missed", "Severe events missed"),
        ],
        "stage",
    )
    prose(
        "robustness",
        "You can inspect cohort sensitivity",
        "I excluded prior MVP listings from one sensitivity cut because you have seen some of their outcomes. "
        "I added the offer-price cut after viewing the aggregate results to inspect cohort mix. "
        "You should treat that cut as a diagnostic; I used no subgroup score for model selection.",
    )
    table(
        "robustness-table",
        "Brier error on sensitivity subsets",
        "robustness",
        [
            ("stage", "Stage"),
            ("segment", "Subset"),
            ("n", "Cases"),
            ("logistic", "Logistic"),
            ("catboost", "CatBoost"),
            ("tabnet", "TabNet"),
        ],
        "stage",
    )
    prose(
        "coverage-sensitivity",
        "You can compare training coverage",
        text["coverage_sensitivity"],
        "coverage_sensitivity",
    )
    table(
        "coverage-training-table",
        "Training cohort sizes",
        "coverage_training",
        [
            ("stage", "Stage"),
            ("period", "Test period"),
            ("main", "Train from 2010"),
            ("restricted", "Train from 2018"),
            ("n", "Test cases"),
        ],
        "period",
    )
    table(
        "coverage-sensitivity-table",
        "Brier error on matched 2020–2025 test cases",
        "coverage_sensitivity",
        [
            ("stage", "Stage"),
            ("model", "Model"),
            ("main", "Train from 2010"),
            ("restricted", "Train from 2018"),
            ("change", "Change"),
        ],
        "stage",
    )
    table(
        "coverage-skill-table",
        "Brier skill after training from 2018",
        "coverage_sensitivity",
        [
            ("stage", "Stage"),
            ("model", "Model"),
            ("brier_skill", "Skill vs train rate"),
            ("skill_50", "Skill vs 50%"),
            ("n", "Cases"),
        ],
        "stage",
    )
    prose("rendering", "You need the external plugin to rebuild HTML", text["rendering"])
    prose("limits", "You need stronger evidence for population claims", text["limitations"])
    prose("next", "You need an allocator study before recommending an action", text["next_steps"])
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "generatedAt": result["generated_at"],
            "blocks": blocks,
            "charts": charts,
            "tables": tables,
            "sources": sources,
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": result["generated_at"],
            "datasets": {
                "pre_metrics": [r for r in metric_rows if r["stage"] == "Pre-IPO"],
                "day_metrics": [r for r in metric_rows if r["stage"] == "Day 20"],
                "coverage": coverage_rows,
                "folds": fold_rows,
                "screening": screening_rows,
                "robustness": robustness_rows,
                "coverage_training": training_rows,
                "coverage_sensitivity": sensitivity_rows,
            },
        },
        "sources": sources,
    }
    save_json(directory / "report-artifact.json", artifact)
    save_json(
        directory / "report-notes.json",
        {
            "results_sha256": sha(directory / "results.json.gz"),
            "coverage_results_sha256": sha(directory / "coverage-results.json.gz"),
            "decision_audit": "research/decision/audit.json; methods in expanded/decision_audit.py; executed companion research/decision/audit.ipynb",
            "decision_audit_visual": "Exact comparisons remain in prose because the two stages have different complete-case denominators. The existing full model tables remain below.",
            "html_regeneration": text["rendering"],
            "audience": "technical",
            "delivery": "html",
            "structure": "Definitions precede comparisons. Next steps include transfer questions.",
            "charts": "Use separate Brier bars for the two targets and a coverage line across sixteen IPO years. Tables preserve exact values.",
            "subgroup_selection": "Offer price >= $10 is a post-hoc cohort diagnostic; no refitting or parameter changes.",
            "source_credit": "Field–Ritter dataset of company founding dates; Field and Karpoff (2002), Loughran and Ritter (2004).",
        },
    )

    def markdown_table(headers, rows):
        return "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "|" + "---|" * len(headers),
                *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows],
            ]
        )

    lines = [
        f"# {title}",
        "## Decision relevance audit",
        text["decision_summary"],
        "## Entry-based outcomes",
        text["target_audit"],
        "## Coverage association",
        text["coverage_audit"],
        "## Historical forecast comparison",
        text["summary"],
        "## Targets",
        text["definitions"],
        "## Held-out results",
        markdown_table(
            [
                "Stage",
                "Model",
                "Brier",
                "Skill vs train rate",
                "Skill vs 50%",
                "ROC-AUC",
                "Test cases",
            ],
            [
                [
                    r[k]
                    for k in (
                        "stage",
                        "model",
                        "brier_display",
                        "brier_skill",
                        "skill_50",
                        "auc",
                        "n",
                    )
                ]
                for r in metric_rows
            ],
        ),
        "## Training coverage sensitivity",
        text["coverage_sensitivity"],
        markdown_table(
            ["Stage", "Test period", "Train from 2010", "Train from 2018", "Test cases"],
            [[r[k] for k in ("stage", "period", "main", "restricted", "n")] for r in training_rows],
        ),
        "You read both skill columns below for the models trained from 2018 onward. You compare each model with that run's training-rate forecasts and with constant 50% forecasts.",
        markdown_table(
            [
                "Stage",
                "Model",
                "Brier: train from 2010",
                "Brier: train from 2018",
                "Change",
                "Skill vs train rate",
                "Skill vs 50%",
            ],
            [
                [
                    r[k]
                    for k in (
                        "stage",
                        "model",
                        "main",
                        "restricted",
                        "change",
                        "brier_skill",
                        "skill_50",
                    )
                ]
                for r in sensitivity_rows
            ],
        ),
    ]
    for heading, body in (
        ("Uncertainty", text["uncertainty"]),
        ("Data coverage", text["coverage"]),
        ("Features", text["features"]),
        ("Validation", text["methods"]),
        ("Period sensitivity", text["recency"]),
        ("Limits", text["limitations"]),
        ("Next use", text["next_steps"]),
    ):
        lines.extend([f"\n## {heading}\n", body])
    lines.extend(
        [
            "\n## Reproduce\n",
            "You can rebuild features and refit the study from the frozen input without network access.",
            "\n```sh\nuv sync --locked --extra challengers\nuv run --extra challengers ipo-challengers build\nuv run --extra challengers ipo-challengers train\nuv run --extra challengers ipo-challengers verify\nuv run --extra challengers ipo-challengers report\n```\n",
            text["rendering"],
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
            "Credit the Field–Ritter dataset of company founding dates, as used by Field and Karpoff (2002) and Loughran and Ritter (2004).",
        ]
    )
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "CHALLENGER_STUDY.md").write_text("\n\n".join(line.strip() for line in lines) + "\n")
    print(f"Report input: {directory / 'report-artifact.json'}", flush=True)
    return artifact
