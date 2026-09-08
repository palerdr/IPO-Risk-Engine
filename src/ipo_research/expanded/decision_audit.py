"""Audit frozen forecasts against single-feature rankings and entry-based proxies."""

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .storage import load_json, save_json, sha


def audit(directory: Path, output: Path | None = None):
    data = load_json(directory / "dataset.json.gz")
    forecasts = load_json(directory / "results.json.gz")
    bundle = load_json(directory / "input.json.gz")
    listings = {row["id"]: row for row in bundle["listings"]}
    result = {
        "status": "retrospective_diagnostic",
        "inputs": {
            name: sha(directory / name)
            for name in ("dataset.json.gz", "results.json.gz", "input.json.gz")
        },
        "ranking_selection": "Use the features and directions from the review: negative log offer price and positive daily volatility. These are post-hoc benchmarks; do not describe them as a new held-out test.",
        "price_basis": "Pre-IPO ratios divide vendor-adjusted closes by nominal offer prices. The frozen normalized input lacks a common share-basis bridge. These are audit proxies, not validated allocator returns.",
        "stages": {},
    }
    for stage, feature, direction in (("0", "log_offer_price", -1), ("20", "daily_volatility", 1)):
        features = {row["id"]: row["features"] for row in data["stages"][stage]}
        rows = forecasts["stages"][stage]["predictions"]
        complete = [row for row in rows if features[row["id"]][feature] is not None]
        y = np.array([row["event"] for row in complete])
        score = np.array([direction * features[row["id"]][feature] for row in complete])
        model_aucs = {
            name: float(roc_auc_score(y, [row["probabilities"][name] for row in complete]))
            for name in rows[0]["probabilities"]
            if name != "base_rate"
        }
        returns, minimums, events, pops = [], [], [], []
        for row in rows:
            listing = listings[row["id"]]
            bars = [
                bar for bar in listing["history"]["bars"] if bar["date"] >= listing["listing_date"]
            ]
            if stage == "0":
                entry = listing["offer_price"]
                if entry is None or entry <= 0:
                    continue
                path = [bar["adjusted_close"] for bar in bars[:20]]
            else:
                entry = bars[19]["adjusted_close"]
                path = [bar["adjusted_close"] for bar in bars[20:40]]
            if not (len(path) == 20):
                raise ValueError("Audit mismatch: len(path) == 20")
            returns.append(path[-1] / entry - 1)
            minimums.append(min(path) / entry - 1)
            events.append(row["event"])
            pops.append(bars[0]["adjusted_close"] / entry - 1)
        returns, minimums, events = (
            np.array(returns),
            np.array(minimums),
            np.array(events, dtype=bool),
        )
        yearly = []
        for coverage in data["quality"]["years"]:
            eligible = [
                row
                for row in data["stages"][stage]
                if row["listing_date"].startswith(str(coverage["year"]))
            ]
            yearly.append(
                {
                    "year": coverage["year"],
                    "coverage": coverage["price_coverage"],
                    "eligible": len(eligible),
                    "event_rate": float(np.mean([row["event"] for row in eligible])),
                }
            )
        stage_result = {
            "held_out": len(rows),
            "single_feature": feature,
            "direction": direction,
            "ranking_cases": len(complete),
            "ranking_missing": len(rows) - len(complete),
            "single_feature_auc": float(roc_auc_score(y, score)),
            "model_aucs_same_cases": model_aucs,
            "return_proxy_cases": len(returns),
            "event_cases_with_return_proxy": int(events.sum()),
            "events_with_nonnegative_terminal_proxy": int(np.sum(events & (returns >= 0))),
            "share_events_with_nonnegative_terminal_proxy": float(np.mean(returns[events] >= 0)),
            "events_never_below_entry_proxy": int(np.sum(events & (minimums >= 0))),
            "yearly_coverage_event_correlation": float(
                np.corrcoef([r["coverage"] for r in yearly], [r["event_rate"] for r in yearly])[
                    0, 1
                ]
            ),
            "yearly": yearly,
        }
        if stage == "0":
            stage_result.update(
                median_first_close_offer_proxy=float(np.median(pops)),
                first_close_offer_proxy_event_correlation=float(np.corrcoef(pops, events)[0, 1]),
            )
        result["stages"][stage] = stage_result
    if output is not None:
        save_json(output, result)
    return result
