"""Build dated feature views and audit labels from the frozen price snapshot."""
from collections import Counter
from datetime import date, timedelta
import json
from pathlib import Path
import re

import numpy as np

from ..dataset import FEATURES as PRICE_FEATURES, drawdown, make_features, validate_bars
from .sources import load_json, save_json, sha

PRE_NUMERIC = ("log_offer_price", "age_capped_80", "log_prior_90d_ipo_count",
               "pre_market_return_20", "pre_market_return_60", "pre_market_volatility_20",
               "pre_market_drawdown_60")
PRE_CATEGORICAL = ("adr", "vc_backing", "dual_class")
PRE_FEATURES = PRE_NUMERIC + PRE_CATEGORICAL
# The common feature view omits underwriters: the public workbook ends in September 2020.
# Analysts can inspect that source field in the registry without treating its absence as a firm trait.


def issuer_key(row: dict) -> str:
    if row.get("cusip") and re.fullmatch(r"[A-Z0-9]{8,9}", row["cusip"]):
        return "cusip:" + row["cusip"][:6]
    return "name:" + re.sub(r"[^a-z0-9]", "", row["name"].lower())


def pre_features(listing: dict, market: list[dict], registry: list[dict]) -> tuple[dict, str]:
    prior = [bar for bar in market if bar["date"] < listing["listing_date"]][-61:]
    if len(prior) < 61:
        raise ValueError("Fewer than 61 pre-listing benchmark sessions")
    validate_bars(prior)
    closes = np.array([r["adjusted_close"] for r in prior])
    returns = closes[1:] / closes[:-1] - 1
    first = date.fromisoformat(listing["listing_date"])
    activity_start = (first - timedelta(days=90)).isoformat()
    activity = sum(activity_start <= row["offer_date"] < listing["listing_date"]
                   and row["id"] != listing["id"] and not row["scope_exclusion"] for row in registry)
    year = listing.get("founding_year")
    age = min(80, first.year - year) if year is not None and 1800 <= year <= first.year else None
    price = listing.get("offer_price")
    features = {
        "log_offer_price": float(np.log(price)) if price is not None and price > 0 else None,
        "age_capped_80": age, "log_prior_90d_ipo_count": float(np.log1p(activity)),
        "pre_market_return_20": float(closes[-1] / closes[-21] - 1),
        "pre_market_return_60": float(closes[-1] / closes[0] - 1),
        "pre_market_volatility_20": float(np.std(returns[-20:], ddof=1)),
        "pre_market_drawdown_60": drawdown(closes),
        "adr": "adr" if listing["adr_code"] == 2 else "domestic_share",
        "vc_backing": {0: "none", 1: "venture", 2: "growth"}.get(listing.get("vc"), "unknown"),
        "dual_class": {0: "single", 1: "dual"}.get(listing.get("dual_class"), "unknown"),
    }
    return features, prior[-1]["date"]


def snapshots(bundle: dict, universe: dict, sessions: int) -> tuple[list[dict], list[dict]]:
    if sessions not in (0, 20):
        raise ValueError("Use 0 for pre-IPO or 20 for the current repository convention")
    market = bundle["benchmark"]["bars"]
    validate_bars(market)
    records, rejected = [], []
    for listing in bundle["listings"]:
        try:
            required = sessions + 20
            bars = [r for r in listing["history"]["bars"] if r["date"] >= listing["listing_date"]][:required]
            if len(bars) != required:
                raise ValueError(f"Fewer than {required} sessions")
            validate_bars(bars)
            if bars[0]["date"] != listing["listing_date"]:
                raise ValueError("First session differs from verified listing date")
            calendar = [r["date"] for r in market if bars[0]["date"] <= r["date"] <= bars[-1]["date"]]
            if [r["date"] for r in bars] != calendar:
                raise ValueError("Trading calendar gap")
            pre, market_cutoff = pre_features(listing, market, universe["registry"])
            features = dict(pre)
            if sessions:
                features.update(make_features(bars[:sessions], market))
            future = bars[sessions - 1:] if sessions else bars
            severity = drawdown(np.array([r["adjusted_close"] for r in future]))
            as_of = bars[sessions - 1]["date"] if sessions else listing["listing_date"]
            for key, value in features.items():
                if value is not None and not isinstance(value, str) and not np.isfinite(value):
                    raise ValueError(f"Nonfinite feature: {key}")
            records.append({"id": listing["id"], "issuer_id": issuer_key(listing),
                            "symbol": listing["symbol"], "name": listing["name"], "stage": sessions,
                            "listing_date": listing["listing_date"], "as_of": as_of,
                            "as_of_phase": "close" if sessions else "pre_open_after_pricing",
                            "pre_market_cutoff": market_cutoff,
                            "price_feature_cutoff": as_of if sessions else None,
                            "label_start": future[0]["date"], "label_end": bars[-1]["date"],
                            "drawdown": severity, "event": int(severity >= .20),
                            "features": features, "price_source_sha256": listing["history"]["raw_sha256"],
                            "source_url": listing["history"]["source_url"],
                            "cross_source_date_verified": bool(listing["cross_references"])})
        except (ValueError, KeyError) as exc:
            rejected.append({"id": listing["id"], "symbol": listing["symbol"], "stage": sessions,
                             "listing_date": listing["listing_date"], "reason": str(exc)})
    # Analysts retain the first event for an issuer to prevent repeated-company leakage.
    records.sort(key=lambda r: (r["as_of"], r["id"]))
    unique, seen = [], set()
    for row in records:
        if row["issuer_id"] in seen:
            rejected.append({"id": row["id"], "symbol": row["symbol"], "stage": sessions,
                             "listing_date": row["listing_date"], "reason": "Repeated issuer"})
        else:
            seen.add(row["issuer_id"])
            unique.append(row)
    return unique, rejected


def build(directory: Path) -> dict:
    bundle = load_json(directory / "input.json.gz")
    universe = load_json(directory / "universe.json.gz")
    if bundle["universe_sha256"] != sha(directory / "universe.json.gz"):
        raise ValueError("Input and registry hashes differ")
    stages, exclusions = {}, {}
    for sessions in (0, 20):
        rows, rejected = snapshots(bundle, universe, sessions)
        stages[str(sessions)], exclusions[str(sessions)] = rows, rejected
    years = []
    for year in range(*[universe["years"][0], universe["years"][1] + 1]):
        requested = [r for r in universe["registry"] if r["offer_date"].startswith(str(year))]
        eligible = [r for r in requested if not r["scope_exclusion"]]
        accepted = [r for r in bundle["listings"] if r["offer_date"].startswith(str(year))]
        counts = {key: sum(r["listing_date"].startswith(str(year)) for r in rows) for key, rows in stages.items()}
        years.append({"year": year, "registry": len(requested), "in_scope": len(eligible),
                      "price_matched": len(accepted), "stage_counts": counts,
                      "price_coverage": len(accepted) / len(eligible) if eligible else None})
    result = {"schema_version": "3.0.0", "input_sha256": sha(directory / "input.json.gz"),
              "universe_sha256": sha(directory / "universe.json.gz"),
              "feature_schema": {"pre_numeric": list(PRE_NUMERIC), "categorical": list(PRE_CATEGORICAL),
                                 "day_20_price": list(PRICE_FEATURES)},
              "stages": stages, "stage_exclusions": exclusions,
              "quality": {"years": years, "registry_count": len(universe["registry"]),
                          "scope_exclusions": dict(Counter(r["reason"] for r in bundle["exclusions"] if r["exclusion_stage"] == "scope")),
                          "price_exclusions": dict(Counter(r["reason"].split(":")[0] for r in bundle["exclusions"] if r["exclusion_stage"] == "prices")),
                          "stage_exclusion_counts": {key: dict(Counter(r["reason"] for r in rows)) for key, rows in exclusions.items()},
                          "missing_features": {stage: {key: sum(r["features"][key] is None for r in rows)
                                                        for key in rows[0]["features"]} for stage, rows in stages.items()},
                          "offer_price_conflicts": sum(r["offer_price_conflict"] for r in universe["registry"]),
                          "dated_cross_references": sum(r["cross_source_date_verified"] for r in stages["0"])}}
    save_json(directory / "dataset.json.gz", result)
    print(f"Dataset: pre-IPO={len(stages['0'])}, day-20={len(stages['20'])}", flush=True)
    return result
