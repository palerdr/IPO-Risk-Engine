"""Check source joins, feature cutoffs, and challenger evaluation boundaries."""

import gzip
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ipo_research.dataset import make_features
from ipo_research.expanded.dataset import pre_features, snapshots
from ipo_research.expanded.evaluate import paired_intervals, stats, temporal_folds
from ipo_research.expanded.models import Transform, features_for, fit_baseline, inner_split
from ipo_research.expanded.sources import parse_stockanalysis, scope_exclusion
from ipo_research.expanded.storage import load_json, save_json, sha
from ipo_research.models import fit_models, matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/expanded"


class ExpandedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_json(DATA / "input.json.gz")
        cls.universe = load_json(DATA / "universe.json.gz")
        cls.dataset = load_json(DATA / "dataset.json.gz")
        cls.protocol = json.loads((DATA / "protocol.json").read_text())

    def test_compressed_snapshots_preserve_content_hashes_and_repeatable_bytes(self):
        with TemporaryDirectory() as temporary:
            plain = Path(temporary) / "snapshot.json"
            packed = Path(temporary) / "snapshot.json.gz"
            value = {"issuer": "Test", "prices": [1.25, None, 2.5]}
            save_json(plain, value)
            save_json(packed, value)
            first = packed.read_bytes()
            self.assertEqual(gzip.decompress(first), plain.read_bytes())
            self.assertEqual(load_json(packed), value)
            self.assertEqual(sha(packed), sha(plain))
            save_json(packed, value)
            self.assertEqual(packed.read_bytes(), first)

    def test_brier_skill_references_differ_under_event_rate_shift(self):
        rows = [
            {
                "as_of": "2020-01-01",
                "event": event,
                "drawdown": 0.4 * event,
                "probabilities": {"base_rate": 0.1, "logistic": 0.5},
            }
            for event in (0, 1)
        ]
        metric = stats(rows, "logistic")
        self.assertAlmostEqual(metric["brier"], 0.25)
        self.assertAlmostEqual(metric["training_rate_reference_brier"], 0.41)
        self.assertAlmostEqual(metric["brier_skill"], 1 - 0.25 / 0.41)
        self.assertEqual(metric["brier_skill_vs_constant_50"], 0)
        self.assertEqual(
            paired_intervals(rows, "logistic", 20)["brier_skill_vs_constant_50_ci"], [0, 0]
        )

    def test_coverage_sensitivity_refits_without_changing_held_out_issuers(self):
        protocol = load_json(DATA / "coverage-protocol.json")
        for stage in ("0", "20"):
            main = temporal_folds(self.dataset["stages"][stage], self.protocol)[1:]
            restricted = temporal_folds(self.dataset["stages"][stage], protocol)
            for original, recent in zip(main, restricted, strict=True):
                self.assertEqual(original["test"], recent["test"])
                self.assertLess(len(recent["train"]), len(original["train"]))
                self.assertTrue(all(row["listing_date"] >= "2018-01-01" for row in recent["train"]))
                self.assertLess(max(row["label_end"] for row in recent["train"]), recent["start"])
                fit, valid, _ = inner_split(recent["train"])
                self.assertLess(
                    max(row["label_end"] for row in fit), min(row["as_of"] for row in valid)
                )

    def test_source_parser_excludes_current_prices_and_returns(self):
        html = """<table><thead><tr><th>IPO Date</th><th>Symbol</th><th>Company Name</th><th>IPO Price</th></tr></thead>
        <tbody><tr><td>Mar 21, 2024</td><td>RDDT</td><td>Reddit</td><td>$34.00</td><td>$9999</td><td>999%</td></tr></tbody></table>"""
        parsed = parse_stockanalysis(html, "https://example.test/source")
        self.assertEqual(parsed[0]["offer_price"], 34)
        self.assertNotIn("current_price", parsed[0])
        self.assertNotIn("return", parsed[0])
        with self.assertRaises(ValueError):
            parse_stockanalysis(
                html.replace("IPO Price", "Current Price"), "https://example.test/source"
            )

    def test_scope_rules_quarantine_acquisition_companies_and_direct_listings(self):
        row = {"symbol": "TEST", "name": "Test Corporation", "adr_code": 1}
        self.assertIsNone(scope_exclusion(row))
        self.assertEqual(
            scope_exclusion({**row, "name": "Test Acquisition Corp"}), "acquisition_company_name"
        )
        self.assertEqual(
            scope_exclusion({**row, "symbol": "TESTU"}), "unit_warrant_or_right_ticker"
        )
        self.assertEqual(scope_exclusion({**row, "symbol": "SPOT"}), "direct_listing")
        self.assertEqual(scope_exclusion({**row, "adr_code": 6}), "unsupported_security_code")

    def test_pre_ipo_features_ignore_issuer_prices_and_future_market_prices(self):
        listing = self.bundle["listings"][0]
        market = self.bundle["benchmark"]["bars"]
        before = pre_features(listing, market, self.universe["registry"])
        changed = deepcopy(listing)
        for bar in changed["history"]["bars"]:
            bar["adjusted_close"] *= 100
        changed_market = deepcopy(market)
        for bar in changed_market:
            if bar["date"] >= listing["listing_date"]:
                bar["adjusted_close"] *= 0.01
        self.assertEqual(before, pre_features(changed, changed_market, self.universe["registry"]))
        self.assertLess(before[1], listing["listing_date"])

    def test_activity_counts_come_from_registry_before_listing(self):
        listing = self.bundle["listings"][0]
        registry = deepcopy(self.universe["registry"])
        market = self.bundle["benchmark"]["bars"]
        before, _ = pre_features(listing, market, registry)
        registry.append({"id": "future", "offer_date": "2099-01-01", "scope_exclusion": None})
        after, _ = pre_features(listing, market, registry)
        self.assertEqual(before["log_prior_90d_ipo_count"], after["log_prior_90d_ipo_count"])

    def test_both_targets_use_the_documented_price_windows(self):
        selected = self.bundle["listings"][0]
        bundle = {**self.bundle, "listings": [deepcopy(selected)]}
        bars = bundle["listings"][0]["history"]["bars"]
        for bar in bars:
            bar["adjusted_close"] = 100
            bar["volume"] = 1000
        bars[19]["adjusted_close"] = 70
        bars[20]["adjusted_close"] = 70
        pre, _ = snapshots(bundle, self.universe, 0)
        day, _ = snapshots(bundle, self.universe, 20)
        self.assertAlmostEqual(pre[0]["drawdown"], 0.3)
        self.assertEqual(day[0]["drawdown"], 0)
        bars[39]["adjusted_close"] = 50
        pre_after, _ = snapshots(bundle, self.universe, 0)
        day_after, _ = snapshots(bundle, self.universe, 20)
        self.assertEqual(pre, pre_after)
        self.assertEqual(day[0]["features"], day_after[0]["features"])
        self.assertAlmostEqual(day_after[0]["drawdown"], 0.5)

    def test_day_20_price_features_preserve_the_mvp_definition(self):
        listing = self.bundle["listings"][0]
        rows, rejected = snapshots({**self.bundle, "listings": [listing]}, self.universe, 20)
        self.assertFalse(rejected)
        expected = make_features(listing["history"]["bars"][:20], self.bundle["benchmark"]["bars"])
        self.assertEqual(expected, {key: rows[0]["features"][key] for key in expected})

    def test_missing_sessions_and_invalid_prices_have_explicit_exclusions(self):
        for issue in ("gap", "negative"):
            listing = deepcopy(self.bundle["listings"][0])
            if issue == "gap":
                del listing["history"]["bars"][10]
            else:
                listing["history"]["bars"][10]["adjusted_close"] = -1
            rows, rejected = snapshots({**self.bundle, "listings": [listing]}, self.universe, 20)
            self.assertFalse(rows)
            self.assertTrue(rejected[0]["reason"])

    def test_outer_and_inner_splits_require_mature_labels_and_disjoint_issuers(self):
        for stage in ("0", "20"):
            seen = set()
            for fold in temporal_folds(self.dataset["stages"][stage], self.protocol):
                train, test = fold["train"], fold["test"]
                self.assertLess(max(r["label_end"] for r in train), min(r["as_of"] for r in test))
                self.assertTrue(
                    {r["issuer_id"] for r in train}.isdisjoint(r["issuer_id"] for r in test)
                )
                self.assertTrue(seen.isdisjoint(r["id"] for r in test))
                seen.update(r["id"] for r in test)
                fit, valid, info = inner_split(train)
                self.assertLess(max(r["label_end"] for r in fit), min(r["as_of"] for r in valid))
                self.assertEqual(info["fit_count"], len(fit))

    def test_transform_learns_missing_values_and_categories_from_training_rows(self):
        rows = deepcopy(self.dataset["stages"]["0"][:100])
        features = features_for(0)
        rows[0]["features"]["log_offer_price"] = None
        transform = Transform.fit(rows, features)
        self.assertTrue(np.isfinite(transform.apply(rows, embedding=True)).all())
        future = deepcopy(rows[:1])
        future[0]["features"]["adr"] = "new-category"
        future[0]["features"]["log_offer_price"] = 1e9
        x = transform.apply(future, embedding=True)
        self.assertEqual(x[0, 2 * len(transform.numeric)], 0)
        self.assertNotIn("new-category", transform.categories["adr"])
        column = [
            r["features"]["log_offer_price"]
            for r in rows
            if r["features"]["log_offer_price"] is not None
        ]
        self.assertAlmostEqual(transform.median[0], float(np.median(column)))

    def test_original_baseline_fits_match_on_the_same_expanded_rows(self):
        fold = temporal_folds(self.dataset["stages"]["20"], self.protocol)[0]
        fitted = fit_models(fold["train"])
        for name in ("logistic", "boosted_trees"):
            actual, _ = fit_baseline(fold["train"], fold["test"], 20, name)
            expected = fitted[name].predict_proba(matrix(fold["test"]))[:, 1]
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_bootstrap_preserves_paired_identity(self):
        rows = [
            {"as_of": day, "event": event, "probabilities": {"base_rate": 0.3, "logistic": 0.4}}
            for day in ("2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01")
            for event in (0, 1)
        ]
        result = paired_intervals(rows, "logistic", 100)
        self.assertEqual(result["brier_difference_vs_logistic_ci"], [0, 0])
        self.assertEqual(paired_intervals(rows, "base_rate", 100)["brier_skill_ci"], [0, 0])


if __name__ == "__main__":
    unittest.main()
