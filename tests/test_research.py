"""Verify observation boundaries, chronological fitting, and report reproduction."""
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from ipo_research.dataset import FEATURES, build_dataset, drawdown, make_features, validate_bars
from ipo_research.evaluate import metrics, run_evaluation, screening, temporal_folds
from ipo_research.models import export_logistic, fit_models, matrix, predict_frozen

ROOT = Path(__file__).resolve().parents[1]


def bars(count=45):
    start = date(2024, 1, 2)
    dates = [start + timedelta(days=i) for i in range(count * 2)]
    return [{"date": day.isoformat(), "adjusted_close": 100 + index, "volume": 1000 + index}
            for index, day in enumerate(day for day in dates if day.weekday() < 5)][:count]


def listing_bundle():
    series = bars()
    return {"exclusions": [], "benchmark": {"bars": deepcopy(series)}, "listings": [{
        "symbol": "TEST", "sector": "test", "listing_date": series[0]["date"],
        "source_page": "https://example.test/prices", "retrieved_at": "2026-09-07T00:00:00Z",
        "bars": series,
    }]}


def sample_rows(count=90):
    rng = np.random.default_rng(15)
    start = date(2020, 1, 1)
    return [{"symbol": f"CASE{i}", "as_of": (start + timedelta(days=i * 3)).isoformat(),
             "label_end": (start + timedelta(days=i * 3 + 28)).isoformat(),
             "features": dict(zip(FEATURES, rng.normal(size=len(FEATURES)).tolist())),
             "event": i % 3 == 0} for i in range(count)]


class DatasetTests(unittest.TestCase):
    def test_drawdown_includes_the_prediction_close_and_respects_peak_order(self):
        self.assertAlmostEqual(drawdown(np.array([100, 80, 90])), .2)
        self.assertAlmostEqual(drawdown(np.array([100, 120, 90])), .25)
        self.assertEqual(drawdown(np.array([100, 110, 120])), 0)

    def test_snapshot_uses_twenty_observations_and_twenty_future_sessions(self):
        bundle = listing_bundle()
        bundle['listings'][0]['bars'][20]['adjusted_close'] = 59.5
        rows, excluded = build_dataset(bundle)
        self.assertFalse(excluded)
        self.assertEqual(rows[0]['as_of'], bundle['listings'][0]['bars'][19]['date'])
        self.assertEqual(rows[0]['label_end'], bundle['listings'][0]['bars'][39]['date'])
        self.assertEqual(rows[0]['event'], 1)
        self.assertAlmostEqual(rows[0]['drawdown'], .5)
        self.assertEqual(len(rows[0]['path']), 40)

    def test_future_prices_and_future_benchmark_values_do_not_enter_features(self):
        bundle = listing_bundle()
        before, _ = build_dataset(bundle)
        changed = deepcopy(bundle)
        for row in changed['listings'][0]['bars'][20:]:
            row['adjusted_close'] *= .1
        for row in changed['benchmark']['bars'][20:]:
            row['adjusted_close'] *= 15
        after, _ = build_dataset(changed)
        self.assertEqual(before[0]['features'], after[0]['features'])
        self.assertNotEqual(before[0]['event'], after[0]['event'])
        observed = bundle['listings'][0]['bars'][:20]
        changed['benchmark']['bars'][25]['adjusted_close'] = None
        self.assertEqual(make_features(observed, bundle['benchmark']['bars']),
                         make_features(observed, changed['benchmark']['bars']))

    def test_invalid_history_is_excluded_with_a_reason(self):
        for modification in ('missing', 'duplicate', 'zero', 'short'):
            bundle = listing_bundle()
            series = bundle['listings'][0]['bars']
            if modification == 'missing':
                del series[5]
            elif modification == 'duplicate':
                series[5]['date'] = series[4]['date']
            elif modification == 'zero':
                series[5]['adjusted_close'] = 0
            else:
                del series[35:]
            with self.subTest(modification=modification):
                rows, errors = build_dataset(bundle)
                self.assertFalse(rows)
                self.assertEqual(errors[0]['symbol'], 'TEST')
                self.assertTrue(errors[0]['reason'])

    def test_features_are_invariant_to_a_uniform_price_adjustment(self):
        observed, benchmark = bars(20), bars(20)
        before = make_features(observed, benchmark)
        changed = deepcopy(observed)
        for row in changed:
            row['adjusted_close'] *= 3
        after = make_features(changed, benchmark)
        np.testing.assert_allclose(list(before.values()), list(after.values()), atol=1e-12)


class EvaluationTests(unittest.TestCase):
    def test_split_purges_unmatured_labels_and_never_trains_on_test_issuers(self):
        folds = temporal_folds(sample_rows())
        self.assertEqual(len(folds), 3)
        tested = []
        for fold in folds:
            self.assertGreater(fold['purged'], 0)
            self.assertLess(max(row['label_end'] for row in fold['train']), fold['start'])
            self.assertTrue(set(row['symbol'] for row in fold['train']).isdisjoint(row['symbol'] for row in fold['test']))
            tested.extend(row['symbol'] for row in fold['test'])
        self.assertEqual(len(tested), len(set(tested)))

    def test_equal_dates_stay_in_one_test_block_and_duplicate_issuers_fail(self):
        rows = sample_rows()
        rows[48]['as_of'] = rows[47]['as_of']
        assignments = {row['symbol']: fold['id'] for fold in temporal_folds(rows) for row in fold['test']}
        self.assertEqual(assignments['CASE47'], assignments['CASE48'])
        with self.assertRaises(ValueError):
            temporal_folds(rows + [rows[0]])

    def test_scaler_fits_training_data_and_frozen_model_reproduces_predictions(self):
        train, test = sample_rows()[:40], sample_rows()[70:]
        fitted = fit_models(train)['logistic']
        np.testing.assert_allclose(fitted.steps[0][1].mean_, matrix(train).mean(axis=0))
        frozen = export_logistic(fitted, train)
        predicted, drivers = predict_frozen(frozen, test[0]['features'], test[0]['as_of'])
        self.assertAlmostEqual(predicted, fitted.predict_proba(matrix(test))[0, 1], places=12)
        self.assertEqual(len(drivers), len(FEATURES))
        with self.assertRaises(ValueError):
            predict_frozen(frozen, test[0]['features'], frozen['trained_through'])

    def test_screening_uses_explicit_denominators_and_no_flags_is_undefined(self):
        y, p = np.array([1, 0, 1, 0]), np.array([.7, .8, .2, .1])
        severe = np.array([True, False, True, False])
        result = screening(y, p, severe, .5)
        self.assertEqual(result['precision'], .5)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['missed_severe_rate'], .5)
        self.assertIsNone(screening(y, p, severe, .9)['precision'])

    def test_brier_uses_each_folds_training_rate_baseline(self):
        predictions = [{"event": event, "drawdown": event * .4, "as_of": '2025-01-01',
                        "probabilities": {"base_rate": base, "logistic": p}}
                       for event, base, p in [(1, .2, .8), (0, .4, .1)]]
        result = metrics(predictions, 'logistic', intervals=False)
        self.assertAlmostEqual(result['brier'], .025)
        self.assertAlmostEqual(result['brier_skill'], 1 - .025 / .4)

    def test_single_class_training_fails_without_fabricating_a_score(self):
        rows = sample_rows(25)
        for row in rows:
            row['event'] = 0
        with self.assertRaises(ValueError):
            fit_models(rows)


class FrozenStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_path = ROOT / 'research/input.json'
        cls.report = json.loads((ROOT / 'web/data/research.json').read_text())

    def test_report_matches_a_fresh_run_from_the_frozen_real_data(self):
        with tempfile.TemporaryDirectory() as directory:
            actual = run_evaluation(self.input_path, Path(directory) / 'report.json')
        self.assertEqual(actual, self.report)
        self.assertFalse(actual['synthetic'])
        self.assertEqual(len(actual['predictions']), actual['cohort']['evaluated'])
        self.assertEqual(actual['provenance']['input_sha256'], hashlib.sha256(self.input_path.read_bytes()).hexdigest())

    def test_saved_fold_models_reproduce_replay_probabilities_and_precede_observations(self):
        models = {row['fold']: row for row in self.report['frozen_logistic_models']}
        for row in self.report['predictions']:
            model = models[row['fold']]
            self.assertNotIn(row['symbol'], model['train_symbols'])
            probability, _ = predict_frozen(model, row['features'], row['as_of'])
            self.assertAlmostEqual(probability, row['probabilities']['logistic'], places=12)


if __name__ == '__main__':
    unittest.main()
