"""Check the Python core without market-data credentials."""
import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from ipo_risk_engine.data.ingest import normalize_bars
from ipo_risk_engine.data.schemas import validate_bars
from ipo_risk_engine.data.store import read_parquet, write_parquet
from ipo_risk_engine.features.streets import compute_street_windows
from ipo_risk_engine.policy.actions import ActionThresholds, assign_actions, false_safe_rate
from ipo_risk_engine.snapshots.builder import build_snapshots_for_symbol


class CoreTests(unittest.TestCase):
    def test_existing_drawdown_fixture(self):
        from scripts.test_forward_mdd import test_synthetic
        test_synthetic()

    def test_normalization_and_parquet_roundtrip(self):
        start = datetime(2020, 1, 2, tzinfo=timezone.utc)
        raw = pl.DataFrame({
            "timestamp": [start + timedelta(days=1), start, start],
            "open": [11, 10, 10], "high": [12, 11, 12],
            "low": [10, 9, 9], "close": [11, 10, 11], "volume": [100, 200, 300],
        })
        bars = normalize_bars(raw, "DEMO")
        validate_bars(bars, "1d")
        self.assertEqual(bars.height, 2)
        self.assertEqual(bars["volume"][0], 300)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bars.parquet"
            write_parquet(bars, path)
            self.assertTrue(read_parquet(path).equals(bars))
        with self.assertRaises(ValueError):
            validate_bars(bars.drop("close"), "1d")

    def test_snapshot_builder_uses_daily_windows_and_forward_labels(self):
        start = datetime(2020, 1, 2, tzinfo=timezone.utc)
        days = [start + timedelta(days=n) for n in range(120)]
        sessions = [day for day in days if day.weekday() < 5][:81]
        close = 20 + np.sin(np.arange(81) / 3)
        bars = pl.DataFrame({"ts": sessions, "open": close, "close": close,
                             "high": close + 1, "low": close - 1,
                             "volume": np.full(81, 1000, dtype=np.int64)})
        windows = compute_street_windows(sessions[0].date(), [day.date() for day in sessions])
        self.assertEqual([w.street for w in windows], ["FLOP", "TURN", "RIVER"])
        with patch('ipo_risk_engine.snapshots.builder._ingest_daily_bars', return_value=bars), patch('ipo_risk_engine.snapshots.builder.read_parquet', return_value=bars):
            snapshots = build_snapshots_for_symbol("DEMO", sessions[0].date(), "unknown", [7, 20], None)
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(snapshots[-1].asof_date, sessions[60].date())
        self.assertIn('flop_realized_vol', snapshots[-1].features)
        self.assertIn('river_realized_vol', snapshots[-1].features)
        self.assertIsNotNone(snapshots[-1].labels['forward_mdd_20d'])

    def test_policy_boundaries_and_false_safe_denominator(self):
        actions = assign_actions(np.array([.29, .3, .59, .6]), ActionThresholds())
        self.assertEqual(actions, ['SIZE_UP', 'SMALL_BET', 'SMALL_BET', 'FOLD'])
        self.assertEqual(false_safe_rate(actions, np.array([1, 0, 0, 1])), .5)

    def test_synthetic_feature_model_calibration_policy_flow(self):
        from scripts.demo_risk_engine import run_demo
        result = run_demo()
        self.assertEqual(result['rows'], {'train': 48, 'validation': 16, 'test': 16})
        self.assertEqual(result['features'], 33)
        self.assertEqual(sum(result['actions'].values()), 16)
        self.assertGreaterEqual(result['example']['adverse_probability'], 0)
        self.assertLessEqual(result['example']['adverse_probability'], 1)
        self.assertIn(result['example']['action'], ['FOLD', 'SMALL_BET', 'SIZE_UP'])

    def test_python_files_compile(self):
        for directory in [Path('src'), Path('scripts')]:
            for filename in directory.rglob('*.py'):
                with self.subTest(path=str(filename)):
                    ast.parse(filename.read_text())


if __name__ == '__main__':
    unittest.main()
