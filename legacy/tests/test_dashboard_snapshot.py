"""Check the dashboard's saved synthetic run against the Python engine."""
import json
from pathlib import Path
import unittest

from scripts.demo_risk_engine import run_demo


class DashboardSnapshotTest(unittest.TestCase):
    def test_saved_walkthrough_matches_the_engine(self):
        path = Path(__file__).resolve().parents[1] / 'web/data/risk-demo.json'
        self.assertEqual(json.loads(path.read_text()), run_demo())
