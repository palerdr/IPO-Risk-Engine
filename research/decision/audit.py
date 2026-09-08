"""Reproduce the review of the frozen held-out forecasts."""
import json
from pathlib import Path

from ipo_research.expanded.decision_audit import audit as evaluate_audit

ROOT = Path(__file__).resolve().parents[2]


def audit():
    return evaluate_audit(ROOT / "research/expanded", ROOT / "research/decision/audit.json")


if __name__ == "__main__":
    result = audit()
    print(json.dumps({stage: {key: value for key, value in values.items() if key != "yearly"}
                      for stage, values in result["stages"].items()}, indent=2))
