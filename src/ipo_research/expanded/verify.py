"""Audit frozen inputs and reproduce held-out metrics from saved models."""
import json
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs
from zoneinfo import ZoneInfo

import numpy as np

from .dataset import snapshots
from .evaluate import stats, temporal_folds
from .models import predict_saved
from .sources import load_json, sha, save_json


def verify(directory: Path, artifact_dir: Path, *, sensitivity=False) -> dict:
    result_path = directory / ("coverage-results.json.gz" if sensitivity else "results.json.gz")
    protocol_path = directory / ("coverage-protocol.json" if sensitivity else "protocol.json")
    if sensitivity:
        artifact_dir = artifact_dir / "coverage-2018"
    result = load_json(result_path)
    data = load_json(directory / "dataset.json.gz")
    bundle = load_json(directory / "input.json.gz")
    universe = load_json(directory / "universe.json.gz")
    protocol = load_json(protocol_path)
    assert result["provenance"]["dataset_sha256"] == sha(directory / "dataset.json.gz")
    assert result["provenance"]["protocol_sha256"] == sha(protocol_path)
    assert data["input_sha256"] == sha(directory / "input.json.gz")
    assert data["universe_sha256"] == bundle["universe_sha256"] == sha(directory / "universe.json.gz")
    source_root = Path(__file__).parent.parent
    assert result["provenance"]["evaluation_source_sha256"] == sha(Path(__file__).with_name("evaluate.py"))
    for name, expected in result["provenance"]["training_source_sha256"].items():
        assert sha(source_root / name) == expected, name
    source_checks = 0
    for source in universe["sources"]:
        file = directory / "raw" / source["file"]
        if file.exists():
            assert sha(file) == source["sha256"], file
            source_checks += 1
    price_checks = 0
    for history in [bundle["benchmark"]] + [row["history"] for row in bundle["listings"]]:
        query = parse_qs(urlparse(history["source_url"]).query)
        start, end = [datetime.fromtimestamp(int(query[key][0]), timezone.utc).date().isoformat()
                      for key in ("period1", "period2")]
        file = directory / "raw" / f"{history['symbol']}-{start}-{end}.response.json"
        if not file.exists():
            continue
        assert sha(file) == history["raw_sha256"], file
        chart = json.loads(file.read_text())["chart"]["result"][0]
        zone = ZoneInfo(chart["meta"]["exchangeTimezoneName"])
        actual = [{"date": datetime.fromtimestamp(timestamp, zone).date().isoformat(),
                   "adjusted_close": chart["indicators"]["adjclose"][0]["adjclose"][index],
                   "volume": chart["indicators"]["quote"][0]["volume"][index]}
                  for index, timestamp in enumerate(chart["timestamp"])]
        assert actual == history["bars"], file
        price_checks += 1
    verified = 0
    for stage, report in result["stages"].items():
        rows, excluded = snapshots(bundle, universe, int(stage))
        assert rows == data["stages"][stage]
        assert excluded == data["stage_exclusions"][stage]
        assert len({r["id"] for r in rows}) == len(rows)
        assert len({r["issuer_id"] for r in rows}) == len(rows)
        for row in rows:
            assert row["pre_market_cutoff"] < row["listing_date"]
            assert row["label_end"] > row["as_of"]
            assert row["event"] == int(row["drawdown"] >= .2)
        predicted = {r["id"]: r for r in report["predictions"]}
        for fold in temporal_folds(rows, protocol):
            train, test = fold["train"], fold["test"]
            names = report["models"].keys() - {"base_rate"}
            base = np.mean([r["event"] for r in train])
            for row in test:
                assert predicted[row["id"]]["probabilities"]["base_rate"] == base
            for name in names:
                target = artifact_dir / f"stage-{stage}" / f"fold-{fold['id']}" / name
                manifest = json.loads((target / "manifest.json").read_text())
                assert manifest["train_ids"] == [r["id"] for r in train]
                assert manifest["trained_through"] == max(r["label_end"] for r in train)
                if manifest.get("inner_split"):
                    split = manifest["inner_split"]
                    assert split["fit_label_end"] < split["start"]
                    assert set(split["fit_ids"]).isdisjoint(split["validation_ids"])
                    assert set(split["fit_ids"] + split["validation_ids"]).issubset(manifest["train_ids"])
                expected = [predicted[r["id"]]["probabilities"][name] for r in test]
                np.testing.assert_allclose(predict_saved(target, test), expected, atol=1e-7)
                verified += 1
            stored = report["folds"][fold["id"] - 1]
            for name, metric in stored["models"].items():
                assert stats([predicted[r["id"]] for r in test], name) == metric
        for name, metric in report["models"].items():
            recomputed = stats(report["predictions"], name)
            assert recomputed == {key: metric[key] for key in recomputed}
        assert len(predicted) == len(report["predictions"])
    receipt = {"status": "passed", "saved_model_groups_verified": verified,
               "source_files_hash_checked": source_checks,
               "source_files_expected": len(universe["sources"]),
               "price_responses_reconciled": price_checks,
               "price_responses_expected": len(bundle["listings"]) + 1,
               "dataset_rebuilt": True, "stage_metrics_recomputed": True,
               "results_sha256": sha(result_path)}
    save_json(directory / ("coverage-verification.json" if sensitivity else "verification.json"), receipt)
    print(json.dumps(receipt, indent=2))
    return receipt
