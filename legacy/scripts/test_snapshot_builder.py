"""
Test snapshot builder with a single symbol (CRWV).
"""
from datetime import date
from ipo_risk_engine.config.settings import load_settings
from ipo_risk_engine.snapshots.builder import build_snapshots_for_symbol


def main():
    settings = load_settings()

    symbol = "CRWV"
    ipo_date = date(2025, 3, 28)
    horizons = [7, 20]

    print(f"Building snapshots for {symbol} (IPO: {ipo_date})")
    print("=" * 60)

    snapshots = build_snapshots_for_symbol(symbol, ipo_date, "technology", horizons, settings)

    print(f"\nBuilt {len(snapshots)} snapshots")
    print("=" * 60)

    for snap in snapshots:
        print(f"\n--- {snap.street} (as_of: {snap.asof_date}) ---")
        print(f"  Features ({len(snap.features)}):")
        for k, v in snap.features.items():
            if isinstance(v, float):
                print(f"    {k:30}: {v:.6f}")
            else:
                print(f"    {k:30}: {v}")

        print(f"  Labels ({len(snap.labels)}):")
        for k, v in snap.labels.items():
            if v is not None:
                print(f"    {k:30}: {v:.6f}")
            else:
                print(f"    {k:30}: None")

    # Sanity checks
    print("\n" + "=" * 60)
    print("Sanity Checks:")

    if len(snapshots) >= 1:
        flop = snapshots[0]
        assert flop.street == "FLOP", f"First snapshot should be FLOP, got {flop.street}"
        assert len(flop.features) == 12, f"FLOP should have 12 features, got {len(flop.features)}"
        print(f"  FLOP features count = 12: PASS (7 instrument + 3 regime + 2 calendar)")

    if len(snapshots) >= 2:
        turn = snapshots[1]
        assert turn.street == "TURN", f"Second snapshot should be TURN, got {turn.street}"
        assert len(turn.features) == 21, f"TURN should have 21 features, got {len(turn.features)}"
        print(f"  TURN features count = 21: PASS (accumulated FLOP + TURN + regime)")

    if len(snapshots) >= 3:
        river = snapshots[2]
        assert river.street == "RIVER", f"Third snapshot should be RIVER, got {river.street}"
        assert len(river.features) == 33, f"RIVER should have 33 features, got {len(river.features)}"
        print(f"  RIVER features count = 33: PASS (accumulated all + regime)")

    # Check labels exist
    for snap in snapshots:
        assert "forward_mdd_7d" in snap.labels, f"Missing forward_mdd_7d in {snap.street}"
        assert "forward_mdd_20d" in snap.labels, f"Missing forward_mdd_20d in {snap.street}"
        assert "forward_mfr_7d" in snap.labels, f"Missing forward_mfr_7d in {snap.street}"
        assert "forward_mfr_20d" in snap.labels, f"Missing forward_mfr_20d in {snap.street}"
    print(f"  All labels present: PASS")

    # Check MDD <= 0 and MFR >= 0
    for snap in snapshots:
        for k, v in snap.labels.items():
            if v is not None:
                if "mdd" in k:
                    assert v <= 0, f"{snap.street} {k} should be <= 0, got {v}"
                if "mfr" in k:
                    assert v >= 0, f"{snap.street} {k} should be >= 0, got {v}"
    print(f"  MDD <= 0, MFR >= 0: PASS")

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
