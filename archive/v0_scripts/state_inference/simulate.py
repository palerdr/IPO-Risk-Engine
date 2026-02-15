"""
Loads config, runs SyntheticEventSource.generate(), validates outputs, writes to parquet
"""
from pathlib import Path
from datetime import datetime
from ipo_risk_engine.sim.generator import SyntheticEventSource
from ipo_risk_engine.state_data.schema import validate_events, validate_labels



def main():
    config = {
        "seed":42,
        "events":{
            "base_rate": 5.0,
            "mean_size": 100.0,
            "initial_price": 100.0,
            "buy_prob": 0.5,
        },
        "regime":{
            "vol_mu": 0.05,
            "vol_theta": 0.1,
            "vol_sigma": 0.01,
            "vol_low": 0.01,
            "vol_high": 0.10,
            "liq_mu": 1.0,
            "liq_theta": 0.1,
            "liq_sigma": 0.1,
            "liq_low": 0.5,
            "liq_high": 2.0,
            "p_jump": 0.01,
        },
        "sim": {
            "start_time": datetime(2020, 1, 1, 0, 0, 0),
            "time_delta_ms": 100,
            "n_steps": 10000,
        },
    }

    source = SyntheticEventSource(config)
    events, labels = source.generate(config["sim"]["n_steps"])

    validate_labels(labels)
    validate_events(events)

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    events.collect().write_parquet("data/raw/events.parquet")
    labels.collect().write_parquet("data/raw/labels.parquet")
    #collect() materializes the lazyframe into a df before writing to parquet

    print(f"Events: {events.collect().height}, Labels: {labels.collect().height}")

if __name__ == "__main__":
    main()