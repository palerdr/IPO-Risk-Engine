import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import polars as pl
from ipo_risk_engine.sim.regime import LatentRegimeProcess, RegimeParams
from ipo_risk_engine.sim.events import EventGeneratorConfig, EventGenerator


class SyntheticEventSource:
    """
    Generates a timeline of timestemps, generating latent trajectory using LatentRegimeProcess
    For every timestep calls EventGenerator.emit() to produce an event
    Computes regime_id by tercile volatility binning
    """
    def __init__(self, config: dict):
        self.config = config
        self.rng = np.random.Generator(np.random.PCG64(config["seed"]))

        self.regime_params = RegimeParams(**self.config["regime"])
        self.generator_config = EventGeneratorConfig(**self.config["events"])
        
        self.regime_process = LatentRegimeProcess(self.rng, self.regime_params)
        self.event_generator = EventGenerator(self.rng, self.generator_config)
        
    def generate(self, n_steps: int) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        
        start_time = self.config["sim"]["start_time"]  # a datetime object
        delta = timedelta(milliseconds=self.config["sim"]["time_delta_ms"])  # e.g., 100ms
        ts_array = [start_time + i * delta for i in range(n_steps)]

        vol_array, liq_array = self.regime_process.generate(n_steps)

        p33, p67 = np.percentile(vol_array, [33.33, 66.67])
        all_events = []
        labels = []
        for i in range(n_steps):
            vol = vol_array[i]
            liq = liq_array[i]
            ts = ts_array[i]
            events = self.event_generator.emit(vol, liq, ts)
            all_events.extend(events)
            
            if vol <= p33:
                regime_id = 0
            elif vol <= p67:
                regime_id = 1
            else:
                regime_id = 2

            labels.append({
                "ts": ts,
                "regime_id": regime_id,
                "vol": vol,
                "liq": liq,
            })
        
        events_lf = pl.LazyFrame(all_events)
        labels_lf = pl.LazyFrame(labels)

        return (events_lf, labels_lf)

