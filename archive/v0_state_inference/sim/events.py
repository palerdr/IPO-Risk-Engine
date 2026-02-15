import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EventGeneratorConfig:
    base_rate : float
    mean_size : float
    initial_price : float
    buy_prob : float

@dataclass
class EventGenerator:
    """
    Emits observable events conditional on the latent state of the process
    """
    rng: np.random.Generator
    config : EventGeneratorConfig

    def __post_init__(self):
        self.current_price = self.config.initial_price

    def reset(self):
        self.current_price = self.config.initial_price
    
    def emit(self, volatility: float, liquidity: float, ts: datetime) -> list[dict]:
        n_events = self.rng.poisson(liquidity * self.config.base_rate)
        events = []
        for i in range(n_events):
            if self.rng.random() < self.config.buy_prob:
                side = "buy"
            else:
                side = "sell"
            size = self.rng.exponential(self.config.mean_size)
            price_delta = volatility * self.rng.standard_normal()
            self.current_price += price_delta
            events.append({
                "ts":ts,
                "event_type": "trade",
                "side":side,
                "size":size,
                "price":self.current_price,
                "source":"synthetic"
            })
        
        return events






