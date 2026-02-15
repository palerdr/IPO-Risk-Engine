import numpy as np
from dataclasses import dataclass

@dataclass
class RegimeParams:
    """
    Stores Regime Parameters
    """
    vol_mu: float
    vol_theta: float
    vol_sigma: float
    vol_low: float
    vol_high: float

    liq_mu: float
    liq_theta: float
    liq_sigma: float
    liq_low: float
    liq_high: float

    p_jump: float

    def __post_init__(self):
        assert self.vol_low < self.vol_high, "vol_low must be < vol_high"
        assert self.liq_low < self.liq_high, "liq_low must be < liq_high"
        assert 0.0 <= self.p_jump <= 1.0, "p_jump must be a probability"

@dataclass
class LatentRegimeProcess:
    """
    Generates a time series of continuous latent variables (volatility, liquidity) that evolve smoothly with occasional regime shifts
    """
    rng : np.random.Generator
    parameters : RegimeParams

    #discrete OU x[t+1] = x[t] + 0(mu - x[t]) + Z
    def generate(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        params: RegimeParams = self.parameters
        rng: np.random.Generator = self.rng

        vol,liq = np.empty(n_steps), np.empty(n_steps)
        vol[0] = params.vol_mu
        liq[0] = params.liq_mu
        
        for t in range(1, n_steps):
            jump_occurs = rng.random() < params.p_jump
            if jump_occurs:
                vol[t] = rng.uniform(params.vol_low, params.vol_high)
                liq[t] = rng.uniform(params.liq_low, params.liq_high)
            else:
                vol[t] = vol[t-1] + params.vol_theta*(params.vol_mu - vol[t-1]) + params.vol_sigma*rng.standard_normal()
                liq[t] = liq[t-1] + params.liq_theta*(params.liq_mu - liq[t-1]) + params.liq_sigma*rng.standard_normal()
            
            vol[t] =  np.clip(vol[t], params.vol_low, params.vol_high)
            liq[t] = np.clip(liq[t], params.liq_low, params.liq_high)

        return (vol, liq)


