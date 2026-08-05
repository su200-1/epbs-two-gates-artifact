"""External (CEX) price process sigma_t.

The free option's underlying is the external market price. Within the
commit->reveal window the CEX price moves; the cumulative return ``r_tau`` over
the window drives the builder's withhold/reveal decision.

This is NOT covered by `consensus-specs` — it is the economic layer the
simulation must supply itself (see MODEL_SCOPE.md). Initially a Gaussian
random walk; ``sigma`` is intended to be calibrated to Binance high-frequency
mid-prices later (cf. The Free Option Problem of ePBS, sec. 4).
"""
from __future__ import annotations

import numpy as np


class PriceProcess:
    """Discretised external price over the reveal window.

    ``sigma`` is the std of the *terminal* cumulative return over the whole
    reveal window; ``steps`` intermediate samples let the environment evaluate
    delayed-reveal decisions. Per-step increments are i.i.d. Gaussian with
    variance ``sigma**2 / steps``.
    """

    def __init__(self, sigma: float, drift: float = 0.0, steps: int = 8):
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.sigma = float(sigma)
        self.drift = float(drift)
        self.steps = int(steps)

    def sample_path(self, rng: np.random.Generator) -> np.ndarray:
        """Cumulative return at each of ``steps`` sample points in the window.

        Returns an array of length ``steps``; the last element is the terminal
        return ``r_tau``.
        """
        step_var = self.sigma ** 2 / self.steps
        step_drift = self.drift / self.steps
        increments = rng.normal(step_drift, np.sqrt(step_var), size=self.steps)
        return np.cumsum(increments)

    def sample_return(self, rng: np.random.Generator) -> float:
        """Terminal cumulative return ``r_tau`` over the full reveal window."""
        return float(self.sample_path(rng)[-1])

    def sample_returns(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` i.i.d. terminal returns — for Monte-Carlo option valuation."""
        step_var = self.sigma ** 2 / self.steps
        step_drift = self.drift / self.steps
        incr = rng.normal(step_drift, np.sqrt(step_var), size=(n, self.steps))
        return incr.sum(axis=1)
