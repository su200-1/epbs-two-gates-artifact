"""Free-option economics.

Formulae from *The Free Option Problem of ePBS* (Mazorra et al., FC 2026), sec. 3:

    Pi_tau(y) = Pi_0(y) + r_tau * y          block profit at reveal time tau
    builder exercises (withholds)  iff  Pi_tau < w   (w = withhold_payoff)
    V*        = E[ max(w, Pi_tau) ]          value with the option
    P*        = Pr[ Pi_tau < w ]             exercise (empty-block) probability
    NetOption = E[ max(0, w - Pi_tau) ]      the option's extra value to builder

``withhold_payoff`` w parameterises "does withholding settle the bid?":
w = 0 under spec-style conditional payment; w = -execution_payment under the
Free Option paper's sunk-cost assumption. See `config.SimConfig`.

This layer is NOT covered by `consensus-specs` (see MODEL_SCOPE.md); it is
validated by reproducing the Free Option paper's qualitative results and by
analytical cross-checks, not by differential testing.
"""
from __future__ import annotations

import math

import numpy as np


def realized_profit(pi_0: float, y: float, r_tau: float) -> float:
    """Block profit at reveal time: ``Pi_tau = Pi_0 + r_tau * y``.

    ``pi_0`` is the block value at commit time (positive => profitable block);
    ``y`` is the CEX-DEX position size carried in the block; ``r_tau`` is the
    external cumulative return over the reveal window.
    """
    return pi_0 + r_tau * y


def builder_exercises(pi_tau: float, withhold_payoff: float = 0.0) -> bool:
    """Rational withhold rule: withhold iff revealing beats nothing.

    ``withhold_payoff`` is the builder's payoff when it withholds — ``0`` under
    spec-style conditional payment (the withheld bid never settles), or
    ``-execution_payment`` under the sunk-cost model (the Free Option paper's
    assumption). The builder withholds exactly when ``pi_tau < withhold_payoff``.
    """
    return pi_tau < withhold_payoff


def builder_payoff(pi_0: float, y: float, r_tau: float, *, has_option: bool,
                   withhold_payoff: float = 0.0) -> float:
    """Builder's realized payoff for one reveal-window outcome.

    With the option the builder floors its payoff at ``withhold_payoff`` by
    withholding; without it the builder must reveal and eats ``Pi_tau``.
    """
    pi_tau = realized_profit(pi_0, y, r_tau)
    return max(withhold_payoff, pi_tau) if has_option else pi_tau


def net_option_value(pi_0: float, y: float, r_tau: float,
                     withhold_payoff: float = 0.0) -> float:
    """The option's extra value for one outcome: ``max(0, withhold_payoff - Pi_tau)``.

    Generalises the Free Option paper's "net option value" (sec. 4.2): the gain
    from being free to take ``withhold_payoff`` instead of a bad ``Pi_tau``.
    """
    return max(0.0, withhold_payoff - realized_profit(pi_0, y, r_tau))


# --- Monte-Carlo aggregates -------------------------------------------------
def exercise_probability(pi_0: float, y: float, returns: np.ndarray,
                         withhold_payoff: float = 0.0) -> float:
    """``P*`` — fraction of return draws for which the builder withholds."""
    pi_tau = pi_0 + returns * y
    return float(np.mean(pi_tau < withhold_payoff))


def expected_net_option_value(pi_0: float, y: float, returns: np.ndarray,
                              withhold_payoff: float = 0.0) -> float:
    """``E[max(0, withhold_payoff - Pi_tau)]`` over Monte-Carlo return draws."""
    pi_tau = pi_0 + returns * y
    return float(np.mean(np.maximum(0.0, withhold_payoff - pi_tau)))


def option_value(pi_0: float, y: float, returns: np.ndarray,
                 withhold_payoff: float = 0.0) -> float:
    """``V*`` — expected payoff with the option, ``E[max(withhold_payoff, Pi_tau)]``."""
    pi_tau = pi_0 + returns * y
    return float(np.mean(np.maximum(withhold_payoff, pi_tau)))


# --- analytical reference (Gaussian r_tau) ---------------------------------
def exercise_probability_gaussian(pi_0: float, y: float, sigma: float,
                                  drift: float = 0.0,
                                  withhold_payoff: float = 0.0) -> float:
    """Closed-form ``P*`` when ``r_tau ~ Normal(drift, sigma**2)``.

    Used as an analytical cross-check against the Monte-Carlo estimate.
    """
    if y <= 0 or sigma <= 0:
        return 0.0
    # Pi_tau < withhold_payoff  <=>  r_tau < (withhold_payoff - pi_0) / y
    z = ((withhold_payoff - pi_0) / y - drift) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def optimal_position(sigma: float, liquidity: float, cex_dex_gap: float = 0.0) -> float:
    """Overshooting position ``y*`` — Free Option paper, Example 1:

        y* ~= 0.61 * sigma * L  +  0.8 * delta * L
    """
    return 0.61 * sigma * liquidity + 0.8 * cex_dex_gap * liquidity
