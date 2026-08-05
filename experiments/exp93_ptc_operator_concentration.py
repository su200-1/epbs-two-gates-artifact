"""PTC seat/operator concentration and recruitment-threshold sensitivity.

The experiment is synthetic because no validator-to-operator mapping is bundled
with the artifact.  It treats the 512 PTC seats as independent stake-weighted
draws and reports how many additional seats and distinct honest operators must
be coordinated to reach the exact 257-seat predicate.

Run: python experiments/exp93_ptc_operator_concentration.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
PTC_SIZE = 512
PIVOTAL = 257
N_COMMITTEES = 100_000
SEED = 20260722
OWNED_FRACTIONS = (0.0, 0.10, 0.20, 1 / 3)


def operator_profiles() -> dict[str, np.ndarray | None]:
    zipf = 1 / np.arange(1, 129, dtype=float)
    zipf /= zipf.sum()
    top = np.array([0.25, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03])
    stylized = np.r_[top, np.repeat((1 - top.sum()) / 24, 24)]
    return {
        # Analytical upper bound: each non-owned seat belongs to a different party.
        "distinct-seat upper bound": None,
        "Zipf(128, alpha=1.0)": zipf,
        "stylized concentrated (32 operators)": stylized,
    }


def summarize(x: np.ndarray) -> dict:
    return {
        "mean": float(x.mean()),
        "p05": float(np.quantile(x, 0.05)),
        "median": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
    }


def simulate_profile(
    rng: np.random.Generator, profile: np.ndarray | None, owned_fraction: float,
) -> dict:
    if profile is None:
        owned = rng.binomial(PTC_SIZE, owned_fraction, size=N_COMMITTEES)
        marginal = np.maximum(PIVOTAL - owned, 0)
        operators = marginal.copy()
        contracted_seats = marginal.copy()
    else:
        probs = np.r_[owned_fraction, (1 - owned_fraction) * profile]
        counts = rng.multinomial(PTC_SIZE, probs, size=N_COMMITTEES)
        owned = counts[:, 0]
        marginal = np.maximum(PIVOTAL - owned, 0)
        honest_counts = np.sort(counts[:, 1:], axis=1)[:, ::-1]
        cumulative = np.cumsum(honest_counts, axis=1)
        # Number of largest operators whose assigned seats cover the marginal need.
        operators = np.where(
            marginal == 0, 0, (cumulative < marginal[:, None]).sum(axis=1) + 1
        )
        idx = np.maximum(operators - 1, 0)
        contracted_seats = np.where(
            operators == 0, 0, cumulative[np.arange(N_COMMITTEES), idx]
        )

    return {
        "owned_fraction": owned_fraction,
        "probability_owned_alone_reaches_257": float((owned >= PIVOTAL).mean()),
        "owned_seats": summarize(owned),
        "additional_pivotal_seats": summarize(marginal),
        "distinct_operators_to_coordinate": summarize(operators),
        "seats_controlled_after_whole_operator_contracts": summarize(contracted_seats),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for name, profile in operator_profiles().items():
        for fraction in OWNED_FRACTIONS:
            row = simulate_profile(rng, profile, fraction)
            row["operator_profile"] = name
            rows.append(row)

    result = {
        "schema": "ptc-operator-concentration-v1",
        "ptc_size": PTC_SIZE,
        "pivotal_seats": PIVOTAL,
        "committees_per_cell": N_COMMITTEES,
        "seed": SEED,
        "sampling_model": "iid stake-weighted seats with replacement",
        "scope": (
            "Synthetic sensitivity, not an empirical mainnet operator mapping or "
            "evidence that any operator would accept a bribe."
        ),
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "exp93_ptc_operator_concentration.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    lines = [
        "# PTC operator-concentration sensitivity\n\n",
        "100,000 synthetic 512-seat committees per cell. Values are medians "
        "with 5--95% intervals. This is not a mainnet operator census.\n\n",
        "| Operator profile | Owned stake | Owned seats | Additional seats | Distinct operators |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for r in rows:
        o, a, c = r["owned_seats"], r["additional_pivotal_seats"], r["distinct_operators_to_coordinate"]
        lines.append(
            f"| {r['operator_profile']} | {100*r['owned_fraction']:.1f}% | "
            f"{o['median']:.0f} [{o['p05']:.0f}, {o['p95']:.0f}] | "
            f"{a['median']:.0f} [{a['p05']:.0f}, {a['p95']:.0f}] | "
            f"{c['median']:.0f} [{c['p05']:.0f}, {c['p95']:.0f}] |\n"
        )
    (OUT_DIR / "exp93_ptc_operator_concentration.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
