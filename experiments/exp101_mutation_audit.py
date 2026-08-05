"""Experiment 101 -- mutation audit of the spec-faithfulness test suite.

Why this exists
---------------
A passing differential suite shows the port agrees with the spec on the inputs
the tests happen to exercise. It does not show the tests would *notice* if the
port drifted. Reviewers are right to ask for that separately: the load-bearing
claims of this paper are exact boundary claims (257, not 256; strict `>`, not
`>=`; silence counted as neither side), so the tests must fail when precisely
those details are broken.

This script applies single-token mutations to the ported predicates, re-runs the
spec-relevant differential tests, and records whether each mutant is killed
(at least one test fails). A surviving mutant is a hole in the suite.

Mutations target exactly the properties the paper asserts:
  M1  threshold comparison  >  ->  >=      (off-by-one: pivotal 257 -> 256)
  M2  threshold constant    256 -> 257     (shifts the boundary the other way)
  M3  vote equality  (vote == timely) -> (vote is not (not timely))
                                           (makes silent None count as a vote)
  M4  local short-circuit   return not timely -> return timely
                                           (flips fail-closed unavailability)
  M5  should_build_on_full fail-open `return True` -> `return False`
                                           (flips the default direction)

Run: python experiments/exp101_mutation_audit.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "epbs" / "primitives.py"
OUT = REPO / "experiments" / "figures" / "drl_risk_epbs"
PYTEST_TARGETS = [
    "difftest/test_primitives.py",
    "difftest/test_forkchoice.py",
    "difftest/test_attestation.py",
    "difftest/test_builder_payments.py",
    "difftest/test_committee.py",
]

MUTANTS: list[tuple[str, str, str, str]] = [
    (
        "M1",
        "threshold comparison > -> >= (pivotal 257 -> 256)",
        "return sum(vote == timely for vote in votes) > config.PAYLOAD_TIMELY_THRESHOLD",
        "return sum(vote == timely for vote in votes) >= config.PAYLOAD_TIMELY_THRESHOLD",
    ),
    (
        "M2",
        "threshold constant off-by-one (+1)",
        "return sum(vote == timely for vote in votes) > config.PAYLOAD_TIMELY_THRESHOLD",
        "return sum(vote == timely for vote in votes) > config.PAYLOAD_TIMELY_THRESHOLD + 1",
    ),
    (
        "M3",
        "silence semantics: None counted as a vote",
        "return sum(vote == timely for vote in votes) > config.PAYLOAD_TIMELY_THRESHOLD",
        "return sum(vote is not (not timely) for vote in votes) > config.PAYLOAD_TIMELY_THRESHOLD",
    ),
    (
        "M4",
        "local short-circuit direction flipped",
        "    if not is_payload_verified(store, root):\n        return not timely",
        "    if not is_payload_verified(store, root):\n        return timely",
    ),
]


def run_tests(cwd: Path) -> tuple[bool, str]:
    """Return (all_passed, tail_of_output)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *PYTEST_TARGETS, "-q", "--no-header", "-x"],
        cwd=cwd, capture_output=True, text=True,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-400:]


def main() -> None:
    src = TARGET.read_text()

    print("=== baseline (unmutated) ===", flush=True)
    ok, tail = run_tests(REPO)
    print(f"baseline passes: {ok}")
    if not ok:
        print("!! baseline suite is not green; mutation scores would be meaningless.")
        print(tail)
        return

    results = []
    for mid, desc, old, new in MUTANTS:
        if old not in src:
            results.append({"id": mid, "description": desc,
                            "applied": False, "killed": None,
                            "note": "pattern not found; port may have been edited"})
            print(f"{mid}: PATTERN NOT FOUND -- skipped")
            continue
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / "repo"
            shutil.copytree(REPO, work, ignore=shutil.ignore_patterns(
                "__pycache__", ".git", "experiments/data", "*.parquet", "figures"))
            tgt = work / "epbs" / "primitives.py"
            tgt.write_text(src.replace(old, new, 1))
            passed, tail = run_tests(work)
        killed = not passed
        results.append({"id": mid, "description": desc, "applied": True,
                        "killed": killed})
        print(f"{mid}: {'KILLED' if killed else 'SURVIVED  <-- gap'}  ({desc})")

    applied = [r for r in results if r.get("applied")]
    killed = [r for r in applied if r["killed"]]
    score = len(killed) / len(applied) if applied else float("nan")
    summary = {
        "schema": "mutation-audit-v1",
        "target": str(TARGET.relative_to(REPO)),
        "tests": PYTEST_TARGETS,
        "n_mutants": len(applied),
        "n_killed": len(killed),
        "mutation_score": score,
        "results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp101_mutation_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nmutation score: {len(killed)}/{len(applied)} = {score:.0%}")


if __name__ == "__main__":
    main()
