"""Report — and by default enforce — which consensus-specs commit the pyspec came from.

Every test module resolves the pyspec through

    os.environ.get("EPBS_PYSPEC_DIR", "../consensus-specs/tests/core/pyspec")

so a checkout sitting next to this repository is picked up silently. That is
convenient and it is a trap: run the suite against a commit other than the pinned
one and the constant assertions fail with, for instance,

    AssertionError: assert 5000 == 7500
      +  where 5000 = config.PAYLOAD_DUE_BPS

which says nothing about the real cause. `PAYLOAD_DUE_BPS` was 7500 at
`015d72704` and is 5000 at the pinned `5366cb59e`, so a stale sibling checkout
turns a green suite into two failures with no hint that the spec, not the port,
is the odd one out.

This reports the commit at session start and stops the run when it is not the
pin. Set `EPBS_ALLOW_UNPINNED=1` to proceed anyway -- the paper's spec-evolution
check deliberately re-runs this suite against an earlier HEAD.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

PINNED = "5366cb59e"
DEFAULT_PYSPEC_DIR = "../consensus-specs/tests/core/pyspec"


SPEC_MODULE_CANDIDATES = (
    "eth_consensus_specs.gloas.mainnet",
    "eth_consensus_specs.gloas.minimal",
)


def _pyspec_dir() -> Path | None:
    """Where the spec the tests will actually import lives.

    Three things can supply it and only one is a path we could guess:
    EPBS_PYSPEC_DIR, the sibling checkout, and -- most often, after `make
    _pyspec` -- an `eth_consensus_specs` already importable from the
    environment, which wins over both regardless of sys.path. So resolve it the
    way the tests resolve it, by importing, and fall back to the guessed path
    only when nothing imports.
    """
    guess = Path(os.environ.get("EPBS_PYSPEC_DIR", DEFAULT_PYSPEC_DIR))
    if guess.is_dir() and str(guess) not in sys.path:
        sys.path.insert(0, str(guess))
    for name in SPEC_MODULE_CANDIDATES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        f = getattr(mod, "__file__", None)
        if f:
            return Path(f).resolve().parent
    return guess.resolve() if guess.is_dir() else None


def _head(path: Path) -> str | None:
    """Commit of the repository containing `path`, or None if it is not in one."""
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def pytest_report_header(config) -> str:  # noqa: ARG001
    d = _pyspec_dir()
    if d is None:
        return "pyspec: not importable; spec-executing tests will skip"
    head = _head(d)
    if head is None:
        return f"pyspec: {d} (not a git checkout; commit unverified)"
    mark = "== pinned" if head.startswith(PINNED) else f"!= pinned {PINNED}"
    return f"pyspec: {d}\npyspec commit: {head[:9]} {mark}"


def pytest_sessionstart(session) -> None:
    if os.environ.get("EPBS_ALLOW_UNPINNED"):
        return
    d = _pyspec_dir()
    if d is None:
        return                      # nothing to check; the spec fixture skips
    head = _head(d)
    if head is None or head.startswith(PINNED):
        return                      # unversioned copy, or the pin: proceed
    pytest.exit(
        f"\n\nThe pyspec at {d}\nis generated from consensus-specs {head[:9]}, "
        f"not the pinned {PINNED}.\n"
        "Constants differ between commits -- PAYLOAD_DUE_BPS is 7500 at 015d72704 "
        "and 5000 at the pin --\nso this run would report failures that are the "
        "checkout's, not the port's.\n\n"
        "  cd consensus-specs && git checkout " + PINNED + " && make _pyspec\n"
        "  export EPBS_PYSPEC_DIR=$PWD/tests/core/pyspec\n\n"
        "To audit against a different commit on purpose -- as the paper's "
        "spec-evolution check does --\nset EPBS_ALLOW_UNPINNED=1.\n"
    )
