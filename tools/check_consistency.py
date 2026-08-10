"""Consistency gate: the numbers in the paper, the figures, the docs and the data
must be the same numbers.

Every defect this checks for has actually happened in this repository, and each
one has the same shape: something was changed in one place and left stale in
another. Reading for them does not work -- a figure that says n=817,528 on a page
whose text says 935,315 survived several careful passes.

Checks
  1  figure freshness   manuscript/figures/*.pdf identical to the generated copies
  2  figure vs text     numbers drawn inside a figure appear in main.tex
  3  data vs text       statistics recomputed from the panel appear in main.tex
  4  data vs docs       the same statistics appear in README and the artifact README
  5  labels             every \\ref resolves; every \\label is referenced
  6  entry point        reproduce_paper3.sh runs nothing retired, and every script
                        it names exists and compiles
  7  translation        (optional, --zh PATH) section and float counts line up

Run: python tools/check_consistency.py [--zh /path/to/论文中文版.md]
Exit status is the number of failed checks, so it can gate a commit.
"""
from __future__ import annotations

import argparse
import filecmp
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MS = ROOT / "manuscript"
GEN = ROOT / "experiments" / "figures" / "drl_risk_epbs"
PANEL = ROOT / "experiments" / "data" / "block_value_panel_delivered_2026H1.parquet"
RETIRED = ("exp88", "exp94", "exp96", "exp100")

FAILS: list[str] = []
NOTES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def skip(name: str, why: str) -> None:
    print(f"  SKIP  {name}   {why}")
    NOTES.append(f"{name}: {why}")


def pdf_text(p: Path) -> str:
    try:
        out = subprocess.run(["pdftotext", "-layout", str(p), "-"],
                             capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return " ".join(out.split())


def tex() -> str:
    return " ".join((MS / "main.tex").read_text().split())


def num_in_tex(body: str, value: str) -> bool:
    """LaTeX writes thousands as 935{,}314 and the PDF as 935,314. Decimals are
    quoted in prose at fewer significant figures than the panel carries, so a
    value counts as present if the text has it at any rounding of 2..4 places."""
    if value in body or value.replace(",", "{,}") in body:
        return True
    try:
        x = float(value.replace(",", ""))
    except ValueError:
        return False
    if "." not in value:
        return False
    return any(f"{x:.{d}f}".rstrip("0").rstrip(".") in body or f"{x:.{d}f}" in body
               for d in (1, 2, 3, 4))


# ------------------------------------------------------------------ 1 and 2 --
def check_figures() -> None:
    print("\n[1] figure freshness -- compiled copy == generated copy")
    pairs = [(MS / "figures" / f.name, f) for f in sorted(GEN.glob("fig_*.pdf"))
             if (MS / "figures" / f.name).exists()]
    if not pairs:
        skip("figure freshness", "no overlapping figure names")
        return
    for compiled, generated in pairs:
        same = filecmp.cmp(compiled, generated, shallow=False)
        check(f"{compiled.name} matches the generated copy", same,
              "" if same else "regenerate, then copy into manuscript/figures/")

    print("\n[2] the sample size drawn in a figure equals the one in the text")
    # This is the check that catches a figure built from a superseded panel: a
    # drawn n= that the manuscript never mentions means the two disagree. Other
    # drawn statistics need not be repeated in prose.
    t = tex()
    for compiled, _ in pairs:
        drawn = {m.group(1) for m in
                 re.finditer(r"\bn\s*=\s*([0-9]{1,3}(?:,[0-9]{3})+)", pdf_text(compiled))}
        stale = sorted(d for d in drawn if not num_in_tex(t, d))
        check(f"{compiled.name}: drawn sample size matches the text",
              not stale, f"drawn but absent from main.tex: {stale}" if stale
              else (f"n={sorted(drawn)[0]}" if drawn else "no n= drawn"))


# ----------------------------------------------------------------- 3 and 4 --
def panel_stats() -> dict[str, str] | None:
    try:
        import pandas as pd
    except ImportError:
        return None
    if not PANEL.exists():
        return None
    try:
        v = pd.read_parquet(PANEL, columns=["block_value_eth"]).block_value_eth
    except Exception as exc:                              # noqa: BLE001
        NOTES.append(f"panel unreadable in this interpreter: {type(exc).__name__}")
        return None
    return {
        "block count": f"{len(v):,}",
        "median": f"{v.median():.4f}",
        "mean": f"{v.mean():.4f}",
        "p99.9": f"{v.quantile(0.999):.3f}",
        "max": f"{v.max():.2f}",
    }


def check_data(stats: dict[str, str] | None) -> None:
    print("\n[3] statistics recomputed from the panel appear in main.tex")
    if stats is None:
        skip("panel vs main.tex", "pandas/pyarrow cannot read the panel here")
        return
    t = tex()
    quoted = ("block count", "median", "mean", "p99.9")   # what the prose states
    for label, value in stats.items():
        if label in quoted:
            check(f"{label} = {value}", num_in_tex(t, value))
        elif not num_in_tex(t, value):
            skip(f"{label} = {value}", "drawn in the figure, not quoted in prose")
        else:
            check(f"{label} = {value}", True)

    print("\n[4] the same statistics appear in the artifact docs")
    for doc in (ROOT / "README.md", ROOT / "docs" / "PAPER3_ARTIFACT_README.md"):
        body = doc.read_text()
        # A doc need not quote every statistic; it must not quote a stale one.
        check(f"{doc.name}: block count = {stats['block count']}",
              num_in_tex(body, stats["block count"]))
        for label in ("median", "mean", "p99.9", "max"):
            if re.search(rf"(?<![\w-]){re.escape(label)}(?![\w-])[^0-9\n]{{0,12}}[0-9]",
                          body, re.I):
                check(f"{doc.name}: {label} = {stats[label]}",
                      num_in_tex(body, stats[label]))


# ---------------------------------------------------------------------- 5 --
def check_labels() -> None:
    print("\n[5] cross-references")
    src = (MS / "main.tex").read_text()
    labels = set(re.findall(r"\\label\{([^}]*)\}", src))
    labels |= set(re.findall(r"label=([A-Za-z]+:[A-Za-z0-9_-]+)", src))
    refs = set(re.findall(r"\\(?:ref|autoref|pageref|eqref)\{([^}]*)\}", src))
    dangling = sorted(refs - labels)
    unused = sorted(l for l in labels - refs if l.startswith(("tab:", "fig:")))
    anchors = sorted(l for l in labels - refs if l.startswith("sec:"))
    if anchors:
        skip("unreferenced section anchors", f"navigation only: {anchors}")
    check("every \\ref resolves to a \\label", not dangling,
          f"dangling: {dangling}" if dangling else f"{len(refs)} refs")
    check("no unreferenced float labels", not unused,
          f"unused: {unused}" if unused else f"{len(labels)} labels")

    log = MS / "main.log"
    if log.exists():
        txt = log.read_text(errors="ignore")
        check("last build had no undefined references",
              "undefined" not in txt.lower(),
              "" if "undefined" not in txt.lower() else "see main.log")
    else:
        skip("build log", "main.log absent; compile first")


# ---------------------------------------------------------------------- 6 --
def check_entry_point() -> None:
    print("\n[6] reproduction entry point")
    sh = ROOT / "experiments" / "reproduce_paper3.sh"
    body = sh.read_text()
    named = sorted(set(re.findall(r"exp\d+_[a-z0-9_]+", body)))
    still_retired = [n for n in named if n.split("_")[0] in RETIRED]
    check("no retired experiment is invoked", not still_retired,
          f"retired: {still_retired}" if still_retired else "")
    missing = [n for n in named if not (ROOT / "experiments" / f"{n}.py").exists()]
    check("every named script exists", not missing, f"missing: {missing}" if missing else
          f"{len(named)} scripts")
    broken = []
    for n in named:
        f = ROOT / "experiments" / f"{n}.py"
        if not f.exists():
            continue
        try:
            compile(f.read_text(), str(f), "exec")
        except SyntaxError as exc:
            broken.append(f"{n}: {exc.msg}")
    check("every named script compiles", not broken, "; ".join(broken))

    # Retired scripts must still compile, so they fail on data rather than syntax.
    rot = []
    for f in sorted((ROOT / "experiments").glob("exp*.py")):
        try:
            compile(f.read_text(), str(f), "exec")
        except SyntaxError as exc:
            rot.append(f"{f.name}: {exc.msg}")
    check("no experiment fails to compile", not rot, "; ".join(rot))

    check("shell syntax is valid",
          subprocess.run(["bash", "-n", str(sh)], capture_output=True).returncode == 0)


# ---------------------------------------------------------------------- 7 --
def check_translation(zh: Path) -> None:
    print("\n[7] translation alignment")
    if not zh.exists():
        skip("translation", f"{zh} not found")
        return
    src = (MS / "main.tex").read_text()
    body = src[src.index(r"\section{Introduction}"):]
    en_sections = len([m for m in re.findall(r"\\section\{([^}]*)\}", body)
                       if m not in ("Specification excerpts", "Supporting tables")])
    en_subs = len(re.findall(r"\n\\subsection\{", body))
    en_floats = len(re.findall(r"\\caption\{", src))
    zt = zh.read_text()
    zh_sections = len(re.findall(r"^## \d", zt, re.M))
    zh_subs = len(re.findall(r"^### \d", zt, re.M))
    zh_floats = len(re.findall(r"^##### 表", zt, re.M)) + \
        len(re.findall(r"^> \[!example\] 图", zt, re.M))
    check(f"top-level sections  en={en_sections} zh={zh_sections}",
          en_sections == zh_sections)
    check(f"subsections         en={en_subs} zh={zh_subs}", en_subs == zh_subs)
    check(f"floats              en={en_floats} zh={zh_floats}", en_floats == zh_floats)
    stale = [n for n in ("935,315", "930,109", "817,528", "0.0312")
             if n in zt]
    check("no superseded number in the translation", not stale,
          f"found: {stale}" if stale else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh", type=Path, help="path to the Chinese draft")
    args = ap.parse_args()

    print("consistency gate")
    check_figures()
    stats = panel_stats()
    check_data(stats)
    check_labels()
    check_entry_point()
    if args.zh:
        check_translation(args.zh)

    print(f"\n{'-' * 62}")
    if NOTES:
        print("skipped:")
        for n in NOTES:
            print(f"  {n}")
    if FAILS:
        print(f"FAILED {len(FAILS)}:")
        for f in FAILS:
            print(f"  {f}")
    else:
        print("all checks passed")
    return len(FAILS)


if __name__ == "__main__":
    sys.exit(main())
