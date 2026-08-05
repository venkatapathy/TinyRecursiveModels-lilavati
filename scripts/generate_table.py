#!/usr/bin/env python3
"""
Build the paper's tables and figures from the frozen harness's result files.

Replaces the previous scraper, which:
  - globbed `results/*{run_id}*.json` newest-mtime-wins, so `basicfour_concat`
    could silently match a `basicfour_concat_reverse` file;
  - looked up keys that `evaluate.py` never wrote (`basicfour/id_accuracy_test`
    vs the actual `basicfour_test/basicfour/id_accuracy`) and printed "-" on a
    miss, so Table 1 was blank-by-construction while Table 2 -- which used
    substring matching -- resolved. That asymmetry is why the tables disagreed;
  - reported a single value per cell with no seed aggregation;
  - carried a hardcoded row list including `baseline_transformer_40x`, which is
    not a run_name in any config and so could never resolve.

Every lookup here is exact and every miss is an error. Cells are mean +/- std
over seeds.

Usage:
    python3 scripts/generate_table.py [--results-dir results] [--manifest runs.json]
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


RESULT_RE = re.compile(r"^(?P<run>.+?)__seed(?P<seed>\d+)__(?P<split>val|test)\.json$")

# pretrain.py appends `_s{seed}` to run_name so checkpoint dirs never collide
# across seeds. The seed is already captured from the filename, so strip that
# suffix to recover the bare run identity -- otherwise trm_ns_s1 and trm_ns_s2
# land in separate groups and nothing ever aggregates.
SEED_SUFFIX_RE = re.compile(r"_s\d+$")


class MissingMetric(KeyError):
    pass


def load_results(results_dir: str) -> Dict[str, Dict[str, List[dict]]]:
    """
    Load every result file into {run_name: {split: [payload, ...]}}.

    Filenames are `{run}__seed{N}__{split}.json` -- structured, not globbed,
    so a run can never absorb another run's file.
    """
    if not os.path.isdir(results_dir):
        raise SystemExit(f"No results directory at {results_dir!r}. Run evaluate.py first.")

    out: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    n = 0
    for fn in sorted(os.listdir(results_dir)):
        m = RESULT_RE.match(fn)
        if not m:
            # runs.json is the manifest and leakage_control.json is a Stage 0
            # control artifact -- both live here legitimately and are not runs.
            if fn.endswith(".json") and fn not in ("runs.json", "leakage_control.json"):
                print(f"  ! ignoring unrecognised result file: {fn}", file=sys.stderr)
            continue
        with open(os.path.join(results_dir, fn)) as f:
            payload = json.load(f)
        seed = int(m.group("seed"))
        run = SEED_SUFFIX_RE.sub("", m.group("run"))
        payload["_seed"] = seed
        if any(p["_seed"] == seed for p in out[run][m.group("split")]):
            raise SystemExit(
                f"Duplicate result for run {run!r} seed {seed} split "
                f"{m.group('split')!r} (file {fn}). Two files claim the same "
                f"cell; refusing to average them."
            )
        out[run][m.group("split")].append(payload)
        n += 1
    print(f"Loaded {n} result files for {len(out)} runs from {results_dir}/")
    return out


def agg(runs: Dict[str, Dict[str, List[dict]]], run: str, split: str, key: str) -> tuple:
    """Return (mean, std, n_seeds) for `key`, or raise if anything is missing."""
    if run not in runs:
        raise MissingMetric(f"no results for run {run!r}")
    if split not in runs[run]:
        raise MissingMetric(f"run {run!r} has no {split!r} split")
    vals = []
    for payload in runs[run][split]:
        if key not in payload:
            raise MissingMetric(
                f"run {run!r} seed {payload['_seed']} split {split!r} "
                f"is missing metric {key!r}. Available: {sorted(payload)[:8]}..."
            )
        vals.append(float(payload[key]))
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def fmt(mean: float, std: float, n: int) -> str:
    if n == 1:
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def cell(runs, run, split, key, strict: bool) -> str:
    try:
        return fmt(*agg(runs, run, split, key))
    except MissingMetric as e:
        if strict:
            raise SystemExit(f"ERROR: {e}")
        print(f"  ! {e}", file=sys.stderr)
        return r"\textemdash"


def load_manifest(path: Optional[str]) -> List[dict]:
    """
    Row list comes from a manifest so new runs/seeds/sizes never require
    editing this file.  Each entry: {"name": ..., "run": ...}.
    """
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)["rows"]
    raise SystemExit(
        f"No manifest at {path!r}. Write one listing the runs to tabulate, e.g.\n"
        '  {"rows": [{"name": "TRM (OS-Before)", "run": "trm_os_before"}]}'
    )


def main_table(runs, rows, strict) -> str:
    cols = [
        ("ID Seq", "val", "seq_accuracy"),
        ("ID Digit", "val", "digit_accuracy"),
        ("OOD Seq", "test", "ood_seq_accuracy"),
        ("OOD Digit", "test", "ood_digit_accuracy"),
        ("Trace", "test", "trace_accuracy"),
    ]
    head = " & ".join(f"\\textbf{{{c[0]}}}" for c in cols)
    out = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Arithmetic accuracy. ID = 1--8 max operand digits (held-out "
        r"in-distribution split); OOD = 9--32. Mean $\pm$ std over seeds. "
        r"All numbers from the frozen harness (\texttt{evaluators/harness.py}).}",
        r"\label{tab:main}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
        r"\textbf{Method} & " + head + r" \\", r"\midrule",
    ]
    for r in rows:
        vals = [cell(runs, r["run"], sp, k, strict) for _, sp, k in cols]
        out.append(f"{r['name']} & " + " & ".join(vals) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def length_table(runs, rows, strict, lengths=(8, 12, 16, 24, 32)) -> str:
    out = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Exact-match accuracy by max operand digit length.}",
        r"\label{tab:by_length}",
        r"\begin{tabular}{l" + "c" * len(lengths) + "}", r"\toprule",
        r"\textbf{Method} & " + " & ".join(f"\\textbf{{{d}}}" for d in lengths) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        vals = [cell(runs, r["run"], "test", f"len_{d}_seq_accuracy", strict) for d in lengths]
        out.append(f"{r['name']} & " + " & ".join(vals) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def length_plot(runs, rows, out_path, max_len=32):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for r in rows:
        xs, ys, es = [], [], []
        for d in range(1, max_len + 1):
            try:
                m, s, _ = agg(runs, r["run"], "test", f"len_{d}_seq_accuracy")
            except MissingMetric:
                continue
            xs.append(d); ys.append(m); es.append(s)
        if not xs:
            continue
        ys, es = np.array(ys), np.array(es)
        line, = ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=r["name"])
        ax.fill_between(xs, ys - es, ys + es, alpha=0.18, color=line.get_color())

    ax.axvline(8.5, color="0.4", linestyle="--", linewidth=1)
    ax.annotate("train range ends", xy=(8.5, 1.02), xytext=(8.8, 1.02),
                fontsize=9, color="0.35", va="center")
    ax.set_xlabel("Max operand digit length")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlim(0.5, max_len + 0.5)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--manifest", default="results/runs.json")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--lenient", action="store_true",
                    help="Emit an em-dash for missing metrics instead of failing. "
                         "Never use this for numbers that go in the paper.")
    args = ap.parse_args()

    runs = load_results(args.results_dir)
    rows = load_manifest(args.manifest)
    strict = not args.lenient
    os.makedirs(args.out_dir, exist_ok=True)

    for fname, content in [
        ("main_table.tex", main_table(runs, rows, strict)),
        ("length_table.tex", length_table(runs, rows, strict)),
    ]:
        p = os.path.join(args.out_dir, fname)
        with open(p, "w") as f:
            f.write(content)
        print(f"  wrote {p}")

    length_plot(runs, rows, os.path.join(args.out_dir, "accuracy_by_length.png"))


if __name__ == "__main__":
    main()
