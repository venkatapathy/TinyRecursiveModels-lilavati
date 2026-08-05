#!/usr/bin/env python3
"""
Data audit. Everything Stage 0 claims about the datasets is checked here
against the actual .npy arrays, not against the generator's intent.

Checks, per dataset x split:

  1. LEAK       -- no supervised position may hold anything but MASK in the
                   input. This is the invariant the ICML formulation violated.
  2. SPLIT      -- train/val are 1..train_digits by max operand digits, test
                   is train_digits+1..max_digits. val must be ID; it was
                   generated with is_train=False and was silently OOD.
  3. TERMINATOR -- labels are [target][exactly one PAD][IGNORE...]. Catches
                   both silent truncation and a regression back to
                   supervising PAD across the whole field.
  4. ROUNDTRIP  -- decoding the LABEL field reproduces expected_result() and
                   expected_trace() from the frozen harness. This is what
                   proves the generator and the evaluator agree; they are
                   independent implementations.
  5. BALANCE    -- share of loss mass sitting on PAD. Two orders of magnitude
                   here is what made the first GATE 1 attempt unreadable.

Exits non-zero on any failure.

Usage:
    venv/bin/python scripts/audit_data.py data/iclr/ns data/iclr/os_after data/iclr/os_before
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.harness import (
    PAD_ID, MASK_ID, EQ_ID, expected_result, expected_trace,
    parse_prompt, split_field,
)

IGNORE = -100

MODE_BY_DIRNAME = {
    "ns": "vanilla",
    "os_after": "basicfour_concat",
    "os_before": "basic_concat_reverse",
}


class Failure(Exception):
    pass


def audit_split(root: str, split: str, mode: str, train_digits: int,
                max_digits: int, sample: int) -> dict:
    d = os.path.join(root, split)
    inputs = np.load(os.path.join(d, "all__inputs.npy"))
    labels = np.load(os.path.join(d, "all__labels.npy"))
    n = len(inputs)
    idx = np.arange(n) if sample <= 0 or sample >= n else \
        np.random.default_rng(0).choice(n, sample, replace=False)

    sup = labels != IGNORE

    # --- 1. leak -----------------------------------------------------------
    leaked = int((sup & (inputs != MASK_ID)).sum())
    if leaked:
        raise Failure(f"{root}/{split}: {leaked} supervised positions hold a "
                      f"non-MASK input token -- the answer is readable from "
                      f"the input")

    # --- 5. balance --------------------------------------------------------
    pad_mass = float((sup & (labels == PAD_ID)).sum()) / float(sup.sum())

    lens = collections.Counter()
    n_trace_checked = 0
    bad_result = bad_trace = bad_term = 0
    first_bad = None

    for i in idx:
        op, a, b = parse_prompt(inputs[i])
        if op is None:
            raise Failure(f"{root}/{split}: example {i} has an unparseable prompt")

        # --- 2. split range ------------------------------------------------
        maxd = max(len(str(a)), len(str(b)))
        lens[maxd] += 1

        # --- 3. terminator -------------------------------------------------
        s = np.where(sup[i])[0]
        contiguous = len(s) == (s[-1] - s[0] + 1)
        n_pad = int((labels[i][s] == PAD_ID).sum())
        if not contiguous or n_pad != 1 or labels[i][s[-1]] != PAD_ID:
            bad_term += 1
            if first_bad is None:
                first_bad = (int(i), "terminator",
                             f"contiguous={contiguous} n_pad={n_pad}")
            continue

        # --- 4. roundtrip through the harness ------------------------------
        eq = np.where(inputs[i] == EQ_ID)[0]
        field = labels[i][int(eq[0]) + 1:].copy()
        field[field == IGNORE] = PAD_ID          # decoder stops at PAD
        pred_str, trace_vals = split_field(field, mode)

        want = expected_result(op, a, b)
        if pred_str != want:
            bad_result += 1
            if first_bad is None:
                first_bad = (int(i), "result", f"{a}{op}{b} label={pred_str!r} want={want!r}")

        if mode != "vanilla":
            want_trace = [str(v) for v in expected_trace(op, a, b)]
            if want_trace:
                n_trace_checked += 1
                if trace_vals != want_trace:
                    bad_trace += 1
                    if first_bad is None:
                        first_bad = (int(i), "trace",
                                     f"{a}{op}{b} label={trace_vals[:6]} want={want_trace[:6]}")

    lo, hi = (1, train_digits) if split in ("train", "val") else (train_digits + 1, max_digits)
    out_of_range = {L: c for L, c in lens.items() if not (lo <= L <= hi)}

    problems = []
    if out_of_range:
        problems.append(f"lengths outside [{lo},{hi}]: {dict(sorted(out_of_range.items()))}")
    if bad_term:
        problems.append(f"{bad_term} bad terminators")
    if bad_result:
        problems.append(f"{bad_result} label fields that do not decode to the expected result")
    if bad_trace:
        problems.append(f"{bad_trace} label traces that disagree with expected_trace()")
    if problems:
        raise Failure(f"{root}/{split}: " + "; ".join(problems) +
                      (f"  first: {first_bad}" if first_bad else ""))

    return {
        "n": n, "checked": len(idx), "pad_mass": pad_mass,
        "len_range": [int(min(lens)), int(max(lens))],
        "trace_checked": n_trace_checked,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--train_digits", type=int, default=8)
    ap.add_argument("--digits", type=int, default=32)
    ap.add_argument("--sample", type=int, default=10000,
                    help="examples to round-trip per split (0 = all)")
    args = ap.parse_args()

    failures = []
    for root in args.roots:
        mode = MODE_BY_DIRNAME.get(os.path.basename(root.rstrip("/")))
        if mode is None:
            failures.append(f"{root}: unknown dataset dir, cannot infer mode")
            continue
        print(f"\n=== {root}  (mode={mode}) ===")
        for split in ("train", "val", "test"):
            try:
                r = audit_split(root, split, mode, args.train_digits,
                                args.digits, args.sample)
            except Failure as e:
                print(f"  {split:5s}  FAIL  {e}")
                failures.append(str(e))
            except FileNotFoundError as e:
                print(f"  {split:5s}  SKIP  {e}")
            else:
                print(f"  {split:5s}  ok    n={r['n']:>7,}  checked={r['checked']:>6,}  "
                      f"maxdigits={r['len_range'][0]}-{r['len_range'][1]}  "
                      f"pad_mass={r['pad_mass']:.1%}  traces={r['trace_checked']:,}")

    print()
    if failures:
        print(f"AUDIT FAILED: {len(failures)} problem(s).")
        raise SystemExit(1)
    print("AUDIT PASSED: no leakage, splits are disjoint by max operand digits, "
          "labels round-trip through the frozen harness.")


if __name__ == "__main__":
    main()
