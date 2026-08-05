"""
Frozen evaluation harness for the arithmetic tasks.

This is the SINGLE source of truth for sequence accuracy, digit accuracy and
trace accuracy. Nothing else in the repo may define those metrics.

Design notes -- each of these is a fix for a specific defect in the evaluators
this module replaces (`evaluators/basicfour.py`, `addition.py`,
`multiplication.py`) and in `evaluate_qwen.py`:

1. One example set. Every counter is incremented at the same point, after the
   parse guards. Previously `total` was incremented before three `continue`
   guards while the digit counters were incremented after them, so
   `correct/total` and `correct_digits/total_digits` were computed over
   different populations and `accuracy` was not the weighted average of
   `id_accuracy` and `ood_accuracy`.

2. Length-matched digit denominator. Each example contributes exactly
   `len(expected)` digit slots. The old denominator was
   `max(len(expected), len(pred))`, which let a run-on prediction contribute
   an unbounded number of slots and made aggregate digit accuracy fall below
   sequence accuracy -- the "impossible" result the reviewers flagged.
   Over-generation is still measured, but as its own rate.

3. Per-example digit accuracy is macro-averaged, so it is mathematically
   guaranteed to be >= sequence accuracy. Both invariants are asserted at the
   end of `result()` rather than left to inspection.

4. One ID/OOD definition: max operand digit length vs `train_digits`. The
   sampler in `dataset/build_arithmetic_dataset.py` uses the same rule.

5. Position-aligned decoding. Under the masked-field formulation the model
   predicts the token AT each position, so the output field starts at
   `eq_pos + 1`. (The old format had labels shifted by one and sliced from
   `eq_pos`.)
"""

from typing import Dict, Optional, List, Tuple
import collections

import numpy as np
import torch
import torch.distributed as dist

from dataset.common import PuzzleDatasetMetadata


PAD_ID = 0
MASK_ID = 1
EQ_ID = 16
RES_ID = 22
SEP_ID = 23
CAR_IDS = {18: '+', 19: '-', 20: '*', 21: '/'}


def _build_inv_vocab() -> Dict[int, str]:
    inv = {PAD_ID: 'PAD', MASK_ID: 'MASK', EQ_ID: '=', 17: 'R',
           RES_ID: '<RES>', SEP_ID: '<SEP>'}
    for i in range(10):
        inv[i + 2] = str(i)
    for i, c in enumerate('+-*/'):
        inv[i + 12] = c
    for tok, op in CAR_IDS.items():
        inv[tok] = f'<CAR_{op}>'
    return inv


INV_VOCAB = _build_inv_vocab()


# ---------------------------------------------------------------------------
# Reference implementations of the algorithmic state.
#
# These are the oracle for trace scoring, and the same functions are the
# verifier for method-directed reward in the RL arm. Keep them in sync with
# the generators in dataset/build_arithmetic_dataset.py.
# ---------------------------------------------------------------------------

def expected_result(op: str, a: int, b: int) -> Optional[str]:
    if op == '+':
        return str(a + b)
    if op == '-':
        return str(a - b)
    if op == '*':
        return str(a * b)
    if op == '/':
        if b == 0:
            return None
        q, r = a // b, a % b
        return str(q) if r == 0 else f"{q}R{r}"
    return None


def expected_trace(op: str, a: int, b: int) -> List[int]:
    """Per-step algorithmic state, as a list of integer values."""
    if op == '+':
        a_d = [int(d) for d in str(a)[::-1]]
        b_d = [int(d) for d in str(b)[::-1]]
        carries, c = [], 0
        for i in range(len(str(a + b))):
            s = (a_d[i] if i < len(a_d) else 0) + (b_d[i] if i < len(b_d) else 0) + c
            c = s // 10
            carries.append(c)
        return carries

    if op == '-':
        if a < b:
            return []
        a_d = [int(d) for d in str(a)[::-1]]
        b_d = [int(d) for d in str(b)[::-1]]
        borrows, bin_ = [], 0
        for i in range(len(a_d)):
            diff = a_d[i] - (b_d[i] if i < len(b_d) else 0) - bin_
            bin_ = 1 if diff < 0 else 0
            borrows.append(bin_)
        return borrows[:len(str(a - b))]

    if op == '*':
        a_d = [int(d) for d in str(a)[::-1]]
        b_d = [int(d) for d in str(b)[::-1]]
        carry, cols = 0, []
        for k in range(len(str(a * b))):
            s = carry
            for i in range(k + 1):
                j = k - i
                if i < len(a_d) and j < len(b_d):
                    s += a_d[i] * b_d[j]
            carry = s // 10
            cols.append(carry)
        return cols

    if op == '/':
        if b == 0:
            return []
        rem, rems = 0, []
        for ch in str(a):
            rem = (rem * 10 + int(ch)) % b
            rems.append(rem)
        return rems

    return []


# ---------------------------------------------------------------------------

def decode(seq, stop_at_pad: bool = True) -> str:
    """Decode ids to a display string. Unknown ids become '?'."""
    out = []
    for t in seq:
        t = int(t)
        if stop_at_pad and t in (PAD_ID, MASK_ID):
            break
        out.append(INV_VOCAB.get(t, '?'))
    return ''.join(out)


def parse_prompt(ids) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Parse 'A op B =' from the input ids. Returns (op, a, b)."""
    s = decode(ids)
    lhs = s.split('=')[0]
    for op in ('+', '*', '/'):
        if op in lhs:
            parts = lhs.split(op)
            if len(parts) == 2:
                try:
                    return op, int(parts[0]), int(parts[1])
                except ValueError:
                    return None, None, None
    # subtraction last: '-' can only be an operator, operands are non-negative
    idx = lhs.find('-', 1)
    if idx > 0:
        try:
            return '-', int(lhs[:idx]), int(lhs[idx + 1:])
        except ValueError:
            return None, None, None
    return None, None, None


def split_field(field_ids, dataset_mode: str) -> Tuple[str, List[str]]:
    """
    Split the predicted output field into (answer_string, trace_values).

    Trace values are <SEP>-terminated, so multi-digit carries and remainders
    stay unambiguous -- that separator is why the trace can be scored
    value-by-value instead of digit-by-digit.
    """
    ids = [int(t) for t in field_ids]

    # Decoding stops at the first PAD/MASK. Everything past that terminator is
    # unobservable output and must not be scanned -- otherwise a stray <RES>
    # emitted beyond the terminator makes the parser read an "answer" out of a
    # region the model already ended, which can yield a parsed string LONGER
    # than the decodable field. Truncate first, then locate the markers.
    stop = next((i for i, t in enumerate(ids) if t in (PAD_ID, MASK_ID)), len(ids))
    ids = ids[:stop]

    def take_until_pad(xs):
        out = []
        for t in xs:
            if t in (PAD_ID, MASK_ID):
                break
            out.append(t)
        return out

    if dataset_mode == 'vanilla':
        return decode(take_until_pad(ids)), []

    car_idx = next((i for i, t in enumerate(ids) if t in CAR_IDS), None)
    res_idx = next((i for i, t in enumerate(ids) if t == RES_ID), None)

    if dataset_mode == 'basic_concat_reverse':
        # <CAR_op> trace <RES> Y
        if car_idx is None:
            trace_ids = []
        else:
            trace_ids = ids[car_idx + 1: res_idx] if res_idx is not None else ids[car_idx + 1:]
        answer_ids = take_until_pad(ids[res_idx + 1:]) if res_idx is not None else []
    else:
        # basicfour_concat: Y <CAR_op> trace
        answer_ids = take_until_pad(ids[:car_idx]) if car_idx is not None else take_until_pad(ids)
        trace_ids = ids[car_idx + 1:] if car_idx is not None else []

    # split trace on <SEP>
    values, cur = [], []
    for t in trace_ids:
        if t in (PAD_ID, MASK_ID):
            break
        if t == SEP_ID:
            values.append(''.join(INV_VOCAB.get(x, '?') for x in cur))
            cur = []
        else:
            cur.append(t)
    if cur:
        values.append(''.join(INV_VOCAB.get(x, '?') for x in cur))

    return decode(answer_ids), values


class ArithmeticEvaluator:
    """The frozen harness. Do not fork this class -- extend it."""

    required_outputs = {"inputs", "preds"}

    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata, **kwargs):
        self.data_path = data_path
        self.dataset_mode = kwargs.get("dataset_mode", "vanilla")
        self.train_digits = int(kwargs.get("train_digits", 8))
        self.max_digits = int(kwargs.get("digits", 32))
        self.vocab_size = getattr(eval_metadata, "vocab_size", 24)
        self.reset_counters()

    # -- counters ----------------------------------------------------------
    def reset_counters(self):
        self.n_seen = 0            # every example the loader handed us
        self.n_unparseable = 0     # prompt could not be parsed -> excluded
        self.n_scored = 0          # examples all metrics are computed over

        self.seq_correct = 0
        self.digit_acc_sum = 0.0   # macro-average: sum of per-example ratios
        self.overgen = 0           # predictions longer than the target

        self.op_scored = collections.Counter()
        self.op_correct = collections.Counter()

        self.id_scored = 0
        self.id_correct = 0
        self.id_digit_sum = 0.0
        self.ood_scored = 0
        self.ood_correct = 0
        self.ood_digit_sum = 0.0

        # per-length grid, sized from the config rather than hardcoded at 40
        self.len_scored = collections.Counter()
        self.len_correct = collections.Counter()
        self.len_digit_sum = collections.defaultdict(float)

        self.trace_scored = 0
        self.trace_acc_sum = 0.0
        self.op_trace_scored = collections.Counter()
        self.op_trace_sum = collections.defaultdict(float)

    def begin_eval(self):
        self.reset_counters()

    # -- scoring -----------------------------------------------------------
    @staticmethod
    def _digit_ratio(expected: str, pred: str) -> float:
        """
        Fraction of the TARGET's digits that are correct, aligned from the
        least significant end. Denominator is len(expected) always, so this
        is in [0, 1] and equals 1.0 exactly when pred == expected.
        """
        if not expected:
            return 1.0 if not pred else 0.0
        e, p = expected[::-1], pred[::-1]
        hits = sum(1 for k in range(min(len(e), len(p))) if e[k] == p[k])
        return hits / len(e)

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        inputs = batch["inputs"].cpu().numpy()
        pred_seqs = preds["preds"].cpu().numpy()

        for i in range(len(inputs)):
            self.n_seen += 1

            op, a, b = parse_prompt(inputs[i])
            if op is None:
                self.n_unparseable += 1
                continue

            eq_pos = np.where(inputs[i] == EQ_ID)[0]
            if len(eq_pos) == 0:
                self.n_unparseable += 1
                continue

            expected = expected_result(op, a, b)
            if expected is None:
                self.n_unparseable += 1
                continue

            # Labels are position-aligned with inputs, so the output field
            # begins one position AFTER '='.
            field = pred_seqs[i, int(eq_pos[0]) + 1:]
            pred_str, trace_vals = split_field(field, self.dataset_mode)

            self.score_one(op, a, b, pred_str, trace_vals, expected=expected)

    def score_one(self, op: str, a: int, b: int, pred_str: str,
                  trace_vals: Optional[List[str]] = None,
                  expected: Optional[str] = None) -> bool:
        """
        Score a single decoded prediction. This is THE definition of
        sequence/digit/trace accuracy for the whole project -- token-level
        callers reach it via update_batch(), string-level callers (the SLM
        path in evaluate_qwen.py) call it directly. Do not reimplement it;
        a second copy is how the ICML submission ended up with four mutually
        inconsistent metrics.

        Returns whether the prediction was an exact match.
        """
        if expected is None:
            expected = expected_result(op, a, b)
        if expected is None:
            self.n_unparseable += 1
            return False

        # ---- from here on, every counter moves together --------------
        self.n_scored += 1
        matched = (pred_str == expected)
        ratio = self._digit_ratio(expected, pred_str)

        self.seq_correct += int(matched)
        self.digit_acc_sum += ratio
        self.overgen += int(len(pred_str) > len(expected))

        self.op_scored[op] += 1
        self.op_correct[op] += int(matched)

        max_digits = max(len(str(a)), len(str(b)))
        if max_digits <= self.train_digits:
            self.id_scored += 1
            self.id_correct += int(matched)
            self.id_digit_sum += ratio
        else:
            self.ood_scored += 1
            self.ood_correct += int(matched)
            self.ood_digit_sum += ratio

        self.len_scored[max_digits] += 1
        self.len_correct[max_digits] += int(matched)
        self.len_digit_sum[max_digits] += ratio

        # trace / carry accuracy
        if self.dataset_mode != 'vanilla' and trace_vals is not None:
            exp_trace = [str(v) for v in expected_trace(op, a, b)]
            if exp_trace:
                hits = sum(1 for k in range(min(len(exp_trace), len(trace_vals)))
                           if exp_trace[k] == trace_vals[k])
                t_ratio = hits / len(exp_trace)
                self.trace_scored += 1
                self.trace_acc_sum += t_ratio
                self.op_trace_scored[op] += 1
                self.op_trace_sum[op] += t_ratio

        return matched

    # -- aggregation -------------------------------------------------------
    def _pack(self) -> Tuple[List[float], List[str]]:
        """Flatten counters into a fixed-order vector for dist.reduce."""
        keys, vals = [], []

        def add(k, v):
            keys.append(k)
            vals.append(float(v))

        add("n_seen", self.n_seen)
        add("n_unparseable", self.n_unparseable)
        add("n_scored", self.n_scored)
        add("seq_correct", self.seq_correct)
        add("digit_acc_sum", self.digit_acc_sum)
        add("overgen", self.overgen)
        add("id_scored", self.id_scored)
        add("id_correct", self.id_correct)
        add("id_digit_sum", self.id_digit_sum)
        add("ood_scored", self.ood_scored)
        add("ood_correct", self.ood_correct)
        add("ood_digit_sum", self.ood_digit_sum)
        add("trace_scored", self.trace_scored)
        add("trace_acc_sum", self.trace_acc_sum)
        for op in '+-*/':
            add(f"op_scored_{op}", self.op_scored[op])
            add(f"op_correct_{op}", self.op_correct[op])
            add(f"op_trace_scored_{op}", self.op_trace_scored[op])
            add(f"op_trace_sum_{op}", self.op_trace_sum[op])
        for L in range(1, self.max_digits + 1):
            add(f"len_scored_{L}", self.len_scored[L])
            add(f"len_correct_{L}", self.len_correct[L])
            add(f"len_digit_sum_{L}", self.len_digit_sum[L])
        return vals, keys

    def result(self, save_path: Optional[str], rank: int, world_size: int, group=None) -> Optional[Dict[str, float]]:
        vals, keys = self._pack()

        if world_size > 1:
            t = torch.tensor(vals, device="cuda", dtype=torch.float64)
            dist.reduce(t, dst=0)
            vals = t.tolist()

        if rank != 0:
            return None

        c = dict(zip(keys, vals))

        def ratio(num, den):
            return (num / den) if den > 0 else 0.0

        n = c["n_scored"]
        seq_acc = ratio(c["seq_correct"], n)
        digit_acc = ratio(c["digit_acc_sum"], n)

        out = {
            "n_seen": c["n_seen"],
            "n_scored": n,
            "n_unparseable": c["n_unparseable"],
            "overgeneration_rate": ratio(c["overgen"], n),

            "seq_accuracy": seq_acc,
            "digit_accuracy": digit_acc,

            "id_seq_accuracy": ratio(c["id_correct"], c["id_scored"]),
            "id_digit_accuracy": ratio(c["id_digit_sum"], c["id_scored"]),
            "id_n": c["id_scored"],
            "ood_seq_accuracy": ratio(c["ood_correct"], c["ood_scored"]),
            "ood_digit_accuracy": ratio(c["ood_digit_sum"], c["ood_scored"]),
            "ood_n": c["ood_scored"],

            "trace_accuracy": ratio(c["trace_acc_sum"], c["trace_scored"]),
        }

        for op, name in zip('+-*/', ('add', 'sub', 'mul', 'div')):
            out[f"{name}_seq_accuracy"] = ratio(c[f"op_correct_{op}"], c[f"op_scored_{op}"])
            out[f"{name}_n"] = c[f"op_scored_{op}"]
            out[f"{name}_trace_accuracy"] = ratio(c[f"op_trace_sum_{op}"], c[f"op_trace_scored_{op}"])

        for L in range(1, self.max_digits + 1):
            nL = c[f"len_scored_{L}"]
            if nL > 0:
                out[f"len_{L}_seq_accuracy"] = ratio(c[f"len_correct_{L}"], nL)
                out[f"len_{L}_digit_accuracy"] = ratio(c[f"len_digit_sum_{L}"], nL)
                out[f"len_{L}_n"] = nL

        self._check_invariants(out, c)
        self._print(out)
        return out

    # -- invariants --------------------------------------------------------
    @staticmethod
    def _check_invariants(out: Dict[str, float], c: Dict[str, float], tol: float = 1e-6):
        """
        These are the assertions that answer the reviewer. If either fails the
        run is not reportable, so fail loudly rather than emitting a number.
        """
        n = out["n_scored"]
        if n == 0:
            return

        # 1. digit accuracy is macro-averaged over the same examples as
        #    sequence accuracy, and an exact match scores 1.0, so:
        if out["digit_accuracy"] + tol < out["seq_accuracy"]:
            raise AssertionError(
                f"METRIC INVARIANT VIOLATED: digit_accuracy "
                f"({out['digit_accuracy']:.6f}) < seq_accuracy "
                f"({out['seq_accuracy']:.6f}). Sequence accuracy can never "
                f"exceed digit accuracy; the two metrics have desynchronised."
            )

        # 2. the per-length grid must partition the scored examples exactly
        grid_n = sum(v for k, v in out.items() if k.startswith("len_") and k.endswith("_n"))
        if abs(grid_n - n) > tol:
            raise AssertionError(
                f"METRIC INVARIANT VIOLATED: per-length grid covers {grid_n} "
                f"examples but {n} were scored. Lengths outside "
                f"[1, max_digits] are being silently dropped from the grid."
            )

        # 3. overall accuracy must equal the counts-weighted mean of the grid
        recon = sum(out[f"len_{L}_seq_accuracy"] * out[f"len_{L}_n"]
                    for L in range(1, 10_000) if f"len_{L}_n" in out) / n
        if abs(recon - out["seq_accuracy"]) > 1e-4:
            raise AssertionError(
                f"METRIC INVARIANT VIOLATED: seq_accuracy ({out['seq_accuracy']:.6f}) "
                f"!= counts-weighted mean of the per-length grid ({recon:.6f})."
            )

        # 4. ID and OOD must partition the scored set
        if abs(out["id_n"] + out["ood_n"] - n) > tol:
            raise AssertionError(
                f"METRIC INVARIANT VIOLATED: id_n ({out['id_n']}) + ood_n "
                f"({out['ood_n']}) != n_scored ({n})."
            )

    @staticmethod
    def _print(out: Dict[str, float]):
        print("\n" + "=" * 58)
        print("  Arithmetic evaluation (frozen harness)")
        print("=" * 58)
        print(f"  scored {int(out['n_scored'])} / {int(out['n_seen'])} seen"
              f"  ({int(out['n_unparseable'])} unparseable)")
        print(f"  seq   {out['seq_accuracy']:.4f}    digit {out['digit_accuracy']:.4f}"
              f"    overgen {out['overgeneration_rate']:.4f}")
        print(f"  ID    seq {out['id_seq_accuracy']:.4f}  digit {out['id_digit_accuracy']:.4f}"
              f"   (n={int(out['id_n'])})")
        print(f"  OOD   seq {out['ood_seq_accuracy']:.4f}  digit {out['ood_digit_accuracy']:.4f}"
              f"   (n={int(out['ood_n'])})")
        print(f"  trace {out['trace_accuracy']:.4f}")
        for name in ('add', 'sub', 'mul', 'div'):
            if out[f"{name}_n"] > 0:
                print(f"    {name}: seq {out[f'{name}_seq_accuracy']:.4f}"
                      f"  trace {out[f'{name}_trace_accuracy']:.4f}"
                      f"  (n={int(out[f'{name}_n'])})")
        print("=" * 58)
