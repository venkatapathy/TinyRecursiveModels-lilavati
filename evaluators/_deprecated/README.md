# Deprecated evaluators — do not use

These are kept only so the ICML-era numbers can be traced back to the code
that produced them. They contain the defects documented in the Stage-0 audit:

- `basicfour.py` — digit-accuracy denominator is `max(len(expected), len(pred))`,
  so a run-on prediction contributes unbounded slots and aggregate digit
  accuracy can fall below sequence accuracy. `self.total` is also incremented
  before three `continue` guards, so `accuracy` and `digit_accuracy` are
  computed over different example sets. Calls `np.clip` without importing numpy.
- `addition.py` / `multiplication.py` — fixed-width digit denominators
  (`digits+1`, `2*digits`) against configs where `digits` is 32 or 100, driving
  digit accuracy toward 0 while sequence accuracy stays nonzero. Both also
  slice the result from `eq_idx + 1`, one position too far.

All live metrics come from `evaluators/harness.py`.
