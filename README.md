# Tiny Recursive Models — arithmetic length generalization

Code for the arithmetic length-generalization experiments on Tiny Recursive
Models (TRM) with output-state supervision.

> **Status: the ICML-era pipeline has been retired.** An audit found a label
> leak that made the original headline numbers unsound, along with a val split
> that was not in-distribution and four mutually inconsistent metric
> definitions. Everything below describes the corrected (ICLR) pipeline.
> See [Corrected formulation](#corrected-formulation) for what changed and why,
> and `evaluators/_deprecated/README.md` for the quarantined code.
> **Do not compare numbers produced before and after this change.**

---

## Installation

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install -r specific_requirements.txt
venv/bin/pip install transformers datasets accelerate peft matplotlib
```

Notes:

- `adam-atan2` requires a compiled CUDA extension and may fail to build. If it
  is absent, `pretrain.py` falls back to AdamW automatically. This changes the
  optimizer, so do not mix pre- and post-fallback runs in one table — state
  which was used in the paper's training protocol.
- `pretrain.py` calls `wandb.init` unconditionally; there is no `--no-wandb`
  flag. Export `WANDB_MODE=offline` for local runs.

---

## Corrected formulation

Four defects invalidated the earlier results. All four are fixed, and each fix
has a verification artifact that fails loudly rather than silently.

### 1. Label leakage

Attention is bidirectional (`causal=False` in both `models/recursive_reasoning/trm.py`
and `models/baselines/transformer.py`), but the model input was the *full*
sequence including the answer, supervised on that same sequence shifted left by
one. Position *t* was asked to predict the token at *t+1* while attending
directly to it.

The fix is a **masked output field**: the input is `prompt + [MASK] * F`, where
`F = max_len - len(prompt)` is fixed and independent of the answer length
(sizing the field to the answer would leak its magnitude). Labels are
position-aligned with no shift.

Evidence — `venv/bin/python scripts/leakage_control.py`:

| Formulation | Accuracy as shipped | Answer masked at eval | Drop |
| :-- | --: | --: | --: |
| Old (shift-by-one) | 1.0000 | 0.1249 | **0.8751** |
| New (masked field) | 0.9642 | 0.9642 | 0.0000 |

The old format's accuracy came from reading the answer out of its own input.

**Consequence for the claims.** Under a masked bidirectional decoder there is
no decoding order, so "emit state before vs. after the answer" is not a
meaningful contrast. The OS-Before / OS-After variable is now the *positional
layout* of the algorithmic state relative to the answer field, with supervision
content held identical.

### 2. `val` was not in-distribution

`is_train` was `(split == "train")`, so `val` was drawn from the OOD range
alongside `test`; `pretrain.py` also built its training-time eval loader from
`test`. Every "in-distribution" number ever reported was measured out of
distribution. Now `is_train = split in ("train", "val")`, and the eval loader
reads `config.eval_split` (default `val`).

ID/OOD is bucketed by **max operand digits**, consistently in the sampler and
the evaluator. It was previously bucketed by result length in one and operand
length in the other.

### 3. One frozen metric

There were four independent implementations of "accuracy". The reviewer-visible
symptom was sequence accuracy exceeding digit accuracy, which arose from mixing
a micro-average over digits with a macro-average over examples while using
`max(len(expected), len(pred))` as the denominator — so a run-on prediction
contributed unbounded slots.

All scoring now lives in `evaluators/harness.py`. The digit-accuracy denominator
is always `len(expected)`. `result()` asserts:

- `digit_accuracy >= sequence_accuracy`
- the per-length grid partitions the scored set
- overall accuracy equals the counts-weighted grid mean

These raise rather than emitting a number. `grep -rn "def _digit_ratio" --include=*.py .`
must return exactly one hit.

### 4. Loss dilution (found during the first gate run)

The masked field is ~500 positions wide while a target is 5–20 tokens. The
first version supervised PAD at every unused slot, putting **96–99% of the
cross-entropy mass on "emit PAD"**. Symptom: token accuracy 0.99 alongside
sequence accuracy 0.08.

Now exactly `N_TERMINATOR_PADS = 1` PAD is supervised after the target and the
rest of the field is `IGNORE_LABEL_ID`. Termination is still learned; PAD mass
is 5–17%.

### Other corrections

- **The "40× transformer" baseline is 3.97×.** Measured by instantiating both
  architectures (`venv/bin/python count_params.py`), not read off config names:
  `config/arch/trm.yaml` is labelled 300k but builds 2,977,538 parameters;
  `config/arch/transformer.yaml` is labelled 1M but builds 11,824,512.
- Multi-digit carries and division remainders are `<SEP>`-terminated; they were
  previously concatenated with no boundary, making the trace ambiguous at OOD
  lengths.
- Overlong examples raise instead of being silently truncated.
- Divisors are bounded (`--max_divisor_digits`, default 3) so the remainder
  trace stays position-invariant.
- Seeds are interpolated into run names and checkpoint paths; `random`, `numpy`,
  and `torch.cuda` are all seeded; the unseeded `np.random.choice` in the batch
  sampler is gone.
- Shape-mismatched checkpoint loads raise instead of partially copying.

---

## Reproducing the experiments

### 1. Generate datasets

```bash
for pair in "ns vanilla" "os_after basicfour_concat" "os_before basic_concat_reverse"; do
  set -- $pair
  venv/bin/python dataset/build_arithmetic_dataset.py \
    --output_dir "data/iclr/$1" --dataset_mode "$2" \
    --num_train 100000 --num_val 4000 --num_test 20000 \
    --train_max_digits 8 --test_max_digits 32 \
    --max_len 512 --seed 42
done
```

`--max_len 512` is required: 32-digit division and multiplication traces exceed
256 and the builder now refuses to truncate.

### 2. Audit the data before training on it

```bash
venv/bin/python scripts/audit_data.py data/iclr/ns data/iclr/os_after data/iclr/os_before
```

Checks, per split: no supervised position holds a non-MASK input token; train/val
are 1–8 max operand digits and test is 9–32; labels are `[target][one PAD][ignore…]`;
and the label field round-trips through the harness's independent implementation
of `expected_result()` / `expected_trace()`. Exits non-zero on any failure.

### 3. Train and evaluate

The full gate — 3 supervision arms × TRM, plus 2 transformer arms, each
evaluated on val (ID) and test (OOD):

```bash
bash scripts/run_gate1.sh 1     # seed 1
bash scripts/run_gate1.sh 2
bash scripts/run_gate1.sh 3
```

Individual runs:

| Model | Config | Data |
| :-- | :-- | :-- |
| TRM (No State) | `cfg_vanilla` | `data/iclr/ns` |
| TRM (OS-After) | `cfg_basicfour_concat` | `data/iclr/os_after` |
| TRM (OS-Before) | `cfg_basicfour_concat_reverse` | `data/iclr/os_before` |
| Transformer-300k (No State) | `cfg_transformer_300k` | `data/iclr/ns` |
| Transformer-300k (OS-Before) | `cfg_transformer_300k_concat_reverse` | `data/iclr/os_before` |

```bash
venv/bin/torchrun --nproc-per-node 4 --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_basicfour_concat_reverse \
  data_paths="[data/iclr/os_before]" data_paths_test="[data/iclr/os_before]" \
  epochs=40 global_batch_size=256 max_len=512 digits=32 train_digits=8 \
  project_name=trm-iclr-gate1 run_name=trm_os_before seed=1

venv/bin/python evaluate.py --config-name cfg_basicfour_concat_reverse \
  data_paths="[data/iclr/os_before]" data_paths_test="[data/iclr/os_before]" \
  max_len=512 digits=32 train_digits=8 \
  project_name=trm-iclr-gate1 run_name=trm_os_before seed=1 \
  +checkpoint_folder=checkpoints/trm-iclr-gate1/trm_os_before_s1 \
  +split=test +compile=false
```

Results are written as `results/{run}__seed{N}__{split}.json` — structured, so
one run can never absorb another's file via globbing.

### 4. SLM baselines

```bash
venv/bin/python evaluate_qwen.py \
  --model Qwen/Qwen2.5-Math-1.5B-Instruct \
  --data_dir data/iclr/ns --split test \
  --run_name qwen15b_ns --seed 1 --batch_size 16
```

Scoring delegates to `harness.score_one`, the same code path the TRM runs use,
so SLM and model rows are directly comparable. It reads `structured_{split}.jsonl`
(explicit `op`/`A`/`B` fields) rather than string-splitting the flat `text`
rendering — for the state-supervised datasets that split returns the trace, not
the answer. Pass `--adapter` to evaluate a LoRA fine-tune.

> **Known limitation.** `train_slm.py` still reads only `vanilla_{split}.jsonl`
> (line 34), so the SLM fine-tuning path can currently only train the No-State
> arm. Rendering jsonl for the other supervision arms is required before the
> OS-Before SLM comparison can be run.

### 5. Tables and plots

```bash
venv/bin/python scripts/generate_table.py
```

Rows come from the `results/runs.json` manifest, so adding a run never means
editing the script. Missing metrics raise and exit non-zero rather than printing
a dash — the silent-dash behaviour is what produced the conflicting tables in
the earlier submission. Use `--lenient` only for exploratory looks, never for
numbers that go in the paper.

Outputs: `results/main_table.tex`, `results/length_table.tex`,
`results/accuracy_by_length.png`.

---

## Verification

| # | Check | Command |
| :-- | :-- | :-- |
| 1 | Leakage control | `venv/bin/python scripts/leakage_control.py` |
| 2 | Metric invariants | asserted inside every `evaluate.py` run |
| 3 | Data audit | `venv/bin/python scripts/audit_data.py data/iclr/*` |
| 4 | One metric definition | `grep -rn "def _digit_ratio" --include=*.py .` → 1 hit |
| 5 | Tables fail on missing keys | `venv/bin/python scripts/generate_table.py; echo $?` |
| 6 | Parameter counts | `venv/bin/python count_params.py` |
| 7 | Seed determinism | `bash scripts/check_determinism.sh` |

---

## Repository notes

- `evaluators/harness.py` — the single scoring implementation. Extend it; do not
  fork it.
- `evaluators/_deprecated/` — the old per-operation evaluators, kept only as a
  record of the defects. Not importable by any config.
- `results/_pre_padfix/` — gate results from before the loss-dilution fix,
  retained as a record. Invalid; do not cite.
- `H_layers` is silently ignored by TRM — it uses one shared `L_level` stack
  (`trm.py`). Recursion depth (`H_cycles` × `L_cycles`) costs no parameters.
- `basic_concat_reverse` means *state before answer*, not digit reversal. No
  digit-order (LSD/MSD) rendering is implemented.
