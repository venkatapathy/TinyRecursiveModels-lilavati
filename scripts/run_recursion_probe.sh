#!/bin/bash
# RECURSION PROBE -- did GATE 1 ever actually test TRM's recursion?
#
# The shipped config/arch/trm.yaml runs the architecture with its defining
# machinery switched off:
#
#     H_cycles: 1          outer recursion -- a single pass
#     L_cycles: 4          inner recursion
#     L_layers: 2
#     H_layers: 1          silently ignored by trm.py
#     halt_max_steps: 1    ACT disabled; every eval reported steps == 1.0
#
# So GATE 1 trained a 2-layer block applied four times with no adaptive
# computation at all. If TRM's advantage comes from deep recursion or from ACT
# allocating more compute to harder instances -- the premise of the
# architecture -- GATE 1 tested neither.
#
# This probe turns both on and asks a single question: does OOD accuracy move
# off zero? Measured throughput cost is 2.2x (8.46 vs 18.27 it/s) for the depth
# change; ACT adds more but halts early once confident.
#
# Expectation, stated in advance so the result cannot be rationalised after the
# fact: OOD stays at zero. A model whose OOD digit precision sits at the chance
# floor (0.107 vs 0.100 random) is not usually one recursion sweep away from a
# position-invariant algorithm. The probe is run because "we never enabled the
# recursion" is an indefensible place to conclude from, not because a reversal
# is likely.
#
# Usage: scripts/run_recursion_probe.sh [seed]
set -uo pipefail
cd "$(dirname "$0")/.."

SEED="${1:-1}"
PY=venv/bin/python
PROJECT=trm-recursion-probe

export WANDB_MODE="${WANDB_MODE:-offline}"
export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"
export OMP_NUM_THREADS=8

EPOCHS="${EPOCHS:-40}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
BS="${BS:-256}"
NPROC="${NPROC:-4}"

# Deep recursion + adaptive computation.
H_CYCLES="${H_CYCLES:-3}"
L_CYCLES="${L_CYCLES:-6}"
HALT="${HALT:-4}"
HALT_EXPLORE="${HALT_EXPLORE:-0.1}"

echo "RECURSION PROBE  seed=${SEED}"
echo "  H_cycles=${H_CYCLES} (was 1)   L_cycles=${L_CYCLES} (was 4)   halt_max_steps=${HALT} (was 1)"
echo "  Baseline for comparison: GATE 1 results in results/*_s${SEED}__seed${SEED}__test.json"
echo ""

run_one () {
  local arch_cfg=$1 data=$2 tag=$3 mode=$4
  local run="${tag}_s${SEED}"
  echo "=================================================================="
  echo "  TRAIN ${run}   (arch=${arch_cfg}, data=${data})"
  echo "=================================================================="

  venv/bin/torchrun --nproc-per-node "$NPROC" --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
    --config-name "$arch_cfg" \
    data_paths="[data/iclr/${data}]" data_paths_test="[data/iclr/${data}]" \
    epochs="$EPOCHS" eval_interval="$EVAL_INTERVAL" global_batch_size="$BS" \
    max_len=512 digits=32 train_digits=8 \
    arch.H_cycles="$H_CYCLES" arch.L_cycles="$L_CYCLES" \
    arch.halt_max_steps="$HALT" arch.halt_exploration_prob="$HALT_EXPLORE" \
    project_name="$PROJECT" run_name="$tag" seed="$SEED" \
    checkpoint_every_eval=False || { echo "TRAIN ${run} FAILED"; return 1; }

  for split in val test; do
    echo "--- EVAL ${run} on ${split} ---"
    $PY evaluate.py --config-name "$arch_cfg" \
      data_paths="[data/iclr/${data}]" data_paths_test="[data/iclr/${data}]" \
      max_len=512 digits=32 train_digits=8 \
      arch.H_cycles="$H_CYCLES" arch.L_cycles="$L_CYCLES" \
      arch.halt_max_steps="$HALT" arch.halt_exploration_prob="$HALT_EXPLORE" \
      project_name="$PROJECT" run_name="$tag" seed="$SEED" \
      +checkpoint_folder="checkpoints/${PROJECT}/${run}" \
      +split="$split" +compile=false || echo "EVAL ${run} ${split} FAILED"
  done

  local last
  last=$(ls "checkpoints/${PROJECT}/${run}" 2>/dev/null \
         | grep -oE 'step_[0-9]+$' | grep -oE '[0-9]+' | sort -n | tail -1)
  if [ -n "$last" ]; then
    echo "--- FAILURE MODE ${run} (OOD, precision vs recall) ---"
    CUDA_VISIBLE_DEVICES=0 $PY scripts/inspect_predictions.py \
      --checkpoint "checkpoints/${PROJECT}/${run}/step_${last}" \
      --data "data/iclr/${data}" --mode "$mode" --split test \
      --n 4000 --bucket 6 --examples 0 || echo "INSPECT ${run} FAILED"
  fi
}

run_one cfg_vanilla                  ns        deep_ns        vanilla
run_one cfg_basicfour_concat_reverse os_before deep_os_before basic_concat_reverse

echo ""
echo "=================================================================="
echo "RECURSION PROBE COMPLETE (seed ${SEED})"
echo "=================================================================="
$PY - <<'PYEOF'
import json, glob, os
rows = []
for f in sorted(glob.glob("results/*__seed1__test.json")):
    d = json.load(open(f))
    rows.append((d["_run"], d.get("ood_seq_accuracy", 0), d.get("ood_digit_accuracy", 0)))
print(f"\n{'run':<24}{'OOD seq':>12}{'OOD digit':>12}")
print("-" * 48)
for r, s, dg in rows:
    tag = "  <-- deep recursion" if r.startswith("deep_") else ""
    print(f"{r:<24}{s:>12.5f}{dg:>12.4f}{tag}")
print("\nIf the deep_* rows are not materially above their GATE 1 counterparts,")
print("the recursion was never the limiting factor and the negative result stands.")
PYEOF
