#!/bin/bash
# FAIR TEST -- the controlled version of GATE 1.
#
# GATE 1 answered "does the ICML result reproduce?" (it does not). It did NOT
# answer "does output-state supervision help length generalization?", because
# the arms were not comparable:
#
#   arm         real target tokens/example    ID seq @ 40 epochs   converged?
#   NS                    4.9                      0.6695          yes (flat by ep30)
#   OS-After             17.0                      0.4238          no
#   OS-Before            18.0                      0.4985          no  (still rising)
#
# The state arms carry ~3.7x more target tokens and were still climbing when
# training stopped. Comparing OOD between models whose ID differs by 0.17 is
# not a controlled experiment -- the OOD gap is confounded with how far each
# arm got through its own fitting problem.
#
# This script trains each arm to CONVERGENCE rather than to a fixed epoch
# count, evaluating often enough to see the ID curve flatten. The budget each
# arm needed is then reported separately, so an efficiency claim stays
# separable from a capability claim.
#
# Read the result as: at matched ID accuracy, does state supervision buy
# out-of-distribution accuracy?
#
# Usage: scripts/run_fair_test.sh [seed] [epochs]
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${1:-1}"
EPOCHS="${2:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
BS="${BS:-256}"
NPROC="${NPROC:-4}"
PY=venv/bin/python
PROJECT=trm-iclr-fair

export WANDB_MODE="${WANDB_MODE:-offline}"
export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"
export OMP_NUM_THREADS=8

echo "FAIR TEST  seed=${SEED}  epochs=${EPOCHS}  eval every ${EVAL_INTERVAL}"
echo "Checkpoints are written every eval so the best-ID step can be selected"
echo "post hoc rather than assuming the last step is the best."
echo ""

run_one () {
  local arch_cfg=$1 data=$2 tag=$3
  local run="${tag}_s${SEED}"
  echo "=================================================================="
  echo "  ${run}   (arch=${arch_cfg}, data=${data})"
  echo "=================================================================="
  venv/bin/torchrun --nproc-per-node "$NPROC" --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
    --config-name "$arch_cfg" \
    data_paths="[data/iclr/${data}]" data_paths_test="[data/iclr/${data}]" \
    epochs="$EPOCHS" eval_interval="$EVAL_INTERVAL" global_batch_size="$BS" \
    max_len=512 digits=32 train_digits=8 \
    project_name="$PROJECT" run_name="$tag" seed="$SEED" \
    checkpoint_every_eval=True

  for split in val test; do
    echo "--- EVAL ${run} on ${split} ---"
    CUDA_VISIBLE_DEVICES=0 $PY evaluate.py --config-name "$arch_cfg" \
      data_paths="[data/iclr/${data}]" data_paths_test="[data/iclr/${data}]" \
      max_len=512 digits=32 train_digits=8 \
      project_name="$PROJECT" run_name="$tag" seed="$SEED" \
      +checkpoint_folder="checkpoints/${PROJECT}/${run}" \
      +split="$split" +compile=false
  done

  echo "--- FAILURE MODE ${run} (OOD) ---"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/inspect_predictions.py \
    --checkpoint "checkpoints/${PROJECT}/${run}/step_$(ls checkpoints/${PROJECT}/${run} \
      | grep -oE 'step_[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)" \
    --data "data/iclr/${data}" --mode "$([ "$data" = ns ] && echo vanilla \
      || ([ "$data" = os_after ] && echo basicfour_concat || echo basic_concat_reverse))" \
    --split test --n 4000 --bucket 6 --examples 0
}

run_one cfg_vanilla                  ns        fair_ns
run_one cfg_basicfour_concat         os_after  fair_os_after
run_one cfg_basicfour_concat_reverse os_before fair_os_before

echo ""
echo "FAIR TEST complete for seed ${SEED}."
echo "Before comparing OOD, confirm each arm's ID curve actually flattened."
echo "If any arm is still climbing at epoch ${EPOCHS}, its OOD number is not"
echo "yet interpretable and the budget must be raised."
