#!/bin/bash
# GATE 1: does TRM + OS-Before still extrapolate once the label leak is closed?
#
# Trains {NS, OS-After, OS-Before} x {TRM, Transformer-300k} on the
# regenerated masked-field data, then evaluates each on the ID (val) and
# OOD (test) splits through the frozen harness.
#
# Usage: scripts/run_gate1.sh [seed]
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${1:-1}"
PY=venv/bin/python
export WANDB_MODE="${WANDB_MODE:-offline}"
export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"
export OMP_NUM_THREADS=8

EPOCHS="${EPOCHS:-40}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
BS="${BS:-256}"
NPROC="${NPROC:-4}"
PROJECT=trm-iclr-gate1

run_one () {
  local arch_cfg=$1 data=$2 tag=$3
  local run="${tag}_s${SEED}"
  echo ""
  echo "=================================================================="
  echo "  TRAIN  ${run}   (arch=${arch_cfg}, data=${data})"
  echo "=================================================================="
  venv/bin/torchrun --nproc-per-node "$NPROC" --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --nnodes=1 pretrain.py \
    --config-name "$arch_cfg" \
    data_paths="[data/iclr/${data}]" data_paths_test="[data/iclr/${data}]" \
    epochs="$EPOCHS" eval_interval="$EVAL_INTERVAL" global_batch_size="$BS" \
    max_len=512 digits=32 train_digits=8 \
    project_name="$PROJECT" run_name="$tag" seed="$SEED" \
    checkpoint_every_eval=False

  for split in val test; do
    echo "--- EVAL ${run} on ${split} ---"
    $PY evaluate.py --config-name "$arch_cfg" \
      data_paths="[data/iclr/${data}]" data_paths_test="[data/iclr/${data}]" \
      max_len=512 digits=32 train_digits=8 \
      project_name="$PROJECT" run_name="$tag" seed="$SEED" \
      +checkpoint_folder="checkpoints/${PROJECT}/${run}" \
      +split="$split" +compile=false
  done
}

# TRM arm
run_one cfg_vanilla                  ns        trm_ns
run_one cfg_basicfour_concat         os_after  trm_os_after
run_one cfg_basicfour_concat_reverse os_before trm_os_before

# Transformer-300k arm
run_one cfg_transformer_300k                  ns        xf300k_ns
run_one cfg_transformer_300k_concat_reverse   os_before xf300k_os_before

echo ""
echo "GATE 1 runs complete for seed ${SEED}. Results in results/"
