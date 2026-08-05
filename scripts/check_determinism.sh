#!/bin/bash
# Verification 7: seed determinism.
#
#   (a) Two runs at the SAME seed must produce an identical step-0 loss.
#   (b) Two runs at DIFFERENT seeds must produce different run names and
#       different checkpoint directories -- the ICML code interpolated
#       neither, so multi-seed runs silently overwrote each other and
#       "mean +/- std over 3 seeds" was really one run reported three times.
#
# Uses a 1-epoch run on the smallest config; only the first logged loss is
# compared, so this is cheap.
#
# Usage: scripts/check_determinism.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_MODE=offline DISABLE_COMPILE=1 OMP_NUM_THREADS=8
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

first_loss () {   # $1 = seed, $2 = logfile
  venv/bin/torchrun --nproc-per-node 1 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
    --config-name cfg_vanilla \
    data_paths="[data/iclr/ns]" data_paths_test="[data/iclr/ns]" \
    epochs=1 eval_interval=1000 global_batch_size=256 \
    max_len=512 digits=32 train_digits=8 \
    project_name=trm-determinism run_name=det seed="$1" \
    checkpoint_every_eval=False > "$2" 2>&1 || true
  tr '\r' '\n' < "$2" | grep -oE "^STEP0_LOSS [0-9.]+" | head -1
}

echo "=== (a) same seed twice ==="
A=$(first_loss 1 "$TMP/a.log")
B=$(first_loss 1 "$TMP/b.log")
echo "  run 1 (seed 1): ${A:-<not found>}"
echo "  run 2 (seed 1): ${B:-<not found>}"
if [[ -z "$A" ]]; then
  echo "  INCONCLUSIVE: could not parse a loss from the log ($TMP/a.log)"; exit 2
fi
[[ "$A" == "$B" ]] && echo "  PASS: identical" || { echo "  FAIL: differ"; exit 1; }

echo ""
echo "=== (b) different seeds get separate identities ==="
C1=$(ls -d checkpoints/trm-determinism/det_s1 2>/dev/null || true)
first_loss 2 "$TMP/c.log" > /dev/null
C2=$(ls -d checkpoints/trm-determinism/det_s2 2>/dev/null || true)
echo "  seed 1 checkpoint dir: ${C1:-<missing>}"
echo "  seed 2 checkpoint dir: ${C2:-<missing>}"
if [[ -n "$C1" && -n "$C2" && "$C1" != "$C2" ]]; then
  echo "  PASS: distinct checkpoint directories"
else
  echo "  FAIL: seeds collide on disk"; exit 1
fi

echo ""
echo "DETERMINISM CHECK PASSED"
