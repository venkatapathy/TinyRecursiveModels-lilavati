#!/bin/bash
set -e

# Config
EPOCHS=3
BS=16
ACC=4
LR=2e-4
MAX_LEN=256

echo "Starting SLM Experiments..."

# 1. Vanilla
echo "----------------------------------------------------------------"
echo "Running VANILLA Experiment"
venv/bin/python train_slm.py \
    --data_dir data/icml/vanilla \
    --output_dir checkpoints/slm/vanilla \
    --run_name gemma_270m_vanilla \
    --epochs $EPOCHS \
    --batch_size $BS \
    --grad_acc $ACC \
    --lr $LR \
    --max_seq_len $MAX_LEN

echo "Evaluating VANILLA"
venv/bin/python evaluate_qwen.py \
    --model checkpoints/slm/vanilla/final \
    --data_dirs data/icml/vanilla/test \
    --wandb_project trm-arthmetic-icml \
    --wandb_run_name eval_gemma_vanilla


# 2. Concat
echo "----------------------------------------------------------------"
echo "Running CONCAT Experiment"
venv/bin/python train_slm.py \
    --data_dir data/icml/concat \
    --output_dir checkpoints/slm/concat \
    --run_name gemma_270m_concat \
    --epochs $EPOCHS \
    --batch_size $BS \
    --grad_acc $ACC \
    --lr $LR \
    --max_seq_len $MAX_LEN

echo "Evaluating CONCAT"
venv/bin/python evaluate_qwen.py \
    --model checkpoints/slm/concat/final \
    --data_dirs data/icml/concat/test \
    --wandb_project trm-arthmetic-icml \
    --wandb_run_name eval_gemma_concat


# 3. Reverse
echo "----------------------------------------------------------------"
echo "Running REVERSE Experiment"
venv/bin/python train_slm.py \
    --data_dir data/icml/reverse \
    --output_dir checkpoints/slm/reverse \
    --run_name gemma_270m_reverse \
    --epochs $EPOCHS \
    --batch_size $BS \
    --grad_acc $ACC \
    --lr $LR \
    --max_seq_len $MAX_LEN

echo "Evaluating REVERSE"
venv/bin/python evaluate_qwen.py \
    --model checkpoints/slm/reverse/final \
    --data_dirs data/icml/reverse/test \
    --wandb_project trm-arthmetic-icml \
    --wandb_run_name eval_gemma_reverse

echo "All experiments completed."
