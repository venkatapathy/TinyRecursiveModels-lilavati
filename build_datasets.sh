#!/bin/bash
# Helper script to build datasets for both vanilla and lilavati1 modes
# 
# Usage: ./build_datasets.sh [digits] [sample_ratio] [max_examples] [varied_length]
#   digits: Number of digits (default: 3, e.g., 3 for 000-999, 4 for 0000-9999)
#   sample_ratio: Fraction to sample 0.0-1.0 (default: all combinations)
#   max_examples: Alternative to sample_ratio - exact number of examples
#   varied_length: "true" to enable varied length (numbers can have 1 to N digits), "false" or omit for fixed length
#   
# Examples:
#   ./build_datasets.sh                          # 3 digits, all combinations, fixed length
#   ./build_datasets.sh 3 0.01                   # 3 digits, 1% sample, fixed length
#   ./build_datasets.sh 5 0.0001 true            # 5 digits, 0.01% sample, VARIED LENGTH
#   ./build_datasets.sh 4 0.001                  # 4 digits, 0.1% sample, fixed length
#   ./build_datasets.sh 3 "" 10000               # 3 digits, exactly 10K examples, fixed length

set -e

# Ensure we're in the project root and set PYTHONPATH
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Use python3 (or python if available, or venv python if activated)
if command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    PYTHON_CMD=python3
fi

# Get parameters from command line arguments
DIGITS=${1:-3}
SAMPLE_RATIO=${2:-""}
MAX_EXAMPLES=${3:-""}
VARIED_LENGTH=${4:-"false"}

# Build argument strings
DIGITS_ARG="--digits $DIGITS"
VARIED_SUFFIX=""
if [ "$VARIED_LENGTH" = "true" ]; then
    VARIED_ARG="--varied-length"
    VARIED_SUFFIX="_varied"
    echo "Using VARIED LENGTH mode (numbers can have 1 to ${DIGITS} digits)"
else
    VARIED_ARG=""
    echo "Using FIXED LENGTH mode (all numbers padded to ${DIGITS} digits)"
fi

OUTPUT_DIR_VANILLA="data/addition${DIGITS}_vanilla${VARIED_SUFFIX}"
OUTPUT_DIR_LILAVATI1="data/addition${DIGITS}_lilavati1${VARIED_SUFFIX}"

if [ -n "$SAMPLE_RATIO" ] && [ -n "$MAX_EXAMPLES" ]; then
    echo "Error: Cannot specify both sample_ratio and max_examples. Use one or the other."
    exit 1
fi

if [ -n "$SAMPLE_RATIO" ]; then
    SAMPLE_ARG="--sample-ratio $SAMPLE_RATIO"
    echo "Building ${DIGITS}-digit datasets with ${SAMPLE_RATIO} (${SAMPLE_RATIO}%) sample..."
elif [ -n "$MAX_EXAMPLES" ]; then
    SAMPLE_ARG="--max-examples $MAX_EXAMPLES"
    echo "Building ${DIGITS}-digit datasets with $MAX_EXAMPLES examples..."
else
    SAMPLE_ARG=""
    echo "Building ${DIGITS}-digit datasets with ALL combinations..."
fi

echo "Building vanilla dataset..."
$PYTHON_CMD dataset/build_addition_dataset.py \
    --output-dir $OUTPUT_DIR_VANILLA \
    --seed 42 \
    --test-ratio 0.1 \
    $DIGITS_ARG \
    --dataset-mode vanilla \
    $SAMPLE_ARG \
    $VARIED_ARG

echo "Building lilavati1 dataset..."
$PYTHON_CMD dataset/build_addition_dataset.py \
    --output-dir $OUTPUT_DIR_LILAVATI1 \
    --seed 42 \
    --test-ratio 0.1 \
    $DIGITS_ARG \
    --dataset-mode lilavati1 \
    $SAMPLE_ARG \
    $VARIED_ARG

echo "Datasets built successfully!"
