# Guide: Generating Evaluation Results for the Paper

This guide explains how to generate the evaluation results, tables, and figures described in your evaluation section.

## Overview

The evaluation section requires:
1. **Evaluation Metrics**: Sequence-level accuracy, digit-level accuracy, carry accuracy
2. **Main Results Table**: Comparing vanilla and Lilavati-2 on ID and OOD test sets
3. **Training Curves**: Learning curves showing convergence over training
4. **Generalization Plots**: Accuracy as a function of result digit length

## Step-by-Step Workflow

### Step 1: Train Your Models

First, ensure you have trained both models:

**Vanilla Model:**
```bash
python pretrain.py --config-name pretrain_addition \
    dataset_mode=vanilla \
    digits=4 \
    data_paths=[data/addition6_vanilla_strict_mixed_fixed] \
    data_paths_test=[data/addition6_vanilla_strict_mixed_fixed]
```

**Lilavati-2 Model:**
```bash
python pretrain.py --config-name pretrain_addition \
    dataset_mode=lilavati2 \
    digits=4 \
    data_paths=[data/addition6_lilavati2_strict_mixed_fixed] \
    data_paths_test=[data/addition6_lilavati2_strict_mixed_fixed]
```

Note: Adjust `digits` and data paths based on your actual setup. The methodology section mentions training on $d \leq 4$, so you may need to use `digits=4` or adjust accordingly.

### Step 2: Evaluate Models

You can evaluate models individually using the `evaluate.py` script:

**Evaluate Vanilla:**
```bash
python evaluate.py \
    --checkpoint checkpoints/trm-lilavati/vanilla_trm_d6 \
    --data data/addition6_vanilla_strict_mixed_fixed \
    --output-json results/vanilla_results.json
```

**Evaluate Lilavati-2:**
```bash
python evaluate.py \
    --checkpoint checkpoints/trm-lilavati/lilavati2_trm_d6 \
    --data data/addition6_lilavati2_strict_mixed_fixed \
    --output-json results/lilavati2_results.json
```

The evaluation script automatically computes:
- Sequence-level accuracy
- Digit-level accuracy
- Carry accuracy (for lilavati modes)
- Results broken down by result length (for generalization analysis)

### Step 3: Generate Tables and Figures

Use the `generate_results.py` script to create LaTeX tables and figures:

```bash
python generate_results.py \
    --checkpoint-dir checkpoints/trm-lilavati \
    --output-dir results \
    --train-max-digits 4 \
    --vanilla-checkpoint checkpoints/trm-lilavati/vanilla_trm_d6 \
    --lilavati2-checkpoint checkpoints/trm-lilavati/lilavati2_trm_d6 \
    --vanilla-data data/addition6_vanilla_strict_mixed_fixed \
    --lilavati2-data data/addition6_lilavati2_strict_mixed_fixed \
    --wandb-project trm-lilavati \
    --vanilla-wandb-run vanilla_trm_d6 \
    --lilavati2-wandb-run lilavati2_trm_d6
```

**Arguments:**
- `--checkpoint-dir`: Directory containing checkpoints
- `--output-dir`: Where to save results, tables, and figures
- `--train-max-digits`: Maximum digit length seen during training (for ID/OOD split)
- `--vanilla-checkpoint` / `--lilavati2-checkpoint`: Paths to model checkpoints
- `--vanilla-data` / `--lilavati2-data`: Paths to test datasets
- `--wandb-project`: Wandb project name
- `--vanilla-wandb-run` / `--lilavati2-wandb-run`: Wandb run names for loading training curves

**Output:**
- `results/addition_results_table.tex`: LaTeX table ready to include in your paper
- `results/figures/addition_learning_curves.pdf`: Training curves figure
- `results/figures/addition_generalization.pdf`: Length generalization plot
- `results/vanilla_results.json` and `results/lilavati2_results.json`: Full evaluation results

### Step 4: Using Existing Results

If you've already run evaluation and have JSON files, you can skip evaluation:

```bash
python generate_results.py \
    --checkpoint-dir checkpoints/trm-lilavati \
    --output-dir results \
    --train-max-digits 4 \
    --skip-eval \
    --wandb-project trm-lilavati \
    --vanilla-wandb-run vanilla_trm_d6 \
    --lilavati2-wandb-run lilavati2_trm_d6
```

## Understanding the Results

### In-Distribution vs Out-of-Distribution

The script automatically separates results into:
- **In-Distribution (ID)**: Examples where result length $\leq$ `train_max_digits`
- **Out-of-Distribution (OOD)**: Examples where result length $>$ `train_max_digits`

For addition with $d$-digit operands:
- Results have $d+1$ digit positions (accounting for overflow)
- If training on $d \leq 4$, then ID includes results with $\leq 5$ digits, OOD includes $> 5$ digits

### Metrics Explained

1. **Sequence-Level Accuracy**: Percentage of examples where the entire result sequence is correct
2. **Digit-Level Accuracy**: Average accuracy across all individual digit positions
3. **Carry Accuracy**: For Lilavati modes, accuracy of carry predictions (digit-level and sequence-level)

### Results Structure

The JSON results files contain:
```json
{
  "sequence_accuracy": 0.95,
  "overall_digit_accuracy": 0.98,
  "overall_carry_accuracy": 0.92,  // Only for lilavati modes
  "results_by_length": {
    "3": {"count": 100, "sequence_accuracy": 0.99, "digit_accuracy": 0.99},
    "4": {"count": 200, "sequence_accuracy": 0.97, "digit_accuracy": 0.98},
    "5": {"count": 150, "sequence_accuracy": 0.95, "digit_accuracy": 0.97},
    "6": {"count": 50, "sequence_accuracy": 0.85, "digit_accuracy": 0.90}  // OOD
  }
}
```

## Manual Table Generation

If you need to manually create the table, here's the structure:

1. Run evaluation on both models
2. Extract metrics from JSON files:
   - ID sequence accuracy: from `results_by_length` for lengths $\leq$ train_max_digits
   - ID digit accuracy: weighted average of digit accuracies for ID lengths
   - OOD sequence accuracy: from `results_by_length` for lengths $>$ train_max_digits
   - OOD digit accuracy: weighted average of digit accuracies for OOD lengths
   - Carry accuracy: from `overall_carry_accuracy` (Lilavati-2 only)

3. Fill in the LaTeX table template (see `generate_results.py` for the exact format)

## Manual Figure Generation

### Learning Curves

To create learning curves manually:
1. Load training data from wandb or training logs
2. Extract `val/sequence_accuracy` over training steps/epochs
3. Plot with matplotlib:
   - Vanilla: solid line
   - Lilavati-2: dashed line

### Generalization Plot

To create the generalization plot manually:
1. Extract `results_by_length` from evaluation JSON
2. Plot sequence accuracy vs result digit length
3. Add vertical line at `train_max_digits` to indicate training boundary

## Troubleshooting

### Checkpoint Not Found
- Ensure checkpoint paths are correct
- Check that checkpoints exist in the specified directory
- Use the latest checkpoint (highest step number) if multiple exist

### Dataset Path Issues
- Verify dataset paths match your actual data directories
- Ensure test sets exist in the dataset directories

### Wandb Issues
- If wandb is not available, the script will skip learning curves
- You can manually create learning curves from training logs
- Ensure wandb run names match your actual runs

### Missing Metrics
- If carry accuracy is missing, check that the model was trained with lilavati mode
- Verify that the dataset mode matches the checkpoint's training mode

## Example Workflow

Here's a complete example assuming you have:
- Vanilla checkpoint: `checkpoints/trm-lilavati/vanilla_trm_d6/step_3900`
- Lilavati-2 checkpoint: `checkpoints/trm-lilavati/lilavati2_trm_d6/step_3900`
- Datasets: `data/addition6_vanilla_strict_mixed_fixed` and `data/addition6_lilavati2_strict_mixed_fixed`

```bash
# Create results directory
mkdir -p results/figures

# Generate all results
python generate_results.py \
    --checkpoint-dir checkpoints/trm-lilavati \
    --output-dir results \
    --train-max-digits 4 \
    --vanilla-checkpoint checkpoints/trm-lilavati/vanilla_trm_d6 \
    --lilavati2-checkpoint checkpoints/trm-lilavati/lilavati2_trm_d6 \
    --vanilla-data data/addition6_vanilla_strict_mixed_fixed \
    --lilavati2-data data/addition6_lilavati2_strict_mixed_fixed \
    --wandb-project trm-lilavati \
    --vanilla-wandb-run vanilla_trm_d6 \
    --lilavati2-wandb-run lilavati2_trm_d6

# Check the generated table
cat results/addition_results_table.tex

# View figures (if you have a PDF viewer)
# evince results/figures/addition_learning_curves.pdf
# evince results/figures/addition_generalization.pdf
```

## Next Steps

1. Review the generated LaTeX table and copy it into your paper
2. Include the figures in your paper using `\includegraphics`
3. Adjust formatting as needed (e.g., decimal places, column widths)
4. Verify that the results match your expectations

## Notes

- The script assumes training on $d \leq 4$ digit operands (so results up to 5 digits are ID)
- Adjust `--train-max-digits` if your training setup differs
- For multiplication, the result length is $2d$, so adjust accordingly
- The script currently focuses on addition; multiplication can be added similarly
