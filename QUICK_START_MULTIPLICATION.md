# Quick Start: Generating Multiplication Results

## Quick Command

After training your models, run:

```bash
# Evaluate both models and generate all results
python generate_results_multiplication.py \
    --vanilla-checkpoint checkpoints/trm-lilavati-multiply/vanilla_trm_d3 \
    --lilavati1-checkpoint checkpoints/trm-lilavati-multiply/lilavati1_trm_d3 \
    --vanilla-data data/multiplication3_vanilla \
    --lilavati1-data data/multiplication3_lilavati1 \
    --wandb-project trm-lilavati-multiply \
    --vanilla-wandb-run vanilla_trm_d3 \
    --lilavati1-wandb-run lilavati1_trm_d3 \
    --output-dir results_multiplication \
    --train-max-digits 3
```

## What You Get

1. **`results_multiplication/multiplication_results_table.tex`** - LaTeX table ready to paste into your paper
2. **`results_multiplication/figures/multiplication_learning_curves.pdf`** - Training curves figure
3. **`results_multiplication/figures/multiplication_generalization.pdf`** - Length generalization plot

## Key Differences from Addition

- **Result Length**: For multiplication with $d$-digit operands, results have $2d$ digits (not $d+1$)
- **ID/OOD Split**: If training on 3-digit operands, results up to 6 digits are ID, 7+ digits are OOD
- **Default Digits**: Multiplication typically uses `--train-max-digits 3` (vs 4 for addition)

## Understanding the Output

- **ID (In-Distribution)**: Results with length $\leq 2d$ digits (for $d$-digit operand training)
- **OOD (Out-of-Distribution)**: Results with length $> 2d$ digits
- **Sequence Accuracy**: Percentage of fully correct results
- **Digit Accuracy**: Average accuracy across all digit positions
- **Carry Accuracy**: Only for Lilavati-1, accuracy of carry predictions

## Note on Length Breakdown

The current `MultiplicationEvaluator` doesn't track results by length automatically. The script will use overall metrics for ID/OOD separation. If you need length-specific breakdown, you may need to modify the evaluator to track results by length similar to the addition evaluator.
