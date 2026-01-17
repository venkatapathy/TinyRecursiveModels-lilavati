# Quick Start: Generating Results

## Quick Command

After training your models, run:

```bash
# Evaluate both models and generate all results
# Note: Lilavati-2 uses the same data format as Lilavati-1
python generate_results.py \
    --vanilla-checkpoint checkpoints/trm-lilavati/vanilla_trm_d6 \
    --lilavati2-checkpoint checkpoints/trm-lilavati/lilavati2_trm_d6 \
    --vanilla-data data/addition6_vanilla_strict_mixed_fixed \
    --lilavati2-data data/addition6_lilavati1_strict_mixed_fixed \
    --wandb-project trm-lilavati \
    --vanilla-wandb-run vanilla_trm_d6 \
    --lilavati2-wandb-run lilavati2_trm_d6 \
    --output-dir results \
    --train-max-digits 4
```

## What You Get

1. **`results/addition_results_table.tex`** - LaTeX table ready to paste into your paper
2. **`results/figures/addition_learning_curves.pdf`** - Training curves figure
3. **`results/figures/addition_generalization.pdf`** - Length generalization plot

## Manual Evaluation (Alternative)

If you prefer to evaluate separately:

```bash
# Evaluate vanilla
python evaluate.py \
    --checkpoint checkpoints/trm-lilavati/vanilla_trm_d6 \
    --data data/addition6_vanilla_strict_mixed_fixed \
    --output-json results/vanilla_results.json

# Evaluate Lilavati-2
python evaluate.py \
    --checkpoint checkpoints/trm-lilavati/lilavati2_trm_d6 \
    --data data/addition6_lilavati2_strict_mixed_fixed \
    --output-json results/lilavati2_results.json

# Then generate tables/figures (skip evaluation)
python generate_results.py \
    --skip-eval \
    --wandb-project trm-lilavati \
    --vanilla-wandb-run vanilla_trm_d6 \
    --lilavati2-wandb-run lilavati2_trm_d6 \
    --output-dir results \
    --train-max-digits 4
```

## Understanding the Output

- **ID (In-Distribution)**: Results with length ≤ 5 digits (for 4-digit operand training)
- **OOD (Out-of-Distribution)**: Results with length > 5 digits
- **Sequence Accuracy**: Percentage of fully correct results
- **Digit Accuracy**: Average accuracy across all digit positions
- **Carry Accuracy**: Only for Lilavati-2, accuracy of carry predictions

See `GENERATING_RESULTS.md` for detailed documentation.
