# Tiny Recursive Models (ICML Submission)

This repository contains the code for the ICML submission "Tiny Recursive Models".

## Installation

1. Clone the repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

# Tiny Recursive Models (ICML Submission)

This repository contains the code for the ICML submission "Tiny Recursive Models".

## Installation

1. Clone the repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Reproduction of Experimental Results

Follow these steps to reproduce the results presented in the paper.

### 1. Dataset Generation

Generate the four distinct procedural arithmetic datasets required for the experiments:

```bash
# 1. Vanilla Dataset (for NS models)
python3 dataset/build_arithmetic_dataset.py \
    --output_dir data/icml/vanilla \
    --dataset_mode vanilla \
    --num_train 100000 --num_val 1000 --num_test 10000 \
    --train_max_result_digits 8 --test_max_result_digits 32 \
    --seed 42

# 2. Concat Dataset (for OS-After models)
python3 dataset/build_arithmetic_dataset.py \
    --output_dir data/icml/concat \
    --dataset_mode basicfour_concat \
    --num_train 100000 --num_val 1000 --num_test 10000 \
    --train_max_result_digits 8 --test_max_result_digits 32 \
    --seed 42

# 3. Reverse Dataset (for OS-Before models)
python3 dataset/build_arithmetic_dataset.py \
    --output_dir data/icml/reverse \
    --dataset_mode basic_concat_reverse \
    --num_train 100000 --num_val 1000 --num_test 10000 \
    --train_max_result_digits 8 --test_max_result_digits 32 \
    --seed 42

# 4. Extended (100-digit) Dataset
python3 dataset/build_arithmetic_dataset.py \
    --output_dir data/reverse_100d \
    --dataset_mode basic_concat_reverse \
    --num_train 100000 --num_val 1000 --num_test 10000 \
    --train_max_result_digits 8 --test_max_result_digits 100 \
    --seed 42
```

### 2. Model Training

Train the model variants using the provided Hydra configurations. Ensure `data_paths` in the configs match your generated data directories.

| Paper Model Name | Config File | Command |
| :--- | :--- | :--- |
| **TRM (NS)** | `cfg_vanilla` | `python pretrain.py --config-name cfg_vanilla` |
| **TRM (OS-After)** | `cfg_basicfour_concat` | `python pretrain.py --config-name cfg_basicfour_concat` |
| **TRM (OS-Before)** | `cfg_basicfour_concat_reverse` | `python pretrain.py --config-name cfg_basicfour_concat_reverse` |
| **Transformer (NS)** | `cfg_transformer_300k` | `python pretrain.py --config-name cfg_transformer_300k` |
| **Transformer (OS-Before)** | `cfg_transformer_300k_concat_reverse` | `python pretrain.py --config-name cfg_transformer_300k_concat_reverse` |
| **Transformer (40x)** | `cfg_transformer_1000k` | `python pretrain.py --config-name cfg_transformer_1000k` |
| **TRM (OS-Before-100d)** | `cfg_reverse_100d` | `python pretrain.py --config-name cfg_reverse_100d` |

### 3. Evaluation

Evaluate the trained checkpoints on the OOD (test) sets.

#### TRM and Transformer Baselines
```bash
# Example for TRM (OS-Before)
python evaluate.py --config-name cfg_basicfour_concat_reverse +split=test
```

#### Large-Scale Baselines (Qwen / Gemma)
```bash
# Qwen 2.5 Math (1.5B)
python evaluate_qwen.py --data_dirs data/icml/vanilla --wandb_run_name "Qwen2.5-Math-1.5B-Instruct"

# Gemma 270M (NS)
python train_slm.py --config_path config/slm/gemma_270m_config.json --data_dir data/icml/vanilla ...
python evaluate_qwen.py --model checkpoints/slm/gemma_270m_vanilla/final --data_dirs data/icml/vanilla
```

### 4. Generating Tables and Plots

Once evaluations are complete and metrics are saved in the `results/` directory, generate the LaTeX tables and accuracy plots:

```bash
python3 scripts/generate_table.py
```

The following files will be created in the `results/` folder:
- `addition_results_table_generated.tex` (Table 1)
- `digit_wise_results_table.tex` (Table 2)
- `carry_accuracy_table.tex` (Table 3)
- `digit_wise_accuracy.png` (Accuracy vs. Problem Length plot)
