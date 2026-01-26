# Tiny Recursive Models (ICML Submission)

This repository contains the code for the ICML submission "Tiny Recursive Models".

## Installation

1. Clone the repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Generation

This project uses procedural arithmetic datasets. You need to generate them before training.

### 1. BasicFour Concat Dataset
This dataset is used for the `basicfour_concat` experiment (Chain-of-Thought style serialization).
```bash
python3 dataset/build_arithmetic_dataset.py \
    --output_dir data/basicfour_concat \
    --dataset_mode basicfour_concat \
    --num_train 100000 --num_val 1000 --num_test 1000 \
    --train_max_result_digits 8 --test_max_result_digits 32 \
    --max_len 256
```

### 2. Dual Head / Vanilla Dataset
This dataset contains auxiliary supervision labels and is used for both `dual_head` (multi-task) and `vanilla` experiments.
```bash
python3 dataset/build_arithmetic_dataset.py \
    --output_dir data/arithmetic_dual \
    --dataset_mode vanilla \
    --num_train 100000 --num_val 1000 --num_test 1000 \
    --train_max_result_digits 8 --test_max_result_digits 32 \
### 3. Reverse Dataset
This dataset reverses the input operands for specific experiments (`basic_concat_reverse`).
```bash
python3 dataset/build_arithmetic_dataset.py \
    --output_dir data/reverse \
    --dataset_mode basic_concat_reverse \
    --num_train 100000 --num_val 1000 --num_test 1000 \
    --train_max_result_digits 8 --test_max_result_digits 32 \
    --max_len 256
```

## Training

### Train BasicFour Concat
```bash
python3 pretrain.py config=cfg_basicfour_concat
```

### Train Dual Head
```bash
python3 pretrain.py config=cfg_dual_head
```

### Train Vanilla
```bash
python3 pretrain.py config=cfg_vanilla
### Train Reverse
```bash
python3 pretrain.py config=cfg_basicfour_concat_reverse
```

## Evaluation

To evaluate a trained model:
```bash
python3 evaluate.py --model_path outputs/<run_name>/checkpoints/last.pt --dataset_path data/basicfour_concat/test
```
### Qwen Evaluation
To evaluate using the Qwen-based evaluation script:
```bash
python3 evaluate_qwen.py --data_dir data/reverse --verbose
```
