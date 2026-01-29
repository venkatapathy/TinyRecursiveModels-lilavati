# for 100k

## Dataset Generation

This project uses procedural arithmetic datasets. You need to generate them before training.


### 1. Vanilla Dataset
```bash
python dataset/build_arithmetic_dataset.py \
    --output_dir data/icml/vanilla \
    --dataset_mode vanilla \
    --num_train 100000 --num_val 1000 --num_test 10000 \
    --train_max_result_digits 8 \
    --test_max_result_digits 32 \
    --max_len 256 \
    --seed 42
```

### 2. BasicFour Concat Dataset
This dataset is used for the `basicfour_concat` experiment (Chain-of-Thought style serialization).
```bash
python3 dataset/build_arithmetic_dataset.py \
    --output_dir data/icml/concat \
    --dataset_mode basicfour_concat \
    --num_train 100000 --num_val 1000 --num_test 1000 \
    --train_max_result_digits 8 \
    --test_max_result_digits 32 \
    --max_len 256 \
    --seed 42
```


    

### 3. Dual Head Dataset
This dataset contains auxiliary supervision labels and is used for both `dual_head` (multi-task) and `vanilla` experiments.
```bash
python3 dataset/build_arithmetic_dataset.py \
    --output_dir data/icml/arithmetic_dual \
    --dataset_mode vanilla \
    --num_train 100000 --num_val 1000 --num_test 1000 \
    --train_max_result_digits 8 --test_max_result_digits 32 \
    --max_len 256 \
    --seed 42
```
### 4. Reverse Dataset
This dataset reverses the input operands for specific experiments (`basic_concat_reverse`).
```bash
python3 dataset/build_arithmetic_dataset.py \
    --output_dir data/icml/reverse \
    --dataset_mode basic_concat_reverse \
    --num_train 100000 --num_val 1000 --num_test 1000 \
    --train_max_result_digits 8 --test_max_result_digits 32 \
    

## Training

### reverse
    project_name=trm-arthmetic-icml \
    data_paths=['data/icml/reverse'] \
    data_paths_test=['data/icml/reverse']
```

## vanilla
```bash
python pretrain.py \
    --config-name cfg_vanilla \
    project_name=trm-arthmetic-icml \
    data_paths=['data/icml/vanilla'] \
    data_paths_test=['data/icml/vanilla']
```

##  Concat
```bash
python pretrain.py \
    --config-name cfg_basicfour_concat \
    project_name=trm-arthmetic-icml \
    data_paths=['data/icml/concat'] \
    data_paths_test=['data/icml/concat']
```

## gemma
python3 train_slm.py \
    --config_path config/slm/gemma_270m_config.json \
    --data_dir data/icml/vanilla \
    --output_dir checkpoints/slm/gemma_270m_vanilla \
    --run_name gemma_270m_vanilla_icml \
    --wandb_project trm-arithmetic-icml

# Evaluation
## qwen
```bash
./venv/bin/python evaluate_qwen.py \
    --data_dirs data/icml/vanilla \
    --verbose \
    --wandb_project trm-icml-eval \
    --wandb_run_name "Qwen2.5-Math-1.5B-Instruct-Vanilla"
```

## gemma
```bash
./venv/bin/python evaluate_qwen.py \
    --model checkpoints/slm/gemma_270m_vanilla/final \
    --data_dirs data/icml/vanilla \
    --verbose \
    --wandb_project trm-icml-eval \
    --wandb_run_name "Gemma-270M-Vanilla-SLM"
```

## TRM Vanilla
```bash
# OOD Evaluation (Test Split)
./venv/bin/python evaluate.py \
    --config-name cfg_vanilla \
    project_name=trm-icml-eval \
    +split=test

# ID Evaluation (Val Split)
./venv/bin/python evaluate.py \
    --config-name cfg_vanilla \
    project_name=trm-icml-eval \
    +split=val
```

## TRM Concat
```bash
# OOD Evaluation (Test Split)
./venv/bin/python evaluate.py --config-name cfg_basicfour_concat project_name=trm-icml-eval +split=test +checkpoint_folder=checkpoints/trm-arthmetic-icml/basicfour_concat

# ID Evaluation (Val Split)
./venv/bin/python evaluate.py \
    --config-name cfg_basicfour_concat \
    project_name=trm-icml-eval \
    +split=val
```

## TRM Reverse
```bash
# OOD Evaluation (Test Split)
./venv/bin/python evaluate.py \
    --config-name cfg_basicfour_concat_reverse \
    project_name=trm-icml-eval \
    +split=test

# ID Evaluation (Val Split)
./venv/bin/python evaluate.py \
    --config-name cfg_basicfour_concat_reverse \
    project_name=trm-icml-eval \
    
```

### Transformer
## concat
```bash
python evaluate.py \
    --config-name cfg_transformer_300k_concat_reverse \
    project_name=trm-icml-eval \
    +split=test \
    +checkpoint_folder=checkpoints/trm-arthmetic-icml/baseline_transformer_300k_concat_reverse
```
##reverse
```bash
python evaluate.py \
    --config-name cfg_transformer_300k \
    project_name=trm-arthmetic-icml \
    +split=test \
    +checkpoint_folder=checkpoints/trm-arthmetic-icml/baseline_transformer_300k
```


# Result Compilation
python3 scripts/generate_table.py 