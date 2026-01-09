import os
import json
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from dataset.common import PuzzleDatasetMetadata
from collections import Counter

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/addition"
    seed: int = 42
    test_ratio: float = 0.1
    digits: int = 3  # Number of digits for addition (e.g., 3 for 000-999, 4 for 0000-9999)
    dataset_mode: str = "vanilla"  # "vanilla" or "lilavati1"
    max_examples: int = None  # Maximum total examples. If None, uses all combinations or sample_ratio. If set, randomly samples this many pairs.
    sample_ratio: float = None  # Fraction of total combinations to sample (0.0 to 1.0). If None and max_examples is None, uses all combinations.

@cli.command(singleton=True)
def main(config: DataProcessConfig):
    assert config.dataset_mode in {"vanilla", "lilavati1"}, f"dataset_mode must be 'vanilla' or 'lilavati1', got {config.dataset_mode}"
    assert config.digits >= 1, f"digits must be >= 1, got {config.digits}"
    assert config.sample_ratio is None or 0.0 < config.sample_ratio <= 1.0, f"sample_ratio must be in (0.0, 1.0], got {config.sample_ratio}"
    
    np.random.seed(config.seed)
    
    # Generate all pairs for N-digit addition (000-999 for 3 digits, 0000-9999 for 4 digits, etc.)
    max_val = 10 ** config.digits - 1
    all_pairs = []
    for i in range(max_val + 1):
        for j in range(max_val + 1):
            all_pairs.append((i, j))
    
    total_possible = len(all_pairs)
    
    # Determine how many examples to use
    if config.max_examples is not None and config.max_examples > 0:
        # Use max_examples if specified
        if config.max_examples > total_possible:
            print(f"Warning: max_examples ({config.max_examples:,}) is larger than total combinations ({total_possible:,}). Using all {total_possible:,} examples.")
            sample_size = total_possible
        else:
            sample_size = config.max_examples
    elif config.sample_ratio is not None and 0.0 < config.sample_ratio <= 1.0:
        # Use percentage-based sampling
        sample_size = int(total_possible * config.sample_ratio)
        if sample_size == 0:
            sample_size = 1  # At least 1 example
        print(f"Sampling {sample_size:,} examples ({config.sample_ratio*100:.2f}%) from {total_possible:,} possible combinations")
    else:
        # Use all combinations
        sample_size = total_possible
        print(f"Using all {total_possible:,} combinations")
    
    # Sample if needed
    if sample_size < total_possible:
        # Randomly sample without replacement
        indices = np.random.choice(total_possible, size=sample_size, replace=False)
        pairs = [all_pairs[i] for i in indices]
    else:
        pairs = all_pairs
    
    np.random.shuffle(pairs)
    
    split_idx = int(len(pairs) * (1 - config.test_ratio))
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    splits = {"train": train_pairs, "test": test_pairs}
    
    # Vocabulary
    # 0: PAD
    # 1: MASK (for unknown answer digits)
    # 2: '0'
    # ...
    # 11: '9'
    # 12: '+'
    # 13: '='
    # 14: '<CAR>' (only for lilavati1 mode)
    
    vocab_map = {str(i): i + 2 for i in range(10)}
    vocab_map['+'] = 12
    vocab_map['='] = 13
    
    # Add <CAR> token for lilavati1 mode
    if config.dataset_mode == "lilavati1":
        vocab_map['<CAR>'] = 14
        vocab_size = 15  # 0..14
        CAR_TOKEN_ID = 14
    else:
        vocab_size = 14  # 0..13
        CAR_TOKEN_ID = None
    
    MASK_ID = 1
    IGNORE_LABEL_ID = 255
    
    def encode(s):
        return [vocab_map[c] for c in s]
    
    def compute_carries(a: int, b: int, digits: int) -> list:
        """Compute carry digits from least significant to most significant."""
        carries = []
        carry = 0
        for pos in range(digits):
            a_digit = (a // (10 ** pos)) % 10
            b_digit = (b // (10 ** pos)) % 10
            total = a_digit + b_digit + carry
            carry = total // 10
            carries.append(carry)
        return carries  # [units_carry, tens_carry, hundreds_carry, ...]
    
    # Store max_seq_len for stats
    max_seq_len = None
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    for split_name, current_pairs in splits.items():
        inputs = []
        labels = []
        puzzle_indices = [0]
        group_indices = [0]
        puzzle_identifiers = []
        
        # Format for 3-digit: XXX+YYY=ZZZZ or XXX+YYY=ZZZ <CAR> CCC
        # Input: XXX+YYY=MMMM (M=Mask) or XXX+YYY=MMMM <CAR> MMM for lilavati1
        # Label: IIIIIIIIIZZZZ (I=Ignore) or IIIIIIIIIZZZ <CAR> CCC for lilavati1
        
        for idx, (a, b) in enumerate(current_pairs):
            res = a + b
            s_a = f"{a:0{config.digits}d}"
            s_b = f"{b:0{config.digits}d}"
            
            # Result can have config.digits or config.digits+1 digits (overflow)
            max_result_digits = config.digits + 1
            s_res = f"{res:0{max_result_digits}d}"
            
            prefix = s_a + "+" + s_b + "="
            
            if config.dataset_mode == "vanilla":
                # Input: Prefix + MASKs for result
                inp_seq = encode(prefix) + [MASK_ID] * max_result_digits
                
                # Label: Ignore Prefix + Result digits
                label_seq = [IGNORE_LABEL_ID] * len(encode(prefix)) + encode(s_res)
                
                # Sanity check: No <CAR> in targets
                assert CAR_TOKEN_ID is None or CAR_TOKEN_ID not in label_seq, "Vanilla mode should not contain <CAR> token"
                
            else:  # lilavati1
                # Compute carries
                carries = compute_carries(a, b, config.digits)
                s_carries = ''.join(str(c) for c in carries)  # Least to most significant
                
                # Input: Prefix + MASKs for result + <CAR> + MASKs for carries
                inp_seq = encode(prefix) + [MASK_ID] * max_result_digits + [CAR_TOKEN_ID] + [MASK_ID] * config.digits
                
                # Label: Ignore Prefix + Result digits + <CAR> + Carry digits
                label_seq = [IGNORE_LABEL_ID] * len(encode(prefix)) + encode(s_res) + [CAR_TOKEN_ID] + encode(s_carries)
                
                # Sanity checks
                assert label_seq.count(CAR_TOKEN_ID) == 1, f"Lilavati-1 mode must have exactly one <CAR> token, got {label_seq.count(CAR_TOKEN_ID)}"
                carry_start_idx = label_seq.index(CAR_TOKEN_ID) + 1
                carry_digits = label_seq[carry_start_idx:carry_start_idx + config.digits]
                assert len(carry_digits) == config.digits, f"Carry length must be {config.digits}, got {len(carry_digits)}"
                # After encoding, '0' -> 2, '1' -> 3, so valid carry token IDs are in {2, 3}
                valid_carry_ids = {vocab_map['0'], vocab_map['1']}
                assert all(cid in valid_carry_ids for cid in carry_digits), f"All carry digits must be 0 or 1 (encoded as {valid_carry_ids}), got {set(carry_digits)}"
            
            inputs.append(inp_seq)
            labels.append(label_seq)
            
            puzzle_identifiers.append(0) # 0 is blank identifier
            puzzle_indices.append(idx + 1)
            group_indices.append(idx + 1)
        
        # Convert to numpy
        # Pad all sequences to same length
        split_max_seq_len = max(len(inp) for inp in inputs)
        if max_seq_len is None:
            max_seq_len = split_max_seq_len
        else:
            max_seq_len = max(max_seq_len, split_max_seq_len)
        inputs_padded = []
        labels_padded = []
        for inp, lab in zip(inputs, labels):
            assert len(inp) == len(lab), f"Input and label lengths must match: {len(inp)} != {len(lab)}"
            pad_len = split_max_seq_len - len(inp)
            inputs_padded.append(inp + [0] * pad_len)  # Pad with PAD_ID (0)
            labels_padded.append(lab + [IGNORE_LABEL_ID] * pad_len)  # Pad with IGNORE
        
        inputs = np.array(inputs_padded, dtype=np.uint8)
        labels = np.array(labels_padded, dtype=np.uint8)
        puzzle_indices = np.array(puzzle_indices, dtype=np.int32)
        group_indices = np.array(group_indices, dtype=np.int32)
        puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
        
        # Save
        save_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(os.path.join(save_dir, "all__inputs.npy"), inputs)
        np.save(os.path.join(save_dir, "all__labels.npy"), labels)
        np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(save_dir, "all__group_indices.npy"), group_indices)
        np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
        
        metadata = PuzzleDatasetMetadata(
            seq_len=split_max_seq_len,
            vocab_size=vocab_size,
            pad_id=0,
            ignore_label_id=IGNORE_LABEL_ID,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=len(group_indices)-1,
            mean_puzzle_examples=1.0,
            total_puzzles=len(puzzle_indices)-1,
            sets=["all"]
        )
        
        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
            
    # Identifiers
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    # Generate dataset statistics and README
    generate_dataset_stats(config, splits, vocab_map, vocab_size, max_seq_len, total_possible)


def generate_dataset_stats(config: DataProcessConfig, splits: dict, vocab_map: dict, vocab_size: int, max_seq_len: int, total_possible: int):
    """Generate a README.md file with dataset statistics and information."""
    
    # Compute statistics
    total_examples = sum(len(pairs) for pairs in splits.values())
    train_examples = len(splits["train"])
    test_examples = len(splits["test"])
    
    # Sample a few examples for display
    train_sample = splits["train"][:5]
    test_sample = splits["test"][:3]
    
    def compute_carries(a: int, b: int, digits: int) -> list:
        """Compute carry digits from least significant to most significant."""
        carries = []
        carry = 0
        for pos in range(digits):
            a_digit = (a // (10 ** pos)) % 10
            b_digit = (b // (10 ** pos)) % 10
            total = a_digit + b_digit + carry
            carry = total // 10
            carries.append(carry)
        return carries
    
    # Build vocabulary description
    vocab_desc = []
    vocab_desc.append("- `0`: PAD")
    vocab_desc.append("- `1`: MASK (for unknown answer digits)")
    for i in range(10):
        vocab_desc.append(f"- `{i+2}`: '{i}'")
    vocab_desc.append("- `12`: '+'")
    vocab_desc.append("- `13`: '='")
    if config.dataset_mode == "lilavati1":
        vocab_desc.append("- `14`: '<CAR>' (carry token)")
    
    # Generate examples
    example_lines = []
    for a, b in train_sample[:3]:
        res = a + b
        s_a = f"{a:0{config.digits}d}"
        s_b = f"{b:0{config.digits}d}"
        max_result_digits = config.digits + 1
        s_res = f"{res:0{max_result_digits}d}"
        
        input_str = f"{s_a}+{s_b}="
        
        if config.dataset_mode == "vanilla":
            output_str = s_res
            example_lines.append(f"- Input: `{input_str}MMMM` → Output: `{output_str}`")
            example_lines.append(f"  - Example: `{s_a}+{s_b}={s_res}`")
        else:
            carries = compute_carries(a, b, config.digits)
            s_carries = ''.join(str(c) for c in carries)
            output_str = f"{s_res} <CAR> {s_carries}"
            example_lines.append(f"- Input: `{input_str}MMMM <CAR> MMM` → Output: `{output_str}`")
            example_lines.append(f"  - Example: `{s_a}+{s_b}={s_res} <CAR> {s_carries}` (carries: {carries})")
    
    # Compute result distribution statistics
    all_results = []
    all_carries = []
    for split_name, pairs in splits.items():
        for a, b in pairs:
            res = a + b
            all_results.append(res)
            if config.dataset_mode == "lilavati1":
                carries = compute_carries(a, b, config.digits)
                all_carries.extend(carries)
    
    result_counter = Counter(all_results)
    min_result = min(all_results)
    max_result = max(all_results)
    mean_result = np.mean(all_results)
    median_result = np.median(all_results)
    
    # Count results with different numbers of digits
    digit_counts = Counter(len(str(r)) for r in all_results)
    
    # Build README content
    readme_content = f"""# Addition Dataset - {config.dataset_mode.upper()} Mode ({config.digits}-digit)

## Dataset Information

### Configuration
- **Mode**: {config.dataset_mode}
- **Digits**: {config.digits}
- **Seed**: {config.seed}
- **Test Ratio**: {config.test_ratio}
- **Sampling**: {f"{config.max_examples:,} examples" if config.max_examples is not None else (f"{config.sample_ratio*100:.2f}%" if config.sample_ratio is not None else "All combinations")}
- **Total Possible Combinations**: {total_possible:,}

### Dataset Statistics
- **Total Examples**: {total_examples:,}
- **Training Examples**: {train_examples:,} ({train_examples/total_examples*100:.1f}%)
- **Test Examples**: {test_examples:,} ({test_examples/total_examples*100:.1f}%)
- **Sequence Length**: {max_seq_len}
- **Vocabulary Size**: {vocab_size}
- **Coverage**: {total_examples:,} / {total_possible:,} possible combinations ({total_examples/total_possible*100:.2f}%)

### Result Distribution Statistics
- **Minimum Result**: {min_result}
- **Maximum Result**: {max_result:,}
- **Mean Result**: {mean_result:.2f}
- **Median Result**: {median_result:.2f}
- **Result Digit Counts**:
  - {digit_counts.get(config.digits, 0):,} results with {config.digits} digits ({digit_counts.get(config.digits, 0)/total_examples*100:.1f}%)
  - {digit_counts.get(config.digits+1, 0):,} results with {config.digits+1} digits ({digit_counts.get(config.digits+1, 0)/total_examples*100:.1f}%)
"""
    
    if config.dataset_mode == "lilavati1":
        carry_counter = Counter(all_carries)
        readme_content += f"""
### Carry Distribution Statistics
- **Total Carry Predictions**: {len(all_carries):,}
- **Carry = 0**: {carry_counter[0]:,} ({carry_counter[0]/len(all_carries)*100:.1f}%)
- **Carry = 1**: {carry_counter[1]:,} ({carry_counter[1]/len(all_carries)*100:.1f}%)
"""
    
    min_num = '0' * config.digits
    max_num = 10 ** config.digits - 1
    result_digits_min = config.digits
    result_digits_max = config.digits + 1
    
    readme_content += f"""
## Format Specification

### Input Format
All examples follow the format: `{('X' * config.digits)}+{('Y' * config.digits)}=` where {('X' * config.digits)} and {('Y' * config.digits)} are {config.digits}-digit numbers ({min_num}-{max_num}).

### Output Format
"""
    
    if config.dataset_mode == "vanilla":
        readme_content += f"""
**Vanilla Mode**: Only result digits are predicted.

- Format: `{'R' * result_digits_max}` where result has {result_digits_min} or {result_digits_max} digits (depending on overflow)
- Example: For {config.digits}-digit addition, results can be {config.digits} or {config.digits+1} digits
  - `{('1' * (config.digits-1))}27+{('3' * (config.digits-1))}89={('5' * (config.digits-1))}16` ({config.digits} digits)
  - `{('9' * config.digits)}+001={('1' * config.digits)}000` ({config.digits+1} digits)
"""
    else:
        readme_content += f"""
**Lilavati-1 Mode**: Result digits followed by carry trace after `<CAR>` token.

- Format: `{'R' * result_digits_max} <CAR> {'C' * config.digits}` where:
  - Result has {result_digits_min} or {result_digits_max} digits (depending on overflow)
  - `<CAR>` is a special token
  - Carry trace has {config.digits} digits: units→tens, tens→hundreds, ..., {config.digits-1}s→{config.digits}s
- Each carry digit is either 0 or 1
- Carries are ordered from least significant to most significant
- Example: `127+389=516 <CAR> 101` (for 3-digit) means:
  - Result: 516
  - Units→Tens carry: 1
  - Tens→Hundreds carry: 0
  - Hundreds→Thousands carry: 1
"""
    
    readme_content += f"""
### Vocabulary
The dataset uses the following vocabulary mapping:

{chr(10).join(vocab_desc)}

## Examples

### Training Examples (sample)
"""
    
    readme_content += "\n".join(example_lines)
    
    readme_content += f"""

### Test Examples (sample)
"""
    
    for a, b in test_sample:
        res = a + b
        s_a = f"{a:0{config.digits}d}"
        s_b = f"{b:0{config.digits}d}"
        max_result_digits = config.digits + 1
        s_res = f"{res:0{max_result_digits}d}"
        input_str = f"{s_a}+{s_b}="
        
        if config.dataset_mode == "vanilla":
            output_str = s_res
            readme_content += f"- `{input_str}MMMM` → `{output_str}`\n"
        else:
            carries = compute_carries(a, b, config.digits)
            s_carries = ''.join(str(c) for c in carries)
            output_str = f"{s_res} <CAR> {s_carries}"
            readme_content += f"- `{input_str}MMMM <CAR> MMM` → `{output_str}`\n"
    
    readme_content += f"""
## File Structure

```
{config.output_dir}/
├── train/
│   ├── all__inputs.npy          # Input sequences (shape: [N, {max_seq_len}])
│   ├── all__labels.npy          # Label sequences (shape: [N, {max_seq_len}])
│   ├── all__puzzle_indices.npy  # Puzzle indices
│   ├── all__group_indices.npy   # Group indices
│   ├── all__puzzle_identifiers.npy  # Puzzle identifiers
│   └── dataset.json             # Dataset metadata
├── test/
│   └── [same structure as train/]
├── identifiers.json             # Identifier mapping
└── README.md                    # This file
```

## Usage in Training

This dataset is designed for training Tiny Recursive Models (TRM) on {config.digits}-digit addition.

### Metrics
- **Digit Accuracy**: Per-digit accuracy over all result positions
- **Sequence Accuracy**: Exact match accuracy (all digits correct)
"""
    
    if config.dataset_mode == "lilavati1":
        readme_content += "- **Carry Accuracy**: Accuracy of carry predictions\n"
    
    readme_content += f"""
### Citation
If you use this dataset in your research, please cite:

```
Addition Dataset - {config.dataset_mode} Mode ({config.digits}-digit)
Dataset Mode: {config.dataset_mode}
Digits: {config.digits}
Total Examples: {total_examples:,}
Generated with seed: {config.seed}
```
"""
    
    # Write README
    readme_path = os.path.join(config.output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    print(f"Generated dataset statistics: {readme_path}")


if __name__ == "__main__":
    cli()

