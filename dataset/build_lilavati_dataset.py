"""
Build Lilavati Pati dataset for Neural Abacus model.

Features:
- Reversed digit order (LSB first) - matches how humans compute addition
- Ground truth carry bits for auxiliary supervision
- Columnar alignment with fixed-width padding

Usage:
    python -m dataset.build_lilavati_dataset
    python -m dataset.build_lilavati_dataset --output_dir data/lilavati --max_digits 24

Vocabulary:
    0: PAD
    1: MASK (for unknown answer digits)
    2-11: '0'-'9'
    12: '+'
    13: '='

Example:
    Standard: "123+456=579"
    Lilavati: "321+654=975" (digits reversed)
    Carries:  [0, 0, 0, 0, ...] (carry bits per position)
"""

import os
import json
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from dataset.common import PuzzleDatasetMetadata

cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/lilavati"
    seed: int = 42
    test_ratio: float = 0.1
    num_examples: int = 100000  # Total examples to generate
    max_digits: int = 24  # Max digits per operand (result can be max_digits + 1)


# Constants
PAD_ID = 0
MASK_ID = 1
IGNORE_LABEL_ID = 255
VOCAB_SIZE = 14  # 0-13

VOCAB_MAP = {str(i): i + 2 for i in range(10)}
VOCAB_MAP['+'] = 12
VOCAB_MAP['='] = 13


def encode(s: str) -> list:
    """Encode a string to token IDs."""
    return [VOCAB_MAP[c] for c in s]


def reverse_digits(s: str) -> str:
    """Reverse the digits in a number string."""
    return s[::-1]


def compute_carries(a: int, b: int) -> list:
    """
    Compute carry bits for addition a + b.
    Returns list of carry bits (0 or 1) for each digit position.
    Position 0 is the least significant digit.
    """
    s_a = str(a)
    s_b = str(b)
    max_len = max(len(s_a), len(s_b))
    
    # Pad to same length
    s_a = s_a.zfill(max_len)
    s_b = s_b.zfill(max_len)
    
    carries = []
    carry = 0
    
    # Process right-to-left (LSB first)
    for i in range(max_len - 1, -1, -1):
        d_a = int(s_a[i])
        d_b = int(s_b[i])
        total = d_a + d_b + carry
        carry = 1 if total >= 10 else 0
        carries.append(carry)
    
    # Final carry (if result has extra digit)
    carries.append(carry)
    
    return carries


def generate_random_number(max_digits: int, rng: np.random.Generator) -> int:
    """Generate a random positive integer with 1 to max_digits digits."""
    # Weighted sampling: favor smaller numbers for curriculum learning
    weights = np.exp(-np.arange(1, max_digits + 1) * 0.1)
    weights = weights / weights.sum()
    num_digits = rng.choice(np.arange(1, max_digits + 1), p=weights)
    
    if num_digits == 1:
        return int(rng.integers(0, 10))
    
    # First digit is 1-9, rest are 0-9
    first = rng.integers(1, 10)
    rest = rng.integers(0, 10, size=num_digits - 1)
    return int(str(first) + ''.join(map(str, rest)))


def compute_seq_len(max_digits: int) -> int:
    """Compute sequence length for given max operand digits."""
    # Format: A + B = C (all reversed)
    # A: max_digits, +: 1, B: max_digits, =: 1, C: max_digits + 1
    return max_digits + 1 + max_digits + 1 + (max_digits + 1)


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    rng = np.random.default_rng(config.seed)
    
    seq_len = compute_seq_len(config.max_digits)
    max_result = 10 ** (config.max_digits + 1)
    max_result_digits = config.max_digits + 1
    
    print(f"Generating Lilavati Pati dataset:")
    print(f"  Max operand digits: {config.max_digits}")
    print(f"  Max result digits: {max_result_digits}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of examples: {config.num_examples}")
    print(f"  Features: Reversed digits, carry supervision")
    
    examples = []
    
    # Generate examples
    print("Generating examples...")
    count = 0
    attempts = 0
    max_attempts = config.num_examples * 10
    
    while count < config.num_examples and attempts < max_attempts:
        attempts += 1
        a = generate_random_number(config.max_digits, rng)
        b = generate_random_number(config.max_digits, rng)
        result = a + b
        
        # Check result fits
        if result >= max_result:
            continue
        
        # Compute carries (before reversing)
        carries = compute_carries(a, b)
        
        # Reverse digits for Lilavati format
        s_a = reverse_digits(str(a))
        s_b = reverse_digits(str(b))
        s_res = reverse_digits(str(result))
        
        prefix = s_a + "+" + s_b + "="
        result_len = len(s_res)
        
        # Check total length fits
        if len(prefix) + result_len > seq_len:
            continue
        
        # Encode
        prefix_encoded = encode(prefix)
        result_encoded = encode(s_res)
        
        # Input: prefix + MASKs + PAD
        inp_seq = prefix_encoded + [MASK_ID] * result_len
        pad_len = seq_len - len(inp_seq)
        inp_seq = inp_seq + [PAD_ID] * pad_len
        
        # Label: IGNORE for prefix, result tokens, IGNORE for padding
        label_seq = [IGNORE_LABEL_ID] * len(prefix_encoded) + result_encoded + [IGNORE_LABEL_ID] * pad_len
        
        # Carries: pad to max_result_digits
        carries_padded = carries[:max_result_digits]
        while len(carries_padded) < max_result_digits:
            carries_padded.append(0)
        
        examples.append((inp_seq, label_seq, carries_padded))
        count += 1
        
        if count % 10000 == 0:
            print(f"  Generated {count}/{config.num_examples} examples")
    
    print(f"Generated {len(examples)} examples")
    
    # Shuffle and split
    rng.shuffle(examples)
    split_idx = int(len(examples) * (1 - config.test_ratio))
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]
    
    splits = {"train": train_examples, "test": test_examples}
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    for split_name, current_examples in splits.items():
        inputs = []
        labels = []
        carries_list = []
        puzzle_indices = [0]
        group_indices = [0]
        puzzle_identifiers = []
        
        for idx, (inp_seq, label_seq, carries) in enumerate(current_examples):
            inputs.append(inp_seq)
            labels.append(label_seq)
            carries_list.append(carries)
            puzzle_identifiers.append(0)  # 0 is blank identifier
            puzzle_indices.append(idx + 1)
            group_indices.append(idx + 1)
        
        # Convert to numpy
        inputs = np.array(inputs, dtype=np.uint8)
        labels = np.array(labels, dtype=np.uint8)
        carries_arr = np.array(carries_list, dtype=np.uint8)
        puzzle_indices = np.array(puzzle_indices, dtype=np.int32)
        group_indices = np.array(group_indices, dtype=np.int32)
        puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
        
        # Save
        save_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(os.path.join(save_dir, "all__inputs.npy"), inputs)
        np.save(os.path.join(save_dir, "all__labels.npy"), labels)
        np.save(os.path.join(save_dir, "all__carries.npy"), carries_arr)  # Extra: carry supervision
        np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(save_dir, "all__group_indices.npy"), group_indices)
        np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
        
        metadata = PuzzleDatasetMetadata(
            seq_len=seq_len,
            vocab_size=VOCAB_SIZE,
            pad_id=PAD_ID,
            ignore_label_id=IGNORE_LABEL_ID,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=len(group_indices) - 1,
            mean_puzzle_examples=1.0,
            total_puzzles=len(puzzle_indices) - 1,
            sets=["all"]
        )
        
        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
        
        print(f"Saved {split_name} split: {len(current_examples)} examples")
    
    # Identifiers
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    # Save additional metadata for Lilavati
    lilavati_meta = {
        "format": "lilavati_pati",
        "reversed_digits": True,
        "carry_supervision": True,
        "max_result_digits": max_result_digits,
    }
    with open(os.path.join(config.output_dir, "lilavati_meta.json"), "w") as f:
        json.dump(lilavati_meta, f, indent=2)
    
    print(f"\nDataset saved to {config.output_dir}")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Test: {len(test_examples)} examples")
    print(f"  Carries array shape: {carries_arr.shape}")


if __name__ == "__main__":
    cli()
