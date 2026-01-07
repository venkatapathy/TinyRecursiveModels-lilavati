"""
Build BasicFour dataset: Addition, Subtraction, Multiplication, Division.

Supports arbitrary-length operands with max result of 10^17 (17 digits).

Usage:
    python -m dataset.build_basicfour_dataset
    python -m dataset.build_basicfour_dataset --output_dir data/basicfour --num_examples 100000

Vocabulary:
    0: PAD
    1: MASK
    2-11: '0'-'9'
    12: '+'
    13: '='
    14: '-' (subtraction operator and negative sign)
    15: '*'
    16: '/'
    17: 'R' (remainder indicator)
"""

import os
import json
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from dataset.common import PuzzleDatasetMetadata

cli = ArgParser()

# Constants
MAX_RESULT = 10**17  # Maximum result value
MAX_RESULT_DIGITS = 17
SEQ_LEN = 53  # max: 17 (a) + 1 (op) + 17 (b) + 1 (=) + 17 (result) = 53

# Vocabulary
PAD_ID = 0
MASK_ID = 1
IGNORE_LABEL_ID = 255

VOCAB_MAP = {str(i): i + 2 for i in range(10)}
VOCAB_MAP['+'] = 12
VOCAB_MAP['='] = 13
VOCAB_MAP['-'] = 14
VOCAB_MAP['*'] = 15
VOCAB_MAP['/'] = 16
VOCAB_MAP['R'] = 17

VOCAB_SIZE = 18

# Operation identifiers
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_DIV = 3

OPERATIONS = ['+', '-', '*', '/']


def encode(s: str) -> list:
    """Encode a string to token IDs."""
    return [VOCAB_MAP[c] for c in s]


def generate_random_number(max_digits: int, rng: np.random.Generator) -> int:
    """Generate a random positive integer with 1 to max_digits digits."""
    num_digits = rng.integers(1, max_digits + 1)
    if num_digits == 1:
        return int(rng.integers(0, 10))
    # First digit is 1-9, rest are 0-9
    first = rng.integers(1, 10)
    rest = rng.integers(0, 10, size=num_digits - 1)
    return int(str(first) + ''.join(map(str, rest)))


def result_fits(result: int) -> bool:
    """Check if result fits within MAX_RESULT bounds."""
    return -MAX_RESULT < result < MAX_RESULT


def format_result(op: str, a: int, b: int) -> tuple:
    """
    Compute and format result for given operation.
    Returns (result_string, is_valid).
    """
    if op == '+':
        result = a + b
        if not result_fits(result):
            return None, False
        return str(result), True
    
    elif op == '-':
        result = a - b
        if not result_fits(result):
            return None, False
        # Negative results get '-' prefix
        if result < 0:
            return '-' + str(abs(result)), True
        return str(result), True
    
    elif op == '*':
        result = a * b
        if not result_fits(result):
            return None, False
        return str(result), True
    
    elif op == '/':
        if b == 0:
            return None, False
        quotient = a // b
        remainder = a % b
        if not result_fits(quotient):
            return None, False
        if remainder == 0:
            return str(quotient), True
        return f"{quotient}R{remainder}", True
    
    return None, False


def create_example(a: int, b: int, op: str, op_id: int) -> tuple:
    """
    Create a single training example.
    Returns (input_seq, label_seq, op_id) or None if invalid.
    """
    result_str, valid = format_result(op, a, b)
    if not valid:
        return None
    
    # Build input: "A op B = MASKS"
    prefix = f"{a}{op}{b}="
    result_len = len(result_str)
    
    # Check total length fits
    if len(prefix) + result_len > SEQ_LEN:
        return None
    
    # Encode
    prefix_encoded = encode(prefix)
    result_encoded = encode(result_str)
    
    # Input: prefix + MASKs + PAD
    input_seq = prefix_encoded + [MASK_ID] * result_len
    # Pad to SEQ_LEN
    pad_len = SEQ_LEN - len(input_seq)
    input_seq = input_seq + [PAD_ID] * pad_len
    
    # Label: IGNORE for prefix, result tokens, IGNORE for padding
    label_seq = [IGNORE_LABEL_ID] * len(prefix_encoded) + result_encoded + [IGNORE_LABEL_ID] * pad_len
    
    return input_seq, label_seq, op_id


class DataProcessConfig(BaseModel):
    output_dir: str = "data/basicfour"
    seed: int = 42
    test_ratio: float = 0.1
    num_examples: int = 200000  # Total examples to generate
    # Digit distribution weights (favor smaller numbers for learning)
    max_digits_add_sub: int = 17  # Max digits for addition/subtraction operands
    max_digits_mul: int = 9  # Max digits for multiplication (9*9 can give 18 digits, we limit result)
    max_digits_div_dividend: int = 17  # Max digits for dividend
    max_digits_div_divisor: int = 9  # Max digits for divisor


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    rng = np.random.default_rng(config.seed)
    
    examples = []
    
    # Target examples per operation
    examples_per_op = config.num_examples // 4
    
    print(f"Generating {config.num_examples} examples ({examples_per_op} per operation)...")
    
    # Generate Addition examples
    print("Generating addition examples...")
    count = 0
    attempts = 0
    while count < examples_per_op and attempts < examples_per_op * 10:
        attempts += 1
        a = generate_random_number(config.max_digits_add_sub, rng)
        b = generate_random_number(config.max_digits_add_sub, rng)
        example = create_example(a, b, '+', OP_ADD)
        if example:
            examples.append(example)
            count += 1
    print(f"  Generated {count} addition examples")
    
    # Generate Subtraction examples
    print("Generating subtraction examples...")
    count = 0
    attempts = 0
    while count < examples_per_op and attempts < examples_per_op * 10:
        attempts += 1
        a = generate_random_number(config.max_digits_add_sub, rng)
        b = generate_random_number(config.max_digits_add_sub, rng)
        example = create_example(a, b, '-', OP_SUB)
        if example:
            examples.append(example)
            count += 1
    print(f"  Generated {count} subtraction examples")
    
    # Generate Multiplication examples
    print("Generating multiplication examples...")
    count = 0
    attempts = 0
    while count < examples_per_op and attempts < examples_per_op * 10:
        attempts += 1
        a = generate_random_number(config.max_digits_mul, rng)
        b = generate_random_number(config.max_digits_mul, rng)
        example = create_example(a, b, '*', OP_MUL)
        if example:
            examples.append(example)
            count += 1
    print(f"  Generated {count} multiplication examples")
    
    # Generate Division examples
    print("Generating division examples...")
    count = 0
    attempts = 0
    while count < examples_per_op and attempts < examples_per_op * 10:
        attempts += 1
        a = generate_random_number(config.max_digits_div_dividend, rng)
        b = generate_random_number(config.max_digits_div_divisor, rng)
        if b == 0:
            continue
        example = create_example(a, b, '/', OP_DIV)
        if example:
            examples.append(example)
            count += 1
    print(f"  Generated {count} division examples")
    
    print(f"Total examples: {len(examples)}")
    
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
        puzzle_indices = [0]
        group_indices = [0]
        puzzle_identifiers = []
        
        for idx, (inp_seq, label_seq, op_id) in enumerate(current_examples):
            inputs.append(inp_seq)
            labels.append(label_seq)
            puzzle_identifiers.append(op_id)
            puzzle_indices.append(idx + 1)
            group_indices.append(idx + 1)
        
        # Convert to numpy
        inputs = np.array(inputs, dtype=np.uint8)
        labels = np.array(labels, dtype=np.uint8)
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
            seq_len=SEQ_LEN,
            vocab_size=VOCAB_SIZE,
            pad_id=PAD_ID,
            ignore_label_id=IGNORE_LABEL_ID,
            blank_identifier_id=0,
            num_puzzle_identifiers=4,  # 4 operations
            total_groups=len(group_indices) - 1,
            mean_puzzle_examples=1.0,
            total_puzzles=len(puzzle_indices) - 1,
            sets=["all"]
        )
        
        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
        
        print(f"Saved {split_name} split: {len(current_examples)} examples")
    
    # Save identifiers (operation names)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["addition", "subtraction", "multiplication", "division"], f)
    
    print(f"\nDataset saved to {config.output_dir}")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Test: {len(test_examples)} examples")


if __name__ == "__main__":
    cli()
