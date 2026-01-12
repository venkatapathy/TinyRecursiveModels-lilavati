#!/usr/bin/env python3
"""
Script to inspect varied and unique datapoints from an addition dataset.
Shows examples with different characteristics (small/large numbers, different carry patterns, etc.)
"""

import os
import sys
import json
import numpy as np
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def inspect_dataset(data_dir: str, num_examples: int = 20):
    """Inspect dataset and show varied examples."""
    
    # Load metadata
    train_metadata_path = os.path.join(data_dir, "train", "dataset.json")
    if not os.path.exists(train_metadata_path):
        print(f"Error: Dataset not found at {data_dir}")
        return
    
    with open(train_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    digits = None
    dataset_mode = None
    
    # Try to infer from directory name or metadata
    if "addition" in data_dir:
        parts = data_dir.split("_")
        for part in parts:
            if part.startswith("d") and part[1:].isdigit():
                digits = int(part[1:])
            if part in ["vanilla", "lilavati1"]:
                dataset_mode = part
    
    # Also check if we can infer from vocab_size
    if metadata['vocab_size'] >= 15:
        dataset_mode = "lilavati1"
    elif dataset_mode is None:
        dataset_mode = "vanilla"
    
    # Try to infer digits from sequence length if not found
    if digits is None:
        # For vanilla: seq_len = digits*2 + 1 (for +) + 1 (for =) + (digits+1) for result
        # For lilavati1: seq_len = digits*2 + 1 + 1 + (digits+1) + 1 (CAR) + digits
        seq_len = metadata['seq_len']
        if dataset_mode == "vanilla":
            # seq_len = 2*digits + 2 + (digits+1) = 3*digits + 3
            # digits = (seq_len - 3) / 3
            digits = (seq_len - 3) // 3
        else:
            # seq_len = 2*digits + 2 + (digits+1) + 1 + digits = 4*digits + 4
            # digits = (seq_len - 4) / 4
            digits = (seq_len - 4) // 4
    
    # Load data
    inputs_path = os.path.join(data_dir, "train", "all__inputs.npy")
    labels_path = os.path.join(data_dir, "train", "all__labels.npy")
    
    if not os.path.exists(inputs_path) or not os.path.exists(labels_path):
        print(f"Error: Data files not found")
        return
    
    inputs = np.load(inputs_path, mmap_mode='r')
    labels = np.load(labels_path, mmap_mode='r')
    
    # Vocabulary mapping
    vocab_map_inv = {i+2: str(i) for i in range(10)}
    vocab_map_inv[12] = '+'
    vocab_map_inv[13] = '='
    if metadata['vocab_size'] >= 15:
        vocab_map_inv[14] = '<CAR>'
    vocab_map_inv[0] = 'PAD'
    vocab_map_inv[1] = 'MASK'
    
    def decode(seq):
        res = ""
        for token in seq:
            token = int(token)
            if token in vocab_map_inv:
                res += vocab_map_inv[token]
            else:
                res += "?"
        return res
    
    # Sample diverse examples - use more samples to check uniqueness
    total_examples = len(inputs)
    num_samples = min(num_examples * 10, total_examples)  # Sample more to check uniqueness
    sample_indices = np.linspace(0, total_examples - 1, num_samples, dtype=int)
    
    print(f"\n{'='*80}")
    print(f"Dataset Inspection: {data_dir}")
    print(f"{'='*80}")
    print(f"Total examples: {total_examples:,}")
    print(f"Sequence length: {metadata['seq_len']}")
    print(f"Vocabulary size: {metadata['vocab_size']}")
    print(f"Digits: {digits if digits else 'unknown'}")
    print(f"Mode: {dataset_mode if dataset_mode else 'unknown'}")
    print(f"{'='*80}\n")
    
    # Collect examples with different characteristics
    examples = []
    all_results = []
    all_carries_list = []
    unique_pairs = set()  # Track unique (a, b) pairs
    
    for idx in sample_indices:
        inp = inputs[idx]
        lab = labels[idx]
        
        # Decode
        input_str = decode(inp)
        label_str = decode(lab)
        
        # Parse
        if '=' not in input_str:
            continue
        
        lhs_str = input_str.split('=')[0]
        if '+' not in lhs_str:
            continue
        
        try:
            a_str, b_str = lhs_str.split('+')
            a = int(a_str)
            b = int(b_str)
            res = a + b
            
            # Extract carries from label if lilavati1 mode
            carries = None
            if dataset_mode == "lilavati1" and digits:
                # Find <CAR> token in labels
                car_token_id = 14
                car_idx = None
                for j, tok in enumerate(lab):
                    if tok == car_token_id:
                        car_idx = j
                        break
                
                if car_idx is not None:
                    # Extract carry digits after <CAR>
                    carry_start = car_idx + 1
                    carry_end = carry_start + digits
                    carry_tokens = lab[carry_start:carry_end]
                    carry_str = decode(carry_tokens)
                    # Convert to list of integers
                    carries = [int(c) for c in carry_str if c.isdigit()]
                    if len(carries) == digits:
                        all_carries_list.append(carries)
                    else:
                        # Fallback to computing
                        carries = compute_carries(a, b, digits)
                        all_carries_list.append(carries)
                else:
                    # Fallback to computing
                    carries = compute_carries(a, b, digits)
                    all_carries_list.append(carries)
            
            all_results.append(res)
            unique_pairs.add((a, b))
            
            examples.append({
                'idx': int(idx),
                'a': a,
                'b': b,
                'result': res,
                'input_str': input_str,
                'label_str': label_str,
                'carries': carries
            })
        except Exception as e:
            continue
    
    # Uniqueness analysis
    print(f"Uniqueness Analysis:")
    print(f"  Total examples analyzed: {len(examples):,}")
    print(f"  Unique (a, b) pairs: {len(unique_pairs):,}")
    print(f"  Duplicate rate: {(1 - len(unique_pairs)/len(examples))*100:.2f}%")
    
    # Result diversity
    unique_results = len(set(all_results))
    print(f"  Unique results: {unique_results:,} out of {len(all_results):,} examples")
    print(f"  Result diversity: {unique_results/len(all_results)*100:.2f}%")
    
    if all_carries_list:
        unique_carry_patterns = len(set(tuple(c) for c in all_carries_list))
        print(f"  Unique carry patterns: {unique_carry_patterns:,} out of {len(all_carries_list):,}")
        print(f"  Carry pattern diversity: {unique_carry_patterns/len(all_carries_list)*100:.2f}%")
    
    print()
    
    # Show statistics
    print(f"Result Statistics:")
    print(f"  Min result: {min(all_results):,}")
    print(f"  Max result: {max(all_results):,}")
    print(f"  Mean result: {np.mean(all_results):.2f}")
    print(f"  Median result: {np.median(all_results):.2f}")
    
    if all_carries_list:
        flat_carries = [c for carries in all_carries_list for c in carries]
        carry_counter = Counter(flat_carries)
        print(f"\nCarry Statistics:")
        print(f"  Carry=0: {carry_counter[0]:,} ({carry_counter[0]/len(flat_carries)*100:.1f}%)")
        print(f"  Carry=1: {carry_counter[1]:,} ({carry_counter[1]/len(flat_carries)*100:.1f}%)")
    
    # Show diverse examples
    print(f"\n{'='*80}")
    print(f"Sample Examples (showing variety):")
    print(f"{'='*80}\n")
    
    # Sort by different criteria to show variety
    examples_by_result = sorted(examples, key=lambda x: x['result'])
    examples_by_sum = sorted(examples, key=lambda x: x['a'] + x['b'])
    
    print("1. Examples with smallest results:")
    for ex in examples_by_result[:5]:
        if dataset_mode == "lilavati1" and ex['carries']:
            carries_str = ''.join(str(c) for c in ex['carries'])
            print(f"   {ex['a']:05d} + {ex['b']:05d} = {ex['result']:06d} <CAR> {carries_str} (carries: {ex['carries']})")
        else:
            print(f"   {ex['a']:05d} + {ex['b']:05d} = {ex['result']:06d}")
    
    print("\n2. Examples with largest results:")
    for ex in examples_by_result[-5:]:
        if dataset_mode == "lilavati1" and ex['carries']:
            carries_str = ''.join(str(c) for c in ex['carries'])
            print(f"   {ex['a']:05d} + {ex['b']:05d} = {ex['result']:06d} <CAR> {carries_str} (carries: {ex['carries']})")
        else:
            print(f"   {ex['a']:05d} + {ex['b']:05d} = {ex['result']:06d}")
    
    if dataset_mode == "lilavati1" and all_carries_list:
        # Examples with interesting carry patterns
        print("\n3. Examples with interesting carry patterns:")
        
        # Find examples with all carries = 1
        all_ones = [ex for ex in examples if ex['carries'] and all(c == 1 for c in ex['carries'])]
        if all_ones:
            ex = all_ones[0]
            carries_str = ''.join(str(c) for c in ex['carries'])
            print(f"   All carries = 1: {ex['a']:05d} + {ex['b']:05d} = {ex['result']:06d} <CAR> {carries_str}")
        
        # Find examples with all carries = 0
        all_zeros = [ex for ex in examples if ex['carries'] and all(c == 0 for c in ex['carries'])]
        if all_zeros:
            ex = all_zeros[0]
            carries_str = ''.join(str(c) for c in ex['carries'])
            print(f"   All carries = 0: {ex['a']:05d} + {ex['b']:05d} = {ex['result']:06d} <CAR> {carries_str}")
        
        # Find examples with mixed carries
        mixed = [ex for ex in examples if ex['carries'] and any(c == 0 for c in ex['carries']) and any(c == 1 for c in ex['carries'])]
        if mixed:
            ex = mixed[0]
            carries_str = ''.join(str(c) for c in ex['carries'])
            print(f"   Mixed carries: {ex['a']:05d} + {ex['b']:05d} = {ex['result']:06d} <CAR> {carries_str}")
    
    print("\n4. Random sample of examples:")
    np.random.seed(42)
    random_sample = np.random.choice(len(examples), min(5, len(examples)), replace=False)
    for idx in random_sample:
        ex = examples[idx]
        if dataset_mode == "lilavati1" and ex['carries']:
            carries_str = ''.join(str(c) for c in ex['carries'])
            print(f"   {ex['a']:05d} + {ex['b']:05d} = {ex['result']:06d} <CAR> {carries_str}")
        else:
            print(f"   {ex['a']:05d} + {ex['b']:05d} = {ex['result']:06d}")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect addition dataset examples")
    parser.add_argument("data_dir", help="Path to dataset directory (e.g., data/addition5_vanilla)")
    parser.add_argument("--num-examples", type=int, default=50, help="Number of examples to analyze")
    args = parser.parse_args()
    
    inspect_dataset(args.data_dir, args.num_examples)
