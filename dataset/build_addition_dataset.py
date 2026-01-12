import os
import json
import numpy as np
from typing import Optional
from argdantic import ArgParser
from pydantic import BaseModel
from dataset.common import PuzzleDatasetMetadata
from collections import Counter

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/addition"
    seed: int = 42
    test_ratio: float = 0.1
    digits: int = 3  # Maximum number of digits for addition (e.g., 3 for up to 999, 4 for up to 9999)
    dataset_mode: str = "vanilla"  # "vanilla", "lilavati1", or "lilavati2"
    max_examples: Optional[int] = None  # Maximum total examples. If None, uses all combinations or sample_ratio. If set, randomly samples this many pairs.
    sample_ratio: Optional[float] = None  # Fraction of total combinations to sample (0.0 to 1.0). If None and max_examples is None, uses all combinations.
    varied_length: bool = False  # If True, allows numbers with different lengths (e.g., 2-digit + 3-digit). If False, all numbers are padded to 'digits' length.
    min_carries: Optional[int] = None  # Minimum number of carries (carry=1) required in an example (works for both vanilla and lilavati1 modes)
    min_carry_ratio: Optional[float] = None  # Minimum ratio of carries that are 1 (0.0 to 1.0, works for both vanilla and lilavati1 modes)
    train_max_digits: Optional[int] = None  # Maximum digit length in train set (e.g., 5 means train only has up to 5-digit examples)
    also_output_dir: Optional[str] = None  # If set, also generate the other mode (vanilla/lilavati1) to this directory

@cli.command(singleton=True)
def main(config: DataProcessConfig):
    assert config.dataset_mode in {"vanilla", "lilavati1", "lilavati2"}, f"dataset_mode must be 'vanilla', 'lilavati1', or 'lilavati2', got {config.dataset_mode}"
    assert config.digits >= 1, f"digits must be >= 1, got {config.digits}"
    assert config.sample_ratio is None or 0.0 < config.sample_ratio <= 1.0, f"sample_ratio must be in (0.0, 1.0], got {config.sample_ratio}"
    assert config.min_carries is None or config.min_carries >= 0, f"min_carries must be >= 0, got {config.min_carries}"
    assert config.min_carry_ratio is None or 0.0 <= config.min_carry_ratio <= 1.0, f"min_carry_ratio must be in [0.0, 1.0], got {config.min_carry_ratio}"
    assert config.train_max_digits is None or config.train_max_digits >= 1, f"train_max_digits must be >= 1, got {config.train_max_digits}"
    assert config.train_max_digits is None or config.train_max_digits < config.digits, f"train_max_digits ({config.train_max_digits}) must be < digits ({config.digits})"
    
    np.random.seed(config.seed)
    
    # Define helper functions for filtering (need to be defined early)
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
    
    def filter_by_carries(a: int, b: int, digits: int, min_carries: int = None, min_carry_ratio: float = None) -> bool:
        """Filter pairs by carry criteria (works for both vanilla and lilavati1 modes)."""
        if min_carries is None and min_carry_ratio is None:
            return True  # No filtering
        
        carries = compute_carries(a, b, digits)
        num_carries = sum(carries)  # Count how many carries are 1
        
        # Check minimum number of carries
        if min_carries is not None and num_carries < min_carries:
            return False
        
        # Check minimum carry ratio
        if min_carry_ratio is not None:
            carry_ratio = num_carries / digits
            if carry_ratio < min_carry_ratio:
                return False
        
        return True
    
    def filter_by_digit_length(a: int, b: int, max_digits: int) -> bool:
        """Filter pairs by maximum digit length."""
        max_len = max(len(str(a)), len(str(b)))
        return max_len <= max_digits
    
    # Calculate total possible combinations for N-digit addition
    max_val = 10 ** config.digits - 1
    total_possible = (max_val + 1) ** 2
    
    # If max_examples is set, it means target number of TRAINING examples after filtering
    # Use rejection sampling to get enough valid examples
    target_train_samples = None
    if config.max_examples is not None and config.max_examples > 0:
        target_train_samples = config.max_examples
        print(f"Target: {target_train_samples:,} training examples (test will be {int(target_train_samples * config.test_ratio):,})")
    
    # Determine how many examples to use for initial sampling
    if target_train_samples is not None:
        # Estimate: assume ~0.8% pass all filters (based on observed ratios)
        # Sample more to account for filtering
        estimated_accept_rate = 0.008 if config.train_max_digits is not None else 0.85
        sample_size = int(target_train_samples / estimated_accept_rate * 1.5)  # 1.5x buffer
        sample_size = min(sample_size, total_possible)
        print(f"Sampling {sample_size:,} pairs (estimated {estimated_accept_rate*100:.1f}% will pass filters)")
    elif config.sample_ratio is not None and 0.0 < config.sample_ratio <= 1.0:
        # Use percentage-based sampling
        sample_size = int(total_possible * config.sample_ratio)
        if sample_size == 0:
            sample_size = 1  # At least 1 example
        print(f"Sampling {sample_size:,} examples ({config.sample_ratio*100:.6f}%) from {total_possible:,} possible combinations")
    else:
        # Use all combinations
        sample_size = total_possible
        print(f"Using all {total_possible:,} combinations")
    
    # Generate pairs efficiently: only generate what we need
    # For large datasets, we can't generate all pairs in memory first
    # Threshold: if total_possible > 100M, use direct sampling
    if sample_size < total_possible and total_possible > 100_000_000:
        # For very large datasets (like 5-digit), sample directly without generating all pairs
        print(f"Using memory-efficient direct sampling for large dataset...")
        rng = np.random.default_rng(config.seed)
        
        # Generate random pairs directly
        pairs_set = set()
        batch_size = min(100000, sample_size * 2)  # Generate in batches
        attempts = 0
        max_attempts = sample_size * 10  # Safety limit
        
        while len(pairs_set) < sample_size and attempts < max_attempts:
            a_vals = rng.integers(0, max_val + 1, size=batch_size, dtype=np.int32)
            b_vals = rng.integers(0, max_val + 1, size=batch_size, dtype=np.int32)
            
            # Add pairs to set (automatically handles duplicates)
            new_pairs = [(int(a), int(b)) for a, b in zip(a_vals, b_vals)]
            pairs_set.update(new_pairs)
            attempts += batch_size
            
            if attempts % 1000000 == 0:
                print(f"  Generated {len(pairs_set):,} unique pairs so far...")
        
        # Convert to list and trim to exact sample_size
        pairs = list(pairs_set)[:sample_size]
        if len(pairs) < sample_size:
            print(f"Warning: Only generated {len(pairs):,} unique pairs out of requested {sample_size:,}")
    elif sample_size < total_possible:
        # For smaller datasets, we can still generate indices and sample
        # But for medium datasets, use direct pair generation with sampling
        print(f"Generating random sample using index-based sampling...")
        rng = np.random.default_rng(config.seed)
        # Sample random indices without generating all pairs
        indices = rng.choice(total_possible, size=sample_size, replace=False)
        # Convert indices back to (a, b) pairs
        pairs = []
        for idx in indices:
            a = idx // (max_val + 1)
            b = idx % (max_val + 1)
            pairs.append((a, b))
    else:
        # Only generate all pairs if we need all of them (smaller datasets only)
        print(f"Generating all {total_possible:,} pairs...")
        all_pairs = []
        for i in range(max_val + 1):
            for j in range(max_val + 1):
                all_pairs.append((i, j))
        pairs = all_pairs
    
    # Apply carry filtering (works for both vanilla and lilavati1 modes)
    if config.min_carries is not None or config.min_carry_ratio is not None:
        print(f"Applying carry filtering (min_carries={config.min_carries}, min_carry_ratio={config.min_carry_ratio})...")
        original_count = len(pairs)
        
        # Filter in batches with progress updates for large datasets
        if original_count > 1_000_000:
            print(f"Filtering {original_count:,} pairs in batches...")
            batch_size = 1_000_000
            filtered_pairs = []
            for i in range(0, original_count, batch_size):
                batch = pairs[i:i + batch_size]
                filtered_batch = [p for p in batch if filter_by_carries(p[0], p[1], config.digits, config.min_carries, config.min_carry_ratio)]
                filtered_pairs.extend(filtered_batch)
                if (i + batch_size) % 10_000_000 == 0 or i + batch_size >= original_count:
                    print(f"  Processed {min(i + batch_size, original_count):,} / {original_count:,} pairs ({len(filtered_pairs):,} passed so far)...")
            pairs = filtered_pairs
        else:
            pairs = [p for p in pairs if filter_by_carries(p[0], p[1], config.digits, config.min_carries, config.min_carry_ratio)]
        
        filtered_count = len(pairs)
        print(f"Filtered {original_count:,} pairs to {filtered_count:,} pairs ({filtered_count/original_count*100:.2f}% retained)")
        if filtered_count == 0:
            raise ValueError(f"No pairs passed the carry filtering criteria!")
    
    # Apply train_max_digits logic: filter ≤train_max_digits, split 90/10, then sample ~1% of >train_max_digits
    if config.train_max_digits is not None:
        print(f"Applying train_max_digits filtering (train_max_digits={config.train_max_digits})...")
        # Separate pairs into trainable (≤train_max_digits) and extra (>train_max_digits)
        trainable_pairs = [p for p in pairs if filter_by_digit_length(p[0], p[1], config.train_max_digits)]
        extra_pairs = [p for p in pairs if not filter_by_digit_length(p[0], p[1], config.train_max_digits)]
        print(f"Separated {len(trainable_pairs):,} trainable pairs (≤{config.train_max_digits} digits) and {len(extra_pairs):,} extra pairs (>{config.train_max_digits} digits)")
        if len(trainable_pairs) == 0:
            raise ValueError(f"No pairs passed the train_max_digits filtering!")
        
        # If target_train_samples is set, use rejection sampling to get enough trainable examples
        if target_train_samples is not None and len(trainable_pairs) < target_train_samples * 1.2:  # Need extra for 90/10 split
            print(f"Using batch rejection sampling to reach {target_train_samples:,} trainable examples...")
            rng = np.random.default_rng(config.seed)
            trainable_set = set(trainable_pairs)
            attempts = 0
            max_rejection_attempts = target_train_samples * 300  # Safety limit
            batch_size = 100000  # Process in batches
            
            while len(trainable_set) < target_train_samples * 1.2 and attempts < max_rejection_attempts:
                # Generate batch of pairs
                batch_attempts = min(batch_size, max_rejection_attempts - attempts)
                a_vals = rng.integers(0, max_val + 1, size=batch_attempts, dtype=np.int32)
                b_vals = rng.integers(0, max_val + 1, size=batch_attempts, dtype=np.int32)
                
                # Filter batch
                for a, b in zip(a_vals, b_vals):
                    attempts += 1
                    # Check if pair passes all filters AND is trainable
                    if filter_by_carries(a, b, config.digits, config.min_carries, config.min_carry_ratio):
                        if filter_by_digit_length(a, b, config.train_max_digits):
                            trainable_set.add((a, b))
                
                if attempts % 1000000 == 0:
                    print(f"  Batch rejection sampling: {len(trainable_set):,} trainable pairs after {attempts:,} attempts...")
            
            trainable_pairs = list(trainable_set)
            print(f"Rejection sampling complete: {len(trainable_pairs):,} trainable pairs")
        
        # Split trainable pairs 90/10
        np.random.shuffle(trainable_pairs)
        if target_train_samples is not None:
            # Ensure we get exactly target_train_samples in train (or as close as possible)
            split_idx = min(target_train_samples, int(len(trainable_pairs) * (1 - config.test_ratio)))
        else:
            split_idx = int(len(trainable_pairs) * (1 - config.test_ratio))
        train_pairs = trainable_pairs[:split_idx]
        test_pairs = trainable_pairs[split_idx:]
        
        # Sample ~1% of extra pairs to add to test (to keep overall ratio ~90/10)
        if len(extra_pairs) > 0:
            np.random.shuffle(extra_pairs)
            sample_size = max(1, int(len(train_pairs) * config.test_ratio * 0.1))  # ~1% of train set
            sample_size = min(sample_size, len(extra_pairs))
            sampled_extra = extra_pairs[:sample_size]
            test_pairs.extend(sampled_extra)
            print(f"Split trainable pairs: {len(train_pairs):,} train, {len(test_pairs) - len(sampled_extra):,} test")
            print(f"Added {len(sampled_extra):,} sampled extra pairs (>{config.train_max_digits} digits) to test (final test size: {len(test_pairs):,})")
    else:
        # No train_max_digits filtering - split all pairs normally
        # If target_train_samples is set, use rejection sampling to get enough examples
        if target_train_samples is not None and len(pairs) < target_train_samples * 1.2:
            print(f"Using batch rejection sampling to reach {target_train_samples:,} examples...")
            rng = np.random.default_rng(config.seed)
            pairs_set = set(pairs)
            attempts = 0
            max_rejection_attempts = target_train_samples * 200
            batch_size = 100000  # Process in batches
            
            while len(pairs_set) < target_train_samples * 1.2 and attempts < max_rejection_attempts:
                # Generate batch of pairs
                batch_attempts = min(batch_size, max_rejection_attempts - attempts)
                a_vals = rng.integers(0, max_val + 1, size=batch_attempts, dtype=np.int32)
                b_vals = rng.integers(0, max_val + 1, size=batch_attempts, dtype=np.int32)
                
                # Filter batch
                for a, b in zip(a_vals, b_vals):
                    attempts += 1
                    if filter_by_carries(a, b, config.digits, config.min_carries, config.min_carry_ratio):
                        pairs_set.add((a, b))
                
                if attempts % 1000000 == 0:
                    print(f"  Batch rejection sampling: {len(pairs_set):,} valid pairs after {attempts:,} attempts...")
            
            pairs = list(pairs_set)
            print(f"Rejection sampling complete: {len(pairs):,} valid pairs")
        
        np.random.shuffle(pairs)
        if target_train_samples is not None:
            split_idx = min(target_train_samples, int(len(pairs) * (1 - config.test_ratio)))
        else:
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
    
    # Add <CAR> token for lilavati1/lilavati2 mode
    if config.dataset_mode in {"lilavati1", "lilavati2"}:
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
    
    # Store max_seq_len for stats
    max_seq_len = None
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # If varied_length, regenerate pairs with different digit lengths
    if config.varied_length:
        print(f"Generating varied-length examples (numbers can have 1 to {config.digits} digits)...")
        # Regenerate pairs with varied lengths
        varied_pairs = []
        rng = np.random.default_rng(config.seed)
        
        # For varied_length, max_examples means target number of TRAINING examples after filtering
        # Determine initial sampling size
        if target_train_samples is not None:
            # Estimate: assume ~0.8% pass all filters for varied length too
            estimated_accept_rate = 0.008 if config.train_max_digits is not None else 0.85
            target_size = int(target_train_samples / estimated_accept_rate * 1.5)  # 1.5x buffer
            print(f"Initial sampling target: {target_size:,} pairs for varied-length (estimated {estimated_accept_rate*100:.1f}% will pass filters)")
        elif config.sample_ratio is not None and 0.0 < config.sample_ratio <= 1.0:
            # Estimate: for varied length, we have more combinations
            # Rough estimate: sum of all combinations from 1-digit to N-digit
            estimated_total = sum((10**d) ** 2 for d in range(1, config.digits + 1))
            target_size = int(estimated_total * config.sample_ratio)
            if target_size == 0:
                target_size = 1
        else:
            # Generate all combinations for all digit lengths
            estimated_total = sum((10**d) ** 2 for d in range(1, config.digits + 1))
            target_size = estimated_total
        
        # Generate varied-length pairs
        pairs_set = set()
        max_attempts = target_size * 20
        
        while len(pairs_set) < target_size and len(pairs_set) < max_attempts:
            # Randomly choose digit lengths for a and b (1 to config.digits)
            a_digits = rng.integers(1, config.digits + 1)
            b_digits = rng.integers(1, config.digits + 1)
            
            # Generate random numbers with those digit lengths
            a_min = 10 ** (a_digits - 1) if a_digits > 1 else 0
            a_max = 10 ** a_digits - 1
            b_min = 10 ** (b_digits - 1) if b_digits > 1 else 0
            b_max = 10 ** b_digits - 1
            
            a = rng.integers(a_min, a_max + 1)
            b = rng.integers(b_min, b_max + 1)
            
            pairs_set.add((a, b))
        
        varied_pairs = list(pairs_set)[:target_size]
        print(f"Generated {len(varied_pairs):,} varied-length pairs")
        
        # Apply carry filtering (works for both vanilla and lilavati1 modes)
        if config.min_carries is not None or config.min_carry_ratio is not None:
            print(f"Applying carry filtering (min_carries={config.min_carries}, min_carry_ratio={config.min_carry_ratio})...")
            original_count = len(varied_pairs)
            
            # Filter in batches with progress updates for large datasets
            if original_count > 1_000_000:
                print(f"Filtering {original_count:,} pairs in batches...")
                batch_size = 1_000_000
                filtered_pairs = []
                for i in range(0, original_count, batch_size):
                    batch = varied_pairs[i:i + batch_size]
                    filtered_batch = [p for p in batch if filter_by_carries(p[0], p[1], config.digits, config.min_carries, config.min_carry_ratio)]
                    filtered_pairs.extend(filtered_batch)
                    if (i + batch_size) % 10_000_000 == 0 or i + batch_size >= original_count:
                        print(f"  Processed {min(i + batch_size, original_count):,} / {original_count:,} pairs ({len(filtered_pairs):,} passed so far)...")
                varied_pairs = filtered_pairs
            else:
                varied_pairs = [p for p in varied_pairs if filter_by_carries(p[0], p[1], config.digits, config.min_carries, config.min_carry_ratio)]
            
            filtered_count = len(varied_pairs)
            print(f"Filtered {original_count:,} pairs to {filtered_count:,} pairs ({filtered_count/original_count*100:.2f}% retained)")
            if filtered_count == 0:
                raise ValueError(f"No pairs passed the carry filtering criteria!")
        
        # Apply train_max_digits logic: filter ≤train_max_digits, split 90/10, then sample ~1% of >train_max_digits
        if config.train_max_digits is not None:
            print(f"Applying train_max_digits filtering (train_max_digits={config.train_max_digits})...")
            # Separate pairs into trainable (≤train_max_digits) and extra (>train_max_digits)
            trainable_pairs = [p for p in varied_pairs if filter_by_digit_length(p[0], p[1], config.train_max_digits)]
            extra_pairs = [p for p in varied_pairs if not filter_by_digit_length(p[0], p[1], config.train_max_digits)]
            print(f"Separated {len(trainable_pairs):,} trainable pairs (≤{config.train_max_digits} digits) and {len(extra_pairs):,} extra pairs (>{config.train_max_digits} digits)")
            if len(trainable_pairs) == 0:
                raise ValueError(f"No pairs passed the train_max_digits filtering!")
            
            # If target_train_samples is set, use rejection sampling to get enough trainable examples
            if target_train_samples is not None and len(trainable_pairs) < target_train_samples * 1.2:
                print(f"Using batch rejection sampling to reach {target_train_samples:,} trainable examples...")
                trainable_set = set(trainable_pairs)
                attempts = 0
                max_rejection_attempts = target_train_samples * 300
                batch_size = 100000  # Process in batches
                
                while len(trainable_set) < target_train_samples * 1.2 and attempts < max_rejection_attempts:
                    # Generate batch of pairs
                    batch_attempts = min(batch_size, max_rejection_attempts - attempts)
                    # Randomly choose digit lengths for a and b (1 to config.train_max_digits)
                    a_digits_vals = rng.integers(1, config.train_max_digits + 1, size=batch_attempts)
                    b_digits_vals = rng.integers(1, config.train_max_digits + 1, size=batch_attempts)
                    
                    # Generate random numbers with those digit lengths
                    a_vals = []
                    b_vals = []
                    for a_digits, b_digits in zip(a_digits_vals, b_digits_vals):
                        a_min = 10 ** (a_digits - 1) if a_digits > 1 else 0
                        a_max = 10 ** a_digits - 1
                        b_min = 10 ** (b_digits - 1) if b_digits > 1 else 0
                        b_max = 10 ** b_digits - 1
                        a_vals.append(rng.integers(a_min, a_max + 1))
                        b_vals.append(rng.integers(b_min, b_max + 1))
                    
                    # Filter batch
                    for a, b in zip(a_vals, b_vals):
                        attempts += 1
                        # Check if pair passes all filters AND is trainable
                        if filter_by_carries(a, b, config.digits, config.min_carries, config.min_carry_ratio):
                            if filter_by_digit_length(a, b, config.train_max_digits):
                                trainable_set.add((a, b))
                    
                    if attempts % 1000000 == 0:
                        print(f"  Batch rejection sampling: {len(trainable_set):,} trainable pairs after {attempts:,} attempts...")
                
                trainable_pairs = list(trainable_set)
                print(f"Rejection sampling complete: {len(trainable_pairs):,} trainable pairs")
            
            # Split trainable pairs 90/10
            np.random.shuffle(trainable_pairs)
            if target_train_samples is not None:
                # Ensure we get exactly target_train_samples in train (or as close as possible)
                split_idx = min(target_train_samples, int(len(trainable_pairs) * (1 - config.test_ratio)))
            else:
                split_idx = int(len(trainable_pairs) * (1 - config.test_ratio))
            train_pairs = trainable_pairs[:split_idx]
            test_pairs = trainable_pairs[split_idx:]
            
            # Sample ~1% of extra pairs to add to test (to keep overall ratio ~90/10)
            if len(extra_pairs) > 0:
                np.random.shuffle(extra_pairs)
                sample_size = max(1, int(len(train_pairs) * config.test_ratio * 0.1))  # ~1% of train set
                sample_size = min(sample_size, len(extra_pairs))
                sampled_extra = extra_pairs[:sample_size]
                test_pairs.extend(sampled_extra)
                print(f"Split trainable pairs: {len(train_pairs):,} train, {len(test_pairs) - len(sampled_extra):,} test")
                print(f"Added {len(sampled_extra):,} sampled extra pairs (>{config.train_max_digits} digits) to test (final test size: {len(test_pairs):,})")
        else:
            # No train_max_digits filtering - split all pairs normally
            # If target_train_samples is set, use rejection sampling to get enough examples
            if target_train_samples is not None and len(varied_pairs) < target_train_samples * 1.2:
                print(f"Using batch rejection sampling to reach {target_train_samples:,} examples...")
                pairs_set = set(varied_pairs)
                attempts = 0
                max_rejection_attempts = target_train_samples * 200
                batch_size = 100000  # Process in batches
                
                while len(pairs_set) < target_train_samples * 1.2 and attempts < max_rejection_attempts:
                    # Generate batch of pairs
                    batch_attempts = min(batch_size, max_rejection_attempts - attempts)
                    # Randomly choose digit lengths for a and b (1 to config.digits)
                    a_digits_vals = rng.integers(1, config.digits + 1, size=batch_attempts)
                    b_digits_vals = rng.integers(1, config.digits + 1, size=batch_attempts)
                    
                    # Generate random numbers with those digit lengths
                    a_vals = []
                    b_vals = []
                    for a_digits, b_digits in zip(a_digits_vals, b_digits_vals):
                        a_min = 10 ** (a_digits - 1) if a_digits > 1 else 0
                        a_max = 10 ** a_digits - 1
                        b_min = 10 ** (b_digits - 1) if b_digits > 1 else 0
                        b_max = 10 ** b_digits - 1
                        a_vals.append(rng.integers(a_min, a_max + 1))
                        b_vals.append(rng.integers(b_min, b_max + 1))
                    
                    # Filter batch
                    for a, b in zip(a_vals, b_vals):
                        attempts += 1
                        if filter_by_carries(a, b, config.digits, config.min_carries, config.min_carry_ratio):
                            pairs_set.add((a, b))
                    
                    if attempts % 1000000 == 0:
                        print(f"  Batch rejection sampling: {len(pairs_set):,} valid pairs after {attempts:,} attempts...")
                
                varied_pairs = list(pairs_set)
                print(f"Rejection sampling complete: {len(varied_pairs):,} valid pairs")
            
            np.random.shuffle(varied_pairs)
            if target_train_samples is not None:
                split_idx = min(target_train_samples, int(len(varied_pairs) * (1 - config.test_ratio)))
            else:
                split_idx = int(len(varied_pairs) * (1 - config.test_ratio))
            train_pairs = varied_pairs[:split_idx]
            test_pairs = varied_pairs[split_idx:]
        
        splits = {
            "train": train_pairs,
            "test": test_pairs
        }
    
    # First pass: compute all sequences and find global maximum sequence length
    all_split_data = {}
    for split_name, current_pairs in splits.items():
        inputs = []
        labels = []
        
        for idx, (a, b) in enumerate(current_pairs):
            res = a + b
            
            if config.varied_length:
                s_a = str(a)
                s_b = str(b)
                s_res = str(res)
                max_result_digits = config.digits + 1
                s_res_padded = s_res.zfill(max_result_digits)
                carry_digits = config.digits
            else:
                s_a = f"{a:0{config.digits}d}"
                s_b = f"{b:0{config.digits}d}"
                max_result_digits = config.digits + 1
                s_res_padded = f"{res:0{max_result_digits}d}"
                carry_digits = config.digits
            
            prefix = s_a + "+" + s_b + "="
            
            if config.dataset_mode == "vanilla":
                inp_seq = encode(prefix) + [MASK_ID] * max_result_digits
                label_seq = [IGNORE_LABEL_ID] * len(encode(prefix)) + encode(s_res_padded)
                assert CAR_TOKEN_ID is None or CAR_TOKEN_ID not in label_seq, "Vanilla mode should not contain <CAR> token"
            else:  # lilavati1 or lilavati2
                carries = compute_carries(a, b, carry_digits)
                s_carries = ''.join(str(c) for c in carries)
                inp_seq = encode(prefix) + [MASK_ID] * max_result_digits + [CAR_TOKEN_ID] + [MASK_ID] * carry_digits
                label_seq = [IGNORE_LABEL_ID] * len(encode(prefix)) + encode(s_res_padded) + [CAR_TOKEN_ID] + encode(s_carries)
                assert label_seq.count(CAR_TOKEN_ID) == 1, f"Lilavati mode must have exactly one <CAR> token, got {label_seq.count(CAR_TOKEN_ID)}"
            
            inputs.append(inp_seq)
            labels.append(label_seq)
        
        all_split_data[split_name] = {
            'inputs': inputs,
            'labels': labels,
            'pairs': current_pairs
        }
        
        # Update global max_seq_len
        split_max_seq_len = max(len(inp) for inp in inputs)
        if max_seq_len is None:
            max_seq_len = split_max_seq_len
        else:
            max_seq_len = max(max_seq_len, split_max_seq_len)
    
    # Second pass: pad all sequences to global maximum and save
    for split_name, split_data in all_split_data.items():
        inputs = split_data['inputs']
        labels = split_data['labels']
        current_pairs = split_data['pairs']
        
        puzzle_indices = [0]
        group_indices = [0]
        puzzle_identifiers = []
        
        # Pad all sequences to global max_seq_len
        inputs_padded = []
        labels_padded = []
        for idx, (inp, lab) in enumerate(zip(inputs, labels)):
            assert len(inp) == len(lab), f"Input and label lengths must match: {len(inp)} != {len(lab)}"
            pad_len = max_seq_len - len(inp)
            inputs_padded.append(inp + [0] * pad_len)  # Pad with PAD_ID (0)
            labels_padded.append(lab + [IGNORE_LABEL_ID] * pad_len)  # Pad with IGNORE
            
            puzzle_identifiers.append(0)
            puzzle_indices.append(idx + 1)
            group_indices.append(idx + 1)
        
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
            seq_len=max_seq_len,  # Use global max_seq_len for all splits
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
    
    # If also_output_dir is set, generate the other mode as well (vanilla <-> lilavati1)
    if config.also_output_dir is not None:
        other_mode = "lilavati1" if config.dataset_mode == "vanilla" else "vanilla"
        print(f"\nAlso generating {other_mode} mode dataset to {config.also_output_dir}...")
        
        # Create config for other mode (reuse all settings except mode and output_dir)
        # We'll create a temporary config object for the other mode
        other_mode_config = type(config)(
            output_dir=config.also_output_dir,
            seed=config.seed,
            test_ratio=config.test_ratio,
            digits=config.digits,
            dataset_mode=other_mode,
            max_examples=config.max_examples,
            sample_ratio=config.sample_ratio,
            varied_length=config.varied_length,
            min_carries=config.min_carries,
            min_carry_ratio=config.min_carry_ratio,
            train_max_digits=config.train_max_digits,
            also_output_dir=None  # Don't generate recursively
        )
        
        # Set up vocab for other mode
        other_vocab_map = {str(i): i + 2 for i in range(10)}
        other_vocab_map['+'] = 12
        other_vocab_map['='] = 13
        if other_mode == "lilavati1":
            other_vocab_map['<CAR>'] = 14
            other_vocab_size = 15
            other_CAR_TOKEN_ID = 14
        else:
            other_vocab_size = 14
            other_CAR_TOKEN_ID = None
        
        os.makedirs(other_mode_config.output_dir, exist_ok=True)
        
        def encode_other(s):
            return [other_vocab_map[c] for c in s]
        
        # First pass: compute all sequences and find global maximum sequence length
        other_all_split_data = {}
        other_max_seq_len = None
        for split_name, current_pairs in splits.items():
            inputs = []
            labels = []
            
            for idx, (a, b) in enumerate(current_pairs):
                res = a + b
                
                if other_mode_config.varied_length:
                    s_a = str(a)
                    s_b = str(b)
                    s_res = str(res)
                    max_result_digits = other_mode_config.digits + 1
                    s_res_padded = s_res.zfill(max_result_digits)
                    carry_digits = other_mode_config.digits
                else:
                    s_a = f"{a:0{other_mode_config.digits}d}"
                    s_b = f"{b:0{other_mode_config.digits}d}"
                    max_result_digits = other_mode_config.digits + 1
                    s_res_padded = f"{res:0{max_result_digits}d}"
                    carry_digits = other_mode_config.digits
                
                prefix = s_a + "+" + s_b + "="
                
                if other_mode == "vanilla":
                    inp_seq = encode_other(prefix) + [MASK_ID] * max_result_digits
                    label_seq = [IGNORE_LABEL_ID] * len(encode_other(prefix)) + encode_other(s_res_padded)
                else:  # lilavati1 (other_mode is always lilavati1 or vanilla, not lilavati2)
                    carries = compute_carries(a, b, carry_digits)
                    s_carries = ''.join(str(c) for c in carries)
                    inp_seq = encode_other(prefix) + [MASK_ID] * max_result_digits + [other_CAR_TOKEN_ID] + [MASK_ID] * carry_digits
                    label_seq = [IGNORE_LABEL_ID] * len(encode_other(prefix)) + encode_other(s_res_padded) + [other_CAR_TOKEN_ID] + encode_other(s_carries)
                
                inputs.append(inp_seq)
                labels.append(label_seq)
            
            other_all_split_data[split_name] = {
                'inputs': inputs,
                'labels': labels,
                'pairs': current_pairs
            }
            
            # Update global other_max_seq_len
            split_max_seq_len = max(len(inp) for inp in inputs)
            if other_max_seq_len is None:
                other_max_seq_len = split_max_seq_len
            else:
                other_max_seq_len = max(other_max_seq_len, split_max_seq_len)
        
        # Second pass: pad all sequences to global maximum and save
        for split_name, split_data in other_all_split_data.items():
            inputs = split_data['inputs']
            labels = split_data['labels']
            current_pairs = split_data['pairs']
            
            puzzle_indices = [0]
            group_indices = [0]
            puzzle_identifiers = []
            
            # Pad all sequences to global other_max_seq_len
            inputs_padded = []
            labels_padded = []
            for idx, (inp, lab) in enumerate(zip(inputs, labels)):
                pad_len = other_max_seq_len - len(inp)
                inputs_padded.append(inp + [0] * pad_len)
                labels_padded.append(lab + [IGNORE_LABEL_ID] * pad_len)
                
                puzzle_identifiers.append(0)
                puzzle_indices.append(idx + 1)
                group_indices.append(idx + 1)
            
            inputs = np.array(inputs_padded, dtype=np.uint8)
            labels = np.array(labels_padded, dtype=np.uint8)
            puzzle_indices = np.array(puzzle_indices, dtype=np.int32)
            group_indices = np.array(group_indices, dtype=np.int32)
            puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
            
            save_dir = os.path.join(other_mode_config.output_dir, split_name)
            os.makedirs(save_dir, exist_ok=True)
            
            np.save(os.path.join(save_dir, "all__inputs.npy"), inputs)
            np.save(os.path.join(save_dir, "all__labels.npy"), labels)
            np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), puzzle_indices)
            np.save(os.path.join(save_dir, "all__group_indices.npy"), group_indices)
            np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
            
            metadata = PuzzleDatasetMetadata(
                seq_len=other_max_seq_len,  # Use global other_max_seq_len for all splits
                vocab_size=other_vocab_size,
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
        
        with open(os.path.join(other_mode_config.output_dir, "identifiers.json"), "w") as f:
            json.dump(["<blank>"], f)
        
        generate_dataset_stats(other_mode_config, splits, other_vocab_map, other_vocab_size, other_max_seq_len, total_possible)
        print(f"Generated {other_mode} mode dataset to {config.also_output_dir}")


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
    if config.dataset_mode in {"lilavati1", "lilavati2"}:
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
            if config.dataset_mode in {"lilavati1", "lilavati2"}:
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
- **Varied Length**: {config.varied_length}
- **Sampling**: {f"{config.max_examples:,} examples" if config.max_examples is not None else (f"{config.sample_ratio*100:.2f}%" if config.sample_ratio is not None else "All combinations")}
- **Total Possible Combinations**: {total_possible:,}
{f"- **Min Carries**: {config.min_carries}" if config.min_carries is not None else ""}
{f"- **Min Carry Ratio**: {config.min_carry_ratio}" if config.min_carry_ratio is not None else ""}
{f"- **Train Max Digits**: {config.train_max_digits} (train only contains examples with ≤{config.train_max_digits} digits)" if config.train_max_digits is not None else ""}

### Dataset Statistics
- **Total Examples**: {total_examples:,}
- **Training Examples**: {train_examples:,} ({train_examples/total_examples*100:.1f}%)
- **Test Examples**: {test_examples:,} ({test_examples/total_examples*100:.1f}%)
- **Sequence Length**: {max_seq_len}
- **Vocabulary Size**: {vocab_size}
- **Coverage**: {total_examples:,} / {total_possible:,} possible combinations ({total_examples/total_possible*100:.2f}%)
"""
    
    # Compute digit length distribution for train vs test (especially useful for varied_length and train_max_digits)
    if config.varied_length or config.train_max_digits is not None:
        train_digit_lengths = Counter(max(len(str(a)), len(str(b))) for a, b in splits["train"])
        test_digit_lengths = Counter(max(len(str(a)), len(str(b))) for a, b in splits["test"])
        
        readme_content += """
### Digit Length Distribution
"""
        readme_content += "**Training Set:**\n"
        for digit_len in sorted(train_digit_lengths.keys()):
            count = train_digit_lengths[digit_len]
            readme_content += f"- {count:,} examples with {digit_len} digit(s) ({count/train_examples*100:.1f}%)\n"
        
        readme_content += "\n**Test Set:**\n"
        for digit_len in sorted(test_digit_lengths.keys()):
            count = test_digit_lengths[digit_len]
            readme_content += f"- {count:,} examples with {digit_len} digit(s) ({count/test_examples*100:.1f}%)\n"
    
    readme_content += """

### Result Distribution Statistics
- **Minimum Result**: {min_result}
- **Maximum Result**: {max_result:,}
- **Mean Result**: {mean_result:.2f}
- **Median Result**: {median_result:.2f}
- **Result Digit Counts**:
  - {digit_counts.get(config.digits, 0):,} results with {config.digits} digits ({digit_counts.get(config.digits, 0)/total_examples*100:.1f}%)
  - {digit_counts.get(config.digits+1, 0):,} results with {config.digits+1} digits ({digit_counts.get(config.digits+1, 0)/total_examples*100:.1f}%)
"""
    
    if config.dataset_mode in {"lilavati1", "lilavati2"}:
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
    
    if config.dataset_mode in {"lilavati1", "lilavati2"}:
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

