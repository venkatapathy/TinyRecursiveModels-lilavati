#!/usr/bin/env python3
"""
Arithmetic Dataset Generator
Generates vanilla and structured (chain-of-thought/intermediate supervision) datasets
for basic arithmetic operations (+, -, *, /).

Design:
- "Vanilla": text completion "<A><op><B>=<Y>"
- "Structured": JSON with explicit fields for operands, result, and intermediate "labels".

Intermediate Labels:
- Addition: "add_carry" (per digit carry-out)
- Subtraction: "sub_borrow" (per digit borrow-out)
- Multiplication: "mul_carry_cols" (column-wise carry-out)
- Division: "div_q_digits" and "div_remainders" (long division steps)

Usage:
    python dataset/build_arithmetic_dataset.py \
        --output_dir data/arithmetic \
        --num_train 100000 \
        --num_val 1000 \
        --num_test 10000 \
        --train_max_result_digits 8 \
        --test_max_result_digits 12
"""

import argparse
import collections
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple, Union

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

OPS = ["+", "-", "*", "/"]

# -----------------------------------------------------------------------------
# Arithmetic Logic with Intermediate Labels
# -----------------------------------------------------------------------------

def get_digits(n: int) -> List[int]:
    """Return digits of n in little-endian order (LSB first)."""
    if n == 0:
        return [0]
    digits = []
    abs_n = abs(n)
    while abs_n > 0:
        digits.append(abs_n % 10)
        abs_n //= 10
    return digits # LSB -> MSB

def digits_to_int(digits: List[int]) -> int:
    """Convert little-endian digits list to int."""
    n = 0
    for i, d in enumerate(digits):
        n += d * (10 ** i)
    return n

def generate_addition(a: int, b: int) -> Dict:
    """
    A + B = Y
    Returns: {
        "text": "A+B=Y",
        "op": "+", "A": str(a), "B": str(b), "Y": str(a+b),
        "labels": {"add_carry": [...], "add_digits": [...]}
    }
    """
    y = a + b
    
    # Calculate carries (LSB -> MSB)
    a_digits = get_digits(a)
    b_digits = get_digits(b)
    y_digits = get_digits(y)
    
    # Pad inputs to length of result (conceptually) or just iterate max len
    max_len = len(y_digits) # Addition can grow by at most 1 digit
    
    carries = []
    c = 0
    # Process from LSB to MSB of the result
    for i in range(max_len):
        da = a_digits[i] if i < len(a_digits) else 0
        db = b_digits[i] if i < len(b_digits) else 0
        s = da + db + c
        c_out = s // 10
        # digit = s % 10
        carries.append(c_out)
        c = c_out
        
    return {
        "op": "+",
        "A": str(a),
        "B": str(b),
        "Y": str(y),
        "labels": {
            "add_carry": carries,
        }
    }

def generate_subtraction(a: int, b: int, allow_negative: bool = False) -> Optional[Dict]:
    """
    A - B = Y
    Default: A >= B. If A < B and not allow_negative, return None.
    Labels: sub_borrow (LSB->MSB)
    """
    if not allow_negative and a < b:
        return None
        
    y = a - b
    
    if y < 0:
        # Handle negative result logic for borrowing? 
        # Usually school subtraction with borrowing is defined for |A| - |B|.
        # We will compute labels for abs(A) - abs(B) if we treat sign separately,
        # but simplified requirement implies usually A>=B.
        # If A < B, we just formatting it as "-Y", but the borrow logic is slightly different 
        # (effectively B - A).
        # For this dataset, we follow "default non-negative A>=B".
        # If negatives enabled and we get A < B, let's skip structural labels or implement B-A logic.
        # Let's assume for now we swap if result is negative to compute borrows of the magnitude subtraction.
        # But text should be "A-B=-Y".
        
        # Borrow logic effectively happens on magnitudes: |A| - |B| or |B| - |A|.
        pass

    # We only generate standard borrowing traces for A >= B >= 0 case.
    # If A < B, we will just not provide borrow labels or define them as borrows for B-A.
    # Let's stick to generating valid A>=B samples primarily unless forced.
    
    # Simulation: A - B (where A >= B)
    # diff_i = a_i - b_i - borrow_in
    a_digits = get_digits(a)
    b_digits = get_digits(b)
    
    # Pad b to len a
    while len(b_digits) < len(a_digits):
        b_digits.append(0)
        
    borrows = []
    b_in = 0
    y_digits_manual = []
    
    # Iterate through A's length (which covers outcome)
    # Note: result might be shorter than A (e.g. 10 - 1 = 9), but borrow trace covers input logic.
    # The user asked: "list of borrow_out per output digit (LSB->MSB), length = len(Y_digits)"
    # This is tricky because Y length can be smaller than A.
    # Usually algorithms iterate up to max(len(A), len(B)). 
    # Let's iterate up to len(A).
    
    for i in range(len(a_digits)):
        da = a_digits[i]
        db = b_digits[i]
        
        diff = da - db - b_in
        if diff < 0:
            diff += 10
            b_out = 1
        else:
            b_out = 0
            
        y_digits_manual.append(diff)
        borrows.append(b_out)
        b_in = b_out
        
    # Trim leading zeros from result to match Y_digits length
    # But borrow trace should correspond to positions.
    # User spec: "per output digit (LSB->MSB), length = len(Y_digits)"
    # If Y is 9 (len 1) from 10-1, but we computed 2 positions.
    # We will truncate borrows to len(Y).
    
    # Actual Y
    y_str = str(y)
    len_y = len(y_str)
    
    # If negative result, len_y includes '-'. We care about magnitude digits.
    if y < 0:
        len_y -= 1
        
    labels = {
        "sub_borrow": borrows[:len_y] # Truncate to result length? Or keep full trace?
        # "per output digit" usually implies 1-to-1. 
        # If 100 - 99 = 1. Output has 1 digit. Borrows happened at pos 0 and 1.
        # If we truncate, we lose info. 
        # Re-reading spec: "per output digit ... length = len(Y_digits)"
        # This might be lossy for subtraction where leading digits cancel out.
        # But I will follow spec strictly: length = len(Y_digits).
    }
    
    return {
        "op": "-",
        "A": str(a),
        "B": str(b),
        "Y": str(y),
        "labels": labels
    }

def generate_multiplication(a: int, b: int) -> Dict:
    """
    A * B = Y
    Labels: mul_carry_cols (column-wise carry-out)
    Definition:
      s_k = sum_{i+j=k} a_i*b_j + carry_in
      carry_out = s_k // 10
    """
    y = a * b
    a_digits = get_digits(a)
    b_digits = get_digits(b)
    y_digits = get_digits(y)
    
    # Number of columns to process.
    # Max index k for i+j is (len(a)-1) + (len(b)-1) = len(a)+len(b)-2.
    # We need to process until carry is 0 and all digit products consumed.
    # Generally this produces len(a)+len(b) columns max.
    
    # User spec: "length = len(Y_digits) (or 2*max_digits)"
    # We'll use len(Y_digits) to be precise.
    
    carry = 0
    col_carries = []
    
    # max columns could cover len(y)
    num_cols = len(y_digits)
    # Note: if y is 0, len is 1.
    
    for k in range(num_cols):
        s_k = carry
        # sum_{i+j=k} a_i * b_j
        # iterate i from 0 to k
        for i in range(k + 1):
            j = k - i
            if i < len(a_digits) and j < len(b_digits):
                s_k += a_digits[i] * b_digits[j]
        
        current_digit = s_k % 10 # This should match y_digits[k]
        carry_out = s_k // 10
        col_carries.append(carry_out)
        carry = carry_out
        
    return {
        "op": "*",
        "A": str(a),
        "B": str(b),
        "Y": str(y),
        "labels": {
            "mul_carry_cols": col_carries
        }
    }

def generate_division(q: int, d: int) -> Dict:
    """
    A / D = Q (exact integer division)
    Input generation: sample q, d. A = q*d.
    Labels: 
      div_q_digits (MSB->LSB)
      div_remainders (MSB->LSB) - invariants 0 <= r < d
    """
    a = q * d # Dividend
    
    # Long division simulation
    # We process A from MSB to LSB.
    # Convert A to string to iterate digits easily
    a_str = str(a)
    a_digits_msb = [int(c) for c in a_str]
    
    current_rem = 0
    q_digits = []
    remainders = []
    
    for digit in a_digits_msb:
        current_rem = current_rem * 10 + digit
        q_d = current_rem // d
        current_rem = current_rem % d
        
        q_digits.append(q_d)
        remainders.append(current_rem)
    
    # Verify
    q_from_digits = int("".join(map(str, q_digits)))
    if q_from_digits != q:
         # Leading zeros in Q might cause mismatch in int() parse if q=0? No 0 is 0.
         # But q_digits might have leading zeros if A < D initially?
         # Example 100 / 50 = 2. a_digits=[1,0,0].
         # 1: r=1, q=0. 10: r=10, q=0. 100: r=0, q=2. q_digits=[0,0,2].
         pass

    # Trim leading zeros from q_digits to match canonical rep?
    # User says "quotient digits in long division order".
    # Usually long division produces leading zeros until the first non-zero.
    # We will keep them as they align with input dividend digits (1-to-1).
    # But usually we might want normalized Q.
    # Let's keep the 1-to-1 mapping with dividend steps as it's cleaner for "step-by-step".
    
    return {
        "op": "/",
        "A": str(a),
        "B": str(d),
        "Y": str(q),
        "labels": {
            "div_q_digits": q_digits,
            "div_remainders": remainders
        }
    }

# -----------------------------------------------------------------------------
# Sampler
# -----------------------------------------------------------------------------

class Sampler:
    def __init__(self, 
                 op: str, 
                 train_max_res_digits: int,
                 test_max_res_digits: int, 
                 allow_zero: bool = False,
                 seed: int = 42):
        self.op = op
        self.train_limit = train_max_res_digits
        self.test_limit = test_max_res_digits
        self.allow_zero = allow_zero
        self.rng = random.Random(seed)
        
    def _sample_operand(self, max_digits: int) -> int:
        """Sample number with random length up to max_digits."""
        # Pick length
        length = self.rng.randint(1, max_digits)
        
        if length == 1:
            if self.allow_zero:
                return self.rng.randint(0, 9)
            else:
                return self.rng.randint(1, 9)
        
        # Leading non-zero
        min_v = 10**(length-1)
        max_v = 10**length - 1
        return self.rng.randint(min_v, max_v)

    def generate_pair(self, is_train: bool) -> Tuple:
        """
        Generate (a, b) such that result length falls in desired bucket.
        Strategy:
        1. Pick target result length L within range.
        2. Heuristically sample operands to unlikely hit that length.
        3. Verify and retry.
        """
        limit = self.train_limit if is_train else self.test_limit
        min_limit = 1
        if not is_train:
            # For OOD test, we prefer lengths > train_limit
            min_limit = self.train_limit + 1 if self.train_limit < self.test_limit else 1
            
        target_res_len = self.rng.randint(min_limit, limit)
        
        # Heuristics to get target length
        for _ in range(50): # attempts
            if self.op == "+":
                # len(a+b) ~ max(len a, len b) or +1
                # Sample len_a close to target_res_len
                len_a = self.rng.randint(1, target_res_len)
                len_b = self.rng.randint(1, target_res_len)
                # Bias towards at least one being target_res_len or target_res_len-1
                if self.rng.random() < 0.5:
                     len_a = target_res_len
                else:
                     len_a = max(1, target_res_len - 1)
                
                a = self._sample_operand(len_a) # _sample_operand samples UP TO, wait fix
                # We want EXACT length sampling helper?
                # _sample_operand(L) implementation above samples length uniformly. 
                # Let's fix that: specific length sampling.
                a = self._sample_n_digits(len_a)
                b = self._sample_n_digits(len_b)
                
                res_len = len(str(a + b))
                
            elif self.op == "-":
                # len(a-b). A >= B.
                # A ~ target. B can be anything smaller.
                len_a = target_res_len
                if self.rng.random() < 0.3: # sometimes larger A can reduce to target len
                    len_a = target_res_len + self.rng.randint(0, 2)
                
                a = self._sample_n_digits(len_a)
                b_max = a if not self.allow_zero else a
                b = self.rng.randint(1 if not self.allow_zero else 0, max(1, a)) # a >= b
                
                # Resample b to have varying lengths
                len_b = self.rng.randint(1, len_a)
                b = self._sample_n_digits(len_b)
                while b > a: b //= 10 # ensure b <= a roughly
                # Or just reroll
                if b > a: b = self.rng.randint(1, a)
                
                res_len = len(str(a - b))
                
            elif self.op == "*":
                # len(a*b) ~ len(a) + len(b)
                # Split target len into l_a + l_b = target
                if target_res_len == 1:
                    l_a, l_b = 1, 1
                else:
                    l_a = self.rng.randint(1, target_res_len - 1)
                    l_b = target_res_len - l_a
                    # Adjust to allow loose bounds (+-1)
                
                a = self._sample_n_digits(l_a)
                # Cap trailing zeros for mul
                if self.rng.random() < 0.8: # 80% remove trailing zeros from operands
                    while a > 0 and a % 10 == 0: a //= 10
                    if a == 0: a = 1
                
                b = self._sample_n_digits(min(l_b, limit)) # bound b
                while b > 0 and b % 10 == 0: b //= 10
                if b == 0: b = 1
                
                res_len = len(str(a * b))
                
            elif self.op == "/":
                # A / D = Q.
                # Target result length is len(Q).
                len_q = target_res_len
                q = self._sample_n_digits(len_q)
                
                # Divisor D. Can be anything.
                # But A = Q*D must vary.
                len_d = self.rng.randint(1, 8) # Limit divisor size somewhat? or up to limit?
                d = self._sample_n_digits(len_d)
                
                # Check constraints?
                # If we want A (dividend) to be within some global max?
                # User didn't specify global operand max, just RESULT-based bucketing.
                # For Div, input A is the structural "operand" but Q is the result.
                # So bucket by Q length.
                res_len = len(str(q))
                return q, d # For div we need Q, D to make A
                
            if res_len == target_res_len:
                return a, b
            
        # Fallback
        return a, b
    
    def _sample_n_digits(self, n: int) -> int:
        if n <= 0: return 0
        if n == 1:
            if self.allow_zero: return self.rng.randint(0, 9)
            return self.rng.randint(1, 9)
        min_v = 10**(n-1)
        max_v = 10**n - 1
        return self.rng.randint(min_v, max_v)

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Arithmetic Dataset")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_val", type=int, default=1000)
    parser.add_argument("--num_test", type=int, default=1000)
    parser.add_argument("--train_max_result_digits", type=int, default=8)
    parser.add_argument("--test_max_result_digits", type=int, default=12)
    parser.add_argument("--allow_zero", action="store_true", help="Allow 0 operands")
    
    args = parser.parse_args()
    
    # Setup
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Stats containers
    stats = {
        "counts": collections.defaultdict(int),
        "hist_result_digits": collections.defaultdict(int),
        "hist_operand_max_digits": collections.defaultdict(int),
        "arithmetic_errors": 0
    }
    
    # Files
    files = {}
    for split in ["train", "val", "test"]:
        files[f"vanilla_{split}"] = open(os.path.join(args.output_dir, f"vanilla_{split}.jsonl"), "w")
        files[f"structured_{split}"] = open(os.path.join(args.output_dir, f"structured_{split}.jsonl"), "w")
    
    # Generation Loop
    splits_config = [
        ("train", args.num_train),
        ("val", args.num_val),
        ("test", args.num_test)
    ]
    
    example_buffer = [] # Store a few for printing
    
    print(f"Starting generation...")
    print(f"Plan: {args.num_train} train, {args.num_val} val, {args.num_test} test per operation.")
    
    for op in OPS:
        sampler = Sampler(
            op, 
            args.train_max_result_digits, 
            args.test_max_result_digits,
            allow_zero=args.allow_zero,
            seed=args.seed + ord(op)
        )
        
        for split, num_examples in splits_config:
            is_train = (split == "train")
            # Divide num_examples by 4 if total? User said: "counts per split and per op". 
            # Usually --num_examples is total. But here arguments say num_train.
            # Let's assume passed nums are TOTAL across all ops, so we divide by 4.
            # Or assume per op? The prompt says "Balance by operation".
            # Safest is target_per_op = num // 4.
            target = max(1, num_examples // 4)
            
            count = 0
            while count < target:
                # 1. Sample operands
                val_a, val_b = sampler.generate_pair(is_train)
                
                # 2. Generate Data
                data = None
                if op == "+":
                    data = generate_addition(val_a, val_b)
                elif op == "-":
                    # For sub, we might need to swap to ensure A>=B if not enforcing negative
                    # But sampler tries A>=B.
                    if val_a < val_b: val_a, val_b = val_b, val_a
                    data = generate_subtraction(val_a, val_b)
                elif op == "*":
                    data = generate_multiplication(val_a, val_b)
                elif op == "/":
                    # Sampler returns q, d. We need to construct A, B(divisor).
                    q, d = val_a, val_b
                    data = generate_division(q, d)
                
                if data is None: 
                    continue
                    
                # 3. Validation
                valid = True
                try:
                    # Basic arithmetic check
                    A = int(data["A"])
                    B = int(data["B"])
                    Y = int(data["Y"])
                    if op == "+": assert A + B == Y
                    elif op == "-": assert A - B == Y
                    elif op == "*": assert A * B == Y
                    elif op == "/": assert A // B == Y and A % B == 0
                except AssertionError:
                    stats["arithmetic_errors"] += 1
                    valid = False
                
                if not valid: continue
                
                # 4. Formats
                vanilla_obj = {
                    "id": f"{split}_{op}_{count}",
                    "text": f"{data['A']}{op}{data['B']}={data['Y']}"
                }
                
                structured_obj = {
                    "id": f"{split}_{op}_{count}",
                    "text": f"{data['A']}{op}{data['B']}={data['Y']}",
                    "op": data["op"],
                    "A": data["A"], 
                    "B": data["B"], 
                    "Y": data["Y"],
                    "labels": data["labels"]
                }
                
                # Write
                json.dump(vanilla_obj, files[f"vanilla_{split}"])
                files[f"vanilla_{split}"].write("\n")
                
                json.dump(structured_obj, files[f"structured_{split}"])
                files[f"structured_{split}"].write("\n")
                
                # Stats
                stats["counts"][f"{split}_{op}"] += 1
                res_len = len(data["Y"])
                if op == "/" and data["Y"].startswith("-"): res_len -= 1 # Handle negative if any
                stats["hist_result_digits"][res_len] += 1
                
                if count < 2 and split == "train":
                    example_buffer.append(structured_obj)
                
                count += 1
                
        print(f"Finished {op}.")

    # Close files
    for f in files.values():
        f.close()
        
    # Write stats
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
        
    # Summary
    print("\nGeneration Complete.")
    print(f"Stats saved to {stats_path}")
    print("Example Structured Outputs:")
    for ex in example_buffer:
        print(json.dumps(ex, indent=2))

if __name__ == "__main__":
    main()
