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
import numpy as np

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

OPS = ["+", "-", "*", "/"]

# -----------------------------------------------------------------------------
# Vocabulary & Constants (Aligned with Causal LM)
# -----------------------------------------------------------------------------

VOCAB_MAP = {
    '0': 2, '1': 3, '2': 4, '3': 5, '4': 6,
    '5': 7, '6': 8, '7': 9, '8': 10, '9': 11,
    '+': 12, '-': 13, '*': 14, '/': 15, '=': 16,
    'R': 17
}
PAD_ID = 0
IGNORE_LABEL_ID = -100

def encode_str(s: str) -> List[int]:
    return [VOCAB_MAP[c] for c in s]


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
# Conversion & Saving Logic
# -----------------------------------------------------------------------------

def process_and_save_split(data_items: List[Dict], output_dir: str, split_name: str, max_len: int, dataset_mode: str = "vanilla", vocab_map: Dict = None):
    """
    Converts structured data items to padded numpy arrays and saves them.
    """
    print(f"Processing {split_name} ({len(data_items)} examples)...")
    
    inputs_list = []
    labels_lm_list = []
    labels_aux_list = []

    for item in data_items:
        # data_items contains the "structured_obj" dicts
        
        A = item["A"]
        B = item["B"]
        op = item["op"]
        Y = item["Y"]
        
        # Input Prompt: "A op B ="
        prompt_str = f"{A}{op}{B}="
        prompt_ids = encode_str(prompt_str)
        
        # Result: "Y"
        result_ids = encode_str(Y)
        
        # Full Input: Prompt + Result
        input_ids = prompt_ids + result_ids
        
        # BASICFOUR CONCAT LOGIC
        concat_suffix_ids = []
        if dataset_mode == "basicfour_concat":
            # 1. Determine CAR token
            car_token_id = vocab_map.get(f"<CAR_{op}>")
            if car_token_id is None:
                 # Fallback or error?
                 # Should fail if map is correct.
                 raise ValueError(f"Missing CAR token for op {op}")
            
            # 2. Get Intermediate Data
            # Note: We need digits as strings/ids, not raw ints 0-9 usually? 
            # Current `inter_list` logic below gets ints. The model needs vocab IDs.
            # Digits 0-9 map to vocab IDs 2-11.
            # We should reuse the logic or convert.
            
            inter_val_list = []
            if op == '+': inter_val_list = item["labels"].get("add_carry", [])
            elif op == '-': inter_val_list = item["labels"].get("sub_borrow", [])
            elif op == '*': inter_val_list = item["labels"].get("mul_carry_cols", [])
            elif op == '/': 
                # Division: "div_q_digits", "div_remainders"
                # For now let's just dump remainders? Or both?
                # User prompted "concatinated prediction".
                # Standard Lilavati usually just does carries. 
                # Let's concatenate remainders for division as they are the "state".
                inter_val_list = item["labels"].get("div_remainders", [])
            
            # Map integers to vocab IDs
            # Simple digits 0-9 -> VOCAB_MAP[str(d)]
            # If value > 9 (like carry 12), we treat it as multi-digit? 
            # Or is carry always single digit? 
            # Addition carry <= 1. Sub borrow <= 1. 
            # Mul carry cols can be large! e.g. 9*9 + 8 = 89 -> carry 8.
            # Sum of products can be larger. 
            # If carry > 9, we need multi-token representation or special tokens.
            # For now, let's assume we output digits of the carry value if > 9.
            # But "Lilavati" usually assumes single digit? 
            # Wait, `trm.py` usually predicts a single token. 
            # If carry > 9 (possible in Mul), we might need to separate it. 
            # Let's stringify each value and encode.
            
            suffix_str_parts = []
            for val in inter_val_list:
                suffix_str_parts.append(str(val)) # e.g. "1", "0", "12"
            
            # Join with what? Just sequence? 
            # Usually: <CAR> c1 c2 c3 ...
            # If c_i is "12", it becomes "1" "2". 
            # BUT we lose boundary. 
            # Ideally Mul carries are single "values" but if they exceed 9, we need boundaries or multi-token.
            # Let's treat them as sequence of digits.
            
            suffix_seq_ids = [car_token_id]
            for val in inter_val_list:
                s_val = str(val)
                for char in s_val:
                    suffix_seq_ids.append(VOCAB_MAP[char])
            
            concat_suffix_ids = suffix_seq_ids
            
            # Append to inputs
            input_ids = input_ids + concat_suffix_ids
        
        # Aux Labels: IGNORE on Prompt, Carry ids on Result
        inter_list = []
        if dataset_mode != "basicfour_concat":
            if op == '+':
                inter_list = item["labels"].get("add_carry", [])
            elif op == '-':
                inter_list = item["labels"].get("sub_borrow", [])
            elif op == '*':
                inter_list = item["labels"].get("mul_carry_cols", [])
            elif op == '/':
                inter_list = [] # No aux training for div yet
            
        if len(inter_list) > 0:
            safe_list = []
            for x in inter_list:
                if x >= 20: 
                    safe_list.append(IGNORE_LABEL_ID)
                else:
                    safe_list.append(x)
            inter_list = safe_list
            
        if len(inter_list) == 0:
             aux_labels = [IGNORE_LABEL_ID] * len(input_ids)
        else:
             target_len = len(result_ids)
             if len(inter_list) >= target_len:
                 aligned_inter = inter_list[-target_len:]
             else:
                 diff = target_len - len(inter_list)
                 aligned_inter = [0]*diff + inter_list
            
             aux_labels = [IGNORE_LABEL_ID] * len(prompt_ids) + aligned_inter
        
        # CAUSAL LM SHIFTING
        full_ids = prompt_ids + result_ids + concat_suffix_ids
        shifted_lm_labels = full_ids[1:] + [PAD_ID]
        
        # Masking Prompt (Keep only the '=' prediction which is Result[0])
        # Indices 0 to len(prompt_ids)-2 should be IGNORED.
        # index len(prompt_ids)-1 is '=', predicting Result[0].
        for i in range(len(prompt_ids) - 1):
            shifted_lm_labels[i] = IGNORE_LABEL_ID
            
        lm_labels = shifted_lm_labels
        
        # Shift Aux Labels
        shifted_aux_labels = aux_labels[1:] + [PAD_ID]
        aux_labels = shifted_aux_labels
        
        assert len(input_ids) == len(lm_labels) == len(aux_labels)
        
        inputs_list.append(input_ids)
        labels_lm_list.append(lm_labels)
        labels_aux_list.append(aux_labels)

    # Padding
    padded_inputs = []
    padded_lm = []
    padded_aux = []
    
    puzzle_indices = [0]
    group_indices = [0]
    puzzle_identifiers = []

    for idx, (inp, lm, aux) in enumerate(zip(inputs_list, labels_lm_list, labels_aux_list)):
        pad_len = max_len - len(inp)
        if pad_len < 0:
             # Truncate if too long (should not happen if args correct, but safety)
             # print(f"Warning: Example {idx} exceeds max_len {max_len} (len={len(inp)}). Truncating.")
             inp = inp[:max_len]
             lm = lm[:max_len]
             aux = aux[:max_len]
             pad_len = 0
        
        padded_inputs.append(inp + [PAD_ID] * pad_len)
        padded_lm.append(lm + [IGNORE_LABEL_ID] * pad_len)
        padded_aux.append(aux + [IGNORE_LABEL_ID] * pad_len)
        
        puzzle_indices.append(idx + 1)
        group_indices.append(idx + 1)
        puzzle_identifiers.append(0)

    # Save
    save_dir = os.path.join(output_dir, split_name)
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "all__inputs.npy"), np.array(padded_inputs, dtype=np.uint8))
    np.save(os.path.join(save_dir, "all__labels.npy"), np.array(padded_lm, dtype=np.int64))
    np.save(os.path.join(save_dir, "all__labels_aux.npy"), np.array(padded_aux, dtype=np.int64))
    
    np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), np.array(puzzle_indices, dtype=np.int32))
    np.save(os.path.join(save_dir, "all__group_indices.npy"), np.array(group_indices, dtype=np.int32))
    np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), np.array(puzzle_identifiers, dtype=np.int32))
    
    metadata = {
        "seq_len": max_len,
        "seq_len": max_len,
        "vocab_size": 22,
        "pad_id": PAD_ID,
        "pad_id": PAD_ID,
        "ignore_label_id": IGNORE_LABEL_ID,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": len(inputs_list),
        "mean_puzzle_examples": 1.0,
        "total_puzzles": len(inputs_list),
        "sets": ["all"],
        "total_examples": len(inputs_list)
    }
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"  Saved numpy arrays to {save_dir}")

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
    parser.add_argument("--max_len", type=int, default=64, help="Padded sequence length")
    parser.add_argument("--dataset_mode", type=str, default="basicfour_concat", choices=["vanilla", "lilavati1", "lilavati2", "lilavati3", "basicfour_concat"], help="Dataset mode")
    
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
    data_buffer = {"train": [], "val": [], "test": []}
    dataset_mode = args.dataset_mode
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
    
    # Initialize vocab_map, vocab_size, CAR_TOKEN_ID (assuming these are global or defined here)
    # This part of the code was not provided in the original context, but is implied by the edit.
    # For the purpose of this edit, we'll assume a `vocab_map` and `config` object exist
    # or are implicitly handled by the surrounding code not shown.
    # We will insert the new logic as requested.
    
    # Placeholder for vocab_map and config if they are not global
    # In a real scenario, these would be properly defined.
    vocab_map = {} # Assuming this is defined globally or passed around
    class Config:
        def __init__(self, dataset_mode):
            self.dataset_mode = dataset_mode
    config = Config(dataset_mode) # Using the dataset_mode defined above
    
    vocab_size = 14  # 0..13 (vanilla only)
    CAR_TOKEN_ID = None
    
    # BasicFour Concat mode: unique CAR tokens
    if config.dataset_mode == "basicfour_concat":
        # Add Operation-specific CAR tokens
        # 14: <CAR_+>
        # 15: <CAR_->
        # 16: <CAR_*>
        # 17: <CAR_/>
        vocab_map['<CAR_+>'] = 18
        vocab_map['<CAR_->'] = 19
        vocab_map['<CAR_*>'] = 20
        vocab_map['<CAR_/>'] = 21
        vocab_size = 22
        # We don't have a single CAR_TOKEN_ID anymore, but we can define a map or handle it per op
        CAR_TOKENS = {'+': 18, '-': 19, '*': 20, '/': 21}
        CAR_TOKEN_ID = None # Should not be used generically
    
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
                
                # Buffer for numpy conversion
                data_buffer[split].append(structured_obj)
                
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
    
    # Run conversion
    print("\nStarting Numpy Conversion...")
    for split in ["train", "val", "test"]:
        if len(data_buffer[split]) > 0:
            process_and_save_split(
                data_buffer[split], 
                args.output_dir, 
                split, 
                args.max_len, 
                dataset_mode=config.dataset_mode, # Use config wrapper we made
                vocab_map=vocab_map
            )
        else:
            print(f"Skipping {split}, no data.")

    print("Example Structured Outputs:")
    for ex in example_buffer:
        print(json.dumps(ex, indent=2))

if __name__ == "__main__":
    main()
