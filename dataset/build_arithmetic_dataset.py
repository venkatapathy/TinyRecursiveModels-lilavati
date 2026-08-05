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
    'R': 17,
    '<RES>': 22,
    '<SEP>': 23,
}
PAD_ID = 0
MASK_ID = 1
IGNORE_LABEL_ID = -100

# How many PAD positions after the target are supervised as an explicit
# end-of-answer terminator. Kept small on purpose -- see the label
# construction in build_dataset() for why supervising the whole field fails.
N_TERMINATOR_PADS = 1

# Vocab size once the state/answer tokens are in play (ids 0..23).
VOCAB_SIZE_STATE = 24
# Vocab size for the plain no-scratchpad format (ids 0..17).
VOCAB_SIZE_PLAIN = 18

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
                 seed: int = 42,
                 max_divisor_digits: int = 3):
        self.op = op
        self.train_limit = train_max_res_digits
        self.test_limit = test_max_res_digits
        self.allow_zero = allow_zero
        self.rng = random.Random(seed)
        # Long-division remainders are bounded by the divisor, so an
        # unbounded divisor makes the remainder trace O(L * len(d)) tokens --
        # quadratic in the problem length, and worse, it makes the per-step
        # state size grow with the input. That destroys the position
        # invariance that the whole length-generalization argument rests on.
        # Bounding the divisor keeps each step's state constant-width, which
        # is the standard long-division setting.
        self.max_divisor_digits = max_divisor_digits
        
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
        Sample (a, b) bucketed by MAX OPERAND DIGIT LENGTH.

        This is the single ID/OOD definition used everywhere -- the sampler,
        the eval harness, and the SLM/Qwen path. It replaces the previous
        scheme, which targeted *result* length here while the evaluator
        thresholded on *operand* length, so the ID/OOD boundary in the tables
        never matched the boundary the data was built around.

        `is_train=True` draws the in-distribution range [1, train_limit];
        otherwise the strictly out-of-distribution range
        [train_limit+1, test_limit]. Both the train AND val splits must be
        generated with is_train=True -- a val split drawn from the OOD range
        is not a validation set.

        Sampling is exact rather than rejection-based: one operand gets
        exactly L digits and the other gets 1..L, so the realised length
        histogram matches the requested one by construction and there is no
        silent fallback.
        """
        if is_train:
            lo, hi = 1, self.train_limit
        else:
            lo = self.train_limit + 1 if self.train_limit < self.test_limit else 1
            hi = self.test_limit
        if lo > hi:
            raise ValueError(
                f"Empty length range [{lo}, {hi}] for op {self.op} "
                f"(train_limit={self.train_limit}, test_limit={self.test_limit})"
            )

        target_len = self.rng.randint(lo, hi)

        if self.op == "/":
            # Prompt operands are the dividend A and the divisor D, and
            # A = q*d >= d, so max operand length == len(A). Pick d first,
            # then pick q so that len(q*d) is exactly target_len. Division
            # previously returned before the length check entirely, leaving
            # dividend length uncontrolled.
            len_d = self.rng.randint(1, min(target_len, self.max_divisor_digits))
            d = self._sample_n_digits(len_d)
            if d == 0:
                d = 1

            lo_a = 10 ** (target_len - 1)
            hi_a = 10 ** target_len - 1
            lo_q = -(-lo_a // d)          # ceil(lo_a / d)
            hi_q = hi_a // d              # floor(hi_a / d)
            if lo_q > hi_q:
                # Only possible if d has more digits than target_len, which
                # the len_d draw above excludes.
                lo_q = hi_q = max(1, lo_q)
            q = self.rng.randint(max(1, lo_q), max(1, hi_q))
            return q, d

        # One operand has exactly target_len digits; the other is 1..target_len.
        other_len = self.rng.randint(1, target_len)
        if self.rng.random() < 0.5:
            len_a, len_b = target_len, other_len
        else:
            len_a, len_b = other_len, target_len

        a = self._sample_n_digits(len_a)
        b = self._sample_n_digits(len_b)

        if self.op == "-":
            # Results are kept non-negative; swapping preserves both lengths.
            if a < b:
                a, b = b, a

        if self.op == "*":
            # Trailing zeros make multiplication trivially easy, so strip them
            # most of the time -- but re-pad to preserve the digit length that
            # this example is bucketed under.
            if self.rng.random() < 0.8:
                a = self._strip_trailing_zeros(a, len_a)
                b = self._strip_trailing_zeros(b, len_b)

        return a, b

    def _strip_trailing_zeros(self, n: int, n_digits: int) -> int:
        """Remove trailing zeros while keeping the digit count at n_digits."""
        if n % 10 != 0:
            return n
        s = str(n).rstrip("0")
        if not s:
            s = "1"
        # Re-pad with non-zero digits so the length bucket is preserved.
        while len(s) < n_digits:
            s += str(self.rng.randint(1, 9))
        return int(s)

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

def process_and_save_split(data_items: List[Dict], output_dir: str, split_name: str, max_len: int, dataset_mode: str = "vanilla", vocab_map: Dict = None, vocab_size: int = 22):
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

        # BASICFOUR CONCAT LOGIC
        concat_suffix_ids = []
        if dataset_mode in ["basicfour_concat", "basic_concat_reverse"]:
            # 1. Determine CAR token
            car_token_id = vocab_map.get(f"<CAR_{op}>")
            if car_token_id is None:
                 # Fallback or error?
                 # Should fail if map is correct.
                 raise ValueError(f"Missing CAR token for op {op}")
            
            res_token_id = vocab_map.get('<RES>')
            
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
            
            # Map state values to vocab ids.
            #
            # Addition carries and subtraction borrows are always 0/1, but
            # multiplication column carries and division remainders are
            # unbounded (e.g. 9*9 + 8 -> 89). Writing those as bare digits
            # loses the value boundary, which makes the trace positionally
            # ambiguous at exactly the OOD lengths we care about. Every value
            # is therefore terminated by <SEP>, so the trace is
            #   <CAR_op> v0 <SEP> v1 <SEP> ... v_{n-1} <SEP>
            # and value k is always recoverable regardless of magnitude.
            sep_id = vocab_map['<SEP>']
            suffix_seq_ids = [car_token_id]
            for val in inter_val_list:
                for char in str(val):
                    suffix_seq_ids.append(VOCAB_MAP[char])
                suffix_seq_ids.append(sep_id)

            concat_suffix_ids = suffix_seq_ids

        # Aux Labels: IGNORE on Prompt, Carry ids on Result
        inter_list = []
        if dataset_mode not in ["basicfour_concat", "basic_concat_reverse"]:
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
            
        # ------------------------------------------------------------------
        # MASKED OUTPUT FIELD
        #
        # The model is bidirectional (causal=False in both TRM and the
        # Transformer baseline). Feeding it the answer and asking it to
        # predict a one-position shift of its own input lets position t
        # attend to position t+1 and copy the very token it is scored on.
        # Instead the prompt is followed by a FIXED-WIDTH field of MASK
        # tokens, and the targets live in the label array only.
        #
        # The field width deliberately does NOT depend on the answer length:
        # sizing it to len(result) would leak the answer's magnitude. Every
        # example gets the same field, and the model must learn where its
        # own output stops by predicting PAD.
        # ------------------------------------------------------------------
        target_ids = result_ids + concat_suffix_ids
        if dataset_mode == "basic_concat_reverse":
            target_ids = concat_suffix_ids + [res_token_id] + result_ids

        field_width = max_len - len(prompt_ids)
        if field_width < len(target_ids) + N_TERMINATOR_PADS:
            raise ValueError(
                f"Example exceeds max_len: prompt={len(prompt_ids)} + "
                f"target={len(target_ids)} + terminator={N_TERMINATOR_PADS} "
                f"> max_len={max_len} (op={op}, A={A}, B={B}). Increase "
                f"--max_len; truncating here would silently corrupt the OOD "
                f"examples."
            )

        input_ids = prompt_ids + [MASK_ID] * field_width

        # Labels are position-aligned with inputs (no shift): the prompt span
        # is ignored and the target occupies the head of the field.
        #
        # Termination is taught by supervising exactly N_TERMINATOR_PADS PAD
        # positions immediately after the target -- NOT by supervising PAD
        # across the whole remaining field. The field is ~500 wide while a
        # target is ~5-20 tokens, so supervising every trailing slot puts
        # 96-99% of the cross-entropy mass on "emit PAD" and starves the
        # arithmetic signal by two orders of magnitude. Everything past the
        # terminator is IGNORE_LABEL_ID; decoding stops at the first PAD, so
        # those positions are unobservable anyway.
        n_trailing = field_width - len(target_ids) - N_TERMINATOR_PADS
        lm_labels = (
            [IGNORE_LABEL_ID] * len(prompt_ids)
            + target_ids
            + [PAD_ID] * N_TERMINATOR_PADS
            + [IGNORE_LABEL_ID] * n_trailing
        )

        # Aux (dual-head) labels align to wherever the answer digits sit.
        aux_labels = [IGNORE_LABEL_ID] * max_len
        if len(inter_list) > 0:
            target_len = len(result_ids)
            if len(inter_list) >= target_len:
                aligned_inter = inter_list[-target_len:]
            else:
                aligned_inter = [0] * (target_len - len(inter_list)) + inter_list

            result_start = len(prompt_ids) + (len(target_ids) - len(result_ids))
            aux_labels[result_start:result_start + target_len] = aligned_inter

        assert len(input_ids) == len(lm_labels) == len(aux_labels) == max_len

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

    # Every sequence is already exactly max_len (prompt + fixed mask field),
    # so there is nothing to pad. Verify rather than silently reshape:
    # truncation here was previously how overlong OOD examples got corrupted.
    for idx, (inp, lm, aux) in enumerate(zip(inputs_list, labels_lm_list, labels_aux_list)):
        if not (len(inp) == len(lm) == len(aux) == max_len):
            raise ValueError(
                f"Example {idx} has inconsistent length: "
                f"inputs={len(inp)}, labels={len(lm)}, aux={len(aux)}, expected {max_len}"
            )

        padded_inputs.append(inp)
        padded_lm.append(lm)
        padded_aux.append(aux)

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
        "vocab_size": vocab_size,
        "pad_id": PAD_ID,
        "mask_id": MASK_ID,
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
    # NOTE: these bound the MAX OPERAND digit length, not the result length.
    # The old names are kept so existing command lines keep working, but the
    # semantics changed: operand length is now the single ID/OOD definition,
    # shared by the sampler, the eval harness and the SLM path.
    parser.add_argument("--train_max_result_digits", "--train_max_digits",
                        dest="train_max_digits", type=int, default=8,
                        help="Max operand digits for the train and val (ID) splits")
    parser.add_argument("--test_max_result_digits", "--test_max_digits",
                        dest="test_max_digits", type=int, default=12,
                        help="Max operand digits for the test (OOD) split")
    parser.add_argument("--allow_zero", action="store_true", help="Allow 0 operands")
    parser.add_argument("--max_divisor_digits", type=int, default=3,
                        help="Cap on divisor length, so long-division remainders stay constant-width")
    parser.add_argument("--max_len", type=int, required=True, help="Padded sequence length (e.g., 512 for 32-digit ops)")
    parser.add_argument("--dataset_mode", type=str, default="basicfour_concat", choices=["vanilla", "basicfour_concat", "basic_concat_reverse"], help="Dataset mode")
    
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
    
    vocab_size = VOCAB_SIZE_PLAIN  # 0..17 (no state tokens)

    # BasicFour Concat modes add the op-specific state markers, the answer
    # marker and the value separator. Keep this in sync with VOCAB_MAP and
    # with the inverse map in evaluators/harness.py.
    if config.dataset_mode in ["basicfour_concat", "basic_concat_reverse"]:
        vocab_map['<CAR_+>'] = 18
        vocab_map['<CAR_->'] = 19
        vocab_map['<CAR_*>'] = 20
        vocab_map['<CAR_/>'] = 21
        vocab_map['<RES>'] = 22
        vocab_map['<SEP>'] = 23
        vocab_size = VOCAB_SIZE_STATE  # 0..23

    for op in OPS:
        sampler = Sampler(
            op, 
            args.train_max_digits,
            args.test_max_digits,
            allow_zero=args.allow_zero,
            seed=args.seed + ord(op),
            max_divisor_digits=args.max_divisor_digits,
        )
        
        for split, num_examples in splits_config:
            # val is an IN-DISTRIBUTION held-out split and must be sampled
            # from the same length range as train. Previously only "train"
            # set this flag, so val was drawn from the OOD range and every
            # number reported as "ID" was actually measured out of
            # distribution.
            is_train = split in ("train", "val")
            # Counts are totals across all four operations.
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
                # Operand-length histogram is the one that matches the ID/OOD
                # definition, so it must actually be populated -- it was
                # declared and left empty before, which is why the realised
                # length distribution was never checkable.
                operand_len = max(len(data["A"]), len(data["B"]))
                key = f"{split}_{operand_len}"
                stats["hist_operand_max_digits"][key] = (
                    stats["hist_operand_max_digits"].get(key, 0) + 1
                )
                
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
                vocab_map=vocab_map,
                vocab_size=vocab_size
            )
        else:
            print(f"Skipping {split}, no data.")

    print("Example Structured Outputs:")
    for ex in example_buffer:
        print(json.dumps(ex, indent=2))

if __name__ == "__main__":
    main()
