from typing import Dict, Optional, Sequence
import torch
import numpy as np
import os
import json
import torch.distributed as dist
from dataset.common import PuzzleDatasetMetadata

class AdditionEvaluator:
    required_outputs = {"inputs", "preds"}  # Default, will be updated in __init__ for lilavati1

    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata, dataset_mode: str = "vanilla", digits: int = 3):
        # Vocab mapping (Must match build_addition_dataset.py)
        # 2-11: 0-9
        # 12: +
        # 13: =
        # 14: <CAR> (only for lilavati2/lilavati3)
        self.vocab_map_inv = {i+2: str(i) for i in range(10)}
        self.vocab_map_inv[12] = '+'
        self.vocab_map_inv[13] = '='
        if eval_metadata.vocab_size >= 15:  # Has <CAR> token
            self.vocab_map_inv[14] = '<CAR>'
            self.car_token_id = 14
        else:
            self.car_token_id = None
        self.vocab_map_inv[0] = 'PAD'
        self.vocab_map_inv[1] = 'MASK'
        
        assert dataset_mode in {"vanilla", "lilavati1", "lilavati2", "lilavati3"}, f"dataset_mode must be 'vanilla', 'lilavati1', 'lilavati2', or 'lilavati3', got {dataset_mode}"
        self.dataset_mode = dataset_mode
        self.digits = digits
        
        # For lilavati modes, we need preds (lilavati1 now uses same format as lilavati3)
        self.required_outputs = {"inputs", "preds"}
        
        # Validate dataset mode matches vocabulary
        if dataset_mode in {"lilavati1", "lilavati2", "lilavati3"}:
            # All lilavati modes now use CAR token
            assert eval_metadata.vocab_size >= 15, f"Lilavati modes require vocab_size >= 15 (to include <CAR> token), got {eval_metadata.vocab_size}"
        else:
            # Vanilla mode should not have <CAR> token, but we allow vocab_size >= 15 for compatibility
            pass
        
        self.total = 0
        self.digit_correct = 0  # Digit-level accuracy for result
        self.sequence_correct = 0  # Sequence-level accuracy for result (all digits correct)
        self.carry_correct = 0  # Carry digit-level accuracy (lilavati1/lilavati2/lilavati3)
        self.carry_total = 0  # Total carry predictions (lilavati1/lilavati2/lilavati3)
        self.carry_sequence_correct = 0  # Carry sequence-level accuracy (all carry digits correct)

    def begin_eval(self):
        self.total = 0
        self.digit_correct = 0
        self.sequence_correct = 0
        self.carry_correct = 0
        self.carry_total = 0
        self.carry_sequence_correct = 0

    def decode(self, seq):
        # seq is tensor or numpy
        # decode to string
        res = ""
        for token in seq:
            token = int(token)
            if token in self.vocab_map_inv:
                res += self.vocab_map_inv[token]
            else:
                res += "?"
        return res
    
    def compute_carries(self, a: int, b: int) -> list:
        """Compute carry digits from least significant to most significant."""
        carries = []
        carry = 0
        for pos in range(self.digits):
            a_digit = (a // (10 ** pos)) % 10
            b_digit = (b // (10 ** pos)) % 10
            total = a_digit + b_digit + carry
            carry = total // 10
            carries.append(carry)
        return carries

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # batch["inputs"] has input
        # preds["preds"] has predicted tokens
        # For lilavati2/lilavati3: preds["carry_preds"] may have separate carry predictions
        
        inp = batch["inputs"].cpu().numpy()
        pred_seq = preds["preds"].cpu().numpy()
        carry_preds = preds.get("carry_preds", None)
        if carry_preds is not None:
            carry_preds = carry_preds.cpu().numpy()
        inputs_carry = batch.get("inputs_carry", None)
        if inputs_carry is not None:
            inputs_carry = inputs_carry.cpu().numpy()
        labels = batch.get("labels", None)
        if labels is not None:
            labels = labels.cpu().numpy()
        
        for i in range(len(inp)):
            self.total += 1
            
            # Get LHS from input
            input_tokens = inp[i]
            input_str = self.decode(input_tokens)
            
            try:
                # Parse LHS
                if '=' not in input_str:
                    continue
                lhs_str = input_str.split('=')[0]
                if '+' not in lhs_str:
                    continue
                
                a_str, b_str = lhs_str.split('+')
                a = int(a_str)
                b = int(b_str)
                expected_res = a + b
                
                # Find where result starts (after '=')
                # Input format: "XXX+YYY=MMMM..." or "XXX+YYY=MMMM <CAR> MMM..."
                eq_idx = None
                for j, tok in enumerate(input_tokens):
                    if tok == 13:  # '='
                        eq_idx = j
                        break
                
                if eq_idx is None:
                    continue
                
                if self.dataset_mode == "vanilla":
                    # Result is max_result_digits after '='
                    max_result_digits = self.digits + 1
                    result_start = eq_idx + 1
                    result_end = result_start + max_result_digits
                    
                    pred_result_tokens = pred_seq[i, result_start:result_end]
                    result_str = self.decode(pred_result_tokens)
                    
                    # Check for invalid tokens
                    if '?' in result_str or 'PAD' in result_str or 'MASK' in result_str or '<CAR>' in result_str:
                        continue
                    
                    # Parse predicted result
                    pred_res = int(result_str)
                    
                    # Compute digit-level accuracy
                    expected_str = f"{expected_res:0{max_result_digits}d}"
                    pred_str = result_str.rjust(max_result_digits, '0')[:max_result_digits]
                    
                    digit_matches = sum(1 for j in range(max_result_digits) 
                                      if j < len(expected_str) and j < len(pred_str) 
                                      and expected_str[j] == pred_str[j])
                    self.digit_correct += digit_matches
                    
                    # Sequence-level accuracy
                    if pred_res == expected_res:
                        self.sequence_correct += 1
                
                elif self.dataset_mode == "lilavati1":
                    # Lilavati1: same format as lilavati3 (single sequence with CAR token), but uses only lm_head
                    max_result_digits = self.digits + 1
                    result_start = eq_idx + 1
                    result_end = result_start + max_result_digits
                    
                    pred_result_tokens = pred_seq[i, result_start:result_end]
                    result_str = self.decode(pred_result_tokens)
                    
                    # Check for invalid tokens in result
                    if '?' in result_str or 'PAD' in result_str or 'MASK' in result_str:
                        continue
                    
                    # Find <CAR> token position (from input, not predictions, as model may not predict it yet)
                    car_idx = None
                    for j in range(result_end, len(input_tokens)):
                        if input_tokens[j] == self.car_token_id:
                            car_idx = j
                            break
                    
                    if car_idx is None:
                        continue
                    
                    # Parse predicted result
                    try:
                        pred_res = int(result_str)
                    except ValueError:
                        continue
                    
                    # Get carry predictions from same sequence (after CAR token, from lm_head)
                    carry_start = car_idx + 1
                    carry_end = carry_start + self.digits
                    pred_carry_tokens = pred_seq[i, carry_start:carry_end]
                    carry_str = self.decode(pred_carry_tokens)
                    
                    if '?' not in carry_str and 'PAD' not in carry_str and 'MASK' not in carry_str:
                        # Compute expected carries
                        expected_carries = self.compute_carries(a, b)
                        expected_carry_str = ''.join(str(c) for c in expected_carries)
                        
                        # Compute carry digit-level accuracy
                        carry_matches = sum(1 for j in range(self.digits)
                                          if j < len(expected_carry_str) and j < len(carry_str)
                                          and expected_carry_str[j] == carry_str[j])
                        self.carry_correct += carry_matches
                        self.carry_total += self.digits
                        
                        # Compute carry sequence-level accuracy (all carry digits correct)
                        if carry_matches == self.digits:
                            self.carry_sequence_correct += 1
                    
                    # Compute digit-level accuracy (for result digits only)
                    expected_res_str = f"{expected_res:0{max_result_digits}d}"
                    pred_res_str = result_str.rjust(max_result_digits, '0')[:max_result_digits]
                    
                    digit_matches = sum(1 for j in range(max_result_digits)
                                      if j < len(expected_res_str) and j < len(pred_res_str)
                                      and expected_res_str[j] == pred_res_str[j])
                    self.digit_correct += digit_matches
                    
                    # Sequence-level accuracy (only result needs to be correct; carries are tracked separately)
                    if pred_res == expected_res:
                        self.sequence_correct += 1
                
                else:  # lilavati2 or lilavati3 (with CAR token)
                    # Result is max_result_digits after '=', then <CAR>, then carries
                    max_result_digits = self.digits + 1
                    result_start = eq_idx + 1
                    result_end = result_start + max_result_digits
                    
                    pred_result_tokens = pred_seq[i, result_start:result_end]
                    result_str = self.decode(pred_result_tokens)
                    
                    # Check for invalid tokens in result
                    if '?' in result_str or 'PAD' in result_str or 'MASK' in result_str:
                        continue
                    
                    # Find <CAR> token
                    car_idx = None
                    for j in range(result_end, len(pred_seq[i])):
                        if pred_seq[i, j] == self.car_token_id:
                            car_idx = j
                            break
                    
                    if car_idx is None:
                        continue
                    
                    # Parse predicted result
                    pred_res = int(result_str)
                    
                    # Get predicted carries
                    carry_start = car_idx + 1
                    carry_end = carry_start + self.digits
                    pred_carry_tokens = pred_seq[i, carry_start:carry_end]
                    carry_str = self.decode(pred_carry_tokens)
                    
                    if '?' in carry_str or 'PAD' in carry_str or 'MASK' in carry_str:
                        continue
                    
                    # Compute expected carries
                    expected_carries = self.compute_carries(a, b)
                    expected_carry_str = ''.join(str(c) for c in expected_carries)
                    
                    # Compute digit-level accuracy (for result digits only)
                    expected_res_str = f"{expected_res:0{max_result_digits}d}"
                    pred_res_str = result_str.rjust(max_result_digits, '0')[:max_result_digits]
                    
                    digit_matches = sum(1 for j in range(max_result_digits)
                                      if j < len(expected_res_str) and j < len(pred_res_str)
                                      and expected_res_str[j] == pred_res_str[j])
                    self.digit_correct += digit_matches
                    
                    # Compute carry digit-level accuracy
                    carry_matches = sum(1 for j in range(self.digits)
                                      if j < len(expected_carry_str) and j < len(carry_str)
                                      and expected_carry_str[j] == carry_str[j])
                    self.carry_correct += carry_matches
                    self.carry_total += self.digits
                    
                    # Compute carry sequence-level accuracy (all carry digits correct)
                    if carry_matches == self.digits:
                        self.carry_sequence_correct += 1
                    
                    # Sequence-level accuracy (only result needs to be correct; carries are tracked separately)
                    if pred_res == expected_res:
                        self.sequence_correct += 1
                
            except Exception as e:
                # Skip invalid examples
                continue
                
    def result(self, save_path: Optional[str], rank: int, world_size: int, group=None):
        # Aggregate metrics
        if world_size > 1:
            if self.dataset_mode in {"lilavati1", "lilavati2", "lilavati3"}:
                t = torch.tensor([self.total, self.digit_correct, self.sequence_correct, self.carry_correct, self.carry_total, self.carry_sequence_correct], device="cuda", dtype=torch.long)
            else:
                t = torch.tensor([self.total, self.digit_correct, self.sequence_correct, 0, 0, 0], device="cuda", dtype=torch.long)
            dist.reduce(t, dst=0)
            total = t[0].item()
            digit_corr = t[1].item()
            seq_corr = t[2].item()
            carry_corr = t[3].item()
            carry_tot = t[4].item()
            carry_seq_corr = t[5].item()
        else:
            total = self.total
            digit_corr = self.digit_correct
            seq_corr = self.sequence_correct
            carry_corr = self.carry_correct
            carry_tot = self.carry_total
            carry_seq_corr = self.carry_sequence_correct
            
        if rank == 0:
            # Compute accuracies
            # Digit accuracy: average over all digit positions
            max_result_digits = self.digits + 1
            total_digit_positions = total * max_result_digits
            digit_acc = digit_corr / total_digit_positions if total_digit_positions > 0 else 0.0
            
            # Sequence accuracy
            seq_acc = seq_corr / total if total > 0 else 0.0
            
            # Return metrics with "val/" prefix for consistency with W&B logging
            metrics = {
                "val/digit_accuracy": digit_acc,
                "val/sequence_accuracy": seq_acc,
            }
            
            if self.dataset_mode in {"lilavati1", "lilavati2", "lilavati3"}:
                carry_digit_acc = carry_corr / carry_tot if carry_tot > 0 else 0.0
                carry_seq_acc = carry_seq_corr / total if total > 0 else 0.0
                metrics["val/carry_digit_accuracy"] = carry_digit_acc
                metrics["val/carry_sequence_accuracy"] = carry_seq_acc
                # Keep backward compatibility
                metrics["val/carry_accuracy"] = carry_digit_acc
                print(f"Addition Eval ({self.dataset_mode}): digit_acc={digit_acc:.4f}, seq_acc={seq_acc:.4f}, carry_digit_acc={carry_digit_acc:.4f}, carry_seq_acc={carry_seq_acc:.4f} ({seq_corr}/{total})")
            else:
                print(f"Addition Eval ({self.dataset_mode}): digit_acc={digit_acc:.4f}, seq_acc={seq_acc:.4f} ({seq_corr}/{total})")
            
            return metrics
        return None

