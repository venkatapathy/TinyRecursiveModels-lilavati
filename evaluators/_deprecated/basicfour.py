"""
Evaluator for BasicFour dataset: Addition, Subtraction, Multiplication, Division.

Tracks overall accuracy and per-operation accuracy.
"""

from typing import Dict, Optional
import torch
import torch.distributed as dist
from dataset.common import PuzzleDatasetMetadata


class BasicFourEvaluator:
    required_outputs = {"inputs", "preds"}

    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata, **kwargs):
        # Vocabulary mapping (must match build_basicfour_dataset.py)
        # Standard tokens 0-17
        self.vocab_map_inv = {i + 2: str(i) for i in range(10)}
        self.vocab_map_inv[12] = '+'
        self.vocab_map_inv[13] = '-'
        self.vocab_map_inv[14] = '*'
        self.vocab_map_inv[15] = '/'
        self.vocab_map_inv[16] = '='
        self.vocab_map_inv[17] = 'R'
        self.vocab_map_inv[0] = 'PAD'
        self.vocab_map_inv[1] = 'MASK'
        
        # Determine model vocab size from metadata
        self.vocab_size = getattr(eval_metadata, "vocab_size", 23)
        
        # BasicFour Concat tokens (18-22)
        # ONLY add them if model vocab supports them to avoid index errors
        self.car_token_ids = set()
        self.res_token_id = None
        
        if self.vocab_size > 18:
            self.vocab_map_inv[18] = '<CAR_+>'
            self.car_token_ids.add(18)
        if self.vocab_size > 19:
            self.vocab_map_inv[19] = '<CAR_->'
            self.car_token_ids.add(19)
        if self.vocab_size > 20:
            self.vocab_map_inv[20] = '<CAR_*>'
            self.car_token_ids.add(20)
        if self.vocab_size > 21:
            self.vocab_map_inv[21] = '<CAR_/>'
            self.car_token_ids.add(21)
        if self.vocab_size > 22:
            self.vocab_map_inv[22] = '<RES>'
            self.res_token_id = 22

        self.dataset_mode = kwargs.get("dataset_mode", "vanilla")
        self.train_digits = kwargs.get("train_digits", 8)
        
        # Per-operation counters
        self.reset_counters()
    
    def reset_counters(self):
        """Reset all counters."""
        self.total = 0
        self.correct = 0
        
        # Per-operation tracking
        self.op_total = {'+': 0, '-': 0, '*': 0, '/': 0}
        self.op_correct = {'+': 0, '-': 0, '*': 0, '/': 0}
        
        # ID/OOD counters (ID: <= 8 digit operands)
        self.id_total = 0
        self.id_correct = 0
        self.ood_total = 0
        self.ood_correct = 0
        
        # Digit Accuracy Counters
        self.total_digits = 0
        self.correct_digits = 0
        
        self.id_total_digits = 0
        self.id_correct_digits = 0
        self.ood_total_digits = 0
        self.ood_correct_digits = 0

        # Per-digit length accuracy (tracking max operand length)
        # We track 1-32 digits
        self.digit_len_total = {i: 0 for i in range(1, 41)}
        self.digit_len_correct = {i: 0 for i in range(1, 41)}

        # Carry/Trace Accuracy Counters
        self.carry_total = 0
        self.carry_correct = 0
        self.op_carry_total = {'+': 0, '-': 0, '*': 0, '/': 0}
        self.op_carry_correct = {'+': 0, '-': 0, '*': 0, '/': 0}

    def begin_eval(self):
        """Called at the start of evaluation."""
        self.reset_counters()

    def decode(self, seq) -> str:
        """Decode token sequence to string."""
        result = ""
        for token in seq:
            token = int(token)
            if token in self.vocab_map_inv:
                char = self.vocab_map_inv[token]
                if char in ('PAD', 'MASK'):
                    break  # Stop at padding/mask
                if char.startswith('<CAR_'):
                    # For both reverse and concat modes, we want to skip the trace markers
                    # and read the actual result digits.
                    # In concat mode: <CAR_*> Trace... Result
                    # In reverse mode: Trace... Result
                    continue 
                result += char
            else:
                result += "?"
        return result

    def detect_operation(self, input_str: str) -> tuple:
        """
        Detect operation and parse operands from input string.
        Returns (op, a, b) or (None, None, None) if parsing fails.
        """
        # Remove everything after '='
        lhs = input_str.split('=')[0]
        
        # Try each operation (order matters: check '-' last since it can be negative sign)
        for op in ['+', '*', '/']:
            if op in lhs:
                parts = lhs.split(op)
                if len(parts) == 2:
                    try:
                        a = int(parts[0])
                        b = int(parts[1])
                        return op, a, b
                    except ValueError:
                        continue
        
        # Handle subtraction (could have negative operands, but we don't generate those)
        if '-' in lhs:
            # Find the subtraction operator (not a negative sign at start)
            for i, c in enumerate(lhs):
                if c == '-' and i > 0:
                    try:
                        a = int(lhs[:i])
                        b = int(lhs[i+1:])
                        return '-', a, b
                    except ValueError:
                        continue
        
        return None, None, None

    def compute_expected(self, op: str, a: int, b: int) -> str:
        """Compute expected result string."""
        if op == '+':
            return str(a + b)
        elif op == '-':
            result = a - b
            if result < 0:
                return '-' + str(abs(result))
            return str(result)
        elif op == '*':
            return str(a * b)
        elif op == '/':
            if b == 0:
                return None
            quotient = a // b
            remainder = a % b
            if remainder == 0:
                return str(quotient)
            return f"{quotient}R{remainder}"
        return None

    def compute_expected_trace(self, op: str, a: int, b: int) -> str:
        """Compute expected trace/carry string."""
        if op == '+':
            # Addition carries (LSB -> MSB)
            y = a + b
            a_digits = [int(d) for d in str(a)[::-1]]
            b_digits = [int(d) for d in str(b)[::-1]]
            max_len = len(str(y))
            carries = []
            c = 0
            for i in range(max_len):
                da = a_digits[i] if i < len(a_digits) else 0
                db = b_digits[i] if i < len(b_digits) else 0
                s = da + db + c
                c_out = s // 10
                carries.append(c_out)
                c = c_out
            return "".join(map(str, carries))
        
        elif op == '-':
            # Subtraction borrows (LSB -> MSB)
            if a < b: return ""
            a_digits = [int(d) for d in str(a)[::-1]]
            b_digits = [int(d) for d in str(b)[::-1]]
            y_str = str(a - b)
            len_y = len(y_str)
            borrows = []
            b_in = 0
            for i in range(len(a_digits)):
                da = a_digits[i]
                db = b_digits[i] if i < len(b_digits) else 0
                diff = da - db - b_in
                if diff < 0:
                    b_out = 1
                else:
                    b_out = 0
                borrows.append(b_out)
                b_in = b_out
            return "".join(map(str, borrows[:len_y]))

        elif op == '*':
            # Multiplication carries (LSB -> MSB)
            y = a * b
            a_digits = [int(d) for d in str(a)[::-1]]
            b_digits = [int(d) for d in str(b)[::-1]]
            num_cols = len(str(y))
            carry = 0
            col_carries = []
            for k in range(num_cols):
                s_k = carry
                for i in range(k + 1):
                    j = k - i
                    if i < len(a_digits) and j < len(b_digits):
                        s_k += a_digits[i] * b_digits[j]
                carry_out = s_k // 10
                col_carries.append(carry_out)
                carry = carry_out
            return "".join(map(str, col_carries))

        elif op == '/':
            # Division remainders (MSB -> LSB)
            # A / B = Q. Here A = a, B = b, Y = a // b.
            # Long division on a by b.
            a_str = str(a)
            current_rem = 0
            remainders = []
            for char in a_str:
                digit = int(char)
                current_rem = current_rem * 10 + digit
                current_rem = current_rem % b
                remainders.append(current_rem)
            return "".join(map(str, remainders))
            
        return ""

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """Process a batch of predictions."""
        inputs = batch["inputs"].cpu().numpy()
        pred_seqs = preds["preds"].cpu().numpy()
        
        for i in range(len(inputs)):
            self.total += 1
            
            # Decode input
            input_str = self.decode(inputs[i])
            
            # Detect operation and operands
            op, a, b = self.detect_operation(input_str)
            if op is None:
                continue
            
            self.op_total[op] += 1
            
            # Find position of '=' token in the input token sequence
            # CRITICAL: We need the TOKEN position, not the decoded string position!
            # The decoded string skips CAR tokens, so positions don't match.
            eq_token_pos = None
            for idx, token in enumerate(inputs[i]):
                if int(token) == 16:  # '=' token ID
                    eq_token_pos = idx
                    break
            
            if eq_token_pos is None:
                continue
            
            # Get predicted result and trace based on dataset mode
            # preds are next-token predictions (aligned with input positions).
            # The prediction made AT the position of `=` (eq_token_pos) is the first token after '='.
            
            # DEBUG: Check for out-of-range tokens in this specific sequence
            cur_preds = pred_seqs[i, eq_token_pos:]
            if (cur_preds >= self.vocab_size).any() or (cur_preds < 0).any():
                print(f"CRITICAL: Found out-of-range prediction tokens in example {i}!")
                print(f"          Vocab Size: {self.vocab_size}")
                print(f"          Max Pred:    {cur_preds.max()}")
                print(f"          Min Pred:    {cur_preds.min()}")
                print(f"          Input:       {input_str}")
                # CLIP to avoid assert
                cur_preds = np.clip(cur_preds, 0, self.vocab_size - 1)
            
            pred_tokens_raw = cur_preds
            
            pred_tokens = []
            trace_tokens = []
            
            # Extract result and trace portions based on mode
            if self.dataset_mode == "basicfour_concat":
                # Format: = Y <CAR_op> trace
                # Result comes FIRST, before the CAR token
                car_token_idx = None
                for idx, token in enumerate(pred_tokens_raw):
                    if int(token) in self.car_token_ids:
                        car_token_idx = idx
                        break
                
                if car_token_idx is not None:
                    pred_tokens = pred_tokens_raw[:car_token_idx]
                    trace_tokens = pred_tokens_raw[car_token_idx+1:]
                else:
                    pred_tokens = pred_tokens_raw
                    trace_tokens = []

            elif self.dataset_mode == "basic_concat_reverse":
                # Format: = <CAR_op> trace <RES> Y
                # Result comes LAST, after the RES token
                car_token_idx = None
                res_token_idx = None
                
                for idx, token in enumerate(pred_tokens_raw):
                    t_int = int(token)
                    if car_token_idx is None and t_int in self.car_token_ids:
                        car_token_idx = idx
                    elif res_token_idx is None and t_int == self.res_token_id:
                        res_token_idx = idx
                
                if car_token_idx is not None and res_token_idx is not None:
                    trace_tokens = pred_tokens_raw[car_token_idx+1:res_token_idx]
                    pred_tokens = pred_tokens_raw[res_token_idx+1:]
                elif car_token_idx is not None:
                    trace_tokens = pred_tokens_raw[car_token_idx+1:]
                    pred_tokens = []
                elif res_token_idx is not None:
                    trace_tokens = []
                    pred_tokens = pred_tokens_raw[res_token_idx+1:]
                else:
                    trace_tokens = []
                    pred_tokens = []
            else:
                # Vanilla mode: everything after '=' is the result, no trace
                pred_tokens = pred_tokens_raw
                trace_tokens = []
            
            # Decode the extracted portions
            pred_str = self.decode(pred_tokens)
            trace_str = self.decode(trace_tokens)
            
            # Compute expected result
            expected = self.compute_expected(op, a, b)
            if expected is None:
                continue
            
            # Compare - now use exact match for all modes
            matched = (pred_str == expected)
            
            if matched:
                self.correct += 1
                self.op_correct[op] += 1
            
            # ID vs OOD (Threshold train_digits)
            max_digits = max(len(str(a)), len(str(b)))
            is_id = max_digits <= self.train_digits
            
            if is_id:
                self.id_total += 1
                if matched: self.id_correct += 1
            else:
                self.ood_total += 1
                if matched: self.ood_correct += 1
            
            # Per-digit length tracking
            if max_digits in self.digit_len_total:
                self.digit_len_total[max_digits] += 1
                if matched:
                    self.digit_len_correct[max_digits] += 1
                
            # Digit Accuracy - compare result portion directly
            if expected is not None:
                # Right-aligned comparison
                t_rev = expected[::-1]
                p_rev = pred_str[::-1]
                match_count = 0
                max_len = max(len(t_rev), len(p_rev))
                if max_len > 0:
                    for k in range(min(len(t_rev), len(p_rev))):
                        if t_rev[k] == p_rev[k]:
                            match_count += 1
                
                self.total_digits += max_len
                self.correct_digits += match_count
                
                if is_id:
                    self.id_total_digits += max_len
                    self.id_correct_digits += match_count
                else:
                    self.ood_total_digits += max_len
                    self.ood_correct_digits += match_count
            
            # Carry/Trace Accuracy
            if self.dataset_mode in ["basicfour_concat", "basic_concat_reverse"]:
                expected_trace = self.compute_expected_trace(op, a, b)
                if expected_trace:
                    # Digit-level matching for trace
                    t_rev = expected_trace[::-1]
                    p_rev = trace_str[::-1]
                    trace_match_count = 0
                    trace_max_len = max(len(t_rev), len(p_rev))
                    if trace_max_len > 0:
                        for k in range(min(len(t_rev), len(p_rev))):
                            if t_rev[k] == p_rev[k]:
                                trace_match_count += 1
                        
                        self.carry_total += trace_max_len
                        self.carry_correct += trace_match_count
                        self.op_carry_total[op] += trace_max_len
                        self.op_carry_correct[op] += trace_match_count
                
            if self.total <= 20:
                 print(f"DEBUG: Input='{input_str}' Op='{op}' A={a} B={b}")
                 print(f"       Expected='{expected}'")
                 print(f"       eq_token_pos={eq_token_pos}")
                 print(f"       PredTokens (raw first 20)={list(pred_tokens_raw[:20])}")
                 if self.dataset_mode == "basic_concat_reverse":
                     print(f"       Looking for RES token (ID {self.res_token_id}) in predictions...")
                     res_positions = [i for i, t in enumerate(pred_tokens_raw) if int(t) == self.res_token_id]
                     print(f"       RES token found at positions: {res_positions}")
                 print(f"       PredTokens (extracted)={list(pred_tokens)}")
                 print(f"       PredStr='{pred_str}'")
                 print(f"       Matched={matched}")
                 print("-" * 30)

    def result(self, save_path: Optional[str], rank: int, world_size: int, group=None) -> Optional[Dict[str, float]]:
        """Aggregate and return metrics."""
        # Prepare tensors for reduction
        # Order: total, correct, +_total, +_correct, -_total, -_correct, *_total, *_correct, /_total, /_correct
        values = [
            self.total, self.correct,
            self.op_total['+'], self.op_correct['+'],
            self.op_total['-'], self.op_correct['-'],
            self.op_total['*'], self.op_correct['*'],
            self.op_total['/'], self.op_correct['/'],
            # New counters
            self.id_total, self.id_correct,
            self.ood_total, self.ood_correct,
            self.id_total_digits, self.id_correct_digits,
            self.ood_total_digits, self.ood_correct_digits,
            self.total_digits, self.correct_digits,
            # Carry counters
            self.carry_total, self.carry_correct,
            self.op_carry_total['+'], self.op_carry_correct['+'],
            self.op_carry_total['-'], self.op_carry_correct['-'],
            self.op_carry_total['*'], self.op_carry_correct['*'],
            self.op_carry_total['/'], self.op_carry_correct['/']
        ]
        
        # Add digit-wise counters to values for reduction
        # (Using a sorted list of keys to ensure consistent order across processes)
        digit_keys = sorted(self.digit_len_total.keys())
        for k in digit_keys:
            values.append(self.digit_len_total[k])
            values.append(self.digit_len_correct[k])
        
        if world_size > 1:
            t = torch.tensor(values, device="cuda", dtype=torch.long)
            dist.reduce(t, dst=0)
            values = t.tolist()
        
        if rank == 0:
            total, correct = values[0], values[1]
            add_total, add_correct = values[2], values[3]
            sub_total, sub_correct = values[4], values[5]
            mul_total, mul_correct = values[6], values[7]
            div_total, div_correct = values[8], values[9]
            
            id_total, id_correct = values[10], values[11]
            ood_total, ood_correct = values[12], values[13]
            id_total_digits, id_correct_digits = values[14], values[15]
            ood_total_digits, ood_correct_digits = values[16], values[17]
            total_digits, correct_digits = values[18], values[19]
            
            carry_total, carry_correct = values[20], values[21]
            add_carry_total, add_carry_correct = values[22], values[23]
            sub_carry_total, sub_carry_correct = values[24], values[25]
            mul_carry_total, mul_carry_correct = values[26], values[27]
            div_carry_total, div_carry_correct = values[28], values[29]
            
            # Start of digit-wise metrics in values list
            idx = 30
            digit_keys = sorted(self.digit_len_total.keys())
            digit_metrics = {}
            for k in digit_keys:
                d_total = values[idx]
                d_correct = values[idx+1]
                idx += 2
                if d_total > 0:
                    acc = d_correct / d_total
                    digit_metrics[f"basicfour/accuracy_digit_len_{k}"] = acc
                    # Also keep a copy for easier summary printing if needed
                    # but we'll just log them all.
            
            # Compute accuracies
            overall_acc = correct / total if total > 0 else 0.0
            add_acc = add_correct / add_total if add_total > 0 else 0.0
            sub_acc = sub_correct / sub_total if sub_total > 0 else 0.0
            mul_acc = mul_correct / mul_total if mul_total > 0 else 0.0
            div_acc = div_correct / div_total if div_total > 0 else 0.0
            
            id_acc = id_correct / id_total if id_total > 0 else 0.0
            ood_acc = ood_correct / ood_total if ood_total > 0 else 0.0
            
            id_digit_acc = id_correct_digits / id_total_digits if id_total_digits > 0 else 0.0
            ood_digit_acc = ood_correct_digits / ood_total_digits if ood_total_digits > 0 else 0.0
            overall_digit_acc = correct_digits / total_digits if total_digits > 0 else 0.0
            
            print(f"\nBasicFour Evaluation Results:")
            print(f"  Overall Seq:        {correct}/{total} = {overall_acc:.4f}")
            print(f"  Overall Digit:      {correct_digits}/{total_digits} = {overall_digit_acc:.4f}")
            print(f"  ID Seq (`val` proxy): {id_correct}/{id_total} = {id_acc:.4f}")
            print(f"  ID Digit:           {id_correct_digits}/{id_total_digits} = {id_digit_acc:.4f}")
            print(f"  OOD Seq (`test`):   {ood_correct}/{ood_total} = {ood_acc:.4f}")
            print(f"  OOD Digit:          {ood_correct_digits}/{ood_total_digits} = {ood_digit_acc:.4f}")
            
            print(f"  Addition (+):   {add_correct}/{add_total} = {add_acc:.4f}")
            print(f"  Subtraction (-): {sub_correct}/{sub_total} = {sub_acc:.4f}")
            print(f"  Multiplication (*): {mul_correct}/{mul_total} = {mul_acc:.4f}")
            print(f"  Division (/):   {div_correct}/{div_total} = {div_acc:.4f}")
            
            results = {
                "basicfour/accuracy": overall_acc,
                "basicfour/add_accuracy": add_acc,
                "basicfour/sub_accuracy": sub_acc,
                "basicfour/mul_accuracy": mul_acc,
                "basicfour/div_accuracy": div_acc,
                
                "basicfour/id_accuracy": id_acc,
                "basicfour/ood_accuracy": ood_acc,
                "basicfour/id_digit_accuracy": id_digit_acc,
                "basicfour/ood_digit_accuracy": ood_digit_acc,
                "basicfour/digit_accuracy": overall_digit_acc,
            }
            
            # Merge digit-wise metrics
            results.update(digit_metrics)
            
            # Carry metrics
            results.update({
                "basicfour/carry_accuracy": carry_correct / carry_total if carry_total > 0 else 0.0,
                "basicfour/add_carry_accuracy": add_carry_correct / add_carry_total if add_carry_total > 0 else 0.0,
                "basicfour/sub_carry_accuracy": sub_carry_correct / sub_carry_total if sub_carry_total > 0 else 0.0,
                "basicfour/mul_carry_accuracy": mul_carry_correct / mul_carry_total if mul_carry_total > 0 else 0.0,
                "basicfour/div_carry_accuracy": div_carry_correct / div_carry_total if div_carry_total > 0 else 0.0,
            })
            
            return results
        
        return None
