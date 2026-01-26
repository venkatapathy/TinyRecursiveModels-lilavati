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
        self.vocab_map_inv = {i + 2: str(i) for i in range(10)}
        self.vocab_map_inv[12] = '+'
        self.vocab_map_inv[13] = '-'
        self.vocab_map_inv[14] = '*'
        self.vocab_map_inv[15] = '/'
        self.vocab_map_inv[16] = '='
        self.vocab_map_inv[17] = 'R'
        self.vocab_map_inv[0] = 'PAD'
        self.vocab_map_inv[0] = 'PAD'
        self.vocab_map_inv[1] = 'MASK'
        
        # BasicFour Concat tokens
        # We can map them to explicit strings or special markers
        self.vocab_map_inv[18] = '<CAR_+>'
        self.vocab_map_inv[19] = '<CAR_->'
        self.vocab_map_inv[20] = '<CAR_*>'
        self.vocab_map_inv[21] = '<CAR_/>'

        self.dataset_mode = kwargs.get("dataset_mode", "vanilla")
        
        # Per-operation counters
        self.reset_counters()
    
    def reset_counters(self):
        """Reset all counters."""
        self.total = 0
        self.correct = 0
        
        # Per-operation tracking
        self.op_total = {'+': 0, '-': 0, '*': 0, '/': 0}
        self.op_correct = {'+': 0, '-': 0, '*': 0, '/': 0}

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
                    if self.dataset_mode == "basic_concat_reverse":
                        continue # Skip marker, continue reading (Trace + Result)
                    else:
                        break # Stop at CAR token (standard concat mode stops here)
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
            
            # Find position of '=' to extract prediction
            try:
                eq_pos = input_str.index('=')
            except ValueError:
                continue
            
            # Get predicted result (tokens after '=')
            # preds are next-token predictions (aligned with input positions).
            # The prediction made AT the position of `=` (eq_pos) is the first token of the result.
            pred_tokens = pred_seqs[i, eq_pos:]
            pred_str = self.decode(pred_tokens)
            
            # Compute expected result
            expected = self.compute_expected(op, a, b)
            if expected is None:
                continue
            
            # Compare
            matched = False
            if self.dataset_mode == "basic_concat_reverse":
                # In reverse mode, output is Trace + Result.
                # Since we don't have a separator, we check if the output *ends with* the expected result.
                # This approximates checking "Trace followed by Result".
                if pred_str.endswith(expected):
                    matched = True
            else:
                if pred_str == expected:
                    matched = True
            
            if matched:
                self.correct += 1
                self.op_correct[op] += 1
                
            if self.total <= 5:
                 print(f"DEBUG: Input='{input_str}' Op='{op}' A={a} B={b} Expected='{expected}' PredTokens={pred_tokens} PredStr='{pred_str}' Correct={pred_str == expected}")

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
        ]
        
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
            
            # Compute accuracies
            overall_acc = correct / total if total > 0 else 0.0
            add_acc = add_correct / add_total if add_total > 0 else 0.0
            sub_acc = sub_correct / sub_total if sub_total > 0 else 0.0
            mul_acc = mul_correct / mul_total if mul_total > 0 else 0.0
            div_acc = div_correct / div_total if div_total > 0 else 0.0
            
            print(f"\nBasicFour Evaluation Results:")
            print(f"  Overall:        {correct}/{total} = {overall_acc:.4f}")
            print(f"  Addition (+):   {add_correct}/{add_total} = {add_acc:.4f}")
            print(f"  Subtraction (-): {sub_correct}/{sub_total} = {sub_acc:.4f}")
            print(f"  Multiplication (*): {mul_correct}/{mul_total} = {mul_acc:.4f}")
            print(f"  Division (/):   {div_correct}/{div_total} = {div_acc:.4f}")
            
            return {
                "basicfour/accuracy": overall_acc,
                "basicfour/add_accuracy": add_acc,
                "basicfour/sub_accuracy": sub_acc,
                "basicfour/mul_accuracy": mul_acc,
                "basicfour/div_accuracy": div_acc,
            }
        
        return None
