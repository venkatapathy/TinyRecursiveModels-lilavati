"""
Evaluator for Lilavati Pati model.

Handles reversed-digit format and tracks both addition accuracy
and carry prediction accuracy.
"""

from typing import Dict, Optional
import torch
import torch.distributed as dist
from dataset.common import PuzzleDatasetMetadata


class LilavatiEvaluator:
    """
    Evaluator for Lilavati model with reversed digits.
    
    Metrics:
    - lilavati/accuracy: Overall addition accuracy
    - lilavati/carry_accuracy: Carry prediction accuracy (if available)
    """
    
    required_outputs = {"inputs", "preds", "carry_logits"}
    
    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata):
        # Vocabulary mapping
        self.vocab_map_inv = {i + 2: str(i) for i in range(10)}
        self.vocab_map_inv[12] = '+'
        self.vocab_map_inv[13] = '='
        self.vocab_map_inv[0] = 'PAD'
        self.vocab_map_inv[1] = 'MASK'
        
        self.reset_counters()
    
    def reset_counters(self):
        """Reset all counters."""
        self.total = 0
        self.correct = 0
        self.carry_total = 0
        self.carry_correct = 0
    
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
                    break
                result += char
            else:
                result += "?"
        return result
    
    def reverse_digits(self, s: str) -> str:
        """Reverse digits in a number string."""
        return s[::-1]
    
    def parse_reversed_input(self, input_str: str) -> tuple:
        """
        Parse reversed input like "321+654=".
        Returns (a, b) in standard (non-reversed) form, or (None, None) if parsing fails.
        """
        try:
            lhs = input_str.split('=')[0]
            if '+' not in lhs:
                return None, None
            
            parts = lhs.split('+')
            if len(parts) != 2:
                return None, None
            
            # Reverse back to standard form
            a = int(self.reverse_digits(parts[0]))
            b = int(self.reverse_digits(parts[1]))
            return a, b
        except (ValueError, IndexError):
            return None, None
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """Process a batch of predictions."""
        inputs = batch["inputs"].cpu().numpy()
        pred_seqs = preds["preds"].cpu().numpy()
        
        # Get carries if available
        carries = batch.get("carries", None)
        carry_logits = preds.get("carry_logits", None)
        
        if carries is not None:
            carries = carries.cpu().numpy()
        if carry_logits is not None:
            carry_logits = carry_logits.cpu().numpy()
        
        for i in range(len(inputs)):
            self.total += 1
            
            # Decode input (reversed format)
            input_str = self.decode(inputs[i])
            
            # Parse operands (reverse back to standard)
            a, b = self.parse_reversed_input(input_str)
            if a is None:
                continue
            
            # Find '=' position
            try:
                eq_pos = input_str.index('=')
            except ValueError:
                continue
            
            # Get predicted result (reversed)
            pred_tokens = pred_seqs[i, eq_pos + 1:]
            pred_str_reversed = self.decode(pred_tokens)
            
            # Reverse back to standard form
            try:
                pred_result = int(self.reverse_digits(pred_str_reversed))
            except ValueError:
                continue
            
            # Check correctness
            expected = a + b
            if pred_result == expected:
                self.correct += 1
            
            # Carry accuracy (if available)
            if carries is not None and carry_logits is not None:
                carry_gt = carries[i]
                carry_pred = (carry_logits[i] > 0).astype(int)
                
                # Count matching carries (up to length of ground truth)
                min_len = min(len(carry_gt), len(carry_pred))
                self.carry_total += min_len
                self.carry_correct += (carry_gt[:min_len] == carry_pred[:min_len]).sum()
    
    def result(
        self, 
        save_path: Optional[str], 
        rank: int, 
        world_size: int, 
        group=None
    ) -> Optional[Dict[str, float]]:
        """Aggregate and return metrics."""
        values = [self.total, self.correct, self.carry_total, self.carry_correct]
        
        if world_size > 1:
            t = torch.tensor(values, device="cuda", dtype=torch.long)
            dist.reduce(t, dst=0)
            values = t.tolist()
        
        if rank == 0:
            total, correct, carry_total, carry_correct = values
            
            accuracy = correct / total if total > 0 else 0.0
            carry_accuracy = carry_correct / carry_total if carry_total > 0 else 0.0
            
            print(f"\nLilavati Evaluation Results:")
            print(f"  Addition Accuracy: {correct}/{total} = {accuracy:.4f}")
            if carry_total > 0:
                print(f"  Carry Accuracy: {carry_correct}/{carry_total} = {carry_accuracy:.4f}")
            
            return {
                "lilavati/accuracy": accuracy,
                "lilavati/carry_accuracy": carry_accuracy,
            }
        
        return None
