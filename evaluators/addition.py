from typing import Dict, Optional, Sequence
import torch
import numpy as np
import os
import json
import torch.distributed as dist
from dataset.common import PuzzleDatasetMetadata

class AdditionEvaluator:
    required_outputs = {"inputs", "preds"}

    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata):
        # Vocab mapping (Must match build_addition_dataset.py)
        # 2-11: 0-9
        # 12: +
        # 13: =
        self.vocab_map_inv = {i+2: str(i) for i in range(10)}
        self.vocab_map_inv[12] = '+'
        self.vocab_map_inv[13] = '='
        self.vocab_map_inv[0] = 'PAD'
        self.vocab_map_inv[1] = 'MASK'
        
        self.total = 0
        self.arithmetic_correct = 0

    def begin_eval(self):
        self.total = 0
        self.arithmetic_correct = 0

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

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # batch["inputs"] has input
        # preds["preds"] has predicted tokens
        
        inp = batch["inputs"].cpu().numpy()
        pred_seq = preds["preds"].cpu().numpy()
        
        # Debug: print first sample of first batch
        if self.total == 0:
            print(f"[AdditionEvaluator] First sample input: {self.decode(inp[0])}")
            print(f"[AdditionEvaluator] First sample pred:  {self.decode(pred_seq[0])}")
        
        for i in range(len(inp)):
            self.total += 1
            
            # Get LHS from input
            input_tokens = inp[i]
            input_str = self.decode(input_tokens)
            
            try:
                # Parse input: "A+B=MMM..."
                lhs_str = input_str.split('=')[0]
                if '+' not in lhs_str: 
                    continue
                
                a_str, b_str = lhs_str.split('+')
                a = int(a_str)
                b = int(b_str)
                
                # Find position of '=' to extract result
                # The '=' token is 13
                eq_pos = -1
                for pos, tok in enumerate(input_tokens):
                    if tok == 13:  # '=' token
                        eq_pos = pos
                        break
                
                if eq_pos < 0:
                    continue
                
                # Get RHS from prediction (tokens after '=')
                pred_rhs_tokens = pred_seq[i, eq_pos + 1:]
                rhs_str = self.decode(pred_rhs_tokens)
                
                # Remove PAD/MASK from end
                rhs_str = rhs_str.replace('PAD', '').replace('MASK', '').strip()
                
                # Check for invalid tokens in result
                if '?' in rhs_str or not rhs_str:
                    continue
                     
                res = int(rhs_str)
                
                if a + b == res:
                    self.arithmetic_correct += 1
            except:
                continue
                
    def result(self, save_path: Optional[str], rank: int, world_size: int, group=None):
        # Aggregate metrics
        if world_size > 1:
            t = torch.tensor([self.total, self.arithmetic_correct], device="cuda", dtype=torch.long)
            dist.reduce(t, dst=0)
            total = t[0].item()
            corr = t[1].item()
        else:
            total = self.total
            corr = self.arithmetic_correct
            
        if rank == 0:
            acc = corr / total if total > 0 else 0.0
            print(f"\n{'='*50}")
            print(f"[AdditionEvaluator] RESULT: {corr}/{total} correct = {acc*100:.2f}%")
            print(f"{'='*50}\n")
            return {"addition/accuracy": acc, "addition/correct": float(corr), "addition/total": float(total)}
        return None

