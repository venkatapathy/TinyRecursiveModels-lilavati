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
        
        for i in range(len(inp)):
            self.total += 1
            
            # Input: XX+YY=MMM
            # Decoded Input: "01+05=???"
            
            # Get LHS from input
            input_tokens = inp[i]
            input_str = self.decode(input_tokens) # "01+05=MASKMASKMASK"
            
            try:
                # Basic parsing
                # Assuming fixed format XX+YY=...
                # We can also just strip non-digits/+ from LHS
                lhs_str = input_str.split('=')[0] # "01+05"
                if '+' not in lhs_str: 
                    continue
                
                a_str, b_str = lhs_str.split('+')
                a = int(a_str)
                b = int(b_str)
                
                # Get RHS from prediction
                # Prediction corresponds to whole sequence
                # We want the part after '='.
                # Fixed length 9: "XX+YY=ZZZ"
                # LHS is 0..5 (6 chars). RHS is 6..8 (3 chars).
                
                pred_rhs_tokens = pred_seq[i, 6:]
                rhs_str = self.decode(pred_rhs_tokens)
                
                # Check for invalid tokens in result
                if '?' in rhs_str or 'PAD' in rhs_str or 'MASK' in rhs_str:
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
            print(f"Addition Eval: {corr}/{total} = {acc:.4f}")
            return {"addition/accuracy": acc}
        return None

