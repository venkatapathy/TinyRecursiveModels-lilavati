#!/usr/bin/env python3
"""
Interactive inference script for the 25-digit Addition TRM model.

Usage:
    python inference_addition25.py
    python inference_addition25.py --checkpoint checkpoints/Addition25-TRM/addition25-v1/step_XXXX

Input Format:
    Enter two numbers separated by '+'. Examples:
        12+34                    -> model predicts 46
        123456789+987654321      -> model predicts 1111111110
        999999999999+1           -> model predicts 1000000000000
    
    Max operand: 24 digits
    Max result: 25 digits
"""

import os
import argparse
import torch
from typing import Dict

from utils.functions import load_model_class


# Vocabulary mapping
VOCAB_MAP = {str(i): i + 2 for i in range(10)}
VOCAB_MAP['+'] = 12
VOCAB_MAP['='] = 13

VOCAB_INV = {v: k for k, v in VOCAB_MAP.items()}
VOCAB_INV[0] = '<PAD>'
VOCAB_INV[1] = '<MASK>'

PAD_ID = 0
MASK_ID = 1
VOCAB_SIZE = 14

# Max digits config
MAX_OPERAND_DIGITS = 24
MAX_RESULT_DIGITS = 25
SEQ_LEN = MAX_OPERAND_DIGITS + 1 + MAX_OPERAND_DIGITS + 1 + MAX_RESULT_DIGITS  # 75


def encode(s: str) -> list:
    """Encode a string to token IDs."""
    return [VOCAB_MAP[c] for c in s]


def decode(tokens) -> str:
    """Decode token IDs to a string."""
    result = ""
    for t in tokens:
        t = int(t)
        if t in VOCAB_INV:
            char = VOCAB_INV[t]
            if char in ('<PAD>', '<MASK>'):
                break
            result += char
        else:
            result += "?"
    return result


def parse_input(user_input: str) -> tuple:
    """
    Parse user input like '123+456'.
    Returns (a, b) as integers, or raises ValueError.
    """
    user_input = user_input.strip()
    if '+' not in user_input:
        raise ValueError("Input must contain '+' (e.g., '12+34')")
    
    parts = user_input.split('+')
    if len(parts) != 2:
        raise ValueError("Input must have exactly one '+' (e.g., '12+34')")
    
    a = int(parts[0])
    b = int(parts[1])
    
    if len(str(a)) > MAX_OPERAND_DIGITS or len(str(b)) > MAX_OPERAND_DIGITS:
        raise ValueError(f"Each operand must be at most {MAX_OPERAND_DIGITS} digits")
    
    if a < 0 or b < 0:
        raise ValueError("Operands must be non-negative")
    
    return a, b


def create_input_tensor(a: int, b: int) -> Dict[str, torch.Tensor]:
    """
    Create input tensor for the model.
    Format: "A+B=MASKS" padded to SEQ_LEN
    """
    s_a = str(a)
    s_b = str(b)
    result_len = len(str(a + b))
    
    prefix = s_a + "+" + s_b + "="
    prefix_encoded = encode(prefix)
    
    # Input: prefix + MASKs + PAD
    inp_seq = prefix_encoded + [MASK_ID] * result_len
    pad_len = SEQ_LEN - len(inp_seq)
    inp_seq = inp_seq + [PAD_ID] * pad_len
    
    inputs = torch.tensor([inp_seq], dtype=torch.uint8, device="cuda")
    puzzle_identifiers = torch.tensor([0], dtype=torch.int32, device="cuda")
    labels = torch.full((1, SEQ_LEN), -100, dtype=torch.long, device="cuda")
    
    return {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_identifiers,
        "labels": labels
    }


def load_model(checkpoint_path: str):
    """Load the trained model from checkpoint."""
    
    model_cfg = {
        "batch_size": 1,
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "num_puzzle_identifiers": 1,
        "causal": False,
        
        "halt_exploration_prob": 0.05,
        "halt_max_steps": 16,
        "H_cycles": 2,
        "L_cycles": 4,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4,
        "puzzle_emb_ndim": 512,
        "pos_encodings": "rope",
        "forward_dtype": "bfloat16",
        "mlp_t": False,
        "puzzle_emb_len": 16,
        "no_ACT_continue": True,
    }
    
    model_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
    loss_head_cls = load_model_class("losses@ACTLossHead")
    
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, loss_type="stablemax_cross_entropy")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_key = k[len("_orig_mod."):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.cuda()
    model.eval()
    return model


@torch.inference_mode()
def run_inference(model, batch: Dict[str, torch.Tensor], max_steps: int = 16):
    """Run inference on a batch."""
    inner_model = model.model
    
    with torch.device("cuda"):
        carry = inner_model.initial_carry(batch)
    
    for step in range(max_steps):
        carry, outputs = inner_model(carry=carry, batch=batch)
        if carry.halted.all():
            break
    
    logits = outputs["logits"]
    predictions = torch.argmax(logits, dim=-1)
    return predictions, step + 1


def main():
    parser = argparse.ArgumentParser(description="Interactive inference for 25-digit Addition TRM model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/Addition25-TRM/addition25-v1/step_10000",
        help="Path to model checkpoint"
    )
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Error: CUDA is required for inference.")
        return
    
    print("Loading model...")
    try:
        model = load_model(args.checkpoint)
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have trained a model first:")
        print("  1. python -m dataset.build_addition_dataset --output_dir data/addition25")
        print("  2. python pretrain.py --config-name pretrain_addition25")
        return
    
    print("=" * 60)
    print("25-Digit Addition Model Interactive Inference")
    print("=" * 60)
    print(f"\nMax operand digits: {MAX_OPERAND_DIGITS}")
    print(f"Max result digits: {MAX_RESULT_DIGITS}")
    print("\nExamples:")
    print("  12+34")
    print("  123456789+987654321")
    print("  999999999999+1")
    print("\nType 'quit' or 'exit' to stop.\n")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nEnter addition (e.g., 12+34): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                a, b = parse_input(user_input)
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue
            
            batch = create_input_tensor(a, b)
            
            input_str = decode(batch["inputs"][0].cpu().numpy())
            print(f"Model input: {input_str}")
            
            predictions, steps = run_inference(model, batch)
            
            pred_tokens = predictions[0].cpu().numpy()
            eq_pos = input_str.index('=')
            result_tokens = pred_tokens[eq_pos + 1:]
            result_str = decode(result_tokens)
            
            try:
                predicted_sum = int(result_str)
                actual_sum = a + b
                status = "✓ Correct!" if predicted_sum == actual_sum else f"✗ Wrong (expected {actual_sum})"
            except ValueError:
                predicted_sum = result_str
                status = "Could not parse result"
            
            print(f"Result: {a} + {b} = {result_str}  {status}")
            print(f"(Inference steps: {steps})")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
