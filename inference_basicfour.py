#!/usr/bin/env python3
"""
Interactive inference script for the BasicFour TRM model.

Supports: Addition (+), Subtraction (-), Multiplication (*), Division (/)

Usage:
    python inference_basicfour.py
    python inference_basicfour.py --checkpoint checkpoints/BasicFour-TRM/basicfour-v1/step_XXXX

Input Format:
    Enter expressions like:
        123+456     ->  model predicts 579
        100-250     ->  model predicts -150
        12*34       ->  model predicts 408
        17/5        ->  model predicts 3R2 (quotient 3, remainder 2)
    
    Max result: 10^17 (17 digits)
"""

import os
import argparse
import torch
from typing import Dict

from utils.functions import load_model_class


# Vocabulary mapping (from build_basicfour_dataset.py)
# 0: PAD
# 1: MASK
# 2-11: '0'-'9'
# 12: '+'
# 13: '='
# 14: '-'
# 15: '*'
# 16: '/'
# 17: 'R' (remainder)

VOCAB_MAP = {str(i): i + 2 for i in range(10)}
VOCAB_MAP['+'] = 12
VOCAB_MAP['='] = 13
VOCAB_MAP['-'] = 14
VOCAB_MAP['*'] = 15
VOCAB_MAP['/'] = 16
VOCAB_MAP['R'] = 17

VOCAB_INV = {v: k for k, v in VOCAB_MAP.items()}
VOCAB_INV[0] = '<PAD>'
VOCAB_INV[1] = '<MASK>'

PAD_ID = 0
MASK_ID = 1
VOCAB_SIZE = 18
SEQ_LEN = 53  # max: 17 + 1 + 17 + 1 + 17
MAX_RESULT = 10**17


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
                break  # Stop at padding/mask
            result += char
        else:
            result += "?"
    return result


def parse_input(user_input: str) -> tuple:
    """
    Parse user input like '123+456', '100-250', '12*34', '17/5'.
    Returns (a, b, op) as (int, int, str), or raises ValueError.
    """
    user_input = user_input.strip()
    
    # Detect operation (check +, *, / first, then - to avoid negative sign confusion)
    op = None
    op_pos = -1
    
    for candidate_op in ['+', '*', '/']:
        if candidate_op in user_input:
            op = candidate_op
            op_pos = user_input.index(candidate_op)
            break
    
    # Handle subtraction (find '-' that's not at position 0)
    if op is None:
        for i, c in enumerate(user_input):
            if c == '-' and i > 0:
                op = '-'
                op_pos = i
                break
    
    if op is None:
        raise ValueError("Input must contain an operator (+, -, *, /). Example: '12+34'")
    
    # Parse operands
    try:
        a = int(user_input[:op_pos])
        b = int(user_input[op_pos + 1:])
    except ValueError:
        raise ValueError(f"Could not parse operands from '{user_input}'")
    
    # Validate
    if a < 0:
        raise ValueError("First operand cannot be negative")
    if b < 0 and op != '-':
        raise ValueError("Second operand cannot be negative (except for subtraction)")
    if op == '/' and b == 0:
        raise ValueError("Division by zero is not allowed")
    
    return a, b, op


def compute_expected(op: str, a: int, b: int) -> str:
    """Compute expected result string."""
    if op == '+':
        result = a + b
        return str(result)
    elif op == '-':
        result = a - b
        if result < 0:
            return '-' + str(abs(result))
        return str(result)
    elif op == '*':
        result = a * b
        return str(result)
    elif op == '/':
        quotient = a // b
        remainder = a % b
        if remainder == 0:
            return str(quotient)
        return f"{quotient}R{remainder}"
    return "?"


def estimate_result_digits(op: str, a: int, b: int) -> int:
    """Estimate the number of digits in the result."""
    if op == '+':
        result = a + b
        return len(str(abs(result)))
    elif op == '-':
        result = a - b
        # Account for negative sign
        if result < 0:
            return len(str(abs(result))) + 1
        return len(str(result))
    elif op == '*':
        result = a * b
        return len(str(abs(result)))
    elif op == '/':
        quotient = a // b
        remainder = a % b
        if remainder == 0:
            return len(str(quotient))
        # Format: "quotientRremainder"
        return len(f"{quotient}R{remainder}")
    return 17  # Fallback


def create_input_tensor(a: int, b: int, op: str) -> Dict[str, torch.Tensor]:
    """
    Create input tensor for the model.
    Format: "A op B = MASKS" padded to SEQ_LEN
    """
    # Build prefix
    prefix = f"{a}{op}{b}="
    
    # Estimate result length for MASK tokens
    result_len = estimate_result_digits(op, a, b)
    result_len = min(result_len, SEQ_LEN - len(prefix))  # Safety cap
    
    # Encode prefix
    prefix_encoded = encode(prefix)
    
    # Build input sequence: prefix + MASKs + PAD
    inp_seq = prefix_encoded + [MASK_ID] * result_len
    pad_len = SEQ_LEN - len(inp_seq)
    inp_seq = inp_seq + [PAD_ID] * pad_len
    
    # Create tensors
    inputs = torch.tensor([inp_seq], dtype=torch.uint8, device="cuda")
    
    # Determine operation ID (0=add, 1=sub, 2=mul, 3=div)
    op_id = {'+': 0, '-': 1, '*': 2, '/': 3}[op]
    puzzle_identifiers = torch.tensor([op_id], dtype=torch.int32, device="cuda")
    
    # Labels (not used in inference, but required by model)
    labels = torch.full((1, SEQ_LEN), -100, dtype=torch.long, device="cuda")
    
    return {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_identifiers,
        "labels": labels
    }


def load_model(checkpoint_path: str):
    """Load the trained model from checkpoint."""
    
    # Model configuration (matching trm.yaml and pretrain_basicfour.yaml)
    model_cfg = {
        "batch_size": 1,
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "num_puzzle_identifiers": 4,  # 4 operations
        "causal": False,
        
        # Architecture config
        "halt_exploration_prob": 0.05,
        "halt_max_steps": 16,  # Increased for harder problems
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
    
    # Load model class
    model_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
    loss_head_cls = load_model_class("losses@ACTLossHead")
    
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, loss_type="stablemax_cross_entropy")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    
    # Handle compiled model checkpoints (keys prefixed with '_orig_mod.')
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
    """
    Run inference on a batch.
    Returns the predicted tokens and number of steps taken.
    """
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
    parser = argparse.ArgumentParser(description="Interactive inference for BasicFour TRM model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/BasicFour-TRM/basicfour-v1/step_10000",
        help="Path to model checkpoint"
    )
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is required for inference.")
        return
    
    # Load model
    print("Loading model...")
    try:
        model = load_model(args.checkpoint)
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have trained a model first:")
        print("  1. python -m dataset.build_basicfour_dataset")
        print("  2. python pretrain.py --config-name pretrain_basicfour")
        return
    
    # Print instructions
    print("=" * 60)
    print("BasicFour Model Interactive Inference")
    print("=" * 60)
    print("\nSupported operations:")
    print("  Addition:       12+34      ->  46")
    print("  Subtraction:    100-250    ->  -150")
    print("  Multiplication: 12*34      ->  408")
    print("  Division:       17/5       ->  3R2 (quotient R remainder)")
    print("\nMax result: 10^17 (17 digits)")
    print("\nType 'quit' or 'exit' to stop.\n")
    print("=" * 60)
    
    # Operation names for display
    op_names = {'+': 'Addition', '-': 'Subtraction', '*': 'Multiplication', '/': 'Division'}
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nEnter expression (e.g., 12+34): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Parse input
            try:
                a, b, op = parse_input(user_input)
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue
            
            # Create input tensor
            batch = create_input_tensor(a, b, op)
            
            # Show what we're feeding to the model
            input_tokens = batch["inputs"][0].cpu().numpy()
            input_str = decode(input_tokens)
            print(f"Operation: {op_names[op]}")
            print(f"Model input: {input_str}")
            
            # Run inference
            predictions, steps = run_inference(model, batch)
            
            # Decode output - find '=' position and extract result
            pred_tokens = predictions[0].cpu().numpy()
            
            # Find '=' position in input
            eq_pos = input_str.index('=')
            
            # Extract result tokens (after '=')
            result_tokens = pred_tokens[eq_pos + 1:]
            result_str = decode(result_tokens)
            
            # Compute expected result
            expected = compute_expected(op, a, b)
            
            # Display result
            if op == '/':
                print(f"Result: {a} {op} {b} = {result_str}")
                if 'R' in result_str:
                    try:
                        q, r = result_str.split('R')
                        print(f"        (quotient: {q}, remainder: {r})")
                    except:
                        pass
            else:
                print(f"Result: {a} {op} {b} = {result_str}")
            
            # Check correctness
            if result_str == expected:
                print(f"Status: ✓ Correct!")
            else:
                print(f"Status: ✗ Wrong (expected {expected})")
            
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
