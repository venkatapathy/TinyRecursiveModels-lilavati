#!/usr/bin/env python3
"""
Interactive inference script for the Addition TRM model.

Usage:
    python inference.py
    python inference.py --checkpoint checkpoints/Addition-TRM/addition-v3/step_10536

Input Format:
    Enter two numbers (0-99) separated by '+'. Examples:
        12+34   ->  model predicts 046
        05+99   ->  model predicts 104
        7+8     ->  model predicts 015
    
    The model will predict the 3-digit sum (zero-padded).
"""

import os
import argparse
import torch
from typing import Dict

from utils.functions import load_model_class


# Vocabulary mapping (from build_addition_dataset.py)
# 0: PAD
# 1: MASK (for unknown answer digits)
# 2-11: '0'-'9'
# 12: '+'
# 13: '='

VOCAB_MAP = {str(i): i + 2 for i in range(10)}
VOCAB_MAP['+'] = 12
VOCAB_MAP['='] = 13

VOCAB_INV = {v: k for k, v in VOCAB_MAP.items()}
VOCAB_INV[0] = '<PAD>'
VOCAB_INV[1] = '<MASK>'

MASK_ID = 1
SEQ_LEN = 9  # "XX+YY=ZZZ"


def encode(s: str) -> list:
    """Encode a string to token IDs."""
    return [VOCAB_MAP[c] for c in s]


def decode(tokens) -> str:
    """Decode token IDs to a string."""
    result = ""
    for t in tokens:
        t = int(t)
        if t in VOCAB_INV:
            result += VOCAB_INV[t]
        else:
            result += "?"
    return result


def parse_input(user_input: str) -> tuple:
    """
    Parse user input like '12+34' or '5+99'.
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
    
    if not (0 <= a <= 99 and 0 <= b <= 99):
        raise ValueError("Both numbers must be between 0 and 99")
    
    return a, b


def create_input_tensor(a: int, b: int) -> Dict[str, torch.Tensor]:
    """
    Create input tensor for the model.
    Format: "XX+YY=MMM" where M is MASK token.
    """
    # Format numbers as 2-digit strings
    s_a = f"{a:02d}"
    s_b = f"{b:02d}"
    
    # Build input sequence: XX+YY=MMM
    prefix = s_a + "+" + s_b + "="
    inp_seq = encode(prefix) + [MASK_ID, MASK_ID, MASK_ID]
    
    # Create tensors
    inputs = torch.tensor([inp_seq], dtype=torch.uint8, device="cuda")
    puzzle_identifiers = torch.tensor([0], dtype=torch.int32, device="cuda")
    # Use -100 as IGNORE_LABEL_ID (matches models/losses.py)
    labels = torch.full((1, SEQ_LEN), -100, dtype=torch.long, device="cuda")
    
    return {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_identifiers,
        "labels": labels
    }


def load_model(checkpoint_path: str, config_path: str = None):
    """Load the trained model from checkpoint."""
    
    # Model configuration (matching trm.yaml and pretrain_addition.yaml)
    model_cfg = {
        "batch_size": 1,
        "vocab_size": 14,  # 0-13
        "seq_len": SEQ_LEN,
        "num_puzzle_identifiers": 1,
        "causal": False,
        
        # Architecture config
        "halt_exploration_prob": 0.05,
        "halt_max_steps": 8,
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
    # Strip the prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_key = k[len("_orig_mod."):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # Ensure model is on CUDA
    model = model.cuda()
    model.eval()
    return model


@torch.inference_mode()
def run_inference(model, batch: Dict[str, torch.Tensor], max_steps: int = 8):
    """
    Run inference on a batch.
    Returns the predicted tokens and number of steps taken.
    
    Uses the inner model directly (not the loss head) for cleaner inference.
    """
    # Get the inner model (without loss head wrapper)
    inner_model = model.model
    
    # Create carry with all tensors on CUDA
    with torch.device("cuda"):
        carry = inner_model.initial_carry(batch)
    
    for step in range(max_steps):
        carry, outputs = inner_model(carry=carry, batch=batch)
        if carry.halted.all():
            break
    
    # Get predictions from logits
    logits = outputs["logits"]
    predictions = torch.argmax(logits, dim=-1)
    return predictions, step + 1


def main():
    parser = argparse.ArgumentParser(description="Interactive inference for Addition TRM model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/Addition-TRM/addition-v3/step_10536",
        help="Path to model checkpoint"
    )
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Using CPU (will be slow).")
        # Could add CPU support here if needed
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint)
    print("Model loaded successfully!\n")
    
    # Print instructions
    print("=" * 60)
    print("Addition Model Interactive Inference")
    print("=" * 60)
    print("\nInput format: two numbers (0-99) separated by '+'")
    print("Examples:")
    print("  12+34  ->  model predicts 046")
    print("  05+99  ->  model predicts 104")
    print("  7+8    ->  model predicts 015")
    print("\nType 'quit' or 'exit' to stop.\n")
    print("=" * 60)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nEnter addition (e.g., 12+34): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Parse input
            try:
                a, b = parse_input(user_input)
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue
            
            # Create input tensor
            batch = create_input_tensor(a, b)
            
            # Show what we're feeding to the model
            input_str = decode(batch["inputs"][0].cpu().numpy())
            print(f"Model input: {input_str}")
            
            # Run inference
            predictions, steps = run_inference(model, batch)
            
            # Decode output
            pred_tokens = predictions[0].cpu().numpy()
            
            # Extract the result (last 3 tokens after '=')
            result_tokens = pred_tokens[6:]  # Positions 6, 7, 8 are the answer
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

