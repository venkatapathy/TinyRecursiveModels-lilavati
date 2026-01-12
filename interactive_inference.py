#!/usr/bin/env python3
"""
Interactive inference script that runs infinitely, allowing you to give input and get output.
Usage:
    python interactive_inference.py --checkpoint <checkpoint_path> [--data <dataset_path>] [--config <config_path>]
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from puzzle_dataset import PuzzleDatasetMetadata
from utils.functions import load_model_class
from models.losses import ACTLossHead

def load_model_from_checkpoint(checkpoint_path: str, config_path: str = None, dataset_path: str = None):
    """Load model from checkpoint path."""
    
    # Determine if checkpoint_path is a file or directory
    if os.path.isfile(checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
    else:
        checkpoint_dir = checkpoint_path
    
    # Try to find config file in checkpoint directory
    if config_path is None:
        config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}. Please provide --config or ensure all_config.yaml exists in checkpoint directory.")
    
    # Load config
    config_dict = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config_dict, resolve=True)
    
    # Load dataset metadata to get vocab_size, seq_len, etc.
    # Use provided dataset_path, or try to infer from config
    if dataset_path is None:
        data_paths = config.get('data_paths', ['data/addition'])
        dataset_path = data_paths[0] if isinstance(data_paths, list) else data_paths
    
    # Load metadata
    metadata_path = os.path.join(dataset_path, "train", "dataset.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)
    
    metadata = PuzzleDatasetMetadata(**metadata_dict)
    
    # Get model architecture from config
    # Handle different config structures
    arch_config = config.get('arch', {})
    if isinstance(arch_config, dict):
        arch_name = arch_config.get('name', 'recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1')
        loss_config = arch_config.get('loss', {})
        if isinstance(loss_config, dict):
            loss_name = loss_config.get('name', 'losses@ACTLossHead')
            loss_extra = {k: v for k, v in loss_config.items() if k != 'name'}
        else:
            loss_name = 'losses@ACTLossHead'
            loss_extra = {}
        
        # Get all arch config except 'name' and 'loss'
        arch_extra = {k: v for k, v in arch_config.items() 
                     if k not in ['name', 'loss']}
    else:
        # Fallback defaults
        arch_name = 'recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1'
        loss_name = 'losses@ACTLossHead'
        arch_extra = {}
        loss_extra = {}
    
    # Get dataset mode and digits
    dataset_mode = config.get('dataset_mode', 'vanilla')
    digits = config.get('digits', 3)
    
    model_cfg = dict(
        **arch_extra,
        batch_size=1,  # Inference batch size
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False
    )
    
    # Load model class
    model_cls = load_model_class(arch_name)
    loss_head_cls = load_model_class(loss_name)
    
    # Create model
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **loss_extra)
        
        # Load checkpoint
        checkpoint_file = None
        if os.path.isfile(checkpoint_path):
            checkpoint_file = checkpoint_path
        else:
            # Find latest checkpoint in directory
            checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.startswith('step_')]
            if checkpoint_files:
                # Sort by step number
                checkpoint_files.sort(key=lambda x: int(x.split('_')[1]))
                checkpoint_file = os.path.join(checkpoint_path, checkpoint_files[-1])
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"Loading checkpoint: {checkpoint_file}")
            state_dict = torch.load(checkpoint_file, map_location="cuda")
            
            # Strip _orig_mod. prefix from keys if present (from torch.compile)
            if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                print("Stripping '_orig_mod.' prefix from checkpoint keys...")
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("_orig_mod."):
                        new_key = k[len("_orig_mod."):]
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict
            
            # Handle puzzle embedding resizing if needed
            puzzle_emb_name = "model.inner.puzzle_emb.weights"
            if puzzle_emb_name in state_dict:
                expected_shape = model.model.puzzle_emb.weights.shape
                puzzle_emb = state_dict[puzzle_emb_name]
                if puzzle_emb.shape != expected_shape:
                    print(f"Resetting puzzle embedding: {puzzle_emb.shape} -> {expected_shape}")
                    state_dict[puzzle_emb_name] = (
                        torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                    )
            
            model.load_state_dict(state_dict, assign=True)
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}, using random initialization")
        
        model.eval()
    
    return model, metadata, dataset_mode, digits

def encode_input(input_str: str, dataset_mode: str, digits: int, seq_len: int, vocab_size: int):
    """Encode input string to token sequence."""
    # Vocab mapping (must match build_addition_dataset.py)
    vocab_map = {str(i): i + 2 for i in range(10)}
    vocab_map['+'] = 12
    vocab_map['='] = 13
    MASK_ID = 1
    CAR_TOKEN_ID = 14 if vocab_size >= 15 else None
    
    # Encode the prefix (e.g., "12345+67890=")
    encoded = []
    for c in input_str:
        if c in vocab_map:
            encoded.append(vocab_map[c])
        else:
            raise ValueError(f"Invalid character '{c}' in input. Only digits, '+', and '=' are allowed.")
    
    # Determine how many mask tokens we need
    # Format: "XXX+YYY=MMMM" for vanilla or "XXX+YYY=MMMM <CAR> MMM" for lilavati1
    max_result_digits = digits + 1
    
    if dataset_mode == "vanilla":
        # Add mask tokens for result
        encoded.extend([MASK_ID] * max_result_digits)
    else:  # lilavati1
        # Add mask tokens for result, then <CAR>, then mask tokens for carries
        encoded.extend([MASK_ID] * max_result_digits)
        if CAR_TOKEN_ID is not None:
            encoded.append(CAR_TOKEN_ID)
        encoded.extend([MASK_ID] * digits)
    
    # Pad to seq_len
    pad_id = 0
    while len(encoded) < seq_len:
        encoded.append(pad_id)
    
    # Truncate if too long
    encoded = encoded[:seq_len]
    
    return encoded

def decode_output(tokens, vocab_size: int):
    """Decode token sequence to string."""
    vocab_map_inv = {i+2: str(i) for i in range(10)}
    vocab_map_inv[12] = '+'
    vocab_map_inv[13] = '='
    if vocab_size >= 15:
        vocab_map_inv[14] = '<CAR>'
    vocab_map_inv[0] = 'PAD'
    vocab_map_inv[1] = 'MASK'
    
    result = ""
    for token in tokens:
        token = int(token)
        if token in vocab_map_inv:
            result += vocab_map_inv[token]
        else:
            result += "?"
    return result

def run_inference(model, input_str: str, metadata: PuzzleDatasetMetadata, dataset_mode: str, digits: int):
    """Run inference on a single input string."""
    # Encode input
    try:
        encoded = encode_input(input_str, dataset_mode, digits, metadata.seq_len, metadata.vocab_size)
    except ValueError as e:
        return None, str(e)
    
    # Convert to tensor
    inputs_tensor = torch.tensor([encoded], dtype=torch.int32)
    
    # Create puzzle_identifiers (use blank identifier)
    puzzle_identifiers = torch.tensor([[metadata.blank_identifier_id]], dtype=torch.int32)
    
    # Create dummy labels (not used during inference, but needed for batch structure)
    from models.losses import IGNORE_LABEL_ID
    labels_tensor = torch.full_like(inputs_tensor, IGNORE_LABEL_ID)
    
    # Create batch
    batch = {
        "inputs": inputs_tensor,
        "labels": labels_tensor,
        "puzzle_identifiers": puzzle_identifiers
    }
    
    # Move to device
    batch = {k: v.cuda() for k, v in batch.items()}
    
    # Initialize carry
    with torch.device("cuda"):
        carry = model.initial_carry(batch)
    
    # Forward pass
    return_keys = {"inputs", "preds"}
    inference_steps = 0
    with torch.inference_mode():
        while True:
            carry, loss, metrics, preds, all_finish = model(
                carry=carry, batch=batch, return_keys=return_keys
            )
            inference_steps += 1
            if all_finish:
                break
    
    # Decode output
    preds_np = preds["preds"].cpu().numpy()[0]
    output_str = decode_output(preds_np, metadata.vocab_size)
    
    return output_str, None

def main():
    parser = argparse.ArgumentParser(description="Interactive inference for trained addition models")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file or directory containing checkpoints")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to dataset directory (default: inferred from config)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (default: <checkpoint>/all_config.yaml)")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, metadata, dataset_mode, digits = load_model_from_checkpoint(
        args.checkpoint, args.config, args.data
    )
    
    print(f"\n{'='*80}")
    print(f"Model loaded successfully!")
    print(f"  Mode: {dataset_mode}")
    print(f"  Digits: {digits}")
    print(f"  Vocab size: {metadata.vocab_size}")
    print(f"  Seq length: {metadata.seq_len}")
    print(f"{'='*80}\n")
    
    print("Interactive inference mode. Enter addition problems in the format: 'XXXXX+YYYYY' or 'XXXXX+YYYYY='")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Infinite loop for interactive inference
    while True:
        try:
            # Get user input
            user_input = input("Input: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Add '=' if not present
            if '=' not in user_input:
                user_input = user_input + '='
            
            # Run inference
            output, error = run_inference(model, user_input, metadata, dataset_mode, digits)
            
            if error:
                print(f"Error: {error}\n")
                continue
            
            # Extract and display just the result part (after '=')
            # Find the '=' position in the input
            eq_pos = user_input.find('=')
            if eq_pos >= 0:
                # Get the part after '=' in the output
                # The output should have the same structure as input but with predictions
                output_after_eq = output[eq_pos+1:]
                # Remove mask tokens and other non-digit characters for cleaner display
                result_clean = ''.join(c for c in output_after_eq if c.isdigit())
                if dataset_mode == "lilavati1" and '<CAR>' in output_after_eq:
                    # Extract carries too
                    car_pos = output_after_eq.find('<CAR>')
                    if car_pos >= 0:
                        result_clean = output_after_eq[:car_pos].replace('MASK', '').replace('PAD', '')
                        carries_clean = ''.join(c for c in output_after_eq[car_pos+5:] if c.isdigit())
                        print(f"Result: {result_clean}")
                        print(f"Carries: {carries_clean}")
                    else:
                        print(f"Result: {result_clean}")
                else:
                    print(f"Result: {result_clean}")
                print(f"Full output: {output}\n")
            else:
                print(f"Output: {output}\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
