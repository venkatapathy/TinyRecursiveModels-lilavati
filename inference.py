#!/usr/bin/env python3
"""
Inference script to test trained models on addition datasets.
Usage:
    python inference.py --checkpoint <checkpoint_path> --data <dataset_path> [--num-examples N] [--show-examples]
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class
from models.losses import ACTLossHead
from evaluators.addition import AdditionEvaluator

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
        causal=False,
        # Lilavati2: pass dataset_mode and digits to model
        dataset_mode=dataset_mode,
        digits=digits,
    )
    
    # Load model class
    model_cls = load_model_class(arch_name)
    loss_head_cls = load_model_class(loss_name)
    
    # Create model
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        # Pass dataset_mode and digits to loss head for lilavati2 carry predictions
        loss_extra["dataset_mode"] = dataset_mode
        loss_extra["digits"] = digits
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
            
            # Handle puzzle embedding resizing if needed
            puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
            if puzzle_emb_name not in state_dict:
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
    
    return model, metadata, dataset_mode, digits, dataset_path

def run_inference(model, dataset_path: str, num_examples: int = 100, show_examples: bool = True, digits: int = 5):
    """Run inference on dataset examples."""
    
    # Create dataset
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[dataset_path],
            global_batch_size=1,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        ),
        split="test"
    )
    
    # Vocabulary for decoding
    vocab_map_inv = {i+2: str(i) for i in range(10)}
    vocab_map_inv[12] = '+'
    vocab_map_inv[13] = '='
    if dataset.metadata.vocab_size >= 15:
        vocab_map_inv[14] = '<CAR>'
    vocab_map_inv[0] = 'PAD'
    vocab_map_inv[1] = 'MASK'
    
    def decode(seq):
        res = ""
        for token in seq:
            token = int(token)
            if token in vocab_map_inv:
                res += vocab_map_inv[token]
            else:
                res += "?"
        return res
    
    # Run inference
    examples_shown = 0
    correct = 0
    total = 0
    
    print(f"\n{'='*80}")
    print(f"Running inference on {num_examples} examples...")
    print(f"{'='*80}\n")
    
    with torch.inference_mode():
        for set_name, batch, global_batch_size in dataset:
            if total >= num_examples:
                break
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Initialize carry
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            
            # Forward pass
            return_keys = {"inputs", "preds"}
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1
                if all_finish:
                    break
            
            # Process results
            inputs_np = batch["inputs"].cpu().numpy()
            preds_np = preds["preds"].cpu().numpy()
            labels_np = batch.get("labels", None)
            if labels_np is not None:
                labels_np = labels_np.cpu().numpy()
            
            for i in range(len(inputs_np)):
                if total >= num_examples:
                    break
                
                total += 1
                inp = inputs_np[i]
                pred = preds_np[i]
                
                # Decode
                input_str = decode(inp)
                pred_str = decode(pred)
                
                # Parse input
                if '=' not in input_str:
                    continue
                
                lhs_str = input_str.split('=')[0]
                if '+' not in lhs_str:
                    continue
                
                try:
                    a_str, b_str = lhs_str.split('+')
                    a = int(a_str)
                    b = int(b_str)
                    expected_res = a + b
                    
                    # Extract prediction (after '=')
                    eq_idx = None
                    for j, tok in enumerate(inp):
                        if tok == 13:  # '='
                            eq_idx = j
                            break
                    
                    if eq_idx is None:
                        continue
                    
                    # Get result part (skip to after '=')
                    result_start = eq_idx + 1
                    # Max result digits: can be digits or digits+1 (for overflow)
                    max_result_digits = digits + 1
                    result_end = result_start + max_result_digits
                    
                    pred_result_tokens = pred[result_start:result_end]
                    pred_result_str = decode(pred_result_tokens)
                    
                    # Clean prediction (remove invalid tokens)
                    pred_result_clean = ''.join(c for c in pred_result_str if c.isdigit())
                    
                    if pred_result_clean:
                        try:
                            pred_res = int(pred_result_clean)
                            is_correct = (pred_res == expected_res)
                            if is_correct:
                                correct += 1
                            
                            # Show examples
                            if show_examples and examples_shown < 20:
                                status = "✓" if is_correct else "✗"
                                print(f"{status} {a:05d} + {b:05d} = {expected_res:06d} | Predicted: {pred_res:06d} | Input: {input_str[:30]}...")
                                examples_shown += 1
                        except ValueError:
                            if show_examples and examples_shown < 20:
                                print(f"✗ {a:05d} + {b:05d} = {expected_res:06d} | Predicted: INVALID ({pred_result_str[:20]})")
                                examples_shown += 1
                except Exception as e:
                    continue
            
            if total % 100 == 0:
                acc = correct / total if total > 0 else 0.0
                print(f"  Processed {total} examples, accuracy: {acc:.4f} ({correct}/{total})")
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'='*80}")
    print(f"Final Results:")
    print(f"  Total examples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*80}\n")
    
    return accuracy, correct, total

def main():
    parser = argparse.ArgumentParser(description="Run inference on trained addition models")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file or directory containing checkpoints")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to dataset directory (e.g., data/addition5_vanilla)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (default: <checkpoint>/all_config.yaml)")
    parser.add_argument("--num-examples", type=int, default=100,
                       help="Number of examples to test (default: 100)")
    parser.add_argument("--show-examples", action="store_true",
                       help="Show individual example predictions")
    parser.add_argument("--no-show-examples", dest="show_examples", action="store_false")
    parser.set_defaults(show_examples=True)
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, metadata, dataset_mode, digits, inferred_dataset_path = load_model_from_checkpoint(
        args.checkpoint, args.config, args.data
    )
    
    # Use provided dataset path or inferred one
    dataset_path = args.data if args.data else inferred_dataset_path
    
    print(f"Model loaded successfully!")
    print(f"  Dataset: {dataset_path}")
    print(f"  Mode: {dataset_mode}")
    print(f"  Digits: {digits}")
    print(f"  Vocab size: {metadata.vocab_size}")
    print(f"  Seq length: {metadata.seq_len}")
    
    # Run inference
    accuracy, correct, total = run_inference(
        model, dataset_path, args.num_examples, args.show_examples, digits
    )
    
    return accuracy, correct, total

if __name__ == "__main__":
    main()
