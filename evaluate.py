#!/usr/bin/env python3
"""
Evaluator-only script for testing trained addition models on test set.
Reports digit-wise accuracy, sequence accuracy, and carry accuracy (for lilavati modes).
Automatically logs results to wandb in a separate project (trm-lilavati-eval) with 'eval/' prefix.

Usage:
    python evaluate.py --checkpoint <checkpoint_path> --data <dataset_path>
    
Examples:
    python evaluate.py --checkpoint checkpoints/my_model --data data/addition6_vanilla_strict_mixed_fixed
    python evaluate.py --checkpoint checkpoints/my_model/step_10000 --data data/addition6_lilavati1_strict_mixed_fixed --verbose
    python evaluate.py --checkpoint checkpoints/my_model --data data/addition6_vanilla_strict_mixed_fixed --no-wandb
    python evaluate.py --checkpoint checkpoints/my_model --data data/addition6_vanilla_strict_mixed_fixed --wandb-project my-eval-project
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from typing import Dict, Tuple, Optional

import torch
import numpy as np
from omegaconf import OmegaConf

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class


class DigitWiseEvaluator:
    """Evaluator that computes digit-wise accuracy for addition models."""
    
    def __init__(self, dataset_mode: str = "vanilla", digits: int = 6, vocab_size: int = 15):
        self.dataset_mode = dataset_mode
        self.digits = digits
        self.max_result_digits = digits + 1  # Account for overflow
        
        # Build vocab mapping
        self.vocab_map_inv = {i+2: str(i) for i in range(10)}
        self.vocab_map_inv[12] = '+'
        self.vocab_map_inv[13] = '='
        if vocab_size >= 15:
            self.vocab_map_inv[14] = '<CAR>'
            self.car_token_id = 14
        else:
            self.car_token_id = None
        self.vocab_map_inv[0] = 'PAD'
        self.vocab_map_inv[1] = 'MASK'
        
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.total = 0
        self.valid = 0  # Successfully parsed examples
        
        # Sequence-level
        self.sequence_correct = 0
        
        # Digit-level (per position)
        self.digit_correct = defaultdict(int)  # position -> count correct
        self.digit_total = defaultdict(int)    # position -> count total
        
        # Overall digit accuracy
        self.total_digit_correct = 0
        self.total_digit_count = 0
        
        # Carry accuracy (lilavati modes only)
        self.carry_correct = defaultdict(int)  # position -> count correct
        self.carry_total = defaultdict(int)    # position -> count total
        self.total_carry_correct = 0
        self.total_carry_count = 0
        
        # Examples for debugging
        self.examples = []
    
    def decode(self, seq) -> str:
        """Decode token sequence to string."""
        result = ""
        for token in seq:
            token = int(token)
            if token in self.vocab_map_inv:
                result += self.vocab_map_inv[token]
            else:
                result += "?"
        return result
    
    def compute_carries(self, a: int, b: int) -> list:
        """Compute carry digits from least significant to most significant."""
        carries = []
        carry = 0
        for pos in range(self.digits):
            a_digit = (a // (10 ** pos)) % 10
            b_digit = (b // (10 ** pos)) % 10
            total = a_digit + b_digit + carry
            carry = total // 10
            carries.append(carry)
        return carries
    
    def update(self, inputs: np.ndarray, predictions: np.ndarray, store_examples: bool = False):
        """Update metrics with a batch of predictions."""
        batch_size = len(inputs)
        
        for i in range(batch_size):
            self.total += 1
            
            input_tokens = inputs[i]
            pred_tokens = predictions[i]
            
            input_str = self.decode(input_tokens)
            
            try:
                # Parse input: "XXX+YYY=..."
                if '=' not in input_str:
                    continue
                lhs_str = input_str.split('=')[0]
                if '+' not in lhs_str:
                    continue
                
                a_str, b_str = lhs_str.split('+')
                a = int(a_str)
                b = int(b_str)
                expected_result = a + b
                
                # Find where result starts (after '=')
                eq_idx = None
                for j, tok in enumerate(input_tokens):
                    if tok == 13:  # '=' token
                        eq_idx = j
                        break
                
                if eq_idx is None:
                    continue
                
                # Extract predicted result digits
                result_start = eq_idx + 1
                result_end = result_start + self.max_result_digits
                pred_result_tokens = pred_tokens[result_start:result_end]
                pred_result_str = self.decode(pred_result_tokens)
                
                # Skip if contains invalid tokens
                if '?' in pred_result_str or 'PAD' in pred_result_str or 'MASK' in pred_result_str:
                    continue
                
                # Remove <CAR> from result string if present
                pred_result_str = pred_result_str.replace('<CAR>', '')
                
                if not pred_result_str or not pred_result_str.strip():
                    continue
                
                # Parse predicted result
                try:
                    pred_result = int(pred_result_str)
                except ValueError:
                    continue
                
                self.valid += 1
                
                # Format strings for digit comparison (zero-padded)
                expected_str = f"{expected_result:0{self.max_result_digits}d}"
                pred_str = f"{pred_result:0{self.max_result_digits}d}"
                
                # Digit-wise accuracy (position 0 = MSB)
                all_digits_correct = True
                for pos in range(self.max_result_digits):
                    if pos < len(expected_str) and pos < len(pred_str):
                        self.digit_total[pos] += 1
                        self.total_digit_count += 1
                        if expected_str[pos] == pred_str[pos]:
                            self.digit_correct[pos] += 1
                            self.total_digit_correct += 1
                        else:
                            all_digits_correct = False
                    else:
                        all_digits_correct = False
                
                # Sequence accuracy
                if pred_result == expected_result:
                    self.sequence_correct += 1
                
                # Handle carries for lilavati modes
                if self.dataset_mode in {"lilavati1", "lilavati2"} and self.car_token_id is not None:
                    # Find <CAR> token in predictions
                    car_idx = None
                    for j in range(result_end, len(pred_tokens)):
                        if pred_tokens[j] == self.car_token_id:
                            car_idx = j
                            break
                    
                    if car_idx is not None:
                        carry_start = car_idx + 1
                        carry_end = carry_start + self.digits
                        pred_carry_tokens = pred_tokens[carry_start:carry_end]
                        pred_carry_str = self.decode(pred_carry_tokens)
                        
                        # Skip if invalid
                        if '?' not in pred_carry_str and 'PAD' not in pred_carry_str:
                            expected_carries = self.compute_carries(a, b)
                            expected_carry_str = ''.join(str(c) for c in expected_carries)
                            
                            # Carry accuracy per position (position 0 = LSB carry)
                            for pos in range(min(len(expected_carry_str), len(pred_carry_str))):
                                self.carry_total[pos] += 1
                                self.total_carry_count += 1
                                if expected_carry_str[pos] == pred_carry_str[pos]:
                                    self.carry_correct[pos] += 1
                                    self.total_carry_correct += 1
                
                # Store example for debugging
                if store_examples and len(self.examples) < 50:
                    self.examples.append({
                        'input': f"{a}+{b}",
                        'expected': expected_result,
                        'predicted': pred_result,
                        'correct': pred_result == expected_result,
                        'expected_str': expected_str,
                        'predicted_str': pred_str,
                    })
                    
            except Exception as e:
                continue
    
    def get_results(self) -> Dict:
        """Compute and return all metrics."""
        results = {
            'total_examples': self.total,
            'valid_examples': self.valid,
            'sequence_accuracy': self.sequence_correct / self.valid if self.valid > 0 else 0.0,
            'sequence_correct': self.sequence_correct,
            'overall_digit_accuracy': self.total_digit_correct / self.total_digit_count if self.total_digit_count > 0 else 0.0,
        }
        
        # Per-position digit accuracy
        digit_acc_by_pos = {}
        for pos in sorted(self.digit_total.keys()):
            if self.digit_total[pos] > 0:
                acc = self.digit_correct[pos] / self.digit_total[pos]
                digit_acc_by_pos[f'pos_{pos}'] = acc
        results['digit_accuracy_by_position'] = digit_acc_by_pos
        
        # Carry accuracy for lilavati modes
        if self.dataset_mode in {"lilavati1", "lilavati2"}:
            results['overall_carry_accuracy'] = self.total_carry_correct / self.total_carry_count if self.total_carry_count > 0 else 0.0
            
            carry_acc_by_pos = {}
            for pos in sorted(self.carry_total.keys()):
                if self.carry_total[pos] > 0:
                    acc = self.carry_correct[pos] / self.carry_total[pos]
                    carry_acc_by_pos[f'pos_{pos}'] = acc
            results['carry_accuracy_by_position'] = carry_acc_by_pos
        
        return results


def load_model_from_checkpoint(checkpoint_path: str, config_path: str = None, dataset_path: str = None):
    """Load model from checkpoint."""
    
    # Determine checkpoint directory
    if os.path.isfile(checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
    else:
        checkpoint_dir = checkpoint_path
    
    # Find config file
    if config_path is None:
        config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    config_dict = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config_dict, resolve=True)
    
    # Load dataset metadata
    if dataset_path is None:
        data_paths = config.get('data_paths', ['data/addition'])
        dataset_path = data_paths[0] if isinstance(data_paths, list) else data_paths
    
    metadata_path = os.path.join(dataset_path, "train", "dataset.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)
    
    metadata = PuzzleDatasetMetadata(**metadata_dict)
    
    # Get model architecture from config
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
        
        arch_extra = {k: v for k, v in arch_config.items() if k not in ['name', 'loss']}
    else:
        arch_name = 'recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1'
        loss_name = 'losses@ACTLossHead'
        arch_extra = {}
        loss_extra = {}
    
    # Get dataset mode and digits
    dataset_mode = config.get('dataset_mode', 'vanilla')
    digits = config.get('digits', 6)
    
    model_cfg = dict(
        **arch_extra,
        batch_size=1,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
        dataset_mode=dataset_mode,
        digits=digits,
    )
    
    # Load model
    model_cls = load_model_class(arch_name)
    loss_head_cls = load_model_class(loss_name)
    
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        loss_extra["dataset_mode"] = dataset_mode
        loss_extra["digits"] = digits
        model = loss_head_cls(model, **loss_extra)
        
        # Find checkpoint file
        checkpoint_file = None
        if os.path.isfile(checkpoint_path):
            checkpoint_file = checkpoint_path
        else:
            checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.startswith('step_')]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: int(x.split('_')[1]))
                checkpoint_file = os.path.join(checkpoint_path, checkpoint_files[-1])
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"Loading checkpoint: {checkpoint_file}")
            state_dict = torch.load(checkpoint_file, map_location="cuda")
            
            # Strip _orig_mod. prefix from keys if present (from torch.compile)
            if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                print("Stripping '_orig_mod.' prefix from state_dict keys...")
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("_orig_mod."):
                        new_key = k[len("_orig_mod."):]
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict
            
            # Handle puzzle embedding resizing
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
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        model.eval()
    
    return model, metadata, dataset_mode, digits


def evaluate_model(
    model,
    dataset_path: str,
    dataset_mode: str,
    digits: int,
    vocab_size: int,
    batch_size: int = 32,
    verbose: bool = False,
    max_examples: int = None,
) -> Dict:
    """Evaluate model on test set and return metrics."""
    
    # Create dataset
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[dataset_path],
            global_batch_size=batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        ),
        split="test"
    )
    
    # Create evaluator
    evaluator = DigitWiseEvaluator(
        dataset_mode=dataset_mode,
        digits=digits,
        vocab_size=vocab_size,
    )
    
    print(f"\nEvaluating on test set...")
    print(f"  Dataset: {dataset_path}")
    print(f"  Mode: {dataset_mode}")
    print(f"  Digits: {digits}")
    print()
    
    processed = 0
    with torch.inference_mode():
        for set_name, batch, global_batch_size in dataset:
            if max_examples and processed >= max_examples:
                break
            
            # Move to GPU
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Initialize carry
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            
            # Forward pass until completion
            return_keys = {"inputs", "preds"}
            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                if all_finish:
                    break
            
            # Update evaluator
            inputs_np = batch["inputs"].cpu().numpy()
            preds_np = preds["preds"].cpu().numpy()
            evaluator.update(inputs_np, preds_np, store_examples=verbose)
            
            processed += len(inputs_np)
            
            if processed % 1000 == 0:
                print(f"  Processed {processed} examples...")
    
    # Get results
    results = evaluator.get_results()
    
    return results, evaluator.examples if verbose else []


def init_wandb_for_eval(checkpoint_path: str, dataset_path: str, dataset_mode: str, digits: int, 
                        project_name: str = None, run_name: str = None, tags: list = None):
    """Initialize wandb for evaluation with separate category from training."""
    if not WANDB_AVAILABLE:
        return None
    
    # Determine project name
    if project_name is None:
        project_name = "trm-lilavati-eval"  # Separate project for evaluations
    
    # Determine run name
    if run_name is None:
        checkpoint_name = os.path.basename(os.path.dirname(checkpoint_path) if os.path.isfile(checkpoint_path) else checkpoint_path)
        dataset_name = os.path.basename(dataset_path.rstrip('/'))
        run_name = f"eval_{dataset_mode}_d{digits}_{checkpoint_name}_{dataset_name}"
        # Truncate if too long
        if len(run_name) > 200:
            run_name = run_name[:200]
    
    # Default tags
    if tags is None:
        tags = ["evaluation", "test-set", f"mode-{dataset_mode}", f"d{digits}"]
    
    # Initialize wandb
    wandb.init(
        project=project_name,
        name=run_name,
        tags=tags,
        config={
            "checkpoint_path": checkpoint_path,
            "dataset_path": dataset_path,
            "dataset_mode": dataset_mode,
            "digits": digits,
            "eval_type": "test_set",
        },
        settings=wandb.Settings(_disable_stats=True),
        reinit=True,  # Allow reinitialization if already initialized
    )
    
    return wandb.run


def log_results_to_wandb(results: Dict, checkpoint_path: str = None):
    """Log evaluation results to wandb with 'eval/' prefix."""
    if not WANDB_AVAILABLE:
        return
    try:
        if wandb.run is None:
            return
    except:
        return
    
    # Prepare metrics with eval/ prefix to separate from training logs
    metrics = {
        "eval/sequence_accuracy": results['sequence_accuracy'],
        "eval/sequence_correct": results['sequence_correct'],
        "eval/sequence_total": results['valid_examples'],
        "eval/overall_digit_accuracy": results['overall_digit_accuracy'],
        "eval/total_examples": results['total_examples'],
        "eval/valid_examples": results['valid_examples'],
    }
    
    # Add per-position digit accuracy
    for pos, acc in results['digit_accuracy_by_position'].items():
        pos_num = int(pos.split('_')[1])
        metrics[f"eval/digit_acc_pos_{pos_num}"] = acc
    
    # Add carry accuracy if available
    if 'overall_carry_accuracy' in results:
        metrics["eval/overall_carry_accuracy"] = results['overall_carry_accuracy']
        for pos, acc in results['carry_accuracy_by_position'].items():
            pos_num = int(pos.split('_')[1])
            metrics[f"eval/carry_acc_pos_{pos_num}"] = acc
    
    # Log checkpoint info if available
    if checkpoint_path:
        metrics["eval/checkpoint"] = checkpoint_path
    
    # Log all metrics
    wandb.log(metrics)
    
    # Create summary table for digit accuracy by position
    digit_table_data = []
    for pos, acc in sorted(results['digit_accuracy_by_position'].items()):
        pos_num = int(pos.split('_')[1])
        digit_table_data.append([f"Position {pos_num}", f"{acc:.4f}", f"{acc*100:.2f}%"])
    
    if digit_table_data:
        digit_table = wandb.Table(
            columns=["Position", "Accuracy", "Percentage"],
            data=digit_table_data
        )
        wandb.log({"eval/digit_accuracy_table": digit_table})
    
    # Create summary table for carry accuracy if available
    if 'carry_accuracy_by_position' in results and results['carry_accuracy_by_position']:
        carry_table_data = []
        for pos, acc in sorted(results['carry_accuracy_by_position'].items()):
            pos_num = int(pos.split('_')[1])
            carry_table_data.append([f"Position {pos_num}", f"{acc:.4f}", f"{acc*100:.2f}%"])
        
        if carry_table_data:
            carry_table = wandb.Table(
                columns=["Position", "Accuracy", "Percentage"],
                data=carry_table_data
            )
            wandb.log({"eval/carry_accuracy_table": carry_table})
    
    print(f"\n✓ Results logged to wandb (project: {wandb.run.project}, run: {wandb.run.name})")


def print_results(results: Dict, examples: list = None, verbose: bool = False):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nTotal examples: {results['total_examples']}")
    print(f"Valid examples: {results['valid_examples']}")
    
    print(f"\n{'='*40}")
    print("ACCURACY METRICS")
    print(f"{'='*40}")
    
    print(f"\nSequence Accuracy: {results['sequence_accuracy']:.4f} ({results['sequence_accuracy']*100:.2f}%)")
    print(f"  Correct: {results['sequence_correct']} / {results['valid_examples']}")
    
    print(f"\nOverall Digit Accuracy: {results['overall_digit_accuracy']:.4f} ({results['overall_digit_accuracy']*100:.2f}%)")
    
    print(f"\nDigit Accuracy by Position (position 0 = MSD):")
    for pos, acc in sorted(results['digit_accuracy_by_position'].items()):
        pos_num = int(pos.split('_')[1])
        print(f"  Position {pos_num}: {acc:.4f} ({acc*100:.2f}%)")
    
    if 'overall_carry_accuracy' in results:
        print(f"\nOverall Carry Accuracy: {results['overall_carry_accuracy']:.4f} ({results['overall_carry_accuracy']*100:.2f}%)")
        
        print(f"\nCarry Accuracy by Position (position 0 = LSB carry):")
        for pos, acc in sorted(results['carry_accuracy_by_position'].items()):
            pos_num = int(pos.split('_')[1])
            print(f"  Position {pos_num}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n" + "=" * 70)
    
    # Show examples if verbose
    if verbose and examples:
        print("\nSAMPLE PREDICTIONS:")
        print("-" * 50)
        
        # Show some correct and incorrect examples
        correct_examples = [e for e in examples if e['correct']][:10]
        incorrect_examples = [e for e in examples if not e['correct']][:10]
        
        if correct_examples:
            print("\nCorrect predictions:")
            for ex in correct_examples:
                print(f"  ✓ {ex['input']} = {ex['predicted']} (expected: {ex['expected']})")
        
        if incorrect_examples:
            print("\nIncorrect predictions:")
            for ex in incorrect_examples:
                print(f"  ✗ {ex['input']} = {ex['predicted']} (expected: {ex['expected']})")
                print(f"      pred: {ex['predicted_str']} vs exp: {ex['expected_str']}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained addition models on test set with digit-wise accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --checkpoint checkpoints/my_model --data data/addition6_vanilla_strict_mixed_fixed
  python evaluate.py --checkpoint checkpoints/my_model/step_10000 --data data/addition6_lilavati1_strict_mixed_fixed --verbose
  python evaluate.py --checkpoint checkpoints/my_model --data data/addition6_vanilla_strict_mixed_fixed --max-examples 1000
        """
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file or directory")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (default: <checkpoint>/all_config.yaml)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation (default: 32)")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Maximum number of examples to evaluate (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show sample predictions")
    parser.add_argument("--output-json", type=str, default=None,
                       help="Save results to JSON file")
    parser.add_argument("--wandb", action="store_true", default=True,
                       help="Log results to wandb (default: True)")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false",
                       help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default=None,
                       help="Wandb project name (default: trm-lilavati-eval)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                       help="Wandb run name (default: auto-generated)")
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=None,
                       help="Additional tags for wandb run")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, metadata, dataset_mode, digits = load_model_from_checkpoint(
        args.checkpoint, args.config, args.data
    )
    
    print(f"Model loaded successfully!")
    print(f"  Vocab size: {metadata.vocab_size}")
    print(f"  Seq length: {metadata.seq_len}")
    print(f"  Dataset mode: {dataset_mode}")
    print(f"  Digits: {digits}")
    
    # Initialize wandb if requested
    if args.wandb and WANDB_AVAILABLE:
        init_wandb_for_eval(
            checkpoint_path=args.checkpoint,
            dataset_path=args.data,
            dataset_mode=dataset_mode,
            digits=digits,
            project_name=args.wandb_project,
            run_name=args.wandb_run_name,
            tags=args.wandb_tags,
        )
    elif args.wandb and not WANDB_AVAILABLE:
        print("Warning: --wandb specified but wandb is not installed. Install with: pip install wandb")
    
    # Evaluate
    results, examples = evaluate_model(
        model=model,
        dataset_path=args.data,
        dataset_mode=dataset_mode,
        digits=digits,
        vocab_size=metadata.vocab_size,
        batch_size=args.batch_size,
        verbose=args.verbose,
        max_examples=args.max_examples,
    )
    
    # Print results
    print_results(results, examples, args.verbose)
    
    # Log to wandb if requested
    if args.wandb and WANDB_AVAILABLE:
        try:
            if wandb.run is not None:
                log_results_to_wandb(results, checkpoint_path=args.checkpoint)
                wandb.finish()
        except:
            pass
    
    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")
    
    return results


if __name__ == "__main__":
    main()
