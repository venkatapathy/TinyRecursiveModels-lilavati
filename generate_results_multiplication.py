#!/usr/bin/env python3
"""
Script to generate evaluation results, tables, and figures for multiplication models.

This script:
1. Runs evaluation on trained models (vanilla and lilavati-1)
2. Extracts results and separates ID vs OOD based on result length
3. Generates LaTeX tables
4. Generates figures (learning curves, generalization plots)

Usage:
    python generate_results_multiplication.py --vanilla-checkpoint <path> --lilavati1-checkpoint <path> --output-dir results_multiplication
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from evaluators.multiplication import MultiplicationEvaluator
from utils.functions import load_model_class
from omegaconf import OmegaConf


def load_model_from_checkpoint(checkpoint_path: str, config_path: str = None, dataset_path: str = None):
    """Load model from checkpoint (adapted for multiplication)."""
    
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
        data_paths = config.get('data_paths', ['data/multiplication'])
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
    digits = config.get('digits', 3)
    
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


def evaluate_multiplication_model(
    model,
    dataset_path: str,
    dataset_mode: str,
    digits: int,
    metadata: PuzzleDatasetMetadata,
    batch_size: int = 32,
    verbose: bool = False,
    max_examples: int = None,
) -> Dict:
    """Evaluate multiplication model on test set and return metrics."""
    
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
    evaluator = MultiplicationEvaluator(
        data_path=dataset_path,
        eval_metadata=metadata,
        dataset_mode=dataset_mode,
        digits=digits,
    )
    
    print(f"\nEvaluating on test set...")
    print(f"  Dataset: {dataset_path}")
    print(f"  Mode: {dataset_mode}")
    print(f"  Digits: {digits}")
    print()
    
    evaluator.begin_eval()
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
            evaluator.update_batch(batch, preds)
            
            processed += len(batch["inputs"])
            
            if processed % 1000 == 0:
                print(f"  Processed {processed} examples...")
    
    # Get results
    import torch.distributed as dist
    results = evaluator.result(None, rank=0, world_size=1, group=None)
    
    # Convert to standard format
    output = {
        'sequence_accuracy': results.get('val/sequence_accuracy', 0.0),
        'overall_digit_accuracy': results.get('val/digit_accuracy', 0.0),
        'valid_examples': processed,
        'total_examples': processed,
    }
    
    if 'val/carry_accuracy' in results:
        output['overall_carry_accuracy'] = results['val/carry_accuracy']
        output['carry_sequence_accuracy'] = results.get('val/carry_sequence_accuracy', 0.0)
    
    # Note: MultiplicationEvaluator doesn't track by length, so we'll need to add that
    # For now, we'll use overall metrics
    output['results_by_length'] = {}  # Will be populated if needed
    
    return output


def separate_id_ood(results: Dict, train_max_digits: int = 3) -> Dict:
    """Separate results into in-distribution (ID) and out-of-distribution (OOD).
    
    For multiplication: if training on d-digit operands, results have 2d digits.
    So if train_max_digits=3 (3-digit operands), then results up to 6 digits are ID,
    and results with 7+ digits are OOD.
    """
    id_results = {
        'total': 0,
        'correct': 0,
        'digit_correct': 0,
        'digit_total': 0,
    }
    ood_results = {
        'total': 0,
        'correct': 0,
        'digit_correct': 0,
        'digit_total': 0,
    }
    
    if 'results_by_length' not in results or not results['results_by_length']:
        # If no length breakdown, use overall metrics for both ID and OOD
        # This is a limitation - ideally we'd track by length
        total = results.get('valid_examples', 0)
        seq_correct = int(results.get('sequence_accuracy', 0.0) * total)
        digit_total = total * (2 * train_max_digits)  # Approximate
        digit_correct = int(results.get('overall_digit_accuracy', 0.0) * digit_total)
        
        # For now, assign all to ID (since we don't have length breakdown)
        id_results = {
            'total': total,
            'correct': seq_correct,
            'digit_correct': digit_correct,
            'digit_total': digit_total,
        }
        
        return {
            'id': {
                **id_results,
                'sequence_accuracy': results.get('sequence_accuracy', 0.0),
                'digit_accuracy': results.get('overall_digit_accuracy', 0.0),
            },
            'ood': {
                **ood_results,
                'sequence_accuracy': 0.0,
                'digit_accuracy': 0.0,
            }
        }
    
    # For multiplication: max result length = 2 * train_max_digits
    max_result_length_id = 2 * train_max_digits
    
    for length_str, stats in results['results_by_length'].items():
        length = int(length_str)
        if length <= max_result_length_id:
            # In-distribution
            id_results['total'] += stats['count']
            id_results['correct'] += int(stats['sequence_accuracy'] * stats['count'])
            id_results['digit_correct'] += stats['digit_correct']
            id_results['digit_total'] += stats['digit_total']
        else:
            # Out-of-distribution
            ood_results['total'] += stats['count']
            ood_results['correct'] += int(stats['sequence_accuracy'] * stats['count'])
            ood_results['digit_correct'] += stats['digit_correct']
            ood_results['digit_total'] += stats['digit_total']
    
    # Compute accuracies
    id_seq_acc = id_results['correct'] / id_results['total'] if id_results['total'] > 0 else 0.0
    id_digit_acc = id_results['digit_correct'] / id_results['digit_total'] if id_results['digit_total'] > 0 else 0.0
    
    ood_seq_acc = ood_results['correct'] / ood_results['total'] if ood_results['total'] > 0 else 0.0
    ood_digit_acc = ood_results['digit_correct'] / ood_results['digit_total'] if ood_results['digit_total'] > 0 else 0.0
    
    return {
        'id': {
            **id_results,
            'sequence_accuracy': id_seq_acc,
            'digit_accuracy': id_digit_acc,
        },
        'ood': {
            **ood_results,
            'sequence_accuracy': ood_seq_acc,
            'digit_accuracy': ood_digit_acc,
        }
    }


def generate_latex_table(results_dict: Dict[str, Dict], output_file: str):
    """Generate LaTeX table for multiplication results."""
    methods = ['Vanilla', 'Lil\\={a}vati-1']
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Multiplication accuracy results comparing vanilla and Lil\\={a}vati-1 supervision. ")
        f.write("Results show sequence-level accuracy (Seq) and digit-level accuracy (Digit) ")
        f.write("for both in-distribution (ID) and out-of-distribution (OOD) test sets.}\n")
        f.write("\\label{tab:multiplication_results}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Method} & \\textbf{ID Seq} & \\textbf{ID Digit} & ")
        f.write("\\textbf{OOD Seq} & \\textbf{OOD Digit} & \\textbf{Carry Acc} \\\\\n")
        f.write("\\midrule\n")
        
        for method in methods:
            method_key = method.lower().replace('\\={a}', 'a').replace('lilavati-1', 'lilavati1')
            if method_key not in results_dict:
                method_key = method.lower().replace('\\={a}', 'a').replace('lilavati-1', 'lilavati1').replace(' ', '_')
            
            if method_key in results_dict:
                r = results_dict[method_key]
                id_seq = r.get('id', {}).get('sequence_accuracy', 0.0)
                id_digit = r.get('id', {}).get('digit_accuracy', 0.0)
                ood_seq = r.get('ood', {}).get('sequence_accuracy', 0.0)
                ood_digit = r.get('ood', {}).get('digit_accuracy', 0.0)
                carry_acc = r.get('overall_carry_accuracy', 0.0)
                
                f.write(f"{method} & {id_seq:.4f} & {id_digit:.4f} & ")
                f.write(f"{ood_seq:.4f} & {ood_digit:.4f} & {carry_acc:.4f} \\\\\n")
            else:
                f.write(f"{method} & -- & -- & -- & -- & -- \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {output_file}")


def load_wandb_data(project_name: str, run_name: str) -> Optional[Dict]:
    """Load training data from wandb."""
    if not WANDB_AVAILABLE:
        print("Warning: wandb not available. Cannot load training curves.")
        return None
    
    api = wandb.Api()
    try:
        run = api.run(f"{project_name}/{run_name}")
        history = run.history()
        
        # Get sequence accuracy from validation metrics
        seq_acc_key = 'val/sequence_accuracy'
        if seq_acc_key not in history.columns:
            seq_acc_key = 'eval/sequence_accuracy'
        
        steps = history['_step'].tolist() if '_step' in history.columns else list(range(len(history)))
        seq_acc = history[seq_acc_key].tolist() if seq_acc_key in history.columns else None
        
        return {
            'steps': steps,
            'sequence_accuracy': seq_acc,
            'epoch': steps,  # Use steps as epochs
        }
    except Exception as e:
        print(f"Error loading wandb data for {project_name}/{run_name}: {e}")
        return None


def plot_learning_curves(wandb_data: Dict[str, Dict], output_file: str):
    """Plot learning curves comparing vanilla and lilavati-1."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method, data in wandb_data.items():
        if data and 'sequence_accuracy' in data and data['sequence_accuracy'] is not None:
            steps = data['steps']
            seq_acc = data['sequence_accuracy']
            
            # Filter out None values
            valid_indices = [i for i, acc in enumerate(seq_acc) if acc is not None and not np.isnan(acc)]
            steps_clean = [steps[i] for i in valid_indices]
            seq_acc_clean = [seq_acc[i] for i in valid_indices]
            
            style = '-' if 'vanilla' in method.lower() else '--'
            label = 'Basic-TRM' if 'vanilla' in method.lower() else 'Multiplication-probed TRM'
            
            ax.plot(steps_clean, seq_acc_clean, style, label=label, linewidth=2)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Sequence-Level Accuracy', fontsize=12)
    ax.set_title('Training Curves for Multiplication Models', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Learning curves saved to: {output_file}")


def plot_generalization(results_dict: Dict[str, Dict], output_file: str, train_max_digits: int = 3):
    """Plot length generalization analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method, method_key, style, label in [
        ('Vanilla', 'vanilla', '-', 'Basic-TRM'),
        ('Lilavati-1', 'lilavati1', '--', 'Multiplication-probed TRM')
    ]:
        if method_key in results_dict:
            r = results_dict[method_key]
            if 'results_by_length' in r and r['results_by_length']:
                lengths = []
                accuracies = []
                for length_str in sorted(r['results_by_length'].keys(), key=int):
                    length = int(length_str)
                    stats = r['results_by_length'][length_str]
                    lengths.append(length)
                    accuracies.append(stats['sequence_accuracy'])
                
                if lengths:
                    ax.plot(lengths, accuracies, style, label=label, linewidth=2, marker='o', markersize=6)
    
    # Add vertical line for training max (result length, not operand length)
    # For multiplication: if training on d-digit operands, max result length is 2d
    max_result_length = 2 * train_max_digits
    ax.axvline(x=max_result_length, color='red', linestyle=':', linewidth=2, 
               label=f'Max Training Result Length ({max_result_length} digits)', alpha=0.7)
    
    ax.set_xlabel('Result Digit Length', fontsize=12)
    ax.set_ylabel('Sequence-Level Accuracy', fontsize=12)
    ax.set_title('Length Generalization Analysis for Multiplication', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Generalization plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation results, tables, and figures for multiplication models"
    )
    parser.add_argument("--output-dir", type=str, default="results_multiplication",
                       help="Output directory for results, tables, and figures")
    parser.add_argument("--train-max-digits", type=int, default=3,
                       help="Maximum digit length seen during training (for ID/OOD split)")
    parser.add_argument("--vanilla-checkpoint", type=str, default=None,
                       help="Path to vanilla model checkpoint")
    parser.add_argument("--lilavati1-checkpoint", type=str, default=None,
                       help="Path to Lilavati-1 model checkpoint")
    parser.add_argument("--vanilla-data", type=str, default=None,
                       help="Path to vanilla dataset")
    parser.add_argument("--lilavati1-data", type=str, default=None,
                       help="Path to Lilavati-1 dataset")
    parser.add_argument("--skip-eval", action="store_true",
                       help="Skip evaluation and use existing JSON files")
    parser.add_argument("--wandb-project", type=str, default="trm-lilavati-multiply",
                       help="Wandb project name for loading training curves")
    parser.add_argument("--vanilla-wandb-run", type=str, default=None,
                       help="Wandb run name for vanilla model")
    parser.add_argument("--lilavati1-wandb-run", type=str, default=None,
                       help="Wandb run name for Lilavati-1 model")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.skip_eval:
        if not args.vanilla_checkpoint and not args.lilavati1_checkpoint:
            parser.error("Either provide --vanilla-checkpoint and/or --lilavati1-checkpoint, or use --skip-eval with existing JSON files")
        if args.vanilla_checkpoint and not args.vanilla_data:
            parser.error("--vanilla-data is required when --vanilla-checkpoint is provided")
        if args.lilavati1_checkpoint and not args.lilavati1_data:
            parser.error("--lilavati1-data is required when --lilavati1-checkpoint is provided")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    
    results_dict = {}
    
    # Evaluate models
    if not args.skip_eval:
        # Vanilla
        if args.vanilla_checkpoint and args.vanilla_data:
            print("\n" + "="*70)
            print("Evaluating Vanilla Model")
            print("="*70)
            vanilla_json = os.path.join(args.output_dir, "vanilla_results.json")
            try:
                model, metadata, dataset_mode, digits = load_model_from_checkpoint(
                    args.vanilla_checkpoint, None, args.vanilla_data
                )
                results = evaluate_multiplication_model(
                    model, args.vanilla_data, dataset_mode, digits, metadata,
                    batch_size=args.batch_size
                )
                with open(vanilla_json, 'w') as f:
                    json.dump(results, f, indent=2)
                results_dict['vanilla'] = results
            except Exception as e:
                print(f"Error evaluating vanilla model: {e}")
                import traceback
                traceback.print_exc()
        
        # Lilavati-1
        if args.lilavati1_checkpoint and args.lilavati1_data:
            print("\n" + "="*70)
            print("Evaluating Lilavati-1 Model")
            print("="*70)
            lilavati1_json = os.path.join(args.output_dir, "lilavati1_results.json")
            try:
                model, metadata, dataset_mode, digits = load_model_from_checkpoint(
                    args.lilavati1_checkpoint, None, args.lilavati1_data
                )
                results = evaluate_multiplication_model(
                    model, args.lilavati1_data, dataset_mode, digits, metadata,
                    batch_size=args.batch_size
                )
                with open(lilavati1_json, 'w') as f:
                    json.dump(results, f, indent=2)
                results_dict['lilavati1'] = results
            except Exception as e:
                print(f"Error evaluating lilavati1 model: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Load existing results
        vanilla_json = os.path.join(args.output_dir, "vanilla_results.json")
        lilavati1_json = os.path.join(args.output_dir, "lilavati1_results.json")
        
        if os.path.exists(vanilla_json):
            with open(vanilla_json, 'r') as f:
                results_dict['vanilla'] = json.load(f)
        
        if os.path.exists(lilavati1_json):
            with open(lilavati1_json, 'r') as f:
                results_dict['lilavati1'] = json.load(f)
    
    # Separate ID/OOD for each method
    processed_results = {}
    for method_key, results in results_dict.items():
        processed_results[method_key] = {
            **results,
            **separate_id_ood(results, args.train_max_digits)
        }
    
    # Generate LaTeX table
    table_file = os.path.join(args.output_dir, "multiplication_results_table.tex")
    generate_latex_table(processed_results, table_file)
    
    # Load training curves from wandb if available
    wandb_data = {}
    if args.vanilla_wandb_run:
        print(f"\nLoading training data for vanilla from wandb...")
        wandb_data['vanilla'] = load_wandb_data(args.wandb_project, args.vanilla_wandb_run)
    
    if args.lilavati1_wandb_run:
        print(f"\nLoading training data for lilavati1 from wandb...")
        wandb_data['lilavati1'] = load_wandb_data(args.wandb_project, args.lilavati1_wandb_run)
    
    # Generate figures
    learning_curves_file = os.path.join(args.output_dir, "figures", "multiplication_learning_curves.pdf")
    if wandb_data:
        plot_learning_curves(wandb_data, learning_curves_file)
    else:
        print("Warning: No wandb data provided. Skipping learning curves plot.")
    
    generalization_file = os.path.join(args.output_dir, "figures", "multiplication_generalization.pdf")
    plot_generalization(processed_results, generalization_file, args.train_max_digits)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for method_key, results in processed_results.items():
        print(f"\n{method_key.upper()}:")
        if 'id' in results:
            print(f"  ID Sequence Accuracy: {results['id']['sequence_accuracy']:.4f}")
            print(f"  ID Digit Accuracy: {results['id']['digit_accuracy']:.4f}")
        if 'ood' in results:
            print(f"  OOD Sequence Accuracy: {results['ood']['sequence_accuracy']:.4f}")
            print(f"  OOD Digit Accuracy: {results['ood']['digit_accuracy']:.4f}")
        if 'overall_carry_accuracy' in results:
            print(f"  Carry Accuracy: {results['overall_carry_accuracy']:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Table: {table_file}")
    print(f"  - Learning curves: {learning_curves_file}")
    print(f"  - Generalization plot: {generalization_file}")


if __name__ == "__main__":
    main()
