#!/usr/bin/env python3
"""
Script to generate evaluation results, tables, and figures for the paper.

This script:
1. Runs evaluation on trained models (vanilla and lilavati-2)
2. Extracts results and separates ID vs OOD based on result length
3. Generates LaTeX tables
4. Generates figures (learning curves, generalization plots)

Usage:
    python generate_results.py --checkpoint-dir checkpoints/trm-lilavati --output-dir results
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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def run_evaluation(checkpoint_path: str, dataset_path: str, output_json: str = None) -> Dict:
    """Run evaluation script and return results."""
    cmd = [
        sys.executable, "evaluate.py",
        "--checkpoint", checkpoint_path,
        "--data", dataset_path,
        "--no-wandb"
    ]
    
    if output_json:
        cmd.extend(["--output-json", output_json])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running evaluation:")
        print(result.stderr)
        return None
    
    if output_json and os.path.exists(output_json):
        with open(output_json, 'r') as f:
            return json.load(f)
    
    return None


def separate_id_ood(results: Dict, train_max_digits: int = 4) -> Dict:
    """Separate results into in-distribution (ID) and out-of-distribution (OOD).
    
    For addition: if training on d-digit operands, results can be up to d+1 digits.
    So if train_max_digits=4 (4-digit operands), then results up to 5 digits are ID,
    and results with 6+ digits are OOD.
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
    
    if 'results_by_length' not in results:
        return {'id': id_results, 'ood': ood_results}
    
    # For addition: max result length = train_max_digits + 1 (accounting for overflow)
    max_result_length_id = train_max_digits + 1
    
    for length_str, stats in results['results_by_length'].items():
        length = int(length_str)
        count = stats['count']
        seq_acc = stats['sequence_accuracy']
        digit_acc = stats['digit_accuracy']
        
        # Calculate digit totals from accuracy and count
        # For addition: each example has 'length' digit positions in the result
        # digit_total = count * length (each example has 'length' digit positions)
        digit_total = count * length
        digit_correct = int(digit_acc * digit_total)
        
        if length <= max_result_length_id:
            # In-distribution
            id_results['total'] += count
            id_results['correct'] += int(seq_acc * count)
            id_results['digit_correct'] += digit_correct
            id_results['digit_total'] += digit_total
        else:
            # Out-of-distribution
            ood_results['total'] += count
            ood_results['correct'] += int(seq_acc * count)
            ood_results['digit_correct'] += digit_correct
            ood_results['digit_total'] += digit_total
    
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
    """Generate LaTeX table for addition results."""
    methods = ['Vanilla', 'Lil\\={a}vati-2']
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Addition accuracy results comparing vanilla and Lil\\={a}vati-2 supervision. ")
        f.write("Results show sequence-level accuracy (Seq) and digit-level accuracy (Digit) ")
        f.write("for both in-distribution (ID) and out-of-distribution (OOD) test sets.}\n")
        f.write("\\label{tab:addition_results}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Method} & \\textbf{ID Seq} & \\textbf{ID Digit} & ")
        f.write("\\textbf{OOD Seq} & \\textbf{OOD Digit} & \\textbf{Carry Acc} \\\\\n")
        f.write("\\midrule\n")
        
        for method in methods:
            method_key = method.lower().replace('\\={a}', 'a').replace('lilavati-2', 'lilavati2')
            if method_key not in results_dict:
                method_key = method.lower().replace('\\={a}', 'a').replace('lilavati-2', 'lilavati2').replace(' ', '_')
            
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
    """Plot learning curves comparing vanilla and lilavati-2."""
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
            label = 'Basic-TRM' if 'vanilla' in method.lower() else 'Addition-probed TRM'
            
            ax.plot(steps_clean, seq_acc_clean, style, label=label, linewidth=2)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Sequence-Level Accuracy', fontsize=12)
    ax.set_title('Training Curves for Addition Models', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Learning curves saved to: {output_file}")


def plot_generalization(results_dict: Dict[str, Dict], output_file: str, train_max_digits: int = 4):
    """Plot length generalization analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method, method_key, style, label in [
        ('Vanilla', 'vanilla', '-', 'Basic-TRM'),
        ('Lilavati-2', 'lilavati2', '--', 'Addition-probed TRM')
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
    # For addition: if training on d-digit operands, max result length is d+1
    max_result_length = train_max_digits + 1
    ax.axvline(x=max_result_length, color='red', linestyle=':', linewidth=2, 
               label=f'Max Training Result Length ({max_result_length} digits)', alpha=0.7)
    
    ax.set_xlabel('Result Digit Length', fontsize=12)
    ax.set_ylabel('Sequence-Level Accuracy', fontsize=12)
    ax.set_title('Length Generalization Analysis for Addition', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Generalization plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation results, tables, and figures for the paper"
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Directory containing model checkpoints (optional if using specific checkpoint paths)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results, tables, and figures")
    parser.add_argument("--train-max-digits", type=int, default=4,
                       help="Maximum digit length seen during training (for ID/OOD split)")
    parser.add_argument("--vanilla-checkpoint", type=str, default=None,
                       help="Path to vanilla model checkpoint")
    parser.add_argument("--lilavati2-checkpoint", type=str, default=None,
                       help="Path to Lilavati-2 model checkpoint")
    parser.add_argument("--vanilla-data", type=str, default=None,
                       help="Path to vanilla dataset")
    parser.add_argument("--lilavati2-data", type=str, default=None,
                       help="Path to Lilavati-2 dataset")
    parser.add_argument("--skip-eval", action="store_true",
                       help="Skip evaluation and use existing JSON files")
    parser.add_argument("--wandb-project", type=str, default="trm-lilavati",
                       help="Wandb project name for loading training curves")
    parser.add_argument("--vanilla-wandb-run", type=str, default=None,
                       help="Wandb run name for vanilla model")
    parser.add_argument("--lilavati2-wandb-run", type=str, default=None,
                       help="Wandb run name for Lilavati-2 model")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.skip_eval:
        if not args.vanilla_checkpoint and not args.lilavati2_checkpoint:
            parser.error("Either provide --vanilla-checkpoint and/or --lilavati2-checkpoint, or use --skip-eval with existing JSON files")
        if args.vanilla_checkpoint and not args.vanilla_data:
            parser.error("--vanilla-data is required when --vanilla-checkpoint is provided")
        if args.lilavati2_checkpoint and not args.lilavati2_data:
            parser.error("--lilavati2-data is required when --lilavati2-checkpoint is provided")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    
    # Find checkpoints if not provided
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    results_dict = {}
    
    # Evaluate models
    if not args.skip_eval:
        # Vanilla
        if args.vanilla_checkpoint and args.vanilla_data:
            print("\n" + "="*70)
            print("Evaluating Vanilla Model")
            print("="*70)
            vanilla_json = os.path.join(args.output_dir, "vanilla_results.json")
            results = run_evaluation(args.vanilla_checkpoint, args.vanilla_data, vanilla_json)
            if results:
                results_dict['vanilla'] = results
        
        # Lilavati-2
        if args.lilavati2_checkpoint and args.lilavati2_data:
            print("\n" + "="*70)
            print("Evaluating Lilavati-2 Model")
            print("="*70)
            lilavati2_json = os.path.join(args.output_dir, "lilavati2_results.json")
            results = run_evaluation(args.lilavati2_checkpoint, args.lilavati2_data, lilavati2_json)
            if results:
                results_dict['lilavati2'] = results
            else:
                print(f"WARNING: Lilavati-2 evaluation failed or returned no results!")
                print(f"  Checkpoint: {args.lilavati2_checkpoint}")
                print(f"  Data: {args.lilavati2_data}")
                if os.path.exists(lilavati2_json):
                    print(f"  Results file exists but is empty or invalid: {lilavati2_json}")
                else:
                    print(f"  Results file was not created: {lilavati2_json}")
    else:
        # Load existing results
        vanilla_json = os.path.join(args.output_dir, "vanilla_results.json")
        lilavati2_json = os.path.join(args.output_dir, "lilavati2_results.json")
        
        print(f"\nLoading existing results from {args.output_dir}...")
        if os.path.exists(vanilla_json):
            print(f"  ✓ Found: {vanilla_json}")
            with open(vanilla_json, 'r') as f:
                results_dict['vanilla'] = json.load(f)
        else:
            print(f"  ✗ Missing: {vanilla_json}")
        
        if os.path.exists(lilavati2_json):
            print(f"  ✓ Found: {lilavati2_json}")
            with open(lilavati2_json, 'r') as f:
                results_dict['lilavati2'] = json.load(f)
        else:
            print(f"  ✗ Missing: {lilavati2_json}")
            print(f"    To generate it, run evaluation with --lilavati2-checkpoint and --lilavati2-data")
    
    # Check if we have results
    if not results_dict:
        print("\nERROR: No evaluation results found!")
        print("  Please run evaluation first or provide existing JSON files.")
        return
    
    # Separate ID/OOD for each method
    processed_results = {}
    for method_key, results in results_dict.items():
        processed_results[method_key] = {
            **results,
            **separate_id_ood(results, args.train_max_digits)
        }
    
    # Print what we have
    print(f"\n{'='*70}")
    print("RESULTS FOUND:")
    print(f"{'='*70}")
    for method_key in processed_results.keys():
        print(f"  ✓ {method_key}")
    if 'lilavati2' not in processed_results:
        print(f"  ✗ lilavati2 (missing - check evaluation logs)")
    
    # Generate LaTeX table
    table_file = os.path.join(args.output_dir, "addition_results_table.tex")
    generate_latex_table(processed_results, table_file)
    
    # Load training curves from wandb if available
    wandb_data = {}
    if args.vanilla_wandb_run:
        print(f"\nLoading training data for vanilla from wandb...")
        wandb_data['vanilla'] = load_wandb_data(args.wandb_project, args.vanilla_wandb_run)
    
    if args.lilavati2_wandb_run:
        print(f"\nLoading training data for lilavati2 from wandb...")
        wandb_data['lilavati2'] = load_wandb_data(args.wandb_project, args.lilavati2_wandb_run)
    
    # Generate figures
    learning_curves_file = os.path.join(args.output_dir, "figures", "addition_learning_curves.pdf")
    if wandb_data:
        plot_learning_curves(wandb_data, learning_curves_file)
    else:
        print("Warning: No wandb data provided. Skipping learning curves plot.")
    
    generalization_file = os.path.join(args.output_dir, "figures", "addition_generalization.pdf")
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
