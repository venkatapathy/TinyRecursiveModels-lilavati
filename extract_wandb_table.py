#!/usr/bin/env python3
"""
Extract final validation metrics from wandb runs and generate LaTeX table.
"""

import wandb
import argparse

def extract_final_metrics(project_name: str, run_path: str):
    """Extract final validation metrics from a wandb run."""
    api = wandb.Api()
    try:
        run = api.run(run_path)
        history = run.history(pandas=False)  # Get as list of dicts
        
        # Get the last row (final metrics)
        if len(history) == 0:
            return None
        
        last_row = history[-1]  # Last dict in list
        
        # Try to find the right keys - check summary first (final values)
        summary = run.summary
        
        # Extract metrics - try summary first (final values), then history
        metrics = {
            'sequence_accuracy': summary.get('val/sequence_accuracy', summary.get('eval/sequence_accuracy', 
                last_row.get('val/sequence_accuracy', last_row.get('eval/sequence_accuracy', 0.0)))),
            'digit_accuracy': summary.get('val/digit_accuracy', summary.get('eval/digit_accuracy',
                last_row.get('val/digit_accuracy', last_row.get('eval/digit_accuracy', 0.0)))),
            'carry_accuracy': summary.get('val/carry_accuracy', summary.get('eval/carry_accuracy',
                last_row.get('val/carry_accuracy', last_row.get('eval/carry_accuracy', 0.0)))),
            'carry_sequence_accuracy': summary.get('val/carry_sequence_accuracy', summary.get('eval/carry_sequence_accuracy',
                last_row.get('val/carry_sequence_accuracy', last_row.get('eval/carry_sequence_accuracy', 0.0)))),
        }
        
        return metrics
    except Exception as e:
        print(f"Error loading wandb data for {run_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_latex_table(vanilla_metrics, lilavati1_metrics, output_file: str):
    """Generate LaTeX table from wandb metrics."""
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Multiplication accuracy results comparing vanilla and Lil\\={a}vati-1 supervision. ")
        f.write("Results show sequence-level accuracy (Seq) and digit-level accuracy (Digit) ")
        f.write("from final validation metrics.}\n")
        f.write("\\label{tab:multiplication_results}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Method} & \\textbf{ID Seq} & \\textbf{ID Digit} & ")
        f.write("\\textbf{OOD Seq} & \\textbf{OOD Digit} & \\textbf{Carry Acc} \\\\\n")
        f.write("\\midrule\n")
        
        # Vanilla
        if vanilla_metrics:
            seq_acc = vanilla_metrics.get('sequence_accuracy', 0.0)
            digit_acc = vanilla_metrics.get('digit_accuracy', 0.0)
            carry_acc = vanilla_metrics.get('carry_accuracy', 0.0)
            f.write(f"Vanilla & {seq_acc:.4f} & {digit_acc:.4f} & ")
            f.write(f"0.0000 & 0.0000 & {carry_acc:.4f} \\\\\n")
        else:
            f.write("Vanilla & -- & -- & -- & -- & -- \\\\\n")
        
        # Lilavati-1
        if lilavati1_metrics:
            seq_acc = lilavati1_metrics.get('sequence_accuracy', 0.0)
            digit_acc = lilavati1_metrics.get('digit_accuracy', 0.0)
            carry_acc = lilavati1_metrics.get('carry_accuracy', 0.0)
            f.write(f"Lil\\={{a}}vati-1 & {seq_acc:.4f} & {digit_acc:.4f} & ")
            f.write(f"0.0000 & 0.0000 & {carry_acc:.4f} \\\\\n")
        else:
            f.write("Lil\\={a}vati-1 & -- & -- & -- & -- & -- \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract final validation metrics from wandb runs and generate LaTeX table"
    )
    parser.add_argument("--wandb-project", type=str, default="trm-lilavati-multiply",
                       help="Wandb project name")
    parser.add_argument("--vanilla-wandb-run", type=str, required=True,
                       help="Wandb run path for vanilla model (e.g., venkatapathy/trm-lilavati-multiply/5rvcnu0a)")
    parser.add_argument("--lilavati1-wandb-run", type=str, required=True,
                       help="Wandb run path for Lilavati-1 model (e.g., venkatapathy/trm-lilavati-multiply/nhpxo2n1)")
    parser.add_argument("--output-file", type=str, default="results_multiplication/multiplication_results_table.tex",
                       help="Output LaTeX table file")
    
    args = parser.parse_args()
    
    print("Extracting metrics from wandb...")
    
    # Extract metrics
    vanilla_metrics = extract_final_metrics(args.wandb_project, args.vanilla_wandb_run)
    lilavati1_metrics = extract_final_metrics(args.wandb_project, args.lilavati1_wandb_run)
    
    if vanilla_metrics:
        print(f"\nVanilla metrics:")
        print(f"  Sequence Accuracy: {vanilla_metrics.get('sequence_accuracy', 0.0):.4f}")
        print(f"  Digit Accuracy: {vanilla_metrics.get('digit_accuracy', 0.0):.4f}")
        print(f"  Carry Accuracy: {vanilla_metrics.get('carry_accuracy', 0.0):.4f}")
    
    if lilavati1_metrics:
        print(f"\nLilavati-1 metrics:")
        print(f"  Sequence Accuracy: {lilavati1_metrics.get('sequence_accuracy', 0.0):.4f}")
        print(f"  Digit Accuracy: {lilavati1_metrics.get('digit_accuracy', 0.0):.4f}")
        print(f"  Carry Accuracy: {lilavati1_metrics.get('carry_accuracy', 0.0):.4f}")
    
    # Generate table
    import os
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    generate_latex_table(vanilla_metrics, lilavati1_metrics, args.output_file)
    
    print(f"\nTable generated successfully!")


if __name__ == "__main__":
    main()
