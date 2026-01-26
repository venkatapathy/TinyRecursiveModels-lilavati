import json
import os
import glob

def load_metrics(run_name):
    # Try finding the json file in results/
    pattern = f"results/*{run_name}*.json"
    files = glob.glob(pattern)
    if not files:
        return None
    
    # Sort by time to get latest
    files.sort(key=os.path.getmtime, reverse=True)
    with open(files[0], 'r') as f:
        return json.load(f)

def get_metric(metrics, metric_name, default="-"):
    if metrics and metric_name in metrics:
        val = metrics[metric_name]
        return f"{val:.4f}"
    return default

def generate_latex():
    # Define the rows and corresponding run names / keys
    # Map Method Name -> (ID Metrics Key Prefix, OOD Metrics Key Prefix)
    # TRM run names usually end with _val or _test based on our new split argument
    
    methods = [
        {
            "name": "Vanilla",
            "run_id": "basicfour_vanilla", # Base run name from config
            "id_suffix": "val",
            "ood_suffix": "test"
        },
        {
            "name": "Concat",
            "run_id": "basicfour_concat",
            "id_suffix": "val", # We will match file names like basicfour_concat_val.json
            "ood_suffix": "test"
        },
        {
            "name": "Reverse",
            "run_id": "basicfour_concat_reverse",
            "id_suffix": "val",
            "ood_suffix": "test"
        },
        {
            "name": "Qwen 1.5B",
            "run_id": "Qwen2.5-Math-1.5B-Instruct-Vanilla", # We used this run name for Qwen
            "id_suffix": "val", # We need to run Qwen on Val explicitly? Or verify if we did.
            "ood_suffix": "test" # Default was test
        }
        # Note: Qwen run names might be qwen_val and qwen_test if we enforce that naming in our execution script
    ]
    
    # LaTeX Header
    latex = r"""\begin{table}[h]
\centering
\caption{Addition accuracy results comparing vanilla and Lil\={a}vati-2 supervision. Results show sequence-level accuracy (Seq) and digit-level accuracy (Digit) for both in-distribution (ID) and out-of-distribution (OOD) test sets.}
\label{tab:addition_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{ID Seq} & \textbf{ID Digit} & \textbf{OOD Seq} & \textbf{OOD Digit} & \textbf{Carry Acc} \\
\midrule
"""

    for method in methods:
        name = method["name"]
        
        # Load ID and OOD Metrics from the same file (usually _test.json)
        # because the new evaluator splits ID/OOD internally in every run.
        if name.startswith("Qwen"):
             id_run = f"{method['run_id']}_val"
             ood_run = f"{method['run_id']}_test"
        else:
             # Favor the test split for full table metrics
             run_id = f"{method['run_id']}_test"
             metrics = load_metrics(run_id)
             # fallback to val if test not found
             if not metrics:
                 run_id = f"{method['run_id']}_val"
                 metrics = load_metrics(run_id)
        
        # Extract Values
        id_seq = "-"
        id_digit = "-"
        ood_seq = "-"
        ood_digit = "-"
        carry_acc = "-"

        if name.startswith("Qwen"):
            # Existing logic for Qwen if needed
            metrics_id = load_metrics(id_run)
            metrics_ood = load_metrics(ood_run)
            if metrics_id:
                for k, v in metrics_id.items():
                    if k.endswith("accuracy") and "digit" not in k:
                        id_seq = f"{v:.4f}"
                        break
            if metrics_ood:
                for k, v in metrics_ood.items():
                    if k.endswith("accuracy") and "digit" not in k:
                        ood_seq = f"{v:.4f}"
                        break
        elif metrics:
            # New prioritized extraction for TRM results
            # ID Metrics
            id_seq = get_metric(metrics, "basicfour/id_accuracy_test", 
                               get_metric(metrics, "basicfour/id_accuracy_val", "-"))
            id_digit = get_metric(metrics, "basicfour/id_digit_accuracy_test", 
                                 get_metric(metrics, "basicfour/id_digit_accuracy_val", "-"))
            
            # OOD Metrics
            ood_seq = get_metric(metrics, "basicfour/ood_accuracy_test", 
                                get_metric(metrics, "basicfour/ood_accuracy_val", "-"))
            ood_digit = get_metric(metrics, "basicfour/ood_digit_accuracy_test", 
                                  get_metric(metrics, "basicfour/ood_digit_accuracy_val", "-"))
            
            # Fallback if granular keys not found (e.g., old results)
            if id_seq == "-" or ood_seq == "-":
                for k, v in metrics.items():
                    if k.endswith("accuracy") and "digit" not in k and "carry" not in k and "id_" not in k and "ood_" not in k:
                        # If we only have global accuracy, assign it based on file name or as global
                        if "test" in run_id: ood_seq = f"{v:.4f}"
                        else: id_seq = f"{v:.4f}"
            
            if id_digit == "-" or ood_digit == "-":
                for k, v in metrics.items():
                    if "digit_accuracy" in k and "id_" not in k and "ood_" not in k:
                        if "test" in run_id: ood_digit = f"{v:.4f}"
                        else: id_digit = f"{v:.4f}"

            # Carry Acc
            for k, v in metrics.items():
                if "carry_accuracy" in k:
                    carry_acc = f"{v:.4f}"
                    break
        
        latex += f"{name} & {id_seq} & {id_digit} & {ood_seq} & {ood_digit} & {carry_acc} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    print(latex)
    
    # Save to file
    os.makedirs("results", exist_ok=True)
    with open("results/addition_results_table_generated.tex", "w") as f:
        f.write(latex)
    print("Table saved to results/addition_results_table_generated.tex")

if __name__ == "__main__":
    generate_latex()
