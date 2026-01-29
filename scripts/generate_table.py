import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

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

def get_full_methods_list():
    return [
        {"name": "TRM (NS)", "run_id": "basicfour_vanilla"},
        {"name": "TRM (OS-After)", "run_id": "basicfour_concat"},
        {"name": "TRM (OS-Before)", "run_id": "basicfour_concat_reverse"},
        {"name": "Transformer (NS)", "run_id": "baseline_transformer_300k"},
        {"name": "Transformer (OS-Before)", "run_id": "baseline_transformer_300k_concat_reverse"},
        {"name": "Transformer (40x)", "run_id": "baseline_transformer_40x"},
        {"name": "TRM (OS-Before-100d)", "run_id": "reverse_100d_run"},
        {"name": "Qwen", "run_id": "Qwen2.5-Math-1.5B-Instruct-Vanilla"}
    ]

def generate_latex():
    methods = get_full_methods_list()
    
    # LaTeX Header
    latex = r"""\begin{table}[h]
\centering
\caption{Addition accuracy results. Results show sequence-level accuracy (Seq) and digit-level accuracy (Digit) for both in-distribution (ID) and out-of-distribution (OOD) test sets.}
\label{tab:addition_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{ID Seq} & \textbf{ID Digit} & \textbf{OOD Seq} & \textbf{OOD Digit} & \textbf{Carry Acc} \\
\midrule
"""

    for method in methods:
        name = method["name"]
        
        # Load Metrics
        metrics = load_metrics(f"{method['run_id']}_test")
        if not metrics:
            metrics = load_metrics(method['run_id'])
        if not metrics:
            metrics = load_metrics(f"{method['run_id']}_val")
        
        # Extract Values
        id_seq = "-"
        id_digit = "-"
        ood_seq = "-"
        ood_digit = "-"
        carry_acc = "-"

        if metrics:
            # Map keys based on known patterns
            id_seq = get_metric(metrics, "basicfour/id_accuracy_test", 
                               get_metric(metrics, "basicfour/id_accuracy_val", 
                                         get_metric(metrics, "basicfour/id_accuracy", "-")))
            id_digit = get_metric(metrics, "basicfour/id_digit_accuracy_test", 
                                 get_metric(metrics, "basicfour/id_digit_accuracy_val", 
                                           get_metric(metrics, "basicfour/id_digit_accuracy", "-")))
            
            ood_seq = get_metric(metrics, "basicfour/ood_accuracy_test", 
                                get_metric(metrics, "basicfour/ood_accuracy_val", 
                                          get_metric(metrics, "basicfour/ood_accuracy", "-")))
            ood_digit = get_metric(metrics, "basicfour/ood_digit_accuracy_test", 
                                   get_metric(metrics, "basicfour/ood_digit_accuracy_val", 
                                             get_metric(metrics, "basicfour/ood_digit_accuracy", "-")))

            # Priority 2: Vanilla keys (for Qwen/SLM)
            if id_seq == "-": id_seq = get_metric(metrics, "vanilla/id_accuracy", "-")
            if id_digit == "-": id_digit = get_metric(metrics, "vanilla/id_digit_accuracy", "-")
            if ood_seq == "-": ood_seq = get_metric(metrics, "vanilla/ood_accuracy", "-")
            if ood_digit == "-": ood_digit = get_metric(metrics, "vanilla/ood_digit_accuracy", "-")

            # Carry Acc
            carry_acc = get_metric(metrics, "basicfour/add_carry_accuracy_test", 
                                  get_metric(metrics, "basicfour/add_carry_accuracy", 
                                            get_metric(metrics, "basicfour/carry_accuracy", "-")))
        
        latex += f"{name} & {id_seq} & {id_digit} & {ood_seq} & {ood_digit} & {carry_acc} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    # Save to file
    os.makedirs("results", exist_ok=True)
    with open("results/addition_results_table_generated.tex", "w") as f:
        f.write(latex)
    print("Addition table saved to results/addition_results_table_generated.tex")

def generate_carry_table():
    methods = get_full_methods_list()
    
    latex = r"""\begin{table}[h]
\centering
\caption{Carry/Trace Accuracy per Operation. Accuracy for intermediate carry or borrow digits predicted by models trained with structural supervision.}
\label{tab:carry_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Add (+)} & \textbf{Sub (-)} & \textbf{Mul (*)} & \textbf{Div (/)} \\
\midrule
"""

    for method in methods:
        name = method["name"]
        
        # Load Metrics
        metrics = load_metrics(f"{method['run_id']}_test")
        if not metrics:
            metrics = load_metrics(method['run_id'])
        if not metrics:
            metrics = load_metrics(f"{method['run_id']}_val")
            
        add_carry = "-"
        sub_carry = "-"
        mul_carry = "-"
        div_carry = "-"
        
        if metrics:
            add_carry = get_metric(metrics, "basicfour/add_carry_accuracy_test", 
                                  get_metric(metrics, "basicfour/add_carry_accuracy", "-"))
            sub_carry = get_metric(metrics, "basicfour/sub_carry_accuracy_test", 
                                  get_metric(metrics, "basicfour/sub_carry_accuracy", "-"))
            mul_carry = get_metric(metrics, "basicfour/mul_carry_accuracy_test", 
                                  get_metric(metrics, "basicfour/mul_carry_accuracy", "-"))
            div_carry = get_metric(metrics, "basicfour/div_carry_accuracy_test", 
                                  get_metric(metrics, "basicfour/div_carry_accuracy", "-"))
            
            # fallback for generic carry_accuracy if all per-op are missing
            if all(v == "-" for v in [add_carry, sub_carry, mul_carry, div_carry]):
                gen_carry = get_metric(metrics, "basicfour/carry_accuracy_test", 
                                      get_metric(metrics, "basicfour/carry_accuracy", "-"))
                if gen_carry != "-":
                     add_carry = gen_carry

        latex += f"{name} & {add_carry} & {sub_carry} & {mul_carry} & {div_carry} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open("results/carry_accuracy_table.tex", "w") as f:
        f.write(latex)
    print("Carry table saved to results/carry_accuracy_table.tex")

def generate_digit_wise_table():
    methods = get_full_methods_list()
    
    # Key digit lengths to show in table
    digit_lengths = [8, 12, 16, 24, 32]
    
    latex = r"""\begin{table}[h]
\centering
\caption{Exact Match Accuracy by Operand Digit Length. ID range is 1--8 digits; OOD range is 9--32 digits.}
\label{tab:digit_wise_results}
\begin{tabular}{l""" + "c" * len(digit_lengths) + r"""}
\toprule
\textbf{Method} & """ + " & ".join([f"\\textbf{{{d} Digits}}" for d in digit_lengths]) + r""" \\
\midrule
"""

    for method in methods:
        name = method["name"]
        metrics = load_metrics(f"{method['run_id']}_test")
        if not metrics:
            metrics = load_metrics(method['run_id'])
            
        row_vals = []
        for d in digit_lengths:
            key = f"accuracy_digit_len_{d}"
            val = "-"
            if metrics:
                for k, v in metrics.items():
                    if key in k:
                        val = f"{v:.4f}"
                        break
            row_vals.append(val)
            
        latex += f"{name} & " + " & ".join(row_vals) + " \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open("results/digit_wise_results_table.tex", "w") as f:
        f.write(latex)
    print("Digit-wise table saved to results/digit_wise_results_table.tex")

def generate_digit_wise_plot():
    methods = [
        {"name": "TRM (NS)", "run_id": "basicfour_vanilla", "color": "red", "marker": "o"},
        {"name": "TRM (OS-After)", "run_id": "basicfour_concat", "color": "blue", "marker": "s"},
        {"name": "TRM (OS-Before)", "run_id": "basicfour_concat_reverse", "color": "green", "marker": "^"},
        {"name": "Transformer (NS)", "run_id": "baseline_transformer_300k", "color": "purple", "marker": "d"},
        {"name": "Transformer (OS-Before)", "run_id": "baseline_transformer_300k_concat_reverse", "color": "olive", "marker": "x", "linestyle": "--"},
        {"name": "Transformer (40x)", "run_id": "baseline_transformer_40x", "color": "orange", "marker": "x"},
        {"name": "TRM (OS-Before-100d)", "run_id": "reverse_100d_run", "color": "darkgreen", "marker": "^", "linestyle": "--"},
    ]
    
    plt.figure(figsize=(12, 7))
    
    for method in methods:
        metrics = load_metrics(f"{method['run_id']}_test")
        if not metrics:
            metrics = load_metrics(method['run_id'])
            
        if not metrics:
            continue
            
        x = []
        y = []
        for d in range(1, 33):
            key = f"accuracy_digit_len_{d}"
            for k, v in metrics.items():
                if key in k:
                    x.append(d)
                    y.append(v)
                    break
        
        if x:
            plt.plot(x, y, label=method["name"], color=method["color"], marker=method["marker"], 
                     markersize=4, linewidth=1.5, linestyle=method.get("linestyle", "-"))

    plt.axvline(x=8.5, color='gray', linestyle='--', alpha=0.5, label='ID/OOD Boundary')
    plt.xlabel('Operand Digit Length')
    plt.ylabel('Exact Match Accuracy')
    plt.title('Arithmetic Performance vs. Problem Length')
    plt.xticks(range(1, 33, 2))
    plt.xlim(0.5, 32.5)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    plot_path = "results/digit_wise_accuracy.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Digit-wise plot saved to {plot_path}")

if __name__ == "__main__":
    generate_latex() # Original addition results
    generate_carry_table()
    generate_digit_wise_table()
    generate_digit_wise_plot()
