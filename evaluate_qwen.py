
import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb


def evaluate_dataset(model, tokenizer, data_path, args, dataset_name, sample_table=None):
    print(f"Checking data path: {data_path}")
    if os.path.isdir(data_path):
        # Try finding jsonl in the directory
        possible_files = [
            os.path.join(data_path, "vanilla_test.jsonl"),
            os.path.join(data_path, "test.jsonl"),
            os.path.join(data_path, "vanilla_train.jsonl") # Fallback
        ]
        found = False
        for p in possible_files:
            if os.path.exists(p):
                data_path = p
                found = True
                break
        if not found:
            print(f"Error: Could not find suitable jsonl file in {data_path}")
            return {}
            
    if not os.path.exists(data_path):
         print(f"Error: Path does not exist: {data_path}")
         return {}
    
    # data_path is now the file path if found, or likely the original path if we failed (returned early above)
    print(f"Reading from {data_path}")

    with open(data_path, 'r') as f:
        lines = f.readlines()
        
    if args.limit > 0:
        lines = lines[:args.limit]
        
    print(f"Evaluating {dataset_name} on {len(lines)} examples...")
    
    correct = 0
    total = 0
    by_op = {
        '+': {'correct': 0, 'total': 0, 'digit_correct': 0, 'digit_total': 0}, 
        '-': {'correct': 0, 'total': 0, 'digit_correct': 0, 'digit_total': 0}, 
        '*': {'correct': 0, 'total': 0, 'digit_correct': 0, 'digit_total': 0}, 
        '/': {'correct': 0, 'total': 0, 'digit_correct': 0, 'digit_total': 0}
    }
    
    for line in tqdm(lines):
        item = json.loads(line)
        prompt = item['text'].split('=')[0] + "="
        target = item['text'].split('=')[1]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_tokens,
                do_sample=False, # Greedy
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Simple extraction
        prediction = generated_text.split()[0] if generated_text else ""
        
        # Compare
        is_correct = (prediction == target)
        
        total += 1
        if is_correct:
            correct += 1
            
        # ID vs OOD Split (Threshold 8 digits)
        # Parse operands from item 'A' and 'B' if available, or infer from prompt?
        # item is the source json object
        len_a = len(str(item.get('A', '')))
        len_b = len(str(item.get('B', '')))
        max_digits = max(len_a, len_b)
        
        is_id = max_digits <= 8
        
        if is_id:
            by_op[op]['id_total'] = by_op[op].get('id_total', 0) + 1
            if is_correct:
                by_op[op]['id_correct'] = by_op[op].get('id_correct', 0) + 1
        else:
            by_op[op]['ood_total'] = by_op[op].get('ood_total', 0) + 1
            if is_correct:
                by_op[op]['ood_correct'] = by_op[op].get('ood_correct', 0) + 1

        # Digit-level accuracy (simple character match from right to left)
        t_rev = target[::-1]
        p_rev = prediction[::-1]
        match_count = 0
        max_len = max(len(t_rev), len(p_rev))
        if max_len > 0:
            for i in range(min(len(t_rev), len(p_rev))):
                if t_rev[i] == p_rev[i]:
                    match_count += 1
        
        # Track digit stats for ID/OOD
        if is_id:
             by_op[op]['id_digit_correct'] = by_op[op].get('id_digit_correct', 0) + match_count
             by_op[op]['id_digit_total'] = by_op[op].get('id_digit_total', 0) + max_len
        else:
             by_op[op]['ood_digit_correct'] = by_op[op].get('ood_digit_correct', 0) + match_count
             by_op[op]['ood_digit_total'] = by_op[op].get('ood_digit_total', 0) + max_len

        op = None
        if '+' in prompt: op = '+'
        elif '-' in prompt: op = '-'
        elif '*' in prompt: op = '*'
        elif '/' in prompt: op = '/'
        
        if op and op in by_op:
            by_op[op]['total'] += 1
            by_op[op]['digit_correct'] += match_count
            by_op[op]['digit_total'] += max_len
            if is_correct:
                by_op[op]['correct'] += 1
                
        if args.verbose and total <= 5:
            print(f"[{dataset_name}] Prompt: {prompt} | Target: {target} | Pred: {prediction} [{'CORRECT' if is_correct else 'WRONG'}]")
        
        if sample_table is not None and sample_table.data and len(sample_table.data) < 1000: 
             sample_table.add_data(dataset_name, prompt, target, generated_text, prediction, is_correct, op)

    # Compute Metrics
    metrics = {}
    overall_acc = correct/total if total > 0 else 0.0
    metrics[f"{dataset_name}/accuracy"] = overall_acc
    
    # ID/OOD Aggregation
    total_id_correct = sum(stats.get('id_correct', 0) for stats in by_op.values())
    total_id_total = sum(stats.get('id_total', 0) for stats in by_op.values())
    
    total_ood_correct = sum(stats.get('ood_correct', 0) for stats in by_op.values())
    total_ood_total = sum(stats.get('ood_total', 0) for stats in by_op.values())
    
    total_id_digit_correct = sum(stats.get('id_digit_correct', 0) for stats in by_op.values())
    total_id_digit_total = sum(stats.get('id_digit_total', 0) for stats in by_op.values())
    
    total_ood_digit_correct = sum(stats.get('ood_digit_correct', 0) for stats in by_op.values())
    total_ood_digit_total = sum(stats.get('ood_digit_total', 0) for stats in by_op.values())
    
    id_acc = total_id_correct / total_id_total if total_id_total > 0 else 0.0
    ood_acc = total_ood_correct / total_ood_total if total_ood_total > 0 else 0.0
    
    id_digit_acc = total_id_digit_correct / total_id_digit_total if total_id_digit_total > 0 else 0.0
    ood_digit_acc = total_ood_digit_correct / total_ood_digit_total if total_ood_digit_total > 0 else 0.0
    
    metrics[f"{dataset_name}/id_accuracy"] = id_acc
    metrics[f"{dataset_name}/ood_accuracy"] = ood_acc
    metrics[f"{dataset_name}/id_digit_accuracy"] = id_digit_acc
    metrics[f"{dataset_name}/ood_digit_accuracy"] = ood_digit_acc

    # Overall Digit Accuracy
    total_digit_correct = sum(stats['digit_correct'] for stats in by_op.values())
    total_digit_total = sum(stats['digit_total'] for stats in by_op.values())
    overall_digit_acc = total_digit_correct / total_digit_total if total_digit_total > 0 else 0.0
    metrics[f"{dataset_name}/digit_accuracy"] = overall_digit_acc
    
    print(f"\nResults for {dataset_name}:")
    print(f"Overall Accuracy: {correct}/{total} = {overall_acc:.4f}")
    print(f"ID Accuracy: {total_id_correct}/{total_id_total} = {id_acc:.4f}")
    print(f"OOD Accuracy: {total_ood_correct}/{total_ood_total} = {ood_acc:.4f}")

    for op, stats in by_op.items():
        if stats['total'] > 0:
            op_acc = stats['correct']/stats['total']
            
            # Op splits
            op_id_acc = stats.get('id_correct', 0)/stats.get('id_total', 1) if stats.get('id_total', 0) > 0 else 0.0
            op_ood_acc = stats.get('ood_correct', 0)/stats.get('ood_total', 1) if stats.get('ood_total', 0) > 0 else 0.0
            
            print(f"Op {op}: Seq={stats['correct']}/{stats['total']} ({op_acc:.4f}) | ID={op_id_acc:.4f} | OOD={op_ood_acc:.4f}")
            
            op_name = "unknown"
            if op == '+': op_name = "add"
            elif op == '-': op_name = "sub"
            elif op == '*': op_name = "mul"
            elif op == '/': op_name = "div"
            
            metrics[f"{dataset_name}/{op_name}_accuracy"] = op_acc
            metrics[f"{dataset_name}/{op_name}_id_accuracy"] = op_id_acc
            metrics[f"{dataset_name}/{op_name}_ood_accuracy"] = op_ood_acc
            metrics[f"{dataset_name}/{op_name}_count"] = stats['total']
            
    # Save to JSON
    os.makedirs("results", exist_ok=True)
    run_name = args.wandb_run_name if args.wandb_run_name else "qwen_eval"
    # To handle multiple datasets in one run, we might overwrite or append.
    # Ideally evaluators return metrics and the main loop saves them.
    # But this function returns metrics. We can save in main loop.
            
    return metrics

def evaluate_qwen(args):
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    
    # Initialize WandB
    sample_table = None
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args)
        )
        sample_table = wandb.Table(columns=["dataset", "prompt", "target", "generated", "prediction", "is_correct", "op"])
    
    all_metrics = {}
    
    for data_dir in args.data_dirs:
        # Determine dataset name
        clean_path = data_dir.rstrip('/')
        dataset_name = os.path.basename(clean_path)
        if not dataset_name: 
             dataset_name = "dataset"
             
        # Evaluate
        metrics = evaluate_dataset(model, tokenizer, data_dir, args, dataset_name, sample_table)
        all_metrics.update(metrics)
        
    # Log all metrics to WandB
    if args.wandb_project:
        wandb.log(all_metrics)
        wandb.log({"samples": sample_table})
        wandb.finish()
        
    # Save to local JSON
    os.makedirs("results", exist_ok=True)
    run_name = args.wandb_run_name if args.wandb_run_name else "qwen_eval"
    # slugify
    run_name = run_name.replace("/", "_").replace(" ", "_")
    output_file = os.path.join("results", f"{run_name}.json")
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--data_dirs", type=str, nargs='+', required=True, help="List of data directories")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="trm-icml-eval")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    # Backwards compatibility: if user provides --data_dir instead (not defined here but could be passed legacy)
    # ArgumentParser will fail if we remove it completely, but let's assume user updates command or we handle unknown args if we cared.
    # For now, strict transition to --data_dirs.
    
    args = parser.parse_args()
    
    evaluate_qwen(args)
