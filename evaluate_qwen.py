
import argparse
import json
import os
import torch
import numpy as np
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb


def calculate_metrics(dataset_name, correct, total, by_op):
    metrics = {}
    if total == 0:
        return metrics
        
    overall_acc = correct / total
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

    for op, stats in by_op.items():
        if stats['total'] > 0:
            op_acc = stats['correct'] / stats['total']
            op_id_acc = stats.get('id_correct', 0) / stats.get('id_total', 1) if stats.get('id_total', 0) > 0 else 0.0
            op_ood_acc = stats.get('ood_correct', 0) / stats.get('ood_total', 1) if stats.get('ood_total', 0) > 0 else 0.0

            op_name_map = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}
            op_name = op_name_map.get(op, "unknown")

            metrics[f"{dataset_name}/{op_name}_accuracy"] = op_acc
            metrics[f"{dataset_name}/{op_name}_id_accuracy"] = op_id_acc
            metrics[f"{dataset_name}/{op_name}_ood_accuracy"] = op_ood_acc
            metrics[f"{dataset_name}/{op_name}_count"] = stats['total']
            
    # Add count for progress tracking
    metrics[f"{dataset_name}/processed_samples"] = total
    
    return metrics


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
    
    # Detect if we should use Instruct format
    is_instruct = "Instruct" in args.model
    
    batch_size = args.batch_size
    for i in tqdm(range(0, len(lines), batch_size)):
        batch_lines = lines[i:i+batch_size]
        batch_items = [json.loads(line) for line in batch_lines]
        batch_targets = [item['text'].split('=')[1] for item in batch_items]
        
        batch_prompts = []
        for item in batch_items:
            # Extract equation
            eq = item['text'].split('=')[0]
            if is_instruct:
                messages = [
                    {"role": "user", "content": f"Please calculate the result of the following arithmetic operation: {eq}. Provide only the final numerical answer inside \\boxed{{}}."}
                ]
                # We'll apply template later because it might need different processing per item if we had history, 
                # but here it's fine.
                batch_prompts.append(messages)
            else:
                batch_prompts.append(eq + "=")
        
        if is_instruct:
            # Tokenize using chat template
            tokenized_batches = []
            for msgs in batch_prompts:
                ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
                tokenized_batches.append(ids[0])
            
            # Pad manually since they might have different lengths
            max_len = max(len(b) for b in tokenized_batches)
            padded_ids = torch.full((len(tokenized_batches), max_len), tokenizer.pad_token_id, dtype=torch.long)
            for k, b in enumerate(tokenized_batches):
                # Padding side depends on what was set in main. Usually left for batch gen.
                if tokenizer.padding_side == "left":
                    padded_ids[k, -len(b):] = b
                else:
                    padded_ids[k, :len(b)] = b
            
            inputs = {"input_ids": padded_ids.to(model.device)}
        else:
            inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_tokens,
                do_sample=False, # Greedy
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for j, (generated_text, target, item) in enumerate(zip(generated_texts, batch_targets, batch_items)):
            generated_text = generated_text.strip()
            
            if is_instruct:
                # Robust extraction: look for \boxed{} first
                # Handle commas in boxed answer
                match = re.search(r'\\boxed\{([-?\d,.]+)\}', generated_text)
                if match:
                    prediction = match.group(1).replace(',', '').strip()
                else:
                    # Fallback: take the last number in the text (removing commas first)
                    clean_text = generated_text.replace(',', '')
                    numbers = re.findall(r'-?\d+', clean_text)
                    if numbers:
                        prediction = numbers[-1]
                    else:
                        prediction = generated_text.split()[0] if generated_text else ""
            else:
                # Simple extraction for non-instruct
                prediction = generated_text.split()[0] if generated_text else ""
            
            # Compare
            is_correct = (prediction == target)
            
            total += 1
            if is_correct:
                correct += 1
                
            # Determine operation
            text_part = item.get('text', '').split('=')[0]
            op = None
            if '+' in text_part: op = '+'
            elif '-' in text_part: op = '-'
            elif '*' in text_part: op = '*'
            elif '/' in text_part: op = '/'

            # Parse operands from item 'A' and 'B' if available, or infer from prompt?
            # item is the source json object
            if 'A' in item and 'B' in item:
                len_a = len(str(item['A']))
                len_b = len(str(item['B']))
            else:
                # Parse from text "A+B=Y"
                text_part = item.get('text', '').split('=')[0]
                # Find the op
                eq_op = None
                for op_char in ['+', '-', '*', '/']:
                    if op_char in text_part:
                        eq_op = op_char
                        break
                
                if eq_op:
                    parts = text_part.split(eq_op)
                    len_a = len(parts[0].strip())
                    len_b = len(parts[1].strip())
                else:
                    len_a = 0
                    len_b = 0

            max_digits = max(len_a, len_b)
            
            is_id = max_digits <= 8
            
            if is_id:
                if op and op in by_op:
                    by_op[op]['id_total'] = by_op[op].get('id_total', 0) + 1
                    if is_correct:
                        by_op[op]['id_correct'] = by_op[op].get('id_correct', 0) + 1
            else:
                if op and op in by_op:
                    by_op[op]['ood_total'] = by_op[op].get('ood_total', 0) + 1
                    if is_correct:
                        by_op[op]['ood_correct'] = by_op[op].get('ood_correct', 0) + 1

            # Digit-level accuracy (simple character match from right to left)
            t_rev = target[::-1]
            p_rev = prediction[::-1]
            match_count = 0
            max_len = max(len(t_rev), len(p_rev))
            if max_len > 0:
                for idx in range(min(len(t_rev), len(p_rev))):
                    if t_rev[idx] == p_rev[idx]:
                        match_count += 1
            
            # Track digit stats for ID/OOD
            if is_id:
                 if op and op in by_op:
                     by_op[op]['id_digit_correct'] = by_op[op].get('id_digit_correct', 0) + match_count
                     by_op[op]['id_digit_total'] = by_op[op].get('id_digit_total', 0) + max_len
            else:
                 if op and op in by_op:
                     by_op[op]['ood_digit_correct'] = by_op[op].get('ood_digit_correct', 0) + match_count
                     by_op[op]['ood_digit_total'] = by_op[op].get('ood_digit_total', 0) + max_len
            
            if op and op in by_op:
                by_op[op]['total'] += 1
                by_op[op]['digit_correct'] += match_count
                by_op[op]['digit_total'] += max_len
                if is_correct:
                    by_op[op]['correct'] += 1
                    
            if args.verbose and total <= 5:
                # Show prompt content if list
                prompt_display = batch_prompts[j] if not is_instruct else batch_prompts[j][-1]['content']
                print(f"[{dataset_name}] Prompt: {prompt_display} | Target: {target} | Pred: {prediction} [{'CORRECT' if is_correct else 'WRONG'}]")
                if not is_correct:
                    gen_text_clean = generated_text.replace('\n', ' ')
                    print(f"    Full Response: {gen_text_clean}")
            
            if sample_table is not None and sample_table.data and len(sample_table.data) < 1000: 
                 prompt_display = batch_prompts[j] if not is_instruct else batch_prompts[j][-1]['content']
                 sample_table.add_data(dataset_name, prompt_display, target, generated_text, prediction, is_correct, op)

        # Incremental logging to WandB
        if wandb.run is not None:
            running_metrics = calculate_metrics(f"{dataset_name}_running", correct, total, by_op)
            wandb.log(running_metrics)
            
        # Periodic local result saving (every 100 samples or at the end of dataset)
        if total % 100 == 0 or i + batch_size >= len(lines):
            periodic_metrics = calculate_metrics(f"{dataset_name}_running", correct, total, by_op)
            save_json_results(args, periodic_metrics)

    # Compute Final Metrics
    metrics = calculate_metrics(dataset_name, correct, total, by_op)
    
    print(f"\nResults for {dataset_name}:")
    print(f"Overall Accuracy: {correct}/{total} = {metrics[f'{dataset_name}/accuracy']:.4f}")
    print(f"ID Accuracy: {metrics[f'{dataset_name}/id_accuracy']:.4f}")
    print(f"OOD Accuracy: {metrics[f'{dataset_name}/ood_accuracy']:.4f}")

    for op_char in ['+', '-', '*', '/']:
        op_name_map = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}
        op_name = op_name_map[op_char]
        acc_key = f"{dataset_name}/{op_name}_accuracy"
        if acc_key in metrics:
            id_key = f"{dataset_name}/{op_name}_id_accuracy"
            ood_key = f"{dataset_name}/{op_name}_ood_accuracy"
            print(f"Op {op_char}: Accuracy={metrics[acc_key]:.4f} | ID={metrics.get(id_key, 0.0):.4f} | OOD={metrics.get(ood_key, 0.0):.4f}")
            
    # Save to JSON
    os.makedirs("results", exist_ok=True)
    run_name = args.wandb_run_name if args.wandb_run_name else "qwen_eval"
    # To handle multiple datasets in one run, we might overwrite or append.
    # Ideally evaluators return metrics and the main loop saves them.
    # But this function returns metrics. We can save in main loop.
            
    return metrics

def save_json_results(args, metrics):
    os.makedirs("results", exist_ok=True)
    run_name = args.wandb_run_name if args.wandb_run_name else "qwen_eval"
    run_name = run_name.replace("/", "_").replace(" ", "_")
    output_file = os.path.join("results", f"{run_name}.json")
    
    # Check if file exists to merge
    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                existing_data = json.load(f)
        except:
            pass
            
    existing_data.update(metrics)
    with open(output_file, "w") as f:
        json.dump(existing_data, f, indent=4)

def evaluate_qwen(args):
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # For batched generation
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
    save_json_results(args, all_metrics)
    print(f"Final results saved to results/{args.wandb_run_name if args.wandb_run_name else 'qwen_eval'}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--data_dirs", type=str, nargs='+', required=True, help="List of data directories")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="trm-icml-eval")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    
    # Backwards compatibility: if user provides --data_dir instead (not defined here but could be passed legacy)
    # ArgumentParser will fail if we remove it completely, but let's assume user updates command or we handle unknown args if we cared.
    # For now, strict transition to --data_dirs.
    
    args = parser.parse_args()
    
    evaluate_qwen(args)
