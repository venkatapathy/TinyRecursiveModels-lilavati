
import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_qwen(args):
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    
    # Load dataset
    data_path = os.path.join(args.data_dir, "vanilla_test.jsonl")
    if not os.path.exists(data_path):
        # Fallback to train if test not found (for verifying on small generated sets)
        data_path = os.path.join(args.data_dir, "vanilla_train.jsonl")
    
    print(f"Loading data from {data_path}")
    with open(data_path, 'r') as f:
        lines = f.readlines()
        
    if args.limit > 0:
        lines = lines[:args.limit]
        
    print(f"Evaluating on {len(lines)} examples...")
    
    correct = 0
    total = 0
    by_op = {'+': {'correct': 0, 'total': 0}, '-': {'correct': 0, 'total': 0}, 
             '*': {'correct': 0, 'total': 0}, '/': {'correct': 0, 'total': 0}}
    
    for line in tqdm(lines):
        item = json.loads(line)
        prompt = item['text'].split('=')[0] + "="
        target = item['text'].split('=')[1]
        
        # Prepare input
        # Note: Qwen-Math might expect specific chat template or just few-shot?
        # For base arithmetic, raw completion might work.
        # "Qwen2.5-Math-Instruct" works well with chat template.
        # "Qwen2.5-Math" (Base) works with completion.
        # We will use simple completion formatting matching the training data if possible,
        # but the training data was "A+B=Y".
        
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
        
        # Simple extraction: take first line or space-separated token?
        # The model might output "123\n" or "123"
        prediction = generated_text.split()[0] if generated_text else ""
        
        # Compare
        is_correct = (prediction == target)
        
        total += 1
        if is_correct:
            correct += 1
            
        # Op detection
        op = None
        if '+' in prompt: op = '+'
        elif '-' in prompt: op = '-'
        elif '*' in prompt: op = '*'
        elif '/' in prompt: op = '/'
        
        if op:
            by_op[op]['total'] += 1
            if is_correct:
                by_op[op]['correct'] += 1
                
        if args.verbose and total <= 5:
            print(f"Prompt: {prompt}")
            print(f"Target: {target}")
            print(f"Gen: {generated_text}")
            print(f"Pred: {prediction} [{'CORRECT' if is_correct else 'WRONG'}]")
            print("-" * 20)

    print("\nResults:")
    print(f"Overall Accuracy: {correct}/{total} = {correct/total:.4f}")
    for op, stats in by_op.items():
        if stats['total'] > 0:
            print(f"Op {op}: {stats['correct']}/{stats['total']} = {stats['correct']/stats['total']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    evaluate_qwen(args)
