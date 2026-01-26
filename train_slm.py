
import os
import sys
import argparse
import json
import torch
import wandb
from datasets import Dataset
from transformers import (
    GemmaConfig,
    GemmaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import numpy as np

def load_arithmetic_dataset(data_dir, split, limit=None):
    """
    Loads data from jsonl files in `data_dir` matching the split name.
    Expects format: {"text": "..."} or similar.
    """
    # Try vanilla_{split}.jsonl
    path = os.path.join(data_dir, f"vanilla_{split}.jsonl")
    if not os.path.exists(path):
        # try just {split}.jsonl
        path = os.path.join(data_dir, f"{split}.jsonl")
        
    if not os.path.exists(path):
        raise ValueError(f"Could not find dataset for split {split} in {data_dir}")

    print(f"Loading {split} from {path}")
    data = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            data.append({"text": item["text"]})
    return Dataset.from_list(data)

def compute_metrics(eval_pred):
    # Optional: Implement custom metrics if needed, but HF Trainer logs loss/perplexity by default.
    # For accuracy, we rely on the separate evaluate_qwen.py script.
    return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/slm/gemma_270m_config.json")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/slm/gemma_270m_run")
    parser.add_argument("--run_name", type=str, default="gemma_270m_arithmetic")
    parser.add_argument("--tokenizer_name", type=str, default="alpindale/gemma-2b") 
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="trm-arthmetic-icml")
    
    args = parser.parse_args()
    
    if not args.debug:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # 1. Config
    print(f"Loading config from {args.config_path}")
    with open(args.config_path, 'r') as f:
        config_dict = json.load(f)
    config = GemmaConfig(**config_dict)
    
    # 2. Tokenizer
    print(f"Loading tokenizer: {args.tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    except Exception as e:
        print("Failed to load tokenizer.")
        raise e
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Model
    print("Initializing model from scratch...")
    model = GemmaForCausalLM(config)
    print(f"Model params: {model.num_parameters():,}")

    # 4. Data
    train_dataset = load_arithmetic_dataset(args.data_dir, "train", limit=100 if args.debug else None)
    val_dataset = load_arithmetic_dataset(args.data_dir, "val", limit=20 if args.debug else None)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.max_seq_len)

    print("Tokenizing...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 5. Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200 if not args.debug else 5,
        save_strategy="steps",
        save_steps=500 if not args.debug else 5,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="wandb" if not args.debug else "none",
        run_name=args.run_name,
        remove_unused_columns=False,
    )
    
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=collator,
        compute_metrics=compute_metrics 
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    main()
