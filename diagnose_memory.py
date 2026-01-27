import torch
import os
import sys

def print_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Reserved:  {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Total Capacity: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA not available")

print("--- Initial Memory ---")
print_memory()

from transformers import GemmaConfig, GemmaForCausalLM

config_path = "config/slm/gemma_270m_config.json"
import json
with open(config_path, 'r') as f:
    config_dict = json.load(f)
config = GemmaConfig(**config_dict)

print("\n--- After Config Load ---")
print_memory()

print("\nInitializing model on CPU...")
model = GemmaForCausalLM(config)
print(f"Model params: {model.num_parameters():,}")

print("\n--- After Model Initialized on CPU ---")
print_memory()

if torch.cuda.is_available():
    print("\nMoving model to GPU...")
    try:
        model.to("cuda:0")
        print("Model moved successfully.")
    except Exception as e:
        print(f"Failed to move model: {e}")
    
    print("\n--- After Model Moved to GPU ---")
    print_memory()

    batch_size = 16
    seq_len = 256
    vocab_size = config.vocab_size
    
    print(f"\nAllocating dummy logits: [{batch_size}, {seq_len}, {vocab_size}] (fp32)")
    try:
        logits = torch.randn(batch_size, seq_len, vocab_size, device="cuda:0")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits memory: {logits.element_size() * logits.nelement() / 1024**3:.2f} GB")
    except Exception as e:
        print(f"Failed to allocate logits: {e}")
    
    print("\n--- After Logits Allocation ---")
    print_memory()
