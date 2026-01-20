
import torch
import hydra
from omegaconf import OmegaConf
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config

def verify_model():
    # Helper to create a dummy config
    config_dict = {
        "batch_size": 2,
        "seq_len": 32,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": 100,
        "H_cycles": 1,
        "L_cycles": 1,
        "H_layers": 1,
        "L_layers": 1,
        "hidden_size": 32,
        "expansion": 2,
        "num_heads": 4,
        "pos_encodings": "learned",
        "halt_max_steps": 5,
        "halt_exploration_prob": 0.1,
        "dataset_mode": "dual_head",
        "digits": 3,
        "carry_loss_weight": 1.0,
        "puzzle_emb_len": 0,
        "mlp_t": False,
        "no_ACT_continue": True
    }
    
    # Initialize model
    print("Initializing model...")
    model = TinyRecursiveReasoningModel_ACTV1(config_dict)
    
    # Create dummy batch
    batch_size = config_dict['batch_size']
    seq_len = config_dict['seq_len']
    vocab_size = config_dict['vocab_size']
    
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    puzzle_identifiers = torch.zeros((batch_size,), dtype=torch.long)
    
    batch = {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_identifiers,
        # Add other potential keys if strictly needed, but model usually just needs inputs for basic forward
    }
    
    # Initial carry
    print("Creating initial carry...")
    carry = model.initial_carry(batch)
    
    # Forward pass
    print("Running forward pass...")
    try:
        new_carry, outputs = model(carry, batch)
        print("Forward pass successful!")
        
        # Check shapes
        if "carry_logits" in outputs:
            print("carry_logits shape:", outputs["carry_logits"].shape)
            # Expected shape: [batch, seql_len, vocab_size] (after puzzle_emb_len slicing)
            # Since puzzle_emb_len is 0 here:
            ckpt_shape = (batch_size, seq_len, vocab_size)
            if outputs["carry_logits"].shape == ckpt_shape:
                print("Shape check PASSED.")
            else:
                print(f"Shape check FAILED. Expected {ckpt_shape}, got {outputs['carry_logits'].shape}")
        else:
            print("carry_logits NOT found in outputs!")
            
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        raise e

if __name__ == "__main__":
    verify_model()
