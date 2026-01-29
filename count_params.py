import torch
from models.baselines.transformer import StandardTransformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

trm_config = {
    "batch_size": 256,
    "seq_len": 256,
    "vocab_size": 20,
    "num_puzzle_identifiers": 0,
    "hidden_size": 128,
    "expansion": 2.666,
    "num_heads": 4,
    "num_layers": 8,
    "pos_encodings": "rope",
    "dataset_mode": "vanilla"
}

print("Searching for L=8 Transformer configs:")
for d in [32, 40, 48, 56, 64]:
    # Ensure d is multiple of num_heads (e.g. 4 or 8)
    for h in [4, 8]:
        if d % h == 0:
            cfg = trm_config.copy()
            cfg["hidden_size"] = d
            cfg["num_heads"] = h
            try:
                m = StandardTransformer(cfg)
                params = count_parameters(m)
                print(f"Trans D={d:2}, L=8, heads={h}: {params:,} params")
            except:
                pass
