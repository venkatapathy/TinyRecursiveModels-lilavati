import os
import json
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from dataset.common import PuzzleDatasetMetadata

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/addition"
    seed: int = 42
    test_ratio: float = 0.1

@cli.command(singleton=True)
def main(config: DataProcessConfig):
    np.random.seed(config.seed)
    
    # Generate all pairs
    pairs = []
    for i in range(100):
        for j in range(100):
            pairs.append((i, j))
    
    np.random.shuffle(pairs)
    
    split_idx = int(len(pairs) * (1 - config.test_ratio))
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    splits = {"train": train_pairs, "test": test_pairs}
    
    # Vocabulary
    # 0: PAD
    # 1: MASK (for unknown answer digits)
    # 2: '0'
    # ...
    # 11: '9'
    # 12: '+'
    # 13: '='
    
    vocab_map = {str(i): i + 2 for i in range(10)}
    vocab_map['+'] = 12
    vocab_map['='] = 13
    
    MASK_ID = 1
    IGNORE_LABEL_ID = 255
    
    def encode(s):
        return [vocab_map[c] for c in s]
        
    os.makedirs(config.output_dir, exist_ok=True)
    
    for split_name, current_pairs in splits.items():
        inputs = []
        labels = []
        puzzle_indices = [0]
        group_indices = [0]
        puzzle_identifiers = []
        
        # Format: XX+YY=ZZZ
        # Input: XX+YY=MMM (M=Mask)
        # Label: IIIIIIZZZ (I=Ignore)
        
        for idx, (a, b) in enumerate(current_pairs):
            res = a + b
            s_a = f"{a:02d}"
            s_b = f"{b:02d}"
            s_res = f"{res:03d}"
            
            prefix = s_a + "+" + s_b + "="
            
            # Input: Prefix + MASKs
            inp_seq = encode(prefix) + [MASK_ID, MASK_ID, MASK_ID]
            
            # Label: Ignore Prefix + Result
            label_seq = [IGNORE_LABEL_ID] * len(encode(prefix)) + encode(s_res)
            
            inputs.append(inp_seq)
            labels.append(label_seq)
            
            puzzle_identifiers.append(0) # 0 is blank identifier
            puzzle_indices.append(idx + 1)
            group_indices.append(idx + 1)
            
        # Convert to numpy
        inputs = np.array(inputs, dtype=np.uint8)
        labels = np.array(labels, dtype=np.uint8)
        puzzle_indices = np.array(puzzle_indices, dtype=np.int32)
        group_indices = np.array(group_indices, dtype=np.int32)
        puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
        
        # Save
        save_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(os.path.join(save_dir, "all__inputs.npy"), inputs)
        np.save(os.path.join(save_dir, "all__labels.npy"), labels)
        np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(save_dir, "all__group_indices.npy"), group_indices)
        np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
        
        metadata = PuzzleDatasetMetadata(
            seq_len=9,
            vocab_size=14, # 0..13
            pad_id=0,
            ignore_label_id=IGNORE_LABEL_ID,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=len(group_indices)-1,
            mean_puzzle_examples=1.0,
            total_puzzles=len(puzzle_indices)-1,
            sets=["all"]
        )
        
        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
            
    # Identifiers
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

if __name__ == "__main__":
    cli()

