
import os
import json
import argparse
import numpy as np
from tqdm import tqdm

# vocabulary similar to build_basicfour_dataset.py but for our needs
# 0: PAD
# 1: MASK (not strictly used in causal LM but good to have)
# 2-11: 0-9
# 12: +
# 13: -
# 14: *
# 15: /
# 16: =
# 17: R (Reserved)

VOCAB_MAP = {
    '0': 2, '1': 3, '2': 4, '3': 5, '4': 6,
    '5': 7, '6': 8, '7': 9, '8': 10, '9': 11,
    '+': 12, '-': 13, '*': 14, '/': 15, '=': 16,
    'R': 17
}
PAD_ID = 0
IGNORE_LABEL_ID = -100

def encode_str(s):
    return [VOCAB_MAP[c] for c in s]

def process_split(input_file, output_dir, split_name, args):
    if not os.path.exists(input_file):
        print(f"Skipping {split_name}, file not found: {input_file}")
        return

    print(f"Processing {split_name} from {input_file}...")
    
    inputs_list = []
    labels_lm_list = []
    labels_aux_list = []
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        
        # Parse fields
        # text: "8+54=62"
        # A: "8", B: "54", op: "+", Y: "62"
        A = item["A"]
        B = item["B"]
        op = item["op"]
        Y = item["Y"]
        
        # Input Prompt: "A op B ="
        prompt_str = f"{A}{op}{B}="
        prompt_ids = encode_str(prompt_str)
        
        # Result: "Y"
        result_ids = encode_str(Y)
        
        # Full Input: Prompt + Result
        # We generally train on the full sequence
        # input:  [Prompt ids ..., Result ids ...]
        input_ids = prompt_ids + result_ids
        
        # LM Labels: Will be constructed by shifting input_ids
        
        # Aux Labels: IGNORE on Prompt, Carry ids on Result
        # Get intermediate list
        inter_list = []
        if op == '+':
            inter_list = item["labels"]["add_carry"]
        elif op == '-':
            inter_list = item["labels"]["sub_borrow"]
        elif op == '*':
            inter_list = item["labels"]["mul_carry_cols"]
        elif op == '/':
            # Division remainders are large integers (e.g. 97), exceeding vocab size (20).
            # We cannot train aux head on them as tokens.
            # Leave inter_list empty -> will be padded with 0 -> but we want IGNORE.
            # Better strategy: handle below.
            inter_list = []
            
        if len(inter_list) > 0:
            # Safety Check: Mask any value >= 20 (vocab size)
            # This handles large division remainders and rare multiplication overflow
            safe_list = []
            for x in inter_list:
                if x >= 20: 
                    safe_list.append(IGNORE_LABEL_ID)
                else:
                    safe_list.append(x)
            inter_list = safe_list
            
        if len(inter_list) == 0:
             # Case for Division or empty: Fill with IGNORE
             aux_labels = [IGNORE_LABEL_ID] * len(input_ids)
        else:
             # Alignment:
             # The Aux output must map 1-to-1 with the Result output tokens.
             # len(inter_list) must equal len(result_ids).
             # Strategy: Take the LAST `len(result_ids)` elements.
        
             target_len = len(result_ids)
             if len(inter_list) >= target_len:
                 aligned_inter = inter_list[-target_len:]
             else:
                 # Pad with 0 at the front if short (unlikely for strict math but safe)
                 # For arithmetic, carries usually match or exceed. 
                 # If inputs are short, this keeps lengths equal.
                 diff = target_len - len(inter_list)
                 aligned_inter = [0]*diff + inter_list
            
             aux_labels = [IGNORE_LABEL_ID] * len(prompt_ids) + aligned_inter
        
        # CAUSAL LM SHIFTING
        # Input[t] should predict Label[t] = Input[t+1].
        # Current logic has Label[t] = Input[t] (Copy).
        # We must shift labels (and aux_labels) left by 1.
        
        # 1. Shift LM Labels
        # Original: [Prompt, Result]
        # Shifted:  [Prompt[1:], Result, IGNORE]
        # Masking:  We want to predict Result.
        #           The first token of Result is predicted by the last token of Prompt (=).
        #           So we keep labels starting from index len(prompt_ids) - 1.
        
        # Construct full sequence first
        full_ids = prompt_ids + result_ids
        
        # Shift inputs/labels
        # inputs: [A, B, C, D]
        # labels: [B, C, D, IGN]
        # We keep inputs as is. We modify labels.
        
        shifted_lm_labels = full_ids[1:] + [PAD_ID]
        
        # Masking Prompt
        # We want to ignore predictions for Prompt tokens, EXCEPT the last one (=) which predicts Result[0].
        # prompt_ids length is L.
        # indices 0..L-1 are prompt.
        # index L-1 is '='.
        # shifted_lm_labels[L-1] is Result[0]. This is GOOD.
        # indices 0..L-2 should be IGNORE.
        for i in range(len(prompt_ids) - 1):
            shifted_lm_labels[i] = IGNORE_LABEL_ID
            
        lm_labels = shifted_lm_labels
        
        # 2. Shift Aux Labels
        # Original: [IGN...IGN, Inter0, Inter1...]
        # Shifted:  [IGN...IGN (one less), Inter0, Inter1..., IGN]
        # The first Inter label (Inter0) should be at index L-1 (=).
        
        shifted_aux_labels = aux_labels[1:] + [PAD_ID]
        # Optimization: The shifting naturally moves Inter0 to L-1 because aux_labels had L IGNORES.
        # aux_labels was [IGN]*L + Inter.
        # shifted is [IGN]*(L-1) + Inter + [IGN].
        # So index L-1 is Inter[0]. Correct.
        
        aux_labels = shifted_aux_labels
        
        assert len(input_ids) == len(lm_labels) == len(aux_labels)
        
        inputs_list.append(input_ids)
        labels_lm_list.append(lm_labels)
        labels_aux_list.append(aux_labels)
        
    # Padding all to max length in this split
    # FORCE: Use fixed length 64 to avoid mismatch between Train (32) and Val (40)
    # The model expects a fixed seq_len determined at init.
    max_len = args.max_len
    print(f"  Forced sequence length: {max_len}")
    
    padded_inputs = []
    padded_lm = []
    padded_aux = []
    
    # Structural arrays
    puzzle_indices = [0]
    group_indices = [0]
    puzzle_identifiers = []

    for idx, (inp, lm, aux) in enumerate(zip(inputs_list, labels_lm_list, labels_aux_list)):
        pad_len = max_len - len(inp)
        
        # Pad inputs with PAD_ID
        padded_inputs.append(inp + [PAD_ID] * pad_len)
        # Pad labels with IGNORE_LABEL_ID
        padded_lm.append(lm + [IGNORE_LABEL_ID] * pad_len)
        padded_aux.append(aux + [IGNORE_LABEL_ID] * pad_len)
        
        # Structure: 1 example = 1 puzzle = 1 group
        puzzle_indices.append(idx + 1)
        group_indices.append(idx + 1)
        puzzle_identifiers.append(0) # Default ID

    # Save
    save_dir = os.path.join(output_dir, split_name)
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "all__inputs.npy"), np.array(padded_inputs, dtype=np.uint8))
    np.save(os.path.join(save_dir, "all__labels.npy"), np.array(padded_lm, dtype=np.int64)) # main labels
    np.save(os.path.join(save_dir, "all__labels_aux.npy"), np.array(padded_aux, dtype=np.int64)) # aux labels
    
    np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), np.array(puzzle_indices, dtype=np.int32))
    np.save(os.path.join(save_dir, "all__group_indices.npy"), np.array(group_indices, dtype=np.int32))
    np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), np.array(puzzle_identifiers, dtype=np.int32))
    
    # Metadata
    metadata = {
        "seq_len": max_len,
        "vocab_size": 20, # safe upper bound
        "pad_id": PAD_ID,
        "ignore_label_id": IGNORE_LABEL_ID,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": len(inputs_list),
        "mean_puzzle_examples": 1.0,
        "total_puzzles": len(inputs_list),
        "sets": ["all"],
        "total_examples": len(inputs_list)
    }
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"  Saved {len(inputs_list)} examples to {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing structured_*.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=64, help="Maximum sequence length")
    args = parser.parse_args()
    
    for split in ["train", "val", "test"]:
        jsonl_path = os.path.join(args.input_dir, f"structured_{split}.jsonl")
        process_split(jsonl_path, args.output_dir, split, args)

if __name__ == "__main__":
    main()
