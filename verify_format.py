
import sys
import os
import numpy as np

# Mocking things to import build_arithmetic_dataset functions without running main
# Or I can just import it if main is guarded. Yes it is.
sys.path.append(os.getcwd())
import dataset.build_arithmetic_dataset as build_ds

# Constants
VOCAB_MAP = {
    '0': 2, '1': 3, '2': 4, '3': 5, '4': 6,
    '5': 7, '6': 8, '7': 9, '8': 10, '9': 11,
    '+': 12, '-': 13, '*': 14, '/': 15, '=': 16,
    'R': 17
}
# BasicFour Concat adds these
VOCAB_MAP['<CAR_+>'] = 18
VOCAB_MAP['<CAR_->'] = 19
VOCAB_MAP['<CAR_*>'] = 20
VOCAB_MAP['<CAR_/>'] = 21

def test_format(dataset_mode):
    print(f"\nTesting mode: {dataset_mode}")
    
    # Dummy item
    item = {
        "A": "15",
        "B": "7",
        "op": "+",
        "Y": "22",
        "labels": {
            "add_carry": [1, 1], # 5+7=12 (c=1), 1+0+1=2 (c=0? wait. 1+0=1. 15+7=22. 5+7=12->2,c=1. 1+0+1=2. c=0. )
            # Correct carries for 15+7: LSB first. 
            # digit 0: 5+7=12. carry_out=1.
            # digit 1: 1+0+1=2. carry_out=0.
            # carries = [1, 0]
        }
    }
    item["labels"]["add_carry"] = [1, 0]
    
    data_items = [item]
    
    # We will hijack the save function or copy logic.
    # Let's copy the logic part relevant to ids.
    
    max_len = 20
    
    inputs_list = []
    
    # Logic extracted from process_and_save_split
    for item in data_items:
        A = item["A"]
        B = item["B"]
        op = item["op"]
        Y = item["Y"]
        
        prompt_str = f"{A}{op}{B}="
        prompt_ids = build_ds.encode_str(prompt_str)
        result_ids = build_ds.encode_str(Y)
        input_ids = prompt_ids + result_ids
        
        concat_suffix_ids = []
        if dataset_mode in ["basicfour_concat", "basic_concat_reverse"]:
            car_token_id = VOCAB_MAP.get(f"<CAR_{op}>")
            inter_val_list = item["labels"].get("add_carry", [])
            
            suffix_seq_ids = [car_token_id]
            for val in inter_val_list:
                s_val = str(val)
                for char in s_val:
                    suffix_seq_ids.append(VOCAB_MAP[char])
            concat_suffix_ids = suffix_seq_ids
            
            # input_ids = input_ids + concat_suffix_ids # This was in line 568 but overridden later?
            # Wait, line 604 overrides.
            
        full_ids = prompt_ids + result_ids + concat_suffix_ids
        if dataset_mode == "basic_concat_reverse":
            full_ids = prompt_ids + concat_suffix_ids + result_ids
            
        print(f"Full IDs: {full_ids}")
        # Decode for readability
        decoded = []
        inv_map = {v: k for k, v in VOCAB_MAP.items()}
        for i in full_ids:
            if i in inv_map: decoded.append(inv_map[i])
            else: decoded.append(f"<{i}>")
        print(f"Decoded: {''.join(decoded)}")

test_format("vanilla")
test_format("basicfour_concat")
test_format("basic_concat_reverse") # Just to show CoT difference
