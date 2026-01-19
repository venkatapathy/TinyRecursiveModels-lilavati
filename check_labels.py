
import numpy as np
import os

def check_values():
    path = "data/arithmetic_dual/train/all__labels_aux.npy"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    data = np.load(path)
    print(f"Shape: {data.shape}")
    
    # Ignore -100
    valid_mask = data != -100
    valid_data = data[valid_mask]
    
    if len(valid_data) == 0:
        print("No valid data (all -100).")
        return
        
    max_val = valid_data.max()
    min_val = valid_data.min()
    
    print(f"Min value: {min_val}")
    print(f"Max value: {max_val}")
    
    print(f"Count > 19: {(valid_data > 19).sum()}")
    
    # Check if division ops are the culprit
    # We need inputs to know which op it is.
    
    
    # Check Puzzle Identifiers - TRAIN
    pid_path = "data/arithmetic_dual/train/all__puzzle_identifiers.npy"
    if os.path.exists(pid_path):
        print(f"Loading {pid_path}...")
        pids = np.load(pid_path)
        print(f"PIDs Max: {pids.max()}")
        print(f"PIDs Min: {pids.min()}")
        if pids.max() >= 1:
             print("ALERT: TRAIN PIDs >= 1 (should be 0)!")
        if pids.min() < 0:
             print("ALERT: TRAIN PIDs < 0!")

    # Check Puzzle Identifiers - VAL
    pid_path = "data/arithmetic_dual/val/all__puzzle_identifiers.npy"
    if os.path.exists(pid_path):
        print(f"Loading {pid_path}...")
        pids = np.load(pid_path)
        print(f"PIDs Max: {pids.max()}")
        print(f"PIDs Min: {pids.min()}")
        if pids.max() >= 1:
             print("ALERT: VAL PIDs >= 1 (should be 0)!")
        if pids.min() < 0:
             print("ALERT: VAL PIDs < 0!")

if __name__ == "__main__":
    check_values()
