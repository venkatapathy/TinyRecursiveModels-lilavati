import numpy as np
import os
import glob

def check_dataset(path, split, vocab_size):
    print(f"Checking {path}/{split} with vocab_size={vocab_size}")
    files = glob.glob(os.path.join(path, split, "*__inputs.npy"))
    for f in files:
        print(f"  Loading {f}...")
        try:
            data = np.load(f)
            max_val = data.max()
            min_val = data.min()
            print(f"    Min: {min_val}, Max: {max_val}, Shape: {data.shape}")
            if max_val >= vocab_size:
                print(f"    [ERROR] Max value {max_val} >= vocab_size {vocab_size}!")
                indices = np.where(data >= vocab_size)
                print(f"    Found {len(indices[0])} violations. First few indices: {indices[0][:5]}")
            if min_val < 0:
                print(f"    [ERROR] Min value {min_val} < 0!")
        except Exception as e:
            print(f"    Error reading {f}: {e}")

    # Also check inputs_carry if it exists
    files_carry = glob.glob(os.path.join(path, split, "*__inputs_carry.npy"))
    for f in files_carry:
        print(f"  Loading {f}...")
        try:
            data = np.load(f)
            max_val = data.max()
            print(f"    Max: {max_val}, Shape: {data.shape}")
            if max_val >= vocab_size:
                print(f"    [ERROR] Max value {max_val} >= vocab_size {vocab_size}!")
        except Exception as e:
            print(f"    Error reading {f}: {e}")

base_path = "/home/venkat/TinyRecursiveModels-lilavati/data/icml/vanilla"
vocab_size = 17

check_dataset(base_path, "train", vocab_size)
check_dataset(base_path, "test", vocab_size)
