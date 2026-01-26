
import numpy as np
import os

data_dir = "data/test_reverse/train"
inputs = np.load(os.path.join(data_dir, "all__inputs.npy"))
labels = np.load(os.path.join(data_dir, "all__labels.npy"))

VOCAB_MAP_INV = {
    2: '0', 3: '1', 4: '2', 5: '3', 6: '4',
    7: '5', 8: '6', 9: '7', 10: '8', 11: '9',
    12: '+', 13: '-', 14: '*', 15: '/', 16: '=',
    17: 'R', 0: 'PAD', 1: 'MASK',
    18: '<CAR_+>', 19: '<CAR_->', 20: '<CAR_*>', 21: '<CAR_/>'
}

def decode(seq):
    res = []
    for x in seq:
        if x == 0: break
        res.append(VOCAB_MAP_INV.get(x, '?'))
    return "".join(res)

print("Verifying Basic Concat Reverse Structure:")
for i in range(min(5, len(inputs))):
    print(f"\nExample {i}:")
    print("Input:", decode(inputs[i]))
    print("Labels (shifted):", decode([x if x!=-100 else 0 for x in labels[i]]))
    
    # Check order
    # Expected: ...=<CAR>...
    decoded = decode(inputs[i])
    if "=" in decoded:
        parts = decoded.split("=")
        rhs = parts[1]
        if rhs.startswith("<CAR_"):
            print("Layout OK: Starts with CAR token after =")
        else:
            print("Layout ERROR: Does NOT start with CAR token after =")
