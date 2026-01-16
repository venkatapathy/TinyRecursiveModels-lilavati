# build_multiplication_dataset.py
import os
import json
import numpy as np
from typing import Optional
from argdantic import ArgParser
from pydantic import BaseModel
from dataset.common import PuzzleDatasetMetadata
from collections import Counter, defaultdict

cli = ArgParser()


def compute_carry_trace(a: int, b: int, digits: int) -> list:
    """
    Compute carry trace from long multiplication.
    For each multiplier digit j=0..digits-1:
        carry = 0
        for i=0..digits-1:
            t = a_ds[i] * b_ds[j] + carry
            carry = t // 10
            append carry to carries
    Returns list of exactly digits*digits integers in [0..8] for digits=3.
    """
    # Extract digits from LSD to MSD
    a_ds = [(a // (10**i)) % 10 for i in range(digits)]
    b_ds = [(b // (10**j)) % 10 for j in range(digits)]
    
    carries = []
    for j in range(digits):  # For each multiplier digit
        carry = 0
        for i in range(digits):  # For each multiplicand digit
            t = a_ds[i] * b_ds[j] + carry
            carry = t // 10
            carries.append(carry)
    return carries  # Exactly digits*digits integers


class DataProcessConfig(BaseModel):
    output_dir: str = "data/multiplication"
    seed: int = 42
    test_ratio: float = 0.1

    digits: int = 3  # max digits per operand if varied_length=False (e.g., 3 => 000..999)
    dataset_mode: str = "vanilla"  # "vanilla" or "lilavati1"

    # Sampling controls (same semantics as your addition script)
    max_examples: Optional[int] = None        # target TRAIN examples after filtering
    sample_ratio: Optional[float] = None      # fraction of total combinations to sample (if max_examples is None)

    varied_length: bool = False               # if True, operands can have 1..digits digits (with no fixed padding)
    train_max_digits: Optional[int] = None    # if set, train uses operands with <=train_max_digits; test adds OOD >train_max_digits
    also_output_dir: Optional[str] = None     # auto-generate other mode to this dir (vanilla <-> lilavati1)



@cli.command(singleton=True)
def main(config: DataProcessConfig):
    assert config.dataset_mode in {"vanilla", "lilavati1"}, \
        f"dataset_mode must be 'vanilla' or 'lilavati1', got {config.dataset_mode}"
    assert config.digits >= 1, f"digits must be >= 1, got {config.digits}"
    assert config.sample_ratio is None or 0.0 < config.sample_ratio <= 1.0, \
        f"sample_ratio must be in (0,1], got {config.sample_ratio}"
    assert config.train_max_digits is None or config.train_max_digits >= 1, \
        f"train_max_digits must be >= 1, got {config.train_max_digits}"
    assert config.train_max_digits is None or config.train_max_digits < config.digits, \
        f"train_max_digits ({config.train_max_digits}) must be < digits ({config.digits})"

    np.random.seed(config.seed)

    # -------------------------
    # helpers
    # -------------------------
    def filter_by_digit_length(a: int, b: int, max_digits: int) -> bool:
        max_len = max(len(str(a)), len(str(b)))
        return max_len <= max_digits

    # For 3-digit operands (0..999), product max is 998001 => 6 digits.
    # Generally, if operands up to 10^d-1, product < 10^(2d), so pad to 2d digits.
    prod_digits_fixed = 2 * config.digits  # 6 for digits=3
    # Carry trace has exactly digits*digits digits
    carry_trace_len = config.digits * config.digits  # 9 for digits=3

    # -------------------------
    # enumerate / sample pairs
    # -------------------------
    max_val = 10 ** config.digits - 1
    total_possible = (max_val + 1) ** 2

    target_train_samples = config.max_examples if (config.max_examples is not None and config.max_examples > 0) else None

    if target_train_samples is not None:
        sample_size = min(int(target_train_samples / (1 - config.test_ratio) * 1.2), total_possible)
        print(f"Target: {target_train_samples:,} train; sampling ~{sample_size:,} pairs for split+filters")
    elif config.sample_ratio is not None:
        sample_size = max(1, int(total_possible * config.sample_ratio))
        print(f"Sampling {sample_size:,} ({config.sample_ratio*100:.6f}%) from {total_possible:,}")
    else:
        sample_size = total_possible
        print(f"Using all {total_possible:,} pairs")

    rng = np.random.default_rng(config.seed)

    if sample_size < total_possible:
        # index-based sampling (memory friendly)
        indices = rng.choice(total_possible, size=sample_size, replace=False)
        pairs = []
        base = (max_val + 1)
        for idx in indices:
            a = int(idx // base)
            b = int(idx % base)
            pairs.append((a, b))
    else:
        # only safe for digits=3 (1,000,000)
        pairs = [(i, j) for i in range(max_val + 1) for j in range(max_val + 1)]

    # -------------------------
    # train_max_digits OOD split (same idea as your addition script)
    # -------------------------
    if config.train_max_digits is not None:
        print(f"Applying train_max_digits={config.train_max_digits} (OOD goes to test)...")
        trainable_pairs = [p for p in pairs if filter_by_digit_length(p[0], p[1], config.train_max_digits)]
        extra_pairs = [p for p in pairs if not filter_by_digit_length(p[0], p[1], config.train_max_digits)]
        if len(trainable_pairs) == 0:
            raise ValueError("No pairs passed train_max_digits filtering.")
        np.random.shuffle(trainable_pairs)

        split_idx = int(len(trainable_pairs) * (1 - config.test_ratio))
        if target_train_samples is not None:
            split_idx = min(target_train_samples, split_idx)

        train_pairs = trainable_pairs[:split_idx]
        test_pairs = trainable_pairs[split_idx:]

        # add OOD to test, balanced by operand max-digit length
        if len(extra_pairs) > 0:
            extra_by_digits = defaultdict(list)
            for a, b in extra_pairs:
                L = max(len(str(a)), len(str(b)))
                extra_by_digits[L].append((a, b))

            target_ood = len(test_pairs)
            num_bins = len(extra_by_digits)
            per_bin = max(1, target_ood // max(1, num_bins))

            sampled_extra = []
            for L in sorted(extra_by_digits.keys()):
                bucket = extra_by_digits[L]
                np.random.shuffle(bucket)
                take = min(per_bin, len(bucket))
                sampled_extra.extend(bucket[:take])

            test_pairs.extend(sampled_extra)
            print(f"Train={len(train_pairs):,}, Test(ID)={len(test_pairs)-len(sampled_extra):,}, "
                  f"Added OOD={len(sampled_extra):,} => Test(total)={len(test_pairs):,}")
    else:
        np.random.shuffle(pairs)
        split_idx = int(len(pairs) * (1 - config.test_ratio))
        if target_train_samples is not None:
            split_idx = min(target_train_samples, split_idx)
        train_pairs = pairs[:split_idx]
        test_pairs = pairs[split_idx:]

    splits = {"train": train_pairs, "test": test_pairs}

    # -------------------------
    # vocabulary (char-level + special token like <CAR>)
    # -------------------------
    # IDs:
    # 0: PAD
    # 1: MASK
    # 2..11: '0'..'9'
    # then operators and specials
    vocab_map = {str(i): i + 2 for i in range(10)}
    vocab_map['*'] = 12
    vocab_map['='] = 13

    # For carry trace we need <CAR>
    if config.dataset_mode == "lilavati1":
        vocab_map['<CAR>'] = 14
        vocab_size = 15  # 0..14
        CAR_TOKEN_ID = 14
    else:
        vocab_size = 14  # 0..13
        CAR_TOKEN_ID = None

    PAD_ID = 0
    MASK_ID = 1
    IGNORE_LABEL_ID = 255

    def encode_str(s: str):
        # s is composed only of characters in vocab_map
        return [vocab_map[c] for c in s]

    # -------------------------
    # build sequences (2-pass to get global max seq_len)
    # -------------------------
    os.makedirs(config.output_dir, exist_ok=True)
    all_split_data = {}
    global_max_seq_len = 0

    for split_name, current_pairs in splits.items():
        inputs, labels = [], []

        for (a, b) in current_pairs:
            prod = a * b

            if config.varied_length:
                s_a = str(a)
                s_b = str(b)
            else:
                s_a = f"{a:0{config.digits}d}"
                s_b = f"{b:0{config.digits}d}"

            # Always pad product to 2*digits so sequence length is stable (like addition digits+1).
            s_prod = f"{prod:0{prod_digits_fixed}d}"

            prefix = s_a + "*" + s_b + "="
            prefix_ids = encode_str(prefix)

            if config.dataset_mode == "vanilla":
                inp_seq = prefix_ids + [MASK_ID] * prod_digits_fixed
                lab_seq = [IGNORE_LABEL_ID] * len(prefix_ids) + encode_str(s_prod)
            else:
                # carry trace
                carries = compute_carry_trace(a, b, config.digits)
                s_carries = ''.join(str(c) for c in carries)
                assert len(s_carries) == carry_trace_len, f"Expected {carry_trace_len} carry digits, got {len(s_carries)}"

                inp_seq = (
                    prefix_ids
                    + [MASK_ID] * prod_digits_fixed
                    + [CAR_TOKEN_ID]
                    + [MASK_ID] * carry_trace_len
                )
                lab_seq = (
                    [IGNORE_LABEL_ID] * len(prefix_ids)
                    + encode_str(s_prod)
                    + [CAR_TOKEN_ID]
                    + encode_str(s_carries)
                )
                assert lab_seq.count(CAR_TOKEN_ID) == 1

            assert len(inp_seq) == len(lab_seq)
            inputs.append(inp_seq)
            labels.append(lab_seq)

        split_max = max(len(x) for x in inputs) if inputs else 0
        global_max_seq_len = max(global_max_seq_len, split_max)

        all_split_data[split_name] = {"inputs": inputs, "labels": labels, "pairs": current_pairs}

    # -------------------------
    # pad + save
    # -------------------------
    for split_name, split_data in all_split_data.items():
        inputs = split_data["inputs"]
        labels = split_data["labels"]
        current_pairs = split_data["pairs"]

        puzzle_indices = [0]
        group_indices = [0]
        puzzle_identifiers = []

        inputs_padded, labels_padded = [], []
        for idx, (inp, lab) in enumerate(zip(inputs, labels)):
            pad_len = global_max_seq_len - len(inp)
            inputs_padded.append(inp + [PAD_ID] * pad_len)
            labels_padded.append(lab + [IGNORE_LABEL_ID] * pad_len)

            puzzle_identifiers.append(0)
            puzzle_indices.append(idx + 1)
            group_indices.append(idx + 1)

        inputs_np = np.array(inputs_padded, dtype=np.uint8)
        labels_np = np.array(labels_padded, dtype=np.uint8)
        puzzle_indices_np = np.array(puzzle_indices, dtype=np.int32)
        group_indices_np = np.array(group_indices, dtype=np.int32)
        puzzle_identifiers_np = np.array(puzzle_identifiers, dtype=np.int32)

        save_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "all__inputs.npy"), inputs_np)
        np.save(os.path.join(save_dir, "all__labels.npy"), labels_np)
        np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), puzzle_indices_np)
        np.save(os.path.join(save_dir, "all__group_indices.npy"), group_indices_np)
        np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers_np)

        metadata = PuzzleDatasetMetadata(
            seq_len=global_max_seq_len,
            vocab_size=vocab_size,
            pad_id=PAD_ID,
            ignore_label_id=IGNORE_LABEL_ID,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=len(group_indices_np) - 1,
            mean_puzzle_examples=1.0,
            total_puzzles=len(puzzle_indices_np) - 1,
            sets=["all"],
        )
        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    generate_dataset_stats(config, splits, vocab_map, vocab_size, global_max_seq_len, total_possible,
                           prod_digits_fixed=prod_digits_fixed,
                           carry_trace_len=carry_trace_len)

    # also generate other mode
    if config.also_output_dir is not None:
        other_mode = "lilavati1" if config.dataset_mode == "vanilla" else "vanilla"
        other_cfg = type(config)(
            output_dir=config.also_output_dir,
            seed=config.seed,
            test_ratio=config.test_ratio,
            digits=config.digits,
            dataset_mode=other_mode,
            max_examples=config.max_examples,
            sample_ratio=config.sample_ratio,
            varied_length=config.varied_length,
            train_max_digits=config.train_max_digits,
            also_output_dir=None,
        )
        print(f"\nAlso generating {other_mode} dataset at: {config.also_output_dir}")
        main(other_cfg)


def generate_dataset_stats(
    config: DataProcessConfig,
    splits: dict,
    vocab_map: dict,
    vocab_size: int,
    max_seq_len: int,
    total_possible: int,
    prod_digits_fixed: int,
    carry_trace_len: int,
):
    total_examples = sum(len(v) for v in splits.values())
    train_examples = len(splits["train"])
    test_examples = len(splits["test"])

    # basic stats
    products = []
    all_carries = []
    for split_name, pairs in splits.items():
        for a, b in pairs:
            products.append(a * b)
            if config.dataset_mode == "lilavati1":
                carries = compute_carry_trace(a, b, config.digits)
                all_carries.extend(carries)

    prod_counter = Counter(len(str(p)) for p in products)

    vocab_desc = []
    vocab_desc.append("- `0`: PAD")
    vocab_desc.append("- `1`: MASK")
    for i in range(10):
        vocab_desc.append(f"- `{i+2}`: '{i}'")
    vocab_desc.append("- `12`: '*'")
    vocab_desc.append("- `13`: '='")
    if config.dataset_mode == "lilavati1":
        vocab_desc.append("- `14`: '<CAR>'")

    # sample examples
    def show_example(a, b):
        if config.varied_length:
            s_a, s_b = str(a), str(b)
        else:
            s_a, s_b = f"{a:0{config.digits}d}", f"{b:0{config.digits}d}"
        p = a * b
        s_p = f"{p:0{prod_digits_fixed}d}"
        if config.dataset_mode == "vanilla":
            return f"`{s_a}*{s_b}={s_p}`"
        else:
            carries = compute_carry_trace(a, b, config.digits)
            s_carries = ''.join(str(c) for c in carries)
            return f"`{s_a}*{s_b}={s_p} <CAR> {s_carries}`"

    train_sample = splits["train"][:5]
    test_sample = splits["test"][:3]

    # README
    readme = []
    readme.append(f"# Multiplication Dataset - {config.dataset_mode.upper()} Mode ({config.digits}-digit operands)\n")
    readme.append("## Dataset Information\n")
    readme.append("### Configuration")
    readme.append(f"- **Mode**: {config.dataset_mode}")
    readme.append(f"- **Digits (operand max)**: {config.digits}")
    readme.append(f"- **Seed**: {config.seed}")
    readme.append(f"- **Test Ratio**: {config.test_ratio}")
    readme.append(f"- **Varied Length**: {config.varied_length}")
    readme.append(f"- **Sampling**: {f'{config.max_examples:,} train examples' if config.max_examples is not None else (f'{config.sample_ratio*100:.4f}%' if config.sample_ratio is not None else 'All combinations')}")
    readme.append(f"- **Total Possible Combinations (0..{10**config.digits-1})^2**: {total_possible:,}")
    if config.train_max_digits is not None:
        readme.append(f"- **Train Max Digits**: {config.train_max_digits} (test includes OOD > {config.train_max_digits})")
    if config.dataset_mode == "lilavati1":
        readme.append(f"- **Carry Trace Length**: {carry_trace_len} digits (digits*digits = {config.digits}*{config.digits})")

    readme.append("\n### Dataset Statistics")
    readme.append(f"- **Total Examples**: {total_examples:,}")
    readme.append(f"- **Training Examples**: {train_examples:,}")
    readme.append(f"- **Test Examples**: {test_examples:,}")
    readme.append(f"- **Sequence Length**: {max_seq_len}")
    readme.append(f"- **Vocabulary Size**: {vocab_size}")
    readme.append(f"- **Product Digits (padded)**: {prod_digits_fixed} (since max is < 10^(2*digits))")
    readme.append(f"- **Product digit-count distribution (raw, before padding)**: {dict(sorted(prod_counter.items()))}")

    if config.dataset_mode == "lilavati1" and all_carries:
        carry_counter = Counter(all_carries)
        readme.append("\n### Carry Distribution Statistics")
        readme.append(f"- **Total Carry Predictions**: {len(all_carries):,}")
        for carry_val in sorted(carry_counter.keys()):
            count = carry_counter[carry_val]
            readme.append(f"- **Carry = {carry_val}**: {count:,} ({count/len(all_carries)*100:.1f}%)")

    readme.append("\n## Format Specification\n")
    readme.append("### Input Format")
    if config.varied_length:
        readme.append(f"- `{('X'*config.digits)}*{('Y'*config.digits)}=` where X and Y have 1..{config.digits} digits (no fixed padding)")
    else:
        readme.append(f"- `{('X'*config.digits)}*{('Y'*config.digits)}=` where X and Y are zero-padded to {config.digits} digits")
    readme.append("\n### Output Format")
    if config.dataset_mode == "vanilla":
        readme.append(f"- `{('P'*prod_digits_fixed)}` (product, padded to {prod_digits_fixed} digits)")
    else:
        readme.append(f"- `{('P'*prod_digits_fixed)} <CAR> {('C'*carry_trace_len)}`")
        readme.append(f"  - Carry trace = {carry_trace_len} digits (digits*digits = {config.digits}*{config.digits})")
        readme.append(f"  - Each carry digit is in [0..8] (from long multiplication partial products)")

    readme.append("\n### Vocabulary")
    readme.append("\n".join(vocab_desc))

    readme.append("\n## Examples\n")
    readme.append("### Training Examples (sample)")
    for a, b in train_sample[:3]:
        readme.append(f"- {show_example(a, b)}")
    readme.append("\n### Test Examples (sample)")
    for a, b in test_sample:
        readme.append(f"- {show_example(a, b)}")

    readme.append("\n## File Structure")
    readme.append(f"""""")

    readme_path = os.path.join(config.output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("\n".join(readme))
    print(f"Wrote README: {readme_path}")


if __name__ == "__main__":
    cli()
