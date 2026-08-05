"""
Evaluate a causal SLM (Qwen family) on the arithmetic benchmark.

Scoring is NOT implemented here. Every metric comes from
`evaluators.harness.ArithmeticEvaluator.score_one`, the same code path the
TRM/Transformer runs use, so SLM rows and model rows in the paper are
directly comparable.

The previous version of this file carried its own copy of the metric logic.
It used `max(len(target), len(pred))` as the digit-accuracy denominator (so a
run-on generation could contribute unbounded slots), micro-averaged digits
while macro-averaging sequences, counted an example into `total` before the
op was resolved, and read the target as `text.split('=')[1]` -- which for the
os_after/os_before renderings is the *trace*, not the answer. All four are
gone; see evaluators/_deprecated/README.md.

Output filename matches evaluate.py: `{run}__seed{N}__{split}.json`.
"""

import argparse
import json
import os
import re
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.common import PuzzleDatasetMetadata
from evaluators.harness import ArithmeticEvaluator

OP_NAMES = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}


def load_examples(data_dir: str, split: str, limit: int):
    """
    Read the structured jsonl, which carries op/A/B/Y as explicit fields.

    The plain `vanilla_*.jsonl` rendering is deliberately not accepted: its
    `text` field has to be string-split to recover operands, and for the
    state-supervised renderings that split returns the trace rather than the
    answer. Requiring the structured file removes the whole class of bug.
    """
    path = os.path.join(data_dir, f"structured_{split}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(
            f"No structured jsonl at {path}. Regenerate the dataset with "
            f"dataset/build_arithmetic_dataset.py -- the SLM path needs the "
            f"explicit op/A/B fields, not the flat `text` rendering."
        )
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def extract_prediction(generated_text: str, is_instruct: bool) -> str:
    """Pull the final integer answer out of a generation."""
    generated_text = generated_text.strip()
    if is_instruct:
        m = re.search(r'\\boxed\{([-\d,.]+)\}', generated_text)
        if m:
            return m.group(1).replace(',', '').strip()
        numbers = re.findall(r'-?\d+', generated_text.replace(',', ''))
        if numbers:
            return numbers[-1]
        return generated_text.split()[0] if generated_text else ""
    return generated_text.split()[0] if generated_text else ""


def build_prompts(items, is_instruct: bool):
    prompts = []
    for it in items:
        eq = f"{it['A']}{it['op']}{it['B']}"
        if is_instruct:
            prompts.append([{
                "role": "user",
                "content": (f"Please calculate the result of the following "
                            f"arithmetic operation: {eq}. Provide only the "
                            f"final numerical answer inside \\boxed{{}}."),
            }])
        else:
            prompts.append(eq + "=")
    return prompts


def tokenize(tokenizer, prompts, is_instruct, device):
    if not is_instruct:
        return tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    seqs = [tokenizer.apply_chat_template(m, tokenize=True,
                                          add_generation_prompt=True,
                                          return_tensors="pt")[0]
            for m in prompts]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), tokenizer.pad_token_id, dtype=torch.long)
    for k, s in enumerate(seqs):
        if tokenizer.padding_side == "left":
            ids[k, -len(s):] = s
        else:
            ids[k, :len(s)] = s
    return {"input_ids": ids.to(device)}


@torch.no_grad()
def evaluate_dataset(model, tokenizer, data_dir, args, split, sample_table=None):
    items = load_examples(data_dir, split, args.limit)
    print(f"Evaluating {data_dir} [{split}] on {len(items)} examples...")

    # The harness is told the mode is 'vanilla' because an SLM emits only the
    # final answer -- there is no trace to score. Everything else (ID/OOD
    # bucketing by max operand digits, the per-length grid, the invariant
    # checks) is identical to the model path.
    ev = ArithmeticEvaluator(
        data_path=data_dir,
        eval_metadata=PuzzleDatasetMetadata(
            seq_len=0, vocab_size=24, pad_id=0, ignore_label_id=-100,
            blank_identifier_id=0, num_puzzle_identifiers=1, total_groups=len(items),
            mean_puzzle_examples=1.0, sets=["all"],
        ),
        dataset_mode="vanilla",
        train_digits=args.train_digits,
        digits=args.digits,
    )
    ev.begin_eval()

    is_instruct = "Instruct" in args.model
    bs = args.batch_size

    for i in tqdm(range(0, len(items), bs)):
        batch = items[i:i + bs]
        prompts = build_prompts(batch, is_instruct)
        inputs = tokenize(tokenizer, prompts, is_instruct, model.device)

        out = model.generate(**inputs, max_new_tokens=args.max_tokens,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
        gen = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

        for it, text in zip(batch, gen):
            pred = extract_prediction(text, is_instruct)
            ev.n_seen += 1
            matched = ev.score_one(it['op'], int(it['A']), int(it['B']), pred)

            if sample_table is not None and len(sample_table.data) < 1000:
                sample_table.add_data(split, f"{it['A']}{it['op']}{it['B']}",
                                      it['Y'], text, pred, matched, it['op'])

    # result() runs the invariant checks (digit >= seq, grid partitions the
    # scored set, seq == counts-weighted grid mean) and raises on violation.
    return ev.result(save_path=None, rank=0, world_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Dataset directory containing structured_{split}.jsonl")
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--run_name", type=str, required=True,
                    help="Row identity in results/runs.json, e.g. qwen15b_ns")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train_digits", type=int, default=8)
    ap.add_argument("--digits", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--adapter", type=str, default=None,
                    help="Optional LoRA adapter directory from train_slm.py")
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True)

    if args.adapter:
        from peft import PeftModel
        print(f"Attaching LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    sample_table = None
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=f"{args.run_name}_s{args.seed}",
                   entity=args.wandb_entity, config=vars(args))
        sample_table = wandb.Table(columns=["split", "prompt", "target",
                                            "generated", "prediction",
                                            "is_correct", "op"])

    metrics = evaluate_dataset(model, tokenizer, args.data_dir, args,
                               args.split, sample_table)

    flat = {"_run": args.run_name, "_split": args.split, "_seed": args.seed}
    for k, v in metrics.items():
        if k in flat:
            raise ValueError(f"metric {k!r} collides with a reserved field")
        flat[k] = v

    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", f"{args.run_name}__seed{args.seed}__{args.split}.json")
    with open(path, "w") as f:
        json.dump(flat, f, indent=4)
    print(f"Saved results to {path}")

    if args.wandb_project:
        wandb.log(metrics)
        wandb.log({"samples": sample_table})
        wandb.finish()


if __name__ == "__main__":
    main()
