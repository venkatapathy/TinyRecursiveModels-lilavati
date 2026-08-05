#!/usr/bin/env python3
"""
Look at what the model actually predicts, per digit length.

The point is to discriminate between two failure modes that produce the same
sequence accuracy of zero:

  A. NEAR-MISS -- the answer has the right length and mostly the right digits,
     with errors concentrated in the high-order positions. That is a model
     that has learned the algorithm and is losing carry propagation with
     depth. Undertrained or capacity-limited, not conceptually broken.

  B. STRUCTURAL -- wrong length, misalignment, truncation, or termination
     failure. That is a model that never learned the task's shape, and no
     amount of extra training on the same setup fixes it.

Prints, per length bucket: exact match, digit accuracy, length-correct rate,
and a handful of verbatim examples.

Usage:
  venv/bin/python scripts/inspect_predictions.py \
      --checkpoint checkpoints/trm-iclr-gate1/trm_ns_s1/step_15624 \
      --data data/iclr/ns --mode vanilla --split test
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.harness import (
    EQ_ID, PAD_ID, expected_result, parse_prompt, split_field,
)


def load_model(ckpt_path):
    # pretrain.py pickles the entire TrainState rather than a state_dict, so
    # unpickling needs the class visible under __main__. Same shim evaluate.py
    # uses. (Saving a bare state_dict is on the Stage 0 list.)
    from pretrain import TrainState
    sys.modules['__main__'].TrainState = TrainState

    cfg_path = os.path.join(os.path.dirname(ckpt_path), "all_config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    arch = dict(cfg["arch"])
    name = arch.pop("name")
    arch.pop("loss", None)
    arch.pop("_target_", None)

    if "trm" in name:
        from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1 as M
    else:
        from models.baselines.transformer import StandardTransformer as M

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if hasattr(state, "model"):           # a pickled TrainState
        state = state.model
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if isinstance(state, dict) and "model" in state and not torch.is_tensor(state["model"]):
        state = state["model"]

    # Recover the shapes the checkpoint was actually written with.
    emb = next((v for k, v in state.items() if k.endswith("embed_tokens.embedding_weight")
                or k.endswith("embed_tokens.weight")), None)
    arch["vocab_size"] = emb.shape[0] if emb is not None else 24
    arch.setdefault("batch_size", 256)
    arch["seq_len"] = cfg.get("max_len", 512)
    arch.setdefault("num_puzzle_identifiers", 1)
    arch.setdefault("dataset_mode", cfg.get("dataset_mode", "vanilla"))
    arch.setdefault("digits", cfg.get("digits", 32))

    model = M(arch)
    missing, unexpected = model.load_state_dict(
        {k.replace("_orig_mod.", "").replace("model.", "", 1): v for k, v in state.items()},
        strict=False)
    if missing:
        print(f"  ! {len(missing)} missing keys, first few: {missing[:4]}")
    return model.cuda().eval(), arch


@torch.no_grad()
def run(args):
    model, arch = load_model(args.checkpoint)
    inputs = np.load(os.path.join(args.data, args.split, "all__inputs.npy"))

    rng = np.random.default_rng(0)
    idx = rng.choice(len(inputs), min(args.n, len(inputs)), replace=False)

    buckets = {}
    shown = 0
    for s in range(0, len(idx), args.batch):
        chunk = idx[s:s + args.batch]
        x = torch.from_numpy(inputs[chunk].astype(np.int64)).cuda()
        pid = torch.zeros(len(x), dtype=torch.long, device="cuda")

        inner = model.inner if hasattr(model, "inner") else model
        if hasattr(inner, "reset_carry"):
            carry = inner.reset_carry(
                torch.ones(len(x), dtype=torch.bool, device="cuda"),
                inner.empty_carry(len(x)))
            out = inner(carry, {"inputs": x, "puzzle_identifiers": pid})
            logits = out[1]
        else:
            out = inner({"inputs": x, "puzzle_identifiers": pid})
            logits = out["logits"] if isinstance(out, dict) else out

        preds = logits.argmax(-1).cpu().numpy()

        for row, p in zip(chunk, preds):
            op, a, b = parse_prompt(inputs[row])
            if op is None:
                continue
            want = expected_result(op, a, b)
            if want is None:
                continue
            eq = np.where(inputs[row] == EQ_ID)[0]
            got, _ = split_field(p[int(eq[0]) + 1:], args.mode)

            L = max(len(str(a)), len(str(b)))
            k = (L // args.bucket) * args.bucket
            d = buckets.setdefault(k, dict(n=0, exact=0, lenok=0, digits=0.0,
                                           emitted=0.0, target=0.0, prec=0.0,
                                           prec_n=0))
            d["n"] += 1
            d["exact"] += int(got == want)
            d["lenok"] += int(len(got) == len(want))
            e, g = want[::-1], got[::-1]
            hits = sum(1 for i in range(min(len(e), len(g))) if e[i] == g[i])
            d["digits"] += hits / len(e)

            # Conditional low-order precision: OF THE DIGITS THE MODEL ACTUALLY
            # EMITTED, how many are right?
            #
            # This is the metric that separates the two failure modes. Recall
            # (hits/len(want)) is dominated by digits the model never emitted,
            # so it collapses with length no matter how good the arithmetic is.
            # Precision (hits/len(got)) asks only about what it did produce.
            # High precision + low recall == the algorithm generalises and only
            # length control fails. Low precision == the arithmetic itself
            # fails, and length control is a side issue.
            d["emitted"] += len(got)
            d["target"] += len(want)
            if got:
                d["prec"] += hits / len(got)
                d["prec_n"] += 1

            if shown < args.examples and L >= args.example_min_len:
                flag = "OK " if got == want else ("LEN" if len(got) != len(want) else "DIG")
                print(f"  [{flag}] {a}{op}{b}")
                print(f"        want {want}")
                print(f"        got  {got}")
                shown += 1

    print()
    print(f"{'max digits':>12} {'n':>6} {'exact':>7} {'recall':>8} {'PRECISION':>10} "
          f"{'len ok':>7} {'emitted':>8} {'target':>7}")
    print("-" * 78)
    for k in sorted(buckets):
        d = buckets[k]
        prec = d["prec"] / max(d["prec_n"], 1)
        print(f"{k:>5}-{k+args.bucket-1:<6} {d['n']:>6} {d['exact']/d['n']:>7.4f} "
              f"{d['digits']/d['n']:>8.4f} {prec:>10.4f} {d['lenok']/d['n']:>7.4f} "
              f"{d['emitted']/d['n']:>8.1f} {d['target']/d['n']:>7.1f}")
    print()
    print("  recall    = correct / len(target)   -- penalised for digits never emitted")
    print("  PRECISION = correct / len(emitted)  -- of what it produced, how much is right")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", default="vanilla")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--bucket", type=int, default=4)
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--example_min_len", type=int, default=9)
    run(ap.parse_args())
