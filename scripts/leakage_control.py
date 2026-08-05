#!/usr/bin/env python3
"""
Leakage control: demonstrate that the ICML-era input formulation was solvable
by copying, and that the masked-field formulation is not.

The old pipeline fed the model `prompt + trace + <RES> + answer` as INPUT and
supervised it on that same sequence shifted left by one, while attention was
bidirectional (`causal=False`). Position t is therefore asked to predict the
token sitting at position t+1, which it can attend to directly.

This script trains the same tiny model under both formulations and reports
accuracy on the supervised positions:

  A. OLD format, evaluated as it was  -> if this is high, nothing is proven yet
  B. OLD format, answer region masked at eval time only
       -> collapse here means A was copying, not computing
  C. NEW masked-field format
       -> the honest baseline: the model never sees the answer in either
          training or eval, so there is no gap to measure

Run:  venv/bin/python scripts/leakage_control.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1_Inner

PAD, MASK, EQ, RES, SEP = 0, 1, 16, 22, 23
VOCAB = 24
SEQ = 128          # enough for the small ID examples used here
STEPS = 600
BS = 256
DEV = "cuda"

V = {str(i): i + 2 for i in range(10)}
for i, c in enumerate("+-*/"):
    V[c] = 12 + i
V["="] = EQ
CAR = {"+": 18, "-": 19, "*": 20, "/": 21}


def enc(s):
    return [V[c] for c in s]


def carries(op, a, b):
    if op == "+":
        ad = [int(d) for d in str(a)[::-1]]
        bd = [int(d) for d in str(b)[::-1]]
        out, c = [], 0
        for i in range(len(str(a + b))):
            s = (ad[i] if i < len(ad) else 0) + (bd[i] if i < len(bd) else 0) + c
            c = s // 10
            out.append(c)
        return out
    return []


def make_examples(n, rng, max_digits=6):
    """Addition only, small, ID-range -- enough to expose the mechanism."""
    ex = []
    while len(ex) < n:
        la = rng.integers(1, max_digits + 1)
        lb = rng.integers(1, max_digits + 1)
        a = int(rng.integers(10 ** (la - 1), 10 ** la))
        b = int(rng.integers(10 ** (lb - 1), 10 ** lb))
        ex.append(("+", a, b))
    return ex


def build_old(ex):
    """Legacy leaky layout: input == full sequence, labels == shift-by-one."""
    X = np.zeros((len(ex), SEQ), np.int64)
    Y = np.full((len(ex), SEQ), -100, np.int64)
    for i, (op, a, b) in enumerate(ex):
        prompt = enc(f"{a}{op}{b}=")
        trace = [CAR[op]]
        for v in carries(op, a, b):
            trace += enc(str(v)) + [SEP]
        full = prompt + trace + [RES] + enc(str(a + b))
        full = full[:SEQ]
        lab = full[1:] + [PAD]
        for j in range(len(prompt) - 1):
            lab[j] = -100
        X[i, :len(full)] = full
        Y[i, :len(lab)] = lab
    return torch.from_numpy(X), torch.from_numpy(Y)


def build_new(ex):
    """Masked-field layout: input == prompt + MASK*, labels position-aligned."""
    X = np.zeros((len(ex), SEQ), np.int64)
    Y = np.full((len(ex), SEQ), -100, np.int64)
    for i, (op, a, b) in enumerate(ex):
        prompt = enc(f"{a}{op}{b}=")
        trace = [CAR[op]]
        for v in carries(op, a, b):
            trace += enc(str(v)) + [SEP]
        target = trace + [RES] + enc(str(a + b))
        w = SEQ - len(prompt)
        X[i, :len(prompt)] = prompt
        X[i, len(prompt):] = MASK
        Y[i, len(prompt):len(prompt) + len(target)] = target[:w]
        Y[i, len(prompt) + len(target):] = PAD
    return torch.from_numpy(X), torch.from_numpy(Y)


def mask_after_eq(X):
    """Blank everything after '=' -- the control that removes the copy source."""
    Xm = X.clone()
    for i in range(len(Xm)):
        eq = (Xm[i] == EQ).nonzero()
        if len(eq):
            Xm[i, int(eq[0]) + 1:] = MASK
    return Xm


class Cfg:
    batch_size, seq_len, vocab_size, num_puzzle_identifiers = BS, SEQ, VOCAB, 1
    hidden_size, expansion, num_heads = 256, 2.666, 4
    H_cycles, L_cycles, H_layers, L_layers = 1, 4, 1, 2
    pos_encodings, rms_norm_eps, rope_theta = "rope", 1e-5, 10000.0
    halt_max_steps, halt_exploration_prob = 1, 0.0
    forward_dtype = "bfloat16"
    puzzle_emb_ndim = puzzle_emb_len = 0
    mlp_t = False
    no_ACT_continue = True
    dataset_mode, digits, carry_loss_weight = "basic_concat_reverse", 32, 1.0


def train_and_score(build_fn, label, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    Xtr, Ytr = build_fn(make_examples(20000, rng))
    Xte, Yte = build_fn(make_examples(2000, rng))

    model = TinyRecursiveReasoningModel_ACTV1_Inner(Cfg()).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    model.train()
    for step in range(STEPS):
        idx = torch.randint(0, len(Xtr), (BS,))
        x, y = Xtr[idx].to(DEV), Ytr[idx].to(DEV)
        carry = model.reset_carry(torch.ones(BS, dtype=torch.bool, device=DEV),
                                  model.empty_carry(BS))
        _, logits, _, _ = model(carry, {"inputs": x, "puzzle_identifiers": torch.zeros(len(x), dtype=torch.long, device=DEV)})
        loss = F.cross_entropy(logits.float().view(-1, VOCAB), y.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def acc(X, Y):
        model.eval()
        tot = hit = 0
        for i in range(0, len(X), BS):
            x, y = X[i:i + BS].to(DEV), Y[i:i + BS].to(DEV)
            if len(x) < BS:
                break
            carry = model.reset_carry(torch.ones(len(x), dtype=torch.bool, device=DEV),
                                      model.empty_carry(len(x)))
            _, logits, _, _ = model(carry, {"inputs": x, "puzzle_identifiers": torch.zeros(len(x), dtype=torch.long, device=DEV)})
            p = logits.argmax(-1)
            m = y != -100
            hit += (m & (p == y)).sum().item()
            tot += m.sum().item()
        return hit / max(tot, 1)

    return acc(Xte, Yte), acc(mask_after_eq(Xte), Yte)


if __name__ == "__main__":
    print(f"Training tiny TRM ({STEPS} steps) under each formulation.\n")

    old_normal, old_masked = train_and_score(build_old, "old")
    new_normal, new_masked = train_and_score(build_new, "new")

    print("=" * 66)
    print("  LEAKAGE CONTROL — token accuracy on supervised positions")
    print("=" * 66)
    print(f"  A. OLD format, evaluated as shipped      : {old_normal:.4f}")
    print(f"  B. OLD format, answer masked at eval only: {old_masked:.4f}")
    print(f"     -> drop attributable to copying       : {old_normal - old_masked:+.4f}")
    print()
    print(f"  C. NEW masked-field format               : {new_normal:.4f}")
    print(f"  D. NEW format, masked again (no-op)      : {new_masked:.4f}")
    print(f"     -> drop (should be ~0, already masked): {new_normal - new_masked:+.4f}")
    print("=" * 66)

    verdict = ("LEAK CONFIRMED: the old format's accuracy came from reading the "
               "answer out of its own input.") if (old_normal - old_masked) > 0.25 else \
              ("No large gap measured; inspect before drawing conclusions.")
    print(f"\n  {verdict}\n")

    json.dump({"old_normal": old_normal, "old_masked": old_masked,
               "new_normal": new_normal, "new_masked": new_masked},
              open("results/leakage_control.json", "w"), indent=2)
    print("  wrote results/leakage_control.json")
