"""
Empirical parameter counts for every point on the scale ladder.

Both architectures are actually INSTANTIATED and their parameters counted.
Nothing here is read off a config filename -- that is precisely how the ICML
submission ended up claiming a "40x larger transformer baseline" when the
real ratio was ~4x:

  config/arch/trm.yaml            is named 300k but builds ~2.98M
  config/arch/transformer.yaml    is named 1000k but builds ~11.8M

Two things make naive estimates wrong. SwiGLU quantises its inner dimension
to a multiple of 256 (models/layers.py), so parameter count is very
non-linear in hidden_size at small D. And TRM shares one weight-tied L_level
stack across all H_cycles*L_cycles recursions, so recursion depth costs zero
parameters while `H_layers` is silently ignored entirely (trm.py).

Usage:
    venv/bin/python count_params.py
    venv/bin/python count_params.py --search 300000 1000000 3000000 10000000
"""

import argparse

from models.baselines.transformer import StandardTransformer
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

# Shared across every ladder point so the comparison is apples-to-apples.
COMMON = dict(batch_size=256, seq_len=512, vocab_size=24,
              num_puzzle_identifiers=1, expansion=2.666,
              pos_encodings="rope", dataset_mode="vanilla", digits=32)

# The ladder as configured. `target` is the label used in the paper; `actual`
# is filled in by measurement.
TRM_LADDER = [
    ("~300k", dict(hidden_size=128, num_heads=4, L_layers=2)),
    ("~1M",   dict(hidden_size=192, num_heads=6, L_layers=2)),
    ("~3M",   dict(hidden_size=384, num_heads=6, L_layers=2)),
    ("~10M",  dict(hidden_size=384, num_heads=6, L_layers=7)),
]

XF_LADDER = [
    ("~300k", dict(hidden_size=40,  num_heads=4, num_layers=8)),
    ("~1M",   dict(hidden_size=104, num_heads=4, num_layers=8)),
    ("~3M",   dict(hidden_size=192, num_heads=6, num_layers=8)),
    ("~10M",  dict(hidden_size=352, num_heads=8, num_layers=8)),
]


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_trm(**over) -> int:
    cfg = dict(COMMON, H_cycles=1, L_cycles=4, H_layers=1, L_layers=2,
               halt_max_steps=1, halt_exploration_prob=0.05,
               puzzle_emb_ndim=0, puzzle_emb_len=0, carry_loss_weight=1.0)
    cfg.update(over)
    return n_params(TinyRecursiveReasoningModel_ACTV1(cfg))


def build_xf(**over) -> int:
    cfg = dict(COMMON, num_layers=8, hidden_size=384, num_heads=6)
    cfg.update(over)
    return n_params(StandardTransformer(cfg))


def ladder_table() -> None:
    print(f"{'Target':>8}  {'TRM':>14}  {'cfg':>26}  {'Transformer':>14}  {'cfg':>24}")
    print("-" * 96)
    trm_actual, xf_actual = [], []
    for (tgt, tcfg), (_, xcfg) in zip(TRM_LADDER, XF_LADDER):
        t, x = build_trm(**tcfg), build_xf(**xcfg)
        trm_actual.append(t)
        xf_actual.append(x)
        tdesc = f"D={tcfg['hidden_size']}, L_layers={tcfg['L_layers']}"
        xdesc = f"D={xcfg['hidden_size']}, layers={xcfg['num_layers']}"
        print(f"{tgt:>8}  {t:>14,}  {tdesc:>26}  {x:>14,}  {xdesc:>24}")

    print()
    print("As-shipped configs referenced by the ICML tables:")
    shipped_trm = build_trm(hidden_size=384, num_heads=6, L_layers=2)
    shipped_xf_300k = build_xf(hidden_size=40, num_heads=4, num_layers=8)
    shipped_xf_1000k = build_xf(hidden_size=384, num_heads=6, num_layers=8)
    print(f"  config/arch/trm.yaml            (labelled 300k) : {shipped_trm:>12,}")
    print(f"  config/arch/transformer_300k.yaml               : {shipped_xf_300k:>12,}")
    print(f"  config/arch/transformer.yaml    (labelled 1M)   : {shipped_xf_1000k:>12,}")
    print()
    print(f"  headline ratio, transformer/TRM : "
          f"{shipped_xf_1000k / shipped_trm:.2f}x   <- NOT 40x")


def search(targets) -> None:
    """Find the hidden_size that lands nearest each target for both archs."""
    print("Nearest hidden_size per target (heads chosen to divide D):\n")
    for target in targets:
        best_t = best_x = None
        for d in range(32, 704, 8):
            for h in (4, 6, 8):
                if d % h:
                    continue
                try:
                    t = build_trm(hidden_size=d, num_heads=h, L_layers=2)
                    if best_t is None or abs(t - target) < abs(best_t[0] - target):
                        best_t = (t, d, h)
                    x = build_xf(hidden_size=d, num_heads=h, num_layers=8)
                    if best_x is None or abs(x - target) < abs(best_x[0] - target):
                        best_x = (x, d, h)
                except Exception as e:      # surfaced, never swallowed
                    print(f"    ! D={d} heads={h} failed to build: {e}")
        print(f"  target {target:>10,}")
        print(f"    TRM         D={best_t[1]:>3} heads={best_t[2]} -> {best_t[0]:>12,}")
        print(f"    Transformer D={best_x[1]:>3} heads={best_x[2]} -> {best_x[0]:>12,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", type=int, nargs="*", default=None,
                    help="Target parameter counts to search hidden_size for")
    args = ap.parse_args()

    ladder_table()
    if args.search:
        print()
        search(args.search)
