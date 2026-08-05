# Research Plan — Algorithmic Commitment as the Lever for Length Generalization

**Target:** ICLR 2027 (expected abstract ~Sep 19, full paper ~Sep 24, 2026 AoE; confirm on official CFP)
**Fallback:** TMLR (rolling) if the stretch pillars slip
**Builds on:** Lee et al., *Teaching Arithmetic to Small Transformers* (ICLR 2024); Zhou et al., *What Algorithms can Transformers Learn?* (RASP-L, ICLR 2024); Jolicoeur-Martineau et al., *Tiny Recursive Models* (2025)

---

## 1. One-sentence thesis

Length generalization on an arithmetic operation is governed by **which algorithm the model commits to executing**; that commitment can be induced either by supervising the algorithm's intermediate state in the output *or* by rewarding it, and we characterize when each route works and when it fails.

## 2. Contribution claims (what we will defend)

- **C1 — Architecture × supervision.** A recursive model (TRM) plus minimal output-state supervision (OS-Before) produces length-invariant extrapolation on operations where the plain-transformer scratchpads of Lee et al. (2024) degrade past the training range. *Recursion is what converts a non-extrapolating scratchpad into an extrapolating one.*
- **C2 — Ordering is causal.** Emitting the algorithmic state *before* the answer vs. *after* it, with identical supervision content, produces materially different extrapolation — isolating decoding order as the operative variable (cleaner than any manipulation in prior arithmetic work).
- **C3 — Supervision ≡ directed reward.** The same algorithmic commitment can be imposed by SFT-on-state or by a method-directed (process) reward; both yield extrapolation.
- **C4 — Outcome-only RL does not discover the generalizing algorithm.** Outcome-only RLVR sharpens the training-length distribution and fails to extrapolate (consistent with the "RLVR sharpens, doesn't expand" literature), whereas method-directed reward recovers it. *This is the novel cell not covered by Lee et al., RASP-L, or the RLVR literature.*
- **C5 — Beyond arithmetic (insurance).** The output-state-constraint mechanism transfers to at least one non-arithmetic algorithmic task with a known short program (sorting / parity), answering the "narrow to arithmetic" critique.

## 3. Positioning / delta vs. prior work

| Prior work | What it establishes | Our delta |
|---|---|---|
| Lee et al. 2024, *Teaching Arithmetic* | Format decides in-distribution learning; simplified scratchpad records digit-sum+carry; reverse (LSD) helps; **reports failure to extrapolate** | We **achieve** extrapolation via recursion + output-state supervision; isolate ordering; add RL axis |
| Zhou et al. 2024, RASP-L | Length generalization ⇔ task has a short RASP-L program (algorithm determines generalizability) | We test the *inducement* question: how commitment to such an algorithm gets into the model (supervision vs. reward) |
| Abbe et al. 2024, Globality Barrier / Inductive Scratchpad | Scratchpad design breaks the globality barrier | We compare *minimal* state and study reward-based inducement |
| TRM / HRM (2025) | Recursion helps algorithmic tasks | We combine recursion with output-state supervision and quantify the interaction |
| RLVR "sharpens not expands" (2025–26) | Outcome RL amplifies base behaviors, doesn't add capability | We give a controlled arithmetic demonstration: outcome RL cannot find the length-generalizing method |

**Framing rule:** cite Lee et al. head-on in the intro and related work; never describe the method as "fundamentally different from CoT" (it is a compact, structured, output-space scratchpad). Lead with C1–C4.

## 4. Pillars & experiments

### Pillar 0 — Data integrity & reproducibility (GATE, blocks everything)

- **Resolve the sequence-vs-digit contradiction.** Sequence accuracy cannot exceed digit accuracy. Audit the eval: confirm both metrics use the same examples and denominator; check for a label swap, a padded/length-mismatched digit denominator, or a scoring bug. Document the root cause.
- **Regenerate Tables 1–3 from a single eval harness** so the 100d Table-1-vs-Table-2 conflict disappears (same run feeds all tables).
- **Write the full training protocol**: data distribution & sampling, sample sizes, optimizer/schedule, joint-vs-per-task training, seeds (report mean ± std over ≥3 seeds), tokenizer, exact `<RES>`/state token counts (kills the "blank think-token" confound).
- **Deliverable:** a frozen eval script + a reproducibility checklist. *No result is quoted until it comes from this harness.*

### Pillar 1 — Supervision core (the spine)

- **Baselines (the missing ones):** No-Scratchpad (NS); **detailed CoT scratchpad**; **Lee et al. simplified scratchpad**; OS-Before; OS-After. (The two scratchpad baselines are what three reviewers demanded.)
- **Architectures:** TRM *and* Transformer for every setting (fixes the "why no Transformer OS-After?" gap).
- **Scale:** a **gradual ladder** (~300k → ~1M → ~3M → ~10M → 40×) instead of a single 40× jump, to plot the scale-vs-supervision trade-off honestly. Reframe the claim as *efficiency at fixed budget*, not "beats scale."
- **Operations:** +, −, ×, ÷. ID = 1–8 digits; OOD = 9–32 (stress to 48/64 for the headline models).
- **Headline results:** C1 (TRM+OS-Before extrapolates where the GPT scratchpad flattens) and C2 (the OS-Before vs OS-After ordering ablation, both architectures).

### Pillar 2 — Method-directed RL vs supervision (the novel headline)

Three training signals per operation:
1. **SFT-on-state** (OS-Before) — supervised baseline.
2. **Outcome-only RLVR** (GRPO, reward = final-answer exact match).
3. **Method-directed / process reward** (reward includes correctness of the chosen method's intermediate state).

Two valid methods per operation (the "same operation, different algorithm" axis):

| Op | Method A (expected to generalize — local, position-invariant) | Method B (expected to generalize worse — global/recursive) |
|---|---|---|
| + | LSD-first carry propagation | MSD-first / carry-lookahead |
| − | LSD-first borrow propagation | Ten's-complement (add-complement) |
| × | Schoolbook long multiplication (column carries) | Karatsuba-style divide & conquer |
| ÷ | Long division (remainder chain, MSB-first) | Partial-quotients ("chunking") |

**Sequencing (honoring "all" but de-risking):** build the harness on **division** (your weakest result — biggest upside) and **multiplication** first; add **addition/subtraction** once it works (cheap).

**Core hypotheses:** signal (2) fails to extrapolate for both methods (sharpens to training lengths); signals (1) and (3) extrapolate for Method A and less so for Method B → *the algorithmic commitment, not the signal type, is the operative variable, and outcome reward alone can't find it.*

### Pillar 3 — Non-arithmetic transfer (insurance for "too narrow")

One algorithmic task with a known short program and a natural intermediate state: **sorting** (comparison/position state) or **parity/counting** (running-count state). Run NS vs OS-Before vs the RL signals; show the mechanism is not arithmetic-specific. *Cut first if the schedule slips.*

## 5. Consolidated experimental matrix

- Signals: {NS, detailed-CoT, simplified-scratchpad, OS-After, OS-Before, RL-outcome, RL-method-directed}
- Architectures: {TRM, Transformer} × scale ladder (5 sizes) for the supervised arm
- Operations: {+, −, ×, ÷} × {Method A, Method B} for the RL arm
- Lengths: ID 1–8, OOD 9–32 (+ stress 48/64 for headline)
- Seeds: ≥3, report mean ± std
- Metrics: sequence exact-match, per-digit accuracy, per-operation carry/trace accuracy, **all from the Pillar-0 harness**

## 6. Metrics & evaluation protocol

- **Fix and freeze** sequence vs digit definitions (sequence correct ⟺ all digits correct; digit accuracy over a fixed, length-matched denominator).
- Report an **OOD length grid**, not just aggregates, so extrapolation curves are visible.
- Regenerate **Figure 1** with legible labels; describe NS curves accurately (they flatline — do not call them "oscillatory").
- Add an **error-type breakdown by length** (the analysis reviewer aK2h asked for).

## 7. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pillar-0 fix moves headline numbers materially | Med | Do it FIRST; if numbers collapse, pivot to TMLR and re-scope honestly |
| Process-reward RL hacks / won't converge | High | Keep RL to div+mul first; Pillar 1 is a complete paper without Pillar 2 |
| RL infra + scale ladder overruns compute | Med | Prioritize headline model sizes; ladder can be coarse |
| Broad scope misses September | Med-High | Spine-plus-stretch schedule with go/no-go gates (below) |
| Second-task (Pillar 3) underbaked | Med | It's insurance; cut first |

## 8. Critical-path schedule (backed into ~Sep 24)

- **Weeks 1–2 (through early Aug):** Pillar 0 — fix eval, regenerate tables, write protocol. **GATE 1:** are the numbers real? If not → TMLR track.
- **Weeks 2–4:** Pillar 1 — baselines (incl. Lee et al. scratchpads), both architectures, scale ladder. **A complete paper exists at the end of this.**
- **Weeks 4–6:** Pillar 2 — RL harness on division + multiplication; SFT vs outcome vs method-directed. **GATE 2 (~late Aug):** is the C4 result clean? If yes, it's the headline; if no, demote RL to a negative-result subsection and keep C1–C2 as the spine.
- **Weeks 6–7:** extend RL to +/−; Pillar 3 if on track; theory/positioning writing (RASP-L, Lee et al. contrast, latent-reasoning vs looped-transformer positioning).
- **Week 7–8:** full write-up to 8–9 pp (ICLR format), figures, reproducibility checklist, limitations section, internal red-team review. Abstract in by the abstract deadline regardless.

## 9. Reviewer-wound crosswalk (ICML → fix)

- "Just a compact scratchpad / not different from CoT" → C1–C4 reframing + Lee et al. cited head-on (Pillar 1 baselines).
- "No CoT/scratchpad baselines" → Pillar 1 detailed-CoT + simplified-scratchpad arms.
- "Impossible seq > digit accuracy; inconsistent tables" → Pillar 0.
- "Single 40× jump; supervision-beats-scale unsupported" → Pillar 1 scale ladder + efficiency reframing.
- "Inference-unchanged claim is contradictory" → drop/qualify in writing.
- "Underdescribed training; not reproducible" → Pillar 0 protocol + checklist.
- "Unfair Qwen baseline" → fine-tune on-distribution or demote to labeled off-the-shelf reference.
- "Narrow to arithmetic" → Pillar 3.
- "No theory for why LSD/ordering helps" → RASP-L positioning (Pillar 1 writing).

## 10. Open decisions (need your input)

- Pillar 3 task: **sorting** vs **parity/counting** — which do you prefer, or defer the choice to GATE 2?
- Qwen: fine-tune it on-distribution (extra compute) or drop to a clearly-labeled zero-shot reference?
- Author/compute logistics: who runs which pillar; is the coding agent driving Pillar 0+1 while you scope Pillar 2?
