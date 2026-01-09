# TRM Lilavati: Logic Summary

## Overview
TRM Lilavati is a columnar arithmetic architecture that processes addition by explicitly modeling place-value structure and carry propagation, inspired by Bhāskara II's Lilavati method.

## Core Philosophy
Instead of treating arithmetic as a generic sequence task, TRM Lilavati enforces a **columnar inductive bias**: each digit position (stambha/column) is processed independently with explicit carry coordination.

---

## 1. Data Format: Reversed Digits (LSB First)

**Input Format:**
- Standard: `"13+25=38"` (MSD first)
- **TRM Lilavati: `"31+52=83"`** (LSB first - reversed)

**Rationale:** Process least significant digits first to align with carry propagation direction.

**Token Mapping:**
- Digits 0-9 → Tokens 2-11
- `+` → Token 12
- `=` → Token 13
- PAD → Token 0
- MASK → Token 1

---

## 2. Architecture: SthanaYantra (स्थानयन्त्र - Place-Value Machine)

### 2.1 Columnar State Representation

**State:** `z_sthana[batch, num_stambha, gulika_dim]`
- `num_stambha`: Number of digit positions (columns)
- `gulika_dim`: Per-column embedding dimension
- Each column `i` corresponds to digit position `i` (stambha 0 = ones/LSB)

### 2.2 Digit Extraction from Input

For reversed input `"31+52="`:
- **Stambha 0** (ones): Extracts digit from position 0 of A and position 3 of B → `(3, 5)`
- **Stambha 1** (tens): Extracts digit from position 1 of A and position 4 of B → `(1, 2)`

**Vectorized extraction:**
```python
# Find '+' and '=' positions
plus_pos = find_plus_token(inputs)  # [batch]
eq_pos = find_eq_token(inputs)      # [batch]

# Direct mapping: stambha i → position i for A, position (plus_pos+1+i) for B
a_pos = stambha_idx.unsqueeze(0)  # [batch, num_stambha]
b_pos = (plus_pos + 1).unsqueeze(1) + stambha_idx.unsqueeze(0)  # [batch, num_stambha]

# Extract digit embeddings from input
a_digits = extract_embeddings(input_embeddings, a_pos)  # [batch, num_stambha, hidden_size]
b_digits = extract_embeddings(input_embeddings, b_pos)  # [batch, num_stambha, hidden_size]
```

### 2.3 Column Initialization

Each column is initialized with:
1. **Digit embeddings** from A and B (concatenated)
2. **Previous column state** `z_sthana` (for recursive refinement)
3. **Projected to gulika_dim** via linear layer

```python
# Initial column state
z_init = linear_projection([a_digits, b_digits, z_sthana])  # [batch, num_stambha, gulika_dim]
```

---

## 3. Processing Layers

### 3.1 StambhaLayer (Column Processing)

**Two operations per layer:**

1. **Local Convolution:**
   - Conv1d with kernel_size=3 (neighbors only)
   - Each column sees itself + left/right neighbors
   - Enables local carry interactions

2. **MLP (SwiGLU):**
   - Per-column non-linear processing
   - Independent computation for each stambha

**Padding Masking:** Invalid columns (beyond actual number length) are masked out to prevent garbage propagation.

```python
# Local convolution
z_conv = conv1d(z.transpose(1,2)).transpose(1,2)  # [batch, num_stambha, gulika_dim]
z_conv = z_conv * valid_mask.unsqueeze(-1)  # Mask padded columns

# Residual + norm
z = rms_norm(z + z_conv)

# MLP
z_mlp = mlp(z) * valid_mask.unsqueeze(-1)
z = rms_norm(z + z_mlp)
```

### 3.2 HastaSanchara (हस्तसंचार - Hand Movement/Carry Propagation)

**GRU-based left-to-right sweep:**
- Processes columns from LSB (stambha 0) to MSB
- Each column updates a carry state
- Carry state is residual-added to column embeddings

```python
carry_state = zeros(batch, gulika_dim)

for col in range(num_stambha):
    col_input = z[:, col, :]  # Current column
    carry_state = gru_cell(col_input, carry_state)  # Update carry
    z[:, col, :] += carry_state  # Residual connection
```

**Purpose:** Explicitly model carry propagation across digit positions.

---

## 4. Prediction Heads

### 4.1 Digit Head (Per-Column)
- **Input:** Column state `z_sthana[batch, num_stambha, gulika_dim]`
- **Output:** `digit_logits[batch, num_stambha, 10]` (logits for digits 0-9)
- **Purpose:** Predict digit at each position

### 4.2 Carry Head (Per-Column)
- **Input:** Column state `z_sthana[batch, num_stambha, gulika_dim]`
- **Output:** `carry_logits[batch, num_stambha]` (logits for carry_out)
- **Purpose:** Predict whether column produces carry

### 4.3 Output Construction

**Key Design:** Prevent global leakage by separating result positions from input positions.

**Strategy:**
1. **Base output:** `lm_head(input_embeddings)` for all positions
2. **Digit output:** Map `digit_logits` to result positions (after `=`)
3. **Combine:** Replace base output with digit output at result positions only

```python
# Base output from input embeddings (no global mixing)
base_output = lm_head(input_embeddings)  # [batch, seq_len, vocab_size]

# Map digit_logits to result positions
for col in range(max_result_len):
    seq_pos = eq_pos + 1 + col  # Position after '='
    stambha_idx = col  # Direct mapping (LSB first)
    
    # Get digit logits for this column
    col_digit_logits = digit_logits[:, stambha_idx, :]  # [batch, 10]
    
    # Pad to vocab size [batch, vocab_size]
    vocab_contrib = pad(col_digit_logits, offset=DIGIT_OFFSET)
    
    # Place at result position via one-hot encoding
    contribution = einsum('bs,bv->bsv', one_hot(seq_pos), vocab_contrib)
    digit_contributions.append(contribution)

digit_output = sum(digit_contributions)  # [batch, seq_len, vocab_size]

# Combine: replace base_output with digit_output at result positions
result_mask = (seq_indices >= result_start)  # [batch, seq_len]
output = base_output * (1 - result_mask) + digit_output * result_mask
```

**Critical:** This prevents the model from "memorizing" by mixing global sequence information at result positions. Only column predictions are used for outputs.

---

## 5. Loss Functions

### 5.1 Language Modeling Loss
- **Standard cross-entropy** on token predictions
- **Computed on:** All positions (but result positions use digit_output, not base_output)

### 5.2 Q-Halt Loss (ACT)
- **Signal:** Confidence-based halting (not binary correctness)
  - Compute digit prediction confidence from `digit_logits`
  - Target: halt when confidence > threshold (e.g., 0.9)
- **Penalty:** Higher weight (3x) for incorrect sequences to encourage earlier halting

### 5.3 Carry Loss (Auxiliary)
- **Binary cross-entropy** on carry predictions
- **Target:** Ground truth `carry_out` for each column (loaded from dataset `carries` field)
- **Weight:** 0.5 (auxiliary supervision, increased for better carry reasoning)
- **Shape handling:** Carries array is padded/truncated to match `num_stambha` if dataset has different length

### 5.4 Digit Loss (Auxiliary)
- **Cross-entropy** on per-column digit predictions
- **Target:** Ground truth result digit at each stambha (extracted from labels after '=')
- **Weight:** 0.5 (auxiliary supervision, increased to force digit-by-digit reasoning)
- **Mapping:** Direct mapping from result position to stambha (LSB-first: position 0 → stambha 0)

### 5.5 Ponder Cost
- **Standard ACT ponder cost:** `0.01 * steps.mean()`
- Encourages early halting

**Total Loss:**
```
L = LM_loss + 
    0.5 * (Q_halt_loss + Q_continue_loss) + 
    0.5 * carry_loss +      # Increased from 0.1
    0.5 * digit_loss +      # Increased from 0.1
    0.001 * ponder_cost     # Adjusted: 0.001 per step
```

**Loss Computation:** Loss is computed on ALL sequences at each ACT step (not just halted ones), then accumulated across all steps before backpropagation. This enables learning from intermediate reasoning steps.

---

## 6. Training Dynamics

### 6.1 Recursive Refinement (ACT)
- Model iteratively refines column states over multiple steps
- Each step: forward through SthanaYantra → update `z_sthana`
- Halts when confidence threshold crossed or max steps reached

**Training Loop:** CRITICAL - Training must loop through multiple ACT steps per batch (not just one step). The training loop continues until all sequences halt or max_steps is reached. This enables multi-step learning:
```python
# Training loop for each batch
while step_count < max_act_steps:
    carry, loss, metrics, _, all_halted = model(carry, batch)
    total_loss = total_loss + loss  # Accumulate across steps
    step_count += 1
    if all_halted:
        break
# Backpropagate accumulated loss
total_loss.backward()
```

**Loss Accumulation:** Loss is accumulated (summed) across all ACT steps, not averaged. Each step contributes gradients, allowing the model to learn from all reasoning steps, not just the final one.

### 6.2 Gradient Flow (Training)
- **Result positions:** Block gradients from `base_output`, only flow through `digit_output`
- **Non-result positions:** Gradients flow through `base_output`
- **Purpose:** Force model to learn columnar reasoning, not sequence memorization
- **Carry state:** Carry states are detached between steps (standard ACT practice) to prevent gradient explosion, but loss accumulation ensures learning from all steps

---

## 7. Inference

### 7.1 Forward Pass
1. Extract digits from input (A and B)
2. Initialize column states
3. Process through StambhaLayer → HastaSanchara
4. Predict digits and carries per column
5. Map digit logits to result positions
6. Compute Q-halt logits from pooled column states

### 7.2 Halting Decision
- Pool column states: `pooled = z_sthana.mean(dim=1)` → `[batch, gulika_dim]`
- Q-head: `q_logits = q_head(pooled)` → `[batch, 2]` (halt, continue)
- Halt if: `q_halt > q_continue` OR `steps >= max_steps`

### 7.3 Prediction Extraction
- Get final logits after halting
- `preds = argmax(logits, dim=-1)`
- Extract tokens after `=` position
- Decode and reverse digits to get standard format
- Compare with expected result

---

## 8. Key Design Choices

### 8.1 Why Columnar?
- **Inductive bias:** Aligns with place-value arithmetic structure
- **Interpretability:** Each column has a clear meaning (digit position)
- **Efficiency:** Parallel processing of columns

### 8.2 Why Reversed Digits?
- **Carry direction:** Process LSB → MSB to match natural carry flow
- **Alignment:** Column indices directly map to digit positions

### 8.3 Why Explicit Carry?
- **Supervision:** Direct supervision on carry propagation (auxiliary loss)
- **Structure:** Explicitly enforces arithmetic reasoning pattern

### 8.4 Why No Global Leakage?
- **Prevent shortcuts:** Model can't "memorize" by mixing all sequence information
- **Force reasoning:** Must use column predictions, not global patterns

---

## 9. Differences from Standard TRM

| Aspect | Standard TRM | TRM Lilavati |
|--------|-------------|--------------|
| **State** | Global `z_H`, `z_L` | Columnar `z_sthana[batch, num_stambha, dim]` |
| **Processing** | Full-sequence attention | Per-column local convolutions |
| **Interactions** | Global attention | Local (neighbors) + GRU carry sweep |
| **Inductive Bias** | Task-agnostic recursive reasoning | Place-value arithmetic |
| **Supervision** | Sequence-level (LM + halting) | Multi-level (sequence + digit + carry) |
| **Output** | Direct from global state | Column predictions mapped to positions |

---

## 10. Flow Summary

```
Input: "31+52=83" (reversed, LSB first)
  ↓
Extract digits:
  Stambha 0: A[0]=3, B[3]=5
  Stambha 1: A[1]=1, B[4]=2
  ↓
Initialize columns: z_sthana[batch, 2, gulika_dim]
  ↓
For each ACT step:
  ↓
  StambhaLayer (4 layers):
    Local conv (neighbors) → MLP
    ↓
  HastaSanchara:
    GRU sweep (LSB → MSB) with carry propagation
    ↓
  Predict:
    digit_logits[batch, 2, 10]  (digits 0-9)
    carry_logits[batch, 2]      (carry_out)
    ↓
  Map to output:
    result_pos[0] (position 7) ← stambha 0 digit
    result_pos[1] (position 8) ← stambha 1 digit
    ↓
  Q-head: pooled states → halt decision
  ↓
  If not halted: continue to next step
  ↓
Final predictions: Extract tokens at result positions
  ↓
Output: "83" → reverse → "38" → compare with ground truth
```

---

## 11. Key Implementation Details

1. **Valid Masking:** Pad columns beyond actual number length to prevent garbage
2. **Grammar Validation:** Ensure exactly one `+` and one `=` in correct order
3. **Type Safety:** Convert integers to tensors before `.clamp()` operations
4. **Gradient Blocking:** Detach `base_output` at result positions during training
5. **Direct Mapping:** `col → stambha_idx` is direct (no reversal) in LSB-first format
6. **Training Loop:** CRITICAL - Must loop through multiple ACT steps per batch (not single step). Loop until all sequences halt or max_steps reached.
7. **Loss Accumulation:** Loss is accumulated (summed) across all ACT steps before backpropagation, enabling multi-step learning.
8. **Carries Field:** Dataset must include `carries` field for auxiliary supervision. Dataset loader automatically handles optional loading and padding.
9. **Shape Compatibility:** Carries array shape is automatically padded/truncated to match `num_stambha` during loss computation.
10. **ACT Halting:** During training, sequences can halt early based on q_halt_logits > 0 (with no_ACT_continue=True) or q_halt > q_continue, subject to exploration constraints (min_halt_steps).