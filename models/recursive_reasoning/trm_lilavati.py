"""
TRM Lilavati: Tiny Recursive Model with SthanaYantra (स्थानयन्त्र) architecture.

Implements Bhaskara II's Lilavati addition method (Saṅkalana - संकलन) where
each Stambha (column) processes one digit position with proper carry
propagation via HastaSanchara.

Key features:
- Columnar representation: z_sthana[batch, num_stambha, gulika_dim]
- Direct digit-to-column mapping (no position collapse)
- Local interactions via Conv1d (neighbors only)
- Carry sweep via GRU (left-to-right for LSB to MSB)
- Per-column digit prediction with supervision
- Auxiliary carry prediction head
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, CastedEmbedding, CastedLinear

IGNORE_LABEL_ID = -100

# Vocabulary constants
PLUS_TOKEN = 12  # '+' token
EQ_TOKEN = 13    # '=' token
DIGIT_OFFSET = 2  # Digits 0-9 map to tokens 2-11


@dataclass
class TRM_LilavatiInnerCarry:
    """Carry state for SthanaYantra (स्थानयन्त्र)."""
    z_sthana: torch.Tensor  # [batch, num_stambha, gulika_dim]


@dataclass
class TRM_LilavatiCarry:
    """Full carry state including ACT halting."""
    inner_carry: TRM_LilavatiInnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TRM_LilavatiConfig(BaseModel):
    """Configuration for TRM Lilavati model."""
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    
    # SthanaYantra config (स्थानयन्त्र - Place-value machine)
    num_stambha: int = 25  # Number of columns (स्तम्भ)
    gulika_dim: int = 64   # Bead dimension (गुलिका)
    stambha_layers: int = 4  # Number of column processing layers
    kernel_size: int = 3   # Local convolution kernel size
    
    # Standard config
    hidden_size: int = 512
    expansion: float = 4
    puzzle_emb_ndim: int = 0
    puzzle_emb_len: int = 16
    
    # Halting config
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.05
    
    rms_norm_eps: float = 1e-5
    forward_dtype: str = "bfloat16"
    
    # ACT config
    no_ACT_continue: bool = True
    
    # Backward compatibility aliases
    @property
    def num_columns(self) -> int:
        return self.num_stambha
    
    @property
    def bead_dim(self) -> int:
        return self.gulika_dim
    
    @property
    def abacus_layers(self) -> int:
        return self.stambha_layers


class StambhaLayer(nn.Module):
    """
    Single column processing layer (स्तम्भ परिक्रमा - Stambha Parikrama).
    
    Combines local convolution (neighbor interaction) with MLP processing.
    All operations in float32 for stability with torch.compile.
    """
    
    def __init__(self, gulika_dim: int, kernel_size: int = 3, expansion: float = 4):
        super().__init__()
        
        # Local convolution: each column sees itself + neighbors
        self.local_conv = nn.Conv1d(
            in_channels=gulika_dim,
            out_channels=gulika_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        
        # MLP for per-column processing
        self.mlp = SwiGLU(
            hidden_size=gulika_dim,
            expansion=expansion,
        )
        
        self.norm_eps = 1e-5
    
    def forward(self, z: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process column state.
        
        Args:
            z: [batch, num_stambha, gulika_dim] - float32
            valid_mask: [batch, num_stambha] - True for active columns, None to process all
        
        Returns:
            Updated z: [batch, num_stambha, gulika_dim] - float32
        """
        # Ensure float32 for all operations
        z = z.float()
        
        # Local convolution (transpose for Conv1d)
        z_t = z.transpose(1, 2)  # [batch, gulika_dim, num_stambha]
        z_conv = self.local_conv(z_t).transpose(1, 2)  # [batch, num_stambha, gulika_dim]
        
        # Mask out padded columns if mask provided
        if valid_mask is not None:
            z_conv = z_conv * valid_mask.unsqueeze(-1).float()
        
        z = rms_norm(z + z_conv, variance_epsilon=self.norm_eps)
        
        # MLP
        z_mlp = self.mlp(z)
        if valid_mask is not None:
            z_mlp = z_mlp * valid_mask.unsqueeze(-1).float()
        z = rms_norm(z + z_mlp, variance_epsilon=self.norm_eps)
        
        return z


class HastaSanchara(nn.Module):
    """
    Carry propagation sweep (हस्त सञ्चार - Hasta Sanchara).
    
    Processes columns left-to-right (LSB to MSB) to propagate carry information.
    All operations in float32 for stability.
    """
    
    def __init__(self, gulika_dim: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(gulika_dim, gulika_dim)
        self.norm_eps = 1e-5
    
    def forward(self, z: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sweep carry information across columns.
        
        Args:
            z: [batch, num_stambha, gulika_dim] - float32
            valid_mask: [batch, num_stambha] - True for active columns, None to process all
        
        Returns:
            Updated z with carry information propagated: [batch, num_stambha, gulika_dim] - float32
        """
        # Ensure float32
        z = z.float()
        batch, num_stambha, gulika_dim = z.shape
        device = z.device
        
        # Initialize carry state (float32)
        carry_state = torch.zeros(batch, gulika_dim, device=device, dtype=torch.float32)
        
        # Determine maximum valid column per batch
        if valid_mask is not None:
            max_valid_col = valid_mask.sum(dim=-1).max().item()  # Max across all batches
        else:
            max_valid_col = num_stambha
        
        outputs = []
        # Sweep left-to-right (LSB to MSB) only through valid columns
        for col in range(max_valid_col):
            col_input = z[:, col, :]
            
            # Update carry state with GRU
            new_carry_state = self.gru_cell(col_input, carry_state)
            
            # Only update if column is valid
            if valid_mask is not None:
                col_mask = valid_mask[:, col].unsqueeze(-1).float()
                carry_state = new_carry_state * col_mask + carry_state * (1 - col_mask)
            else:
                carry_state = new_carry_state
            
            outputs.append(carry_state)
        
        # Pad outputs if we stopped early
        while len(outputs) < num_stambha:
            outputs.append(outputs[-1] if outputs else torch.zeros_like(carry_state))
        
        result = torch.stack(outputs, dim=1)
        
        # Mask result to prevent updates to invalid columns
        if valid_mask is not None:
            result = result * valid_mask.unsqueeze(-1).float()
        
        return rms_norm(z + result, variance_epsilon=self.norm_eps)


class SthanaYantra(nn.Module):
    """
    SthanaYantra (स्थानयन्त्र): Neural Place-Value Machine.
    
    Implements Bhaskara II's Lilavati addition method (Saṅkalana) where
    each Stambha (column) processes one digit position with proper
    carry propagation via HastaSanchara.
    
    Unlike the previous mean-pooling approach, this directly extracts
    digit embeddings from their positions in the sequence.
    """
    
    def __init__(self, config: TRM_LilavatiConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        
        # Digit value embedding (0-9) - encodes numerical meaning
        # 11 entries: 0-9 for digits, 10 for "no digit" (padding/invalid)
        self.digit_value_emb = nn.Embedding(11, config.gulika_dim)
        
        # Per-column projection: from hidden_size to gulika_dim
        self.stambha_proj = nn.Linear(config.hidden_size, config.gulika_dim)
        
        # Stambha layers with HastaSanchara
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'stambha': StambhaLayer(config.gulika_dim, config.kernel_size, config.expansion),
                'hasta': HastaSanchara(config.gulika_dim),
            }) for _ in range(config.stambha_layers)
        ])
        
        # REMOVED: Output projection to sequence space (prevents global leakage)
        # We'll map columns directly to result positions instead of global mixing
        
        # Digit prediction head (per column) - predicts result digit (0-9) at each position
        # Each stambha predicts: result_digit[i] = (a[i] + b[i] + carry_in[i]) mod 10
        self.digit_head = nn.Linear(config.gulika_dim, 10)
        
        # Carry prediction head (auxiliary) - predicts carry_out per column
        # carry_out[i] = floor((a[i] + b[i] + carry_in[i]) / 10)
        self.carry_head = nn.Linear(config.gulika_dim, 1)
        
        self.norm_eps = config.rms_norm_eps
    
    def forward(
        self, 
        x: torch.Tensor,
        inputs: torch.Tensor,
        z_sthana: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process input through SthanaYantra with proper digit extraction.
        
        Vectorized implementation for GPU efficiency.
        
        Args:
            x: Input embeddings [batch, seq_len, hidden_size]
            inputs: Raw input tokens [batch, seq_len]
            z_sthana: Previous column state (optional) [batch, num_stambha, gulika_dim]
        
        Returns:
            - z_out: Updated column state [batch, num_stambha, gulika_dim]
            - output: None (removed to prevent global leakage)
            - digit_logits: Per-column digit predictions [batch, num_stambha, 10]
            - carry_logits: Per-column carry predictions [batch, num_stambha] (carry_out)
        """
        batch = x.shape[0]
        seq_len = x.shape[1]
        device = x.device
        num_stambha = self.config.num_stambha
        
        # Parse input to find operator positions
        plus_pos = (inputs == PLUS_TOKEN).long().argmax(dim=1)  # [batch]
        eq_pos = (inputs == EQ_TOKEN).long().argmax(dim=1)      # [batch]
        
        # Validate grammar: ensure exactly one '+' and one '=' in valid positions
        plus_count = (inputs == PLUS_TOKEN).sum(dim=-1)  # [batch]
        eq_count = (inputs == EQ_TOKEN).sum(dim=-1)  # [batch]
        valid_grammar = (plus_count == 1) & (eq_count == 1) & (plus_pos < eq_pos)
        if not valid_grammar.all():
            # This should not happen with proper dataset, but handle gracefully
            pass  # Will be masked out by valid checks below
        
        # Vectorized digit extraction for all stambha at once
        # CRITICAL FIX: For reversed string "31+52=", positions are:
        # Position 0='3' (A[0], ones digit), 1='1' (A[1], tens digit)
        # Position 2='+', 3='5' (B[0], ones digit), 4='2' (B[1], tens digit), 5='='
        # Note: B starts at sequence position 3, which is B[0] (B's ones digit)
        # Stambha 0 = ones digit (LSB) = position 0 in reversed string for A
        # Stambha 1 = tens digit = position 1 in reversed string for A
        stambha_idx = torch.arange(num_stambha, device=device)  # [num_stambha]
        
        # Direct mapping: stambha i maps to position i for A
        # For B: starts after '+' at position (plus_pos + 1), so B[j] is at sequence position (plus_pos + 1 + j)
        # The formula b_pos = (plus_pos + 1) + stambha_idx correctly maps:
        #   stambha 0 → position (plus_pos + 1 + 0) = B[0] position
        #   stambha 1 → position (plus_pos + 1 + 1) = B[1] position
        a_pos = stambha_idx.unsqueeze(0)  # [batch, num_stambha] - direct mapping
        b_pos = (plus_pos + 1).unsqueeze(1) + stambha_idx.unsqueeze(0)  # [batch, num_stambha]
        
        # Bounds check: [batch, num_stambha]
        # A digits must be before '+' position, B digits must be between '+' and '='
        a_valid = (a_pos >= 0) & (a_pos < plus_pos.unsqueeze(1))
        b_valid = (b_pos > plus_pos.unsqueeze(1)) & (b_pos < eq_pos.unsqueeze(1))
        
        # Clamp positions to valid range for gather
        a_pos_clamped = a_pos.clamp(0, seq_len - 1)  # [batch, num_stambha]
        b_pos_clamped = b_pos.clamp(0, seq_len - 1)  # [batch, num_stambha]
        
        # Gather embeddings using advanced indexing
        # x is [batch, seq_len, hidden_size]
        # We need to gather at positions [batch, num_stambha] -> [batch, num_stambha, hidden_size]
        batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1, num_stambha)
        
        a_emb = x[batch_idx, a_pos_clamped]  # [batch, num_stambha, hidden_size]
        b_emb = x[batch_idx, b_pos_clamped]  # [batch, num_stambha, hidden_size]
        
        # Zero out invalid positions
        a_emb = a_emb * a_valid.unsqueeze(-1).float()
        b_emb = b_emb * b_valid.unsqueeze(-1).float()
        
        # Project combined embeddings to column dimension
        # stambha_proj: [hidden_size] -> [gulika_dim]
        combined_emb = (a_emb + b_emb).float()  # [batch, num_stambha, hidden_size]
        z = self.stambha_proj(combined_emb)     # [batch, num_stambha, gulika_dim]
        
        # Extract actual digit values and add digit value embeddings
        # This gives the model explicit numerical information
        # Digit tokens are 2-11, mapping to values 0-9
        a_tokens = inputs[batch_idx, a_pos_clamped]  # [batch, num_stambha]
        b_tokens = inputs[batch_idx, b_pos_clamped]  # [batch, num_stambha]
        
        # Convert tokens to digit values (0-9), use 10 for invalid/non-digit
        a_digit = torch.where(
            a_valid & (a_tokens >= DIGIT_OFFSET) & (a_tokens < DIGIT_OFFSET + 10),
            a_tokens - DIGIT_OFFSET,
            torch.full_like(a_tokens, 10)  # 10 = no digit
        )
        b_digit = torch.where(
            b_valid & (b_tokens >= DIGIT_OFFSET) & (b_tokens < DIGIT_OFFSET + 10),
            b_tokens - DIGIT_OFFSET,
            torch.full_like(b_tokens, 10)  # 10 = no digit
        )
        
        # Get digit value embeddings and add to stambha state
        a_digit_emb = self.digit_value_emb(a_digit)  # [batch, num_stambha, gulika_dim]
        b_digit_emb = self.digit_value_emb(b_digit)  # [batch, num_stambha, gulika_dim]
        
        # Combine token embeddings with digit value embeddings
        z = z + a_digit_emb + b_digit_emb
        
        # Add previous state if available
        if z_sthana is not None:
            z = z + z_sthana.float()
        
        # Compute valid column mask: columns that have at least one valid digit
        valid_columns = a_valid | b_valid  # [batch, num_stambha]
        
        # Process through stambha layers with hasta sanchara
        for layer in self.layers:
            z = layer['stambha'](z, valid_mask=valid_columns)
            z = layer['hasta'](z, valid_mask=valid_columns)
        
        # Digit predictions (per column) - only predict for valid columns
        digit_logits = self.digit_head(z)  # [batch, num_stambha, 10]
        # Mask invalid columns to prevent spurious predictions
        valid_columns_expanded = valid_columns.unsqueeze(-1).float()  # [batch, num_stambha, 1]
        digit_logits = digit_logits * valid_columns_expanded + (1 - valid_columns_expanded) * (-1e9)
        
        # Carry predictions (per column) - only predict for valid columns
        carry_logits = self.carry_head(z).squeeze(-1)  # [batch, num_stambha]
        # Mask invalid columns
        carry_logits = carry_logits * valid_columns.float() + (1 - valid_columns.float()) * (-1e9)
        
        # CRITICAL FIX: Remove global leakage - don't use global projection
        # Instead, return None for output (will be handled in TRM_Lilavati_Inner)
        # The column states are sufficient for per-column predictions
        output = None  # No global mixing
        
        # Convert output back to input dtype for compatibility
        z_out = z.to(x.dtype)
        
        return z_out, output, digit_logits, carry_logits


class TRM_Lilavati_Inner(nn.Module):
    """Inner model for TRM Lilavati with SthanaYantra."""
    
    def __init__(self, config: TRM_LilavatiConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        
        # Embeddings
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        
        self.embed_tokens = CastedEmbedding(
            config.vocab_size, config.hidden_size, 
            init_std=embed_init_std, cast_to=self.forward_dtype
        )
        
        # SthanaYantra (replaces NeuralAbacus)
        self.sthana_yantra = SthanaYantra(config)
        
        # Output heads
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        # Q-head: input from pooled column states (gulika_dim) instead of hidden_size
        self.q_head = CastedLinear(config.gulika_dim, 2, bias=True)
        
        # Initial column state (float32 for stability)
        self.sthana_init = nn.Parameter(
            trunc_normal_init_(
                torch.empty(config.num_stambha, config.gulika_dim, dtype=torch.float32), 
                std=1
            )
        )
        
        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)
    
    def _input_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute input embeddings."""
        embedding = self.embed_tokens(inputs.to(torch.int32))
        return self.embed_scale * embedding
    
    def empty_carry(self, batch_size: int) -> TRM_LilavatiInnerCarry:
        """Create empty carry state."""
        return TRM_LilavatiInnerCarry(
            z_sthana=torch.zeros(
                batch_size, self.config.num_stambha, self.config.gulika_dim, 
                dtype=torch.float32
            )
        )
    
    def reset_carry(
        self, 
        reset_flag: torch.Tensor, 
        carry: TRM_LilavatiInnerCarry
    ) -> TRM_LilavatiInnerCarry:
        """Reset carry for halted sequences."""
        return TRM_LilavatiInnerCarry(
            z_sthana=torch.where(
                reset_flag.view(-1, 1, 1), 
                self.sthana_init.unsqueeze(0).expand(carry.z_sthana.shape[0], -1, -1),
                carry.z_sthana
            )
        )
    
    def forward(
        self, 
        carry: TRM_LilavatiInnerCarry, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRM_LilavatiInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass with direct stambha-to-sequence output mapping.
        
        Returns:
            - new_carry: Updated carry state
            - output: LM logits [batch, seq_len, vocab_size]
            - q_logits: (q_halt, q_continue)
            - digit_logits: [batch, num_stambha, 10]
            - carry_logits: [batch, num_stambha]
        """
        inputs = batch["inputs"]
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        device = inputs.device
        
        # Input embedding
        input_embeddings = self._input_embeddings(inputs)
        
        # Process through SthanaYantra
        new_z_sthana, sthana_output, digit_logits, carry_logits = self.sthana_yantra(
            input_embeddings, inputs, carry.z_sthana
        )
        
        # Find result positions (after '=')
        eq_pos = (inputs == EQ_TOKEN).long().argmax(dim=1)  # [batch]
        
        # CRITICAL FIX: Remove global leakage - only use input embeddings for non-result positions
        # Result positions will be filled with column predictions only
        # Base output from input embeddings only (no global mixing)
        base_output = self.lm_head(input_embeddings)  # [batch, seq_len, vocab_size]
        
        # ALWAYS map digit_logits to output at result positions (both train and eval)
        # Compute result positions from inputs (don't require labels)
        num_stambha = self.config.num_stambha
        vocab_size = base_output.shape[-1]
        result_start = eq_pos + 1  # [batch]
        
        # Determine result length: use labels if available, otherwise estimate
        labels = batch.get("labels", None)
        if labels is not None:
            # Use labels to determine exact result length per batch
            seq_indices = torch.arange(seq_len, device=device).unsqueeze(0)
            result_mask_labels = (seq_indices >= result_start.unsqueeze(1)) & \
                                (labels != IGNORE_LABEL_ID) & (labels != 0)
            num_result_digits = result_mask_labels.sum(dim=1)  # [batch]
            max_result_len = min(num_result_digits.max().item(), num_stambha)
        else:
            # During eval: use MASK tokens to determine result length
            # The input has MASK tokens (ID=1) where the result should be
            MASK_ID = 1
            seq_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
            result_region = (seq_indices >= result_start.unsqueeze(1))  # [batch, seq_len]
            mask_tokens = (inputs == MASK_ID)  # [batch, seq_len]
            # Count consecutive MASK tokens after '='
            result_mask_count = (result_region & mask_tokens).sum(dim=1)  # [batch]
            # Fallback: if no masks, use conservative estimate (max 6 digits for 5-digit addition)
            # This assumes num_stambha is set appropriately (e.g., 8 for 5-digit addition)
            conservative_max = min(6, num_stambha)  # Max result digits = operand_digits + 1
            num_result_digits = torch.where(
                result_mask_count > 0,
                result_mask_count.clamp(1, num_stambha),
                torch.full_like(result_mask_count, conservative_max)
            )  # [batch]
            max_result_len = min(num_result_digits.max().item(), num_stambha)
        
        # Debug: ensure we always have valid result length
        if max_result_len <= 0:
            max_result_len = min(2, num_stambha)  # At least 2 digits for safety
        
        if max_result_len > 0:
            # Build digit output tensor
            digit_contributions = []
            
            for col in range(int(max_result_len)):
                seq_pos = result_start + col  # [batch]
                # CRITICAL FIX: In reversed format, result digits are written LSB first
                # For "31+52=83": result_start = 6 (after '='), so:
                #   col=0 → position 6 → stambha 0 (ones digit) → should predict digit 8
                #   col=1 → position 7 → stambha 1 (tens digit) → should predict digit 3
                # Direct mapping: col → stambha_idx (LSB-first means first col = ones = stambha 0)
                stambha_idx = torch.full((batch_size,), col, dtype=torch.long, device=device)  # [batch]
                
                # Validity check: position must be valid and stambha must exist
                # col is int, num_result_digits is [batch], so broadcast comparison
                valid = (col < num_result_digits) & (seq_pos < seq_len) & \
                       (stambha_idx >= 0) & (stambha_idx < num_stambha)
                
                if valid.any():
                    batch_idx = torch.arange(batch_size, device=device)
                    stambha_idx_clamped = stambha_idx.clamp(0, num_stambha - 1)
                    seq_pos_clamped = seq_pos.clamp(0, seq_len - 1)
                    
                    # Get digit logits: [batch, 10]
                    col_digit_logits = digit_logits[batch_idx, stambha_idx_clamped]
                    
                    # Create one-hot position encoding [batch, seq_len]
                    pos_onehot = F.one_hot(seq_pos_clamped, num_classes=seq_len).float()
                    pos_onehot = pos_onehot * valid.unsqueeze(-1).float()
                    
                    # Pad digit logits to vocab size [batch, vocab_size]
                    vocab_contrib = F.pad(
                        col_digit_logits.to(base_output.dtype), 
                        (DIGIT_OFFSET, vocab_size - DIGIT_OFFSET - 10),
                        value=0
                    )
                    
                    # Outer product: [batch, seq_len, vocab_size]
                    contribution = torch.einsum('bs,bv->bsv', pos_onehot, vocab_contrib)
                    digit_contributions.append(contribution)
            
            if digit_contributions:
                # Sum all contributions
                digit_output = sum(digit_contributions)
                
                # CRITICAL FIX: Create precise result mask - only exact result positions, not all after '='
                # This prevents masking PAD/MASK tokens which causes wrong predictions
                seq_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
                
                # Compute exact result positions per batch element
                # result_mask[i, j] = True if position j is a valid result digit for batch i
                result_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
                
                # Set mask for each valid result position
                for col in range(int(max_result_len)):
                    seq_pos = result_start + col  # [batch]
                    valid = (col < num_result_digits) & (seq_pos < seq_len)
                    
                    if valid.any():
                        # Set mask for valid positions
                        batch_idx = torch.arange(batch_size, device=device)[valid]
                        pos_idx = seq_pos[valid]
                        result_mask[batch_idx, pos_idx] = True
                
                # Refine mask with labels if available (for training - exclude PAD/MASK/IGNORE)
                labels = batch.get("labels", None)
                if labels is not None:
                    valid_labels = (labels != IGNORE_LABEL_ID) & (labels != 0) & \
                                  (labels >= DIGIT_OFFSET) & (labels < DIGIT_OFFSET + 10)
                    result_mask = result_mask & valid_labels
                
                result_mask_3d = result_mask.unsqueeze(-1).float()
                
                # CRITICAL FIX: Force model to learn from digit_logits, not memorize
                # Completely REPLACE base_output with digit_output at result positions
                # Scale digit_output to ensure it dominates (multiply by large factor to ensure argmax picks it)
                # This ensures column predictions are used exclusively
                digit_output_scaled = digit_output * 1.0  # Keep scale as 1.0 for now, but ensure it's used
                output = base_output * (1 - result_mask_3d) + digit_output_scaled * result_mask_3d
                
                if self.training:
                    # During training: block lm_head gradients at result positions
                    # Non-results: base_output (with gradients)
                    # Results: base_output.detach() (no gradients) + digit_output (with gradients)
                    # Reconstruct to ensure gradients flow correctly
                    output_base_grad = base_output * (1 - result_mask_3d)
                    output_result_grad = base_output.detach() * result_mask_3d + digit_output_scaled * result_mask_3d
                    output = output_base_grad + output_result_grad
            else:
                output = base_output
        else:
            output = base_output
        
        # Q-head: compute from column states instead of global output
        # Pool column states to get single representation
        # Average pooling over all columns (they're all processed, invalid ones are masked)
        pooled = new_z_sthana.mean(dim=1)  # [batch, gulika_dim]
        
        # Q-head now takes gulika_dim input directly (changed from hidden_size)
        q_logits = self.q_head(pooled.to(self.forward_dtype)).to(torch.float32)
        
        new_carry = TRM_LilavatiInnerCarry(z_sthana=new_z_sthana.detach())
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), digit_logits, carry_logits


class TRM_Lilavati(nn.Module):
    """
    TRM Lilavati: ACT wrapper for SthanaYantra model.
    
    Implements Bhaskara II's Lilavati addition method with proper
    place-value processing and carry propagation.
    """
    
    def __init__(self, config_dict: dict):
        super().__init__()
        # Handle both old and new config parameter names
        if 'num_columns' in config_dict and 'num_stambha' not in config_dict:
            config_dict['num_stambha'] = config_dict.pop('num_columns')
        if 'bead_dim' in config_dict and 'gulika_dim' not in config_dict:
            config_dict['gulika_dim'] = config_dict.pop('bead_dim')
        if 'abacus_layers' in config_dict and 'stambha_layers' not in config_dict:
            config_dict['stambha_layers'] = config_dict.pop('abacus_layers')
            
        self.config = TRM_LilavatiConfig(**config_dict)
        self.inner = TRM_Lilavati_Inner(self.config)
    
    @property
    def puzzle_emb(self):
        """Puzzle embeddings (not used in Lilavati, returns None)."""
        return None
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRM_LilavatiCarry:
        """Create initial carry state."""
        batch_size = batch["inputs"].shape[0]
        
        return TRM_LilavatiCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
    
    def forward(
        self, 
        carry: TRM_LilavatiCarry, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRM_LilavatiCarry, Dict[str, torch.Tensor]]:
        """
        Forward pass with ACT halting.
        
        Returns:
            - new_carry: Updated carry state
            - outputs: Dictionary with logits and predictions
        """
        # Reset carry for halted sequences
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        
        # Update current data for newly unhalted sequences
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), 
                batch[k], 
                v
            ) for k, v in carry.current_data.items()
        }
        
        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), digit_logits, carry_logits = \
            self.inner(new_inner_carry, new_current_data)
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "digit_logits": digit_logits,
            "carry_logits": carry_logits,
        }
        
        with torch.no_grad():
            # Update step count
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step
            
            # ACT halting during training
            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)
                
                # Exploration
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * 
                    torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)
        
        return TRM_LilavatiCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
