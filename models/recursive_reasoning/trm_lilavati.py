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
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Process column state.
        
        Args:
            z: [batch, num_stambha, gulika_dim] - float32
        
        Returns:
            Updated z: [batch, num_stambha, gulika_dim] - float32
        """
        # Ensure float32 for all operations
        z = z.float()
        
        # Local convolution (transpose for Conv1d)
        z_t = z.transpose(1, 2)  # [batch, gulika_dim, num_stambha]
        z_conv = self.local_conv(z_t).transpose(1, 2)  # [batch, num_stambha, gulika_dim]
        z = rms_norm(z + z_conv, variance_epsilon=self.norm_eps)
        
        # MLP
        z = rms_norm(z + self.mlp(z), variance_epsilon=self.norm_eps)
        
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
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Sweep carry information across columns.
        
        Args:
            z: [batch, num_stambha, gulika_dim] - float32
        
        Returns:
            Updated z with carry information propagated: [batch, num_stambha, gulika_dim] - float32
        """
        # Ensure float32
        z = z.float()
        batch, num_stambha, gulika_dim = z.shape
        device = z.device
        
        # Initialize carry state (float32)
        carry_state = torch.zeros(batch, gulika_dim, device=device, dtype=torch.float32)
        
        outputs = []
        # Sweep left-to-right (LSB to MSB)
        for col in range(num_stambha):
            col_input = z[:, col, :]
            # Update carry state with GRU
            carry_state = self.gru_cell(col_input, carry_state)
            outputs.append(carry_state)
        
        result = torch.stack(outputs, dim=1)
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
        
        # Per-column projection: from hidden_size to gulika_dim
        self.stambha_proj = nn.Linear(config.hidden_size, config.gulika_dim)
        
        # Stambha layers with HastaSanchara
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'stambha': StambhaLayer(config.gulika_dim, config.kernel_size, config.expansion),
                'hasta': HastaSanchara(config.gulika_dim),
            }) for _ in range(config.stambha_layers)
        ])
        
        # Output projection: from all columns to sequence space
        self.output_proj = nn.Linear(config.num_stambha * config.gulika_dim, config.hidden_size)
        
        # Digit prediction head (per column) - predicts 0-9
        self.digit_head = nn.Linear(config.gulika_dim, 10)
        
        # Carry prediction head (auxiliary)
        self.carry_head = nn.Linear(config.gulika_dim, 1)
        
        self.norm_eps = config.rms_norm_eps
    
    def _gather_masked(
        self, 
        x: torch.Tensor, 
        positions: torch.Tensor, 
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Gather embeddings at positions, zeroing invalid ones.
        
        Args:
            x: [batch, seq_len, hidden_size]
            positions: [batch] - positions to gather from
            valid_mask: [batch] - boolean mask for valid positions
        
        Returns:
            gathered: [batch, hidden_size] - embeddings at positions, zeroed if invalid
        """
        batch = x.shape[0]
        # Clamp to valid range to avoid index errors
        positions_clamped = positions.clamp(0, x.shape[1] - 1)
        # Gather embeddings
        gathered = x[torch.arange(batch, device=x.device), positions_clamped]
        # Zero out invalid positions
        return gathered * valid_mask.unsqueeze(-1).float()
    
    def forward(
        self, 
        x: torch.Tensor,
        inputs: torch.Tensor,
        z_sthana: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process input through SthanaYantra with proper digit extraction.
        
        Args:
            x: Input embeddings [batch, seq_len, hidden_size]
            inputs: Raw input tokens [batch, seq_len]
            z_sthana: Previous column state (optional) [batch, num_stambha, gulika_dim]
        
        Returns:
            - z_out: Updated column state [batch, num_stambha, gulika_dim]
            - output: Processed embeddings [batch, hidden_size]
            - digit_logits: Per-column digit predictions [batch, num_stambha, 10]
            - carry_logits: Per-column carry predictions [batch, num_stambha]
        """
        batch = x.shape[0]
        device = x.device
        
        # Parse input to find operator positions
        plus_pos = (inputs == PLUS_TOKEN).long().argmax(dim=1)  # Position of '+'
        eq_pos = (inputs == EQ_TOKEN).long().argmax(dim=1)      # Position of '='
        
        # Initialize each stambha (column) with its digit pair
        z = torch.zeros(batch, self.config.num_stambha, self.config.gulika_dim, 
                        device=device, dtype=torch.float32)
        
        for stambha in range(self.config.num_stambha):
            # Reversed indexing: stambha 0 = rightmost digit (LSB)
            # A's digit at position (plus_pos - 1 - stambha)
            # B's digit at position (eq_pos - 1 - stambha)
            a_pos = plus_pos - 1 - stambha
            b_pos = eq_pos - 1 - stambha
            
            # Bounds check
            a_valid = (a_pos >= 0)
            b_valid = (b_pos > plus_pos)  # B starts after '+'
            
            # Gather embeddings with masking
            a_emb = self._gather_masked(x, a_pos, a_valid)
            b_emb = self._gather_masked(x, b_pos, b_valid)
            
            # Project combined embedding to column dimension
            z[:, stambha, :] = self.stambha_proj((a_emb + b_emb).float())
        
        # Add previous state if available
        if z_sthana is not None:
            z = z + z_sthana.float()
        
        # Process through stambha layers with hasta sanchara
        for layer in self.layers:
            z = layer['stambha'](z)
            z = layer['hasta'](z)
        
        # Digit predictions (per column)
        digit_logits = self.digit_head(z)  # [batch, num_stambha, 10]
        
        # Carry predictions (per column)
        carry_logits = self.carry_head(z).squeeze(-1)  # [batch, num_stambha]
        
        # Project back to sequence space
        z_flat = z.view(batch, -1)  # [batch, num_stambha * gulika_dim]
        output = self.output_proj(z_flat)  # [batch, hidden_size]
        
        # Convert output back to input dtype for compatibility
        output = output.to(x.dtype)
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
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        
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
        eq_pos = (inputs == EQ_TOKEN).long().argmax(dim=1)
        
        # Build output logits - start with base LM output
        output_expanded = sthana_output.unsqueeze(1).expand(-1, seq_len, -1)
        combined = input_embeddings + output_expanded
        output = self.lm_head(combined)
        
        # Direct stambha-to-sequence mapping for result positions
        # Override result positions with column digit predictions
        labels = batch.get("labels", None)
        if labels is not None:
            # Count result digits from labels
            for b in range(batch_size):
                result_start = eq_pos[b] + 1
                # Find how many result digits there are
                result_mask = (labels[b, result_start:] != IGNORE_LABEL_ID) & (labels[b, result_start:] != 0)
                num_result_digits = result_mask.sum().item()
                
                # Map each result position to its corresponding stambha
                for col in range(int(num_result_digits)):
                    seq_pos = result_start + col
                    # Reverse mapping: leftmost result digit = highest stambha index
                    stambha_idx = int(num_result_digits) - 1 - col
                    if stambha_idx < self.config.num_stambha and seq_pos < seq_len:
                        # Map digit logits (0-9) to vocab positions (2-11)
                        output[b, seq_pos, DIGIT_OFFSET:DIGIT_OFFSET+10] = digit_logits[b, stambha_idx, :]
        
        # Q-head (use pooled representation)
        q_logits = self.q_head(sthana_output).to(torch.float32)
        
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
