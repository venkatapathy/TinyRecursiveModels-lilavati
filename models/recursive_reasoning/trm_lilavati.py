"""
TRM Lilavati: Tiny Recursive Model with Neural Abacus architecture.

The Neural Abacus processes numbers in a columnar fashion, mimicking
how the Lilavati algorithm (and humans) perform digit-by-digit addition
with carry propagation.

Key features:
- Columnar representation: z_abacus[batch, num_columns, bead_dim]
- Local interactions via Conv1d (neighbors only)
- Carry sweep via GRU (left-to-right for reversed digits)
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


@dataclass
class TRM_LilavatiInnerCarry:
    """Carry state for Neural Abacus."""
    z_abacus: torch.Tensor  # [batch, num_columns, bead_dim]


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
    
    # Neural Abacus config
    num_columns: int = 25  # Number of digit positions
    bead_dim: int = 64  # Representation dimension per column
    abacus_layers: int = 4  # Number of abacus processing layers
    kernel_size: int = 3  # Local convolution kernel size
    
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


class AbacusLayer(nn.Module):
    """
    Single abacus processing layer.
    
    Combines local convolution (neighbor interaction) with MLP processing.
    All operations in float32 for stability with torch.compile.
    """
    
    def __init__(self, bead_dim: int, kernel_size: int = 3, expansion: float = 4):
        super().__init__()
        
        # Local convolution: each column sees itself + neighbors
        self.local_conv = nn.Conv1d(
            in_channels=bead_dim,
            out_channels=bead_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        
        # MLP for per-column processing
        self.mlp = SwiGLU(
            hidden_size=bead_dim,
            expansion=expansion,
        )
        
        self.norm_eps = 1e-5
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Process abacus state.
        
        Args:
            z: [batch, num_columns, bead_dim] - float32
        
        Returns:
            Updated z: [batch, num_columns, bead_dim] - float32
        """
        # Ensure float32 for all operations
        z = z.float()
        
        # Local convolution (transpose for Conv1d)
        z_t = z.transpose(1, 2)  # [batch, bead_dim, num_columns]
        z_conv = self.local_conv(z_t).transpose(1, 2)  # [batch, num_columns, bead_dim]
        z = rms_norm(z + z_conv, variance_epsilon=self.norm_eps)
        
        # MLP
        z = rms_norm(z + self.mlp(z), variance_epsilon=self.norm_eps)
        
        return z


class CarrySweep(nn.Module):
    """
    Carry propagation via GRU sweep.
    
    Processes columns left-to-right (since digits are reversed, this is LSB to MSB).
    All operations in float32 for stability.
    """
    
    def __init__(self, bead_dim: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(bead_dim, bead_dim)
        self.norm_eps = 1e-5
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Sweep carry information across columns.
        
        Args:
            z: [batch, num_columns, bead_dim] - float32
        
        Returns:
            Updated z with carry information propagated: [batch, num_columns, bead_dim] - float32
        """
        # Ensure float32
        z = z.float()
        batch, num_cols, bead_dim = z.shape
        device = z.device
        
        # Initialize carry state (float32)
        carry_state = torch.zeros(batch, bead_dim, device=device, dtype=torch.float32)
        
        outputs = []
        # Sweep left-to-right (LSB to MSB for reversed digits)
        for col in range(num_cols):
            col_input = z[:, col, :]
            # Update carry state with GRU (already float32)
            carry_state = self.gru_cell(col_input, carry_state)
            outputs.append(carry_state)
        
        result = torch.stack(outputs, dim=1)
        return rms_norm(z + result, variance_epsilon=self.norm_eps)


class NeuralAbacus(nn.Module):
    """
    Neural Abacus: Columnar processing for digit-by-digit arithmetic.
    """
    
    def __init__(self, config: TRM_LilavatiConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        
        # Input projection: from sequence to abacus columns
        self.input_proj = nn.Linear(config.hidden_size, config.num_columns * config.bead_dim)
        
        # Abacus layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'local': AbacusLayer(config.bead_dim, config.kernel_size, config.expansion),
                'sweep': CarrySweep(config.bead_dim),
            }) for _ in range(config.abacus_layers)
        ])
        
        # Output projection: from abacus to sequence
        self.output_proj = nn.Linear(config.num_columns * config.bead_dim, config.hidden_size)
        
        # Digit prediction head (per column)
        self.digit_head = nn.Linear(config.bead_dim, 10)
        
        # Carry prediction head (auxiliary)
        self.carry_head = nn.Linear(config.bead_dim, 1)
        
        self.norm_eps = config.rms_norm_eps
    
    def forward(
        self, 
        x: torch.Tensor, 
        z_abacus: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process input through Neural Abacus.
        
        Args:
            x: Input embeddings [batch, seq_len, hidden_size]
            z_abacus: Previous abacus state (optional) [batch, num_columns, bead_dim]
        
        Returns:
            - z_out: Updated abacus state [batch, num_columns, bead_dim]
            - output: Processed embeddings [batch, hidden_size]
            - digit_logits: Per-column digit predictions [batch, num_columns, 10]
            - carry_logits: Per-column carry predictions [batch, num_columns]
        """
        batch = x.shape[0]
        input_dtype = x.dtype
        
        # Pool sequence and project to abacus (convert to float32 for processing)
        x_pooled = x.mean(dim=1).float()  # [batch, hidden_size]
        z = self.input_proj(x_pooled)  # [batch, num_columns * bead_dim]
        z = z.view(batch, self.config.num_columns, self.config.bead_dim)
        
        # Add previous state if available (convert to float32)
        if z_abacus is not None:
            z = z + z_abacus.float()
        
        # Process through abacus layers (all in float32)
        for layer in self.layers:
            z = layer['local'](z)
            z = layer['sweep'](z)
        
        # Digit predictions (per column)
        digit_logits = self.digit_head(z)  # [batch, num_columns, 10]
        
        # Carry predictions (per column)
        carry_logits = self.carry_head(z).squeeze(-1)  # [batch, num_columns]
        
        # Project back to sequence space
        z_flat = z.view(batch, -1)  # [batch, num_columns * bead_dim]
        output = self.output_proj(z_flat)  # [batch, hidden_size]
        
        # Convert output back to input dtype for compatibility
        output = output.to(input_dtype)
        z_out = z.to(input_dtype)
        
        return z_out, output, digit_logits, carry_logits


class TRM_Lilavati_Inner(nn.Module):
    """Inner model for TRM Lilavati."""
    
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
        
        # Neural Abacus
        self.abacus = NeuralAbacus(config)
        
        # Output heads
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        
        # Initial abacus state (float32 for stability)
        self.abacus_init = nn.Parameter(
            trunc_normal_init_(
                torch.empty(config.num_columns, config.bead_dim, dtype=torch.float32), 
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
            z_abacus=torch.zeros(
                batch_size, self.config.num_columns, self.config.bead_dim, 
                dtype=torch.float32  # Use float32 for abacus state
            )
        )
    
    def reset_carry(
        self, 
        reset_flag: torch.Tensor, 
        carry: TRM_LilavatiInnerCarry
    ) -> TRM_LilavatiInnerCarry:
        """Reset carry for halted sequences."""
        return TRM_LilavatiInnerCarry(
            z_abacus=torch.where(
                reset_flag.view(-1, 1, 1), 
                self.abacus_init.unsqueeze(0).expand(carry.z_abacus.shape[0], -1, -1),
                carry.z_abacus
            )
        )
    
    def forward(
        self, 
        carry: TRM_LilavatiInnerCarry, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRM_LilavatiInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            - new_carry: Updated carry state
            - output: LM logits [batch, seq_len, vocab_size]
            - q_logits: (q_halt, q_continue)
            - digit_logits: [batch, num_columns, 10]
            - carry_logits: [batch, num_columns]
        """
        # Input embedding
        input_embeddings = self._input_embeddings(batch["inputs"])
        
        # Process through abacus
        new_z_abacus, abacus_output, digit_logits, carry_logits = self.abacus(
            input_embeddings, carry.z_abacus
        )
        
        # Broadcast abacus output to sequence length for LM head
        batch_size, seq_len, _ = input_embeddings.shape
        output_expanded = abacus_output.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine with input embeddings for final output
        combined = input_embeddings + output_expanded
        
        # LM output
        output = self.lm_head(combined)
        
        # Q-head (use pooled representation)
        q_logits = self.q_head(abacus_output).to(torch.float32)
        
        new_carry = TRM_LilavatiInnerCarry(z_abacus=new_z_abacus.detach())
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), digit_logits, carry_logits


class TRM_Lilavati(nn.Module):
    """
    TRM Lilavati: ACT wrapper for Neural Abacus model.
    """
    
    def __init__(self, config_dict: dict):
        super().__init__()
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
