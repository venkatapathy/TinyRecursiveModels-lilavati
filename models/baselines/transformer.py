from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
from torch import nn
from pydantic import BaseModel
from models.layers import (
    rms_norm, 
    SwiGLU, 
    Attention, 
    RotaryEmbedding, 
    CosSin, 
    CastedEmbedding, 
    CastedLinear
)

@dataclass
class TransformerCarry:
    halted: torch.Tensor
    steps: torch.Tensor
    current_data: Dict[str, torch.Tensor]

class StandardTransformerConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int = 0
    
    num_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str = "rope"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"
    
    # Dataset mode for compatibility
    dataset_mode: str = "vanilla"
    digits: int = 8

class TransformerBlock(nn.Module):
    def __init__(self, config: StandardTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self Attention + Norm
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), 
            variance_epsilon=self.norm_eps
        )
        # MLP + Norm
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states), 
            variance_epsilon=self.norm_eps
        )
        return hidden_states

class StandardTransformer(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = StandardTransformerConfig(**config_dict)
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        
        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, 
            self.config.hidden_size, 
            init_std=embed_init_std, 
            cast_to=self.forward_dtype
        )
        
        # Consistent with TRM's decoder head logic
        self.lm_head = CastedLinear(self.config.hidden_size * 2, self.config.vocab_size, bias=False)
        
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len,
                base=self.config.rope_theta
            )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(self.config) for _ in range(self.config.num_layers)
        ])
        
        # Initial state for Z_L (TRM uses two states, we'll use Z_H and Z_L for compatibility)
        self.z_l_init = nn.Parameter(torch.randn(self.config.hidden_size) * 0.02)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return TransformerCarry(
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: TransformerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TransformerCarry, Dict[str, torch.Tensor]]:
        # Update current data
        current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}
        
        device = batch["inputs"].device
        inputs = current_data["inputs"]
        batch_size, seq_len = inputs.shape
        
        # Embed
        x = self.embed_tokens(inputs.to(torch.int32)) * self.embed_scale
        
        # RoPE
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        
        # Blocks
        for block in self.blocks:
            x = block(cos_sin, x)
            
        # Standard transformer doesn't have a "halt" state in the same way, 
        # but the evaluator expects q_halt_logits. 
        # We'll just return high logits for "halt" so it always finishes in 1 step.
        q_halt_logits = torch.full((batch_size,), 10.0, device=device, dtype=torch.float32)
        q_continue_logits = torch.full((batch_size,), -10.0, device=device, dtype=torch.float32)
        
        # For compatibility with TRM heads, we might need a "second state"
        # TRM uses concat([z_H, z_L]). We'll just use x and a constant init for the other half.
        z_l = self.z_l_init.view(1, 1, -1).expand(batch_size, seq_len, -1).to(x.dtype)
        logits = self.lm_head(torch.cat([x, z_l], dim=-1))
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        # Always halted after one step
        new_carry = TransformerCarry(
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            steps=carry.steps + 1,
            current_data=current_data
        )
        
        return new_carry, outputs
