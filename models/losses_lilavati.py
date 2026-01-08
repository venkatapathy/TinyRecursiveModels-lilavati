"""
Loss functions for TRM Lilavati model.

Extends the standard ACT loss with auxiliary carry and digit prediction losses.

Losses:
- LM loss: Cross-entropy on token predictions
- Q-halt loss: BCE on halting decision
- Carry loss: BCE on carry predictions (auxiliary)
- Digit loss: Cross-entropy on per-stambha digit predictions (new)
"""

from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.losses import IGNORE_LABEL_ID, stablemax_cross_entropy, softmax_cross_entropy

# Vocabulary constants (must match dataset)
EQ_TOKEN = 13      # '=' token
DIGIT_OFFSET = 2   # Digits 0-9 map to tokens 2-11


class ACTLossHeadLilavati(nn.Module):
    """
    ACT Loss Head for Lilavati model with auxiliary carry and digit supervision.
    
    Losses:
    - LM loss: Cross-entropy on token predictions
    - Q-halt loss: BCE on halting decision
    - Carry loss: BCE on carry predictions (auxiliary)
    - Digit loss: Cross-entropy on per-stambha digit predictions
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        loss_type: str = "stablemax_cross_entropy",
        carry_loss_weight: float = 0.1,
        digit_loss_weight: float = 0.1
    ):
        super().__init__()
        self.model = model
        self.loss_fn = stablemax_cross_entropy if loss_type == "stablemax_cross_entropy" else softmax_cross_entropy
        self.carry_loss_weight = carry_loss_weight
        self.digit_loss_weight = digit_loss_weight
    
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)
    
    def _extract_result_digits(
        self, 
        labels: torch.Tensor, 
        inputs: torch.Tensor, 
        num_stambha: int
    ) -> torch.Tensor:
        """
        Extract ground truth digits for each stambha (column).
        
        The result digits in the sequence are stored left-to-right (MSD first),
        but stambha 0 corresponds to the rightmost digit (LSB).
        
        Args:
            labels: [batch, seq_len] - ground truth labels
            inputs: [batch, seq_len] - input tokens
            num_stambha: Number of columns in the model
        
        Returns:
            targets: [batch, num_stambha] - digit targets (0-9) or -1 for padding
        """
        batch = labels.shape[0]
        device = labels.device
        
        # Find position of '=' in each sequence
        eq_pos = (inputs == EQ_TOKEN).long().argmax(dim=1)
        
        # Initialize targets with -1 (ignore)
        targets = torch.full((batch, num_stambha), -1, dtype=torch.long, device=device)
        
        for b in range(batch):
            result_start = eq_pos[b] + 1
            result_tokens = labels[b, result_start:]
            
            # Find valid result tokens (digits 0-9 are tokens 2-11)
            valid = (result_tokens != IGNORE_LABEL_ID) & (result_tokens >= DIGIT_OFFSET) & (result_tokens < DIGIT_OFFSET + 10)
            valid_indices = valid.nonzero(as_tuple=True)[0]
            
            if len(valid_indices) == 0:
                continue
            
            # Get the valid result digits (convert from vocab to digit)
            result_digits = result_tokens[valid_indices] - DIGIT_OFFSET
            num_result_digits = len(result_digits)
            
            # Reverse to match stambha order (LSB first)
            # stambha 0 = rightmost digit = result_digits[-1]
            for i in range(num_result_digits):
                if i < num_stambha:
                    # Reverse index: rightmost digit goes to stambha 0
                    targets[b, i] = result_digits[num_result_digits - 1 - i]
        
        return targets
    
    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Forward pass with loss computation.
        
        Returns:
            - new_carry: Updated carry state
            - loss: Total loss
            - metrics: Dictionary of metrics
            - detached_outputs: Requested outputs (detached)
            - all_halted: Whether all sequences have halted
        """
        # Forward through model
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        inputs = new_carry.current_data["inputs"]
        
        # Get carries if available (for auxiliary loss)
        carries = new_carry.current_data.get("carries", None)
        
        with torch.no_grad():
            # Predictions
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)
            
            # Correctness computation
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            
            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (for halted sequences only)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics, 
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1), 
                    0
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }
        
        # === Losses ===
        
        # 1. LM loss
        lm_loss = (
            self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) 
            / loss_divisor
        ).sum()
        
        # 2. Q-halt loss
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], 
            seq_is_correct.to(outputs["q_halt_logits"].dtype), 
            reduction="sum"
        )
        
        # 3. Carry loss (auxiliary) - only if carries are provided
        carry_loss = torch.tensor(0.0, device=lm_loss.device)
        if carries is not None and "carry_logits" in outputs:
            carry_logits = outputs["carry_logits"]  # [batch, num_stambha]
            carries_float = carries.to(carry_logits.dtype)
            
            # Only compute loss for valid carry positions
            num_stambha = carry_logits.shape[1]
            carries_truncated = carries_float[:, :num_stambha]
            
            carry_loss = F.binary_cross_entropy_with_logits(
                carry_logits, 
                carries_truncated, 
                reduction="sum"
            )
            
            # Carry accuracy metric
            with torch.no_grad():
                carry_preds = (carry_logits > 0).float()
                carry_correct = (carry_preds == carries_truncated).float().mean()
                metrics["carry_accuracy"] = carry_correct * valid_metrics.sum()
        
        # 4. Digit loss (per-stambha supervision)
        digit_loss = torch.tensor(0.0, device=lm_loss.device)
        if "digit_logits" in outputs:
            digit_logits = outputs["digit_logits"]  # [batch, num_stambha, 10]
            num_stambha = digit_logits.shape[1]
            
            # Extract ground truth result digits for each stambha
            digit_targets = self._extract_result_digits(labels, inputs, num_stambha)
            
            # Only compute loss for valid digits (not -1)
            valid_digits = (digit_targets >= 0)
            
            if valid_digits.any():
                # Flatten for cross entropy
                valid_logits = digit_logits[valid_digits]  # [num_valid, 10]
                valid_targets = digit_targets[valid_digits]  # [num_valid]
                
                digit_loss = F.cross_entropy(
                    valid_logits,
                    valid_targets,
                    reduction="sum"
                )
                
                # Digit accuracy metric
                with torch.no_grad():
                    digit_preds = digit_logits.argmax(dim=-1)  # [batch, num_stambha]
                    digit_correct = (digit_preds == digit_targets) & valid_digits
                    digit_acc = digit_correct.sum().float() / valid_digits.sum().float()
                    metrics["digit_accuracy"] = digit_acc * valid_metrics.sum()
        
        # Update metrics
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "carry_loss": carry_loss.detach(),
            "digit_loss": digit_loss.detach(),
        })
        
        # Q-continue loss (if using bootstrapping)
        q_continue_loss = torch.tensor(0.0, device=lm_loss.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], 
                outputs["target_q_continue"], 
                reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()
        
        # Total loss
        total_loss = (
            lm_loss + 
            0.5 * (q_halt_loss + q_continue_loss) + 
            self.carry_loss_weight * carry_loss +
            self.digit_loss_weight * digit_loss
        )
        
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
