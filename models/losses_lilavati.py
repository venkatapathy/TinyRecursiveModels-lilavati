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
            
            # CRITICAL FIX: In reversed format, result digits are already LSB-first in sequence
            # Position 7 (first result) = LSB = stambha 0
            # Position 8 = tens = stambha 1
            # Position 9 (last result) = MSB = stambha 2
            # So mapping is DIRECT (no reversal needed)
            for i in range(num_result_digits):
                if i < num_stambha:
                    # Direct mapping: position i in sequence → stambha i
                    targets[b, i] = result_digits[i]
        
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
        # NOTE: In ACT, loss is computed on ALL sequences at each step, not just halted ones.
        # This allows learning from all reasoning steps, not just the final halted state.
        # Metrics are computed only on halted sequences (see valid_metrics above).
        
        # Identify result positions (answer digits after '=') - these are CRITICAL
        eq_pos = (inputs == EQ_TOKEN).long().argmax(dim=1)  # [batch]
        seq_len = labels.shape[1]
        seq_indices = torch.arange(seq_len, device=labels.device).unsqueeze(0)  # [1, seq_len]
        result_mask = (seq_indices >= (eq_pos + 1).unsqueeze(1)) & mask  # [batch, seq_len]
        
        # 1. LM loss - computed on all sequences
        # CRITICAL: Weight result positions (answer digits) more heavily
        # Result positions are 5x more important than other positions
        per_token_loss = self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)
        
        # Create position weights: 5.0 for result positions, 1.0 for others
        position_weights = torch.where(result_mask, 5.0, 1.0)
        weighted_loss = per_token_loss * position_weights
        
        # Normalize by weighted token count to keep scale consistent
        weighted_token_count = (mask.float() * position_weights).sum().clamp_min(1.0)
        lm_loss = weighted_loss.sum() / weighted_token_count
        
        # 2. Q-halt loss - computed on all sequences
        # FIXED: Use confidence-based halting instead of binary correctness
        batch_size = outputs["q_halt_logits"].shape[0]
        
        # Compute confidence from digit logits (for valid sequences)
        if "digit_logits" in outputs:
            digit_logits = outputs["digit_logits"]  # [batch, num_stambha, 10]
            digit_probs = F.softmax(digit_logits, dim=-1)  # [batch, num_stambha, 10]
            max_probs = digit_probs.max(dim=-1)[0]  # [batch, num_stambha] - confidence per column
            # Average confidence across valid columns
            # Mask out invalid predictions (very low logits from masking)
            valid_mask = (digit_logits.max(dim=-1)[0] > -1e8).float()  # [batch, num_stambha]
            num_valid_cols = valid_mask.sum(dim=-1).clamp_min(1.0)  # [batch]
            confidence = (max_probs * valid_mask).sum(dim=-1) / num_valid_cols  # [batch]
        else:
            # Fallback: use uniform confidence if digit_logits not available
            confidence = torch.ones(batch_size, device=outputs["q_halt_logits"].device) * 0.5
        
        # Target: halt when confidence > threshold (0.8)
        confidence_threshold = 0.8
        q_halt_targets = (confidence > confidence_threshold).float()
        
        # Still allow binary correctness as additional signal, but weight by confidence
        correctness_weight = seq_is_correct.float()
        confidence_weight = confidence
        
        # Combined target: prefer halting when both correct and confident
        # But also allow halting when very confident (model is sure, even if wrong initially)
        q_halt_targets = torch.where(
            correctness_weight > 0.5,  # If correct
            torch.ones_like(q_halt_targets),  # Definitely halt
            q_halt_targets  # Otherwise use confidence-based halting
        )
        
        q_halt_loss_per_seq = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], 
            q_halt_targets, 
            reduction="none"
        )  # [batch]
        
        # Weight by confidence: more confident predictions should have stronger halting signal
        confidence_weights = 0.5 + 0.5 * confidence  # Between 0.5 and 1.0
        q_halt_loss = (q_halt_loss_per_seq * confidence_weights).sum() / max(batch_size, 1)
        
        # Track confidence in metrics
        metrics["avg_confidence"] = confidence.mean().detach()
        
        # 3. Carry loss (auxiliary) - only if carries are provided
        carry_loss = torch.tensor(0.0, device=lm_loss.device)
        if carries is not None and "carry_logits" in outputs:
            carry_logits = outputs["carry_logits"]  # [batch, num_stambha]
            carries_float = carries.to(carry_logits.dtype)
            
            # Pad or truncate carries to match num_stambha
            num_stambha = carry_logits.shape[1]
            carry_len = carries_float.shape[1] if carries_float.ndim > 1 else 1
            
            if carry_len < num_stambha:
                # Pad with zeros
                padding = torch.zeros(
                    carries_float.shape[0], 
                    num_stambha - carry_len, 
                    dtype=carries_float.dtype, 
                    device=carries_float.device
                )
                carries_padded = torch.cat([carries_float, padding], dim=1)
            elif carry_len > num_stambha:
                # Truncate
                carries_padded = carries_float[:, :num_stambha]
            else:
                carries_padded = carries_float
            
            # Normalize by number of valid carry positions (non-negative)
            carry_loss_per_pos = F.binary_cross_entropy_with_logits(
                carry_logits, 
                carries_padded, 
                reduction="none"
            )  # [batch, num_stambha]
            # Only count non-negative carries as valid (though carries should be 0 or 1, but be safe)
            num_valid_carries = (carries_padded >= 0).sum().float().clamp_min(1.0)
            carry_loss = carry_loss_per_pos.sum() / num_valid_carries
            
            # Carry accuracy metric
            with torch.no_grad():
                carry_preds = (carry_logits > 0).float()
                carry_correct = (carry_preds == carries_padded).float().mean()
                metrics["carry_accuracy"] = carry_correct * valid_metrics.sum()
        
        # 4. Digit loss (per-stambha supervision)
        # CRITICAL: This is the most important auxiliary loss - it directly supervises answer digits
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
                
                # Normalize by number of valid digits
                digit_loss = F.cross_entropy(
                    valid_logits,
                    valid_targets,
                    reduction="sum"
                ) / valid_digits.sum().float().clamp_min(1.0)
                
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
        
        # Step-based regularization: ACT-style ponder cost
        # Encourage early halting when confident (standard ACT formulation)
        # Penalty: small cost per step to encourage efficiency
        ponder_cost = torch.tensor(0.0, device=lm_loss.device)
        if batch_size > 0:
            # Average steps across all sequences (not just halted)
            avg_steps = new_carry.steps.float().mean()
            # Small cost per step (standard ACT: encourage early halting)
            ponder_cost_per_step = 0.001  # Small penalty to encourage efficiency
            ponder_cost = ponder_cost_per_step * avg_steps * batch_size
            
            metrics["avg_steps"] = avg_steps.detach()
            metrics["ponder_cost"] = ponder_cost.detach()
        
        # Total loss with increased auxiliary supervision
        # CRITICAL: Increased weights force the model to learn proper reasoning, not shortcuts
        # Digit loss is most critical - it directly supervises the answer digits
        # Q-halt loss uses confidence-based signal (not binary correctness)
        total_loss = (
            lm_loss + 
            0.5 * (q_halt_loss + q_continue_loss) + 
            self.carry_loss_weight * carry_loss +
            self.digit_loss_weight * digit_loss +
            ponder_cost  # Ponder cost instead of step penalty
        )
        
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
