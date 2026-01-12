from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


CAR_TOKEN_ID = 14  # <CAR> token for lilavati modes


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, dataset_mode: str = "vanilla", digits: int = 3, carry_loss_weight: float = 1.0):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.dataset_mode = dataset_mode
        self.digits = digits
        self.carry_loss_weight = carry_loss_weight
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # For lilavati2/lilavati3: need to compute masks before predictions
        if self.dataset_mode in {"lilavati2", "lilavati3"} and "carry_logits" in outputs:
            # Find CAR token positions from INPUTS (not labels, as labels may be dummy during inference)
            inputs = new_carry.current_data["inputs"]
            car_mask_inputs = (inputs == CAR_TOKEN_ID)
            car_mask_labels = (labels == CAR_TOKEN_ID)
            
            # Create position indices
            batch_size, seq_len = labels.shape
            positions = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, -1)
            
            # Find CAR position from inputs (where CAR token appears)
            car_positions = car_mask_inputs.float().argmax(dim=-1, keepdim=True)  # [B, 1]
            
            # Carry mask: positions after CAR
            carry_pos_mask = (positions > car_positions)
            
            # Combine predictions: use lm_head for result positions, carry_head for carry positions
            lm_preds = torch.argmax(outputs["logits"], dim=-1)
            carry_preds = torch.argmax(outputs["carry_logits"], dim=-1)
            combined_preds = torch.where(carry_pos_mask, carry_preds, lm_preds)
            
            # For loss computation, use labels to find CAR mask
            car_mask = car_mask_labels
        else:
            combined_preds = None
            car_mask = None
            carry_pos_mask = None
            car_positions = None

        with torch.no_grad():
            # Preds - for lilavati2, use combined predictions
            if combined_preds is not None:
                outputs["preds"] = combined_preds
            else:
                outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                # Digit-level accuracy (average over positions)
                "digit_accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                # Sequence-level accuracy (all positions correct)
                "sequence_accuracy": (valid_metrics & seq_is_correct).sum(),
                
                # Keep old names for backward compatibility
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        
        # For lilavati2/lilavati3: compute separate losses for result and carry positions
        if self.dataset_mode in {"lilavati2", "lilavati3"} and "carry_logits" in outputs:
            # Result mask: valid positions at or before CAR (including CAR itself, since lm_head predicts it)
            result_mask = mask & (positions <= car_positions)
            # Carry mask: valid positions after CAR
            carry_mask = mask & carry_pos_mask
            
            # Compute result loss (using lm_head logits)
            result_counts = result_mask.sum(-1)
            result_divisor = result_counts.clamp_min(1).unsqueeze(-1)
            lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=result_mask) / result_divisor).sum()
            
            # Compute carry loss (using carry_head logits)
            carry_counts = carry_mask.sum(-1)
            carry_divisor = carry_counts.clamp_min(1).unsqueeze(-1)
            carry_loss = (self.loss_fn(outputs["carry_logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=carry_mask) / carry_divisor).sum()
            
            # Carry accuracy metrics
            with torch.no_grad():
                carry_correct = carry_mask & (outputs["preds"] == labels)
                carry_accuracy = torch.where(valid_metrics, (carry_correct.to(torch.float32) / carry_divisor).sum(-1), 0).sum()
                metrics["carry_accuracy"] = carry_accuracy
                metrics["carry_loss"] = carry_loss.detach()
            
            total_lm_loss = lm_loss + self.carry_loss_weight * carry_loss
        else:
            # Vanilla / lilavati1: original behavior
            lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
            total_lm_loss = lm_loss

        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()

