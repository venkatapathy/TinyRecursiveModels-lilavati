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


CAR_TOKEN_ID = 14  # <CAR> token for addition lilavati modes
AVY_TOKEN_ID = 16  # <AVY> token for multiplication lilavati modes (legacy)
FACT_TOKEN_ID = 15  # <FACT> token for multiplication factorization


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
        # Single forward pass for all modes (including lilavati1 now)
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        labels_carry = new_carry.current_data.get("labels_carry", None)  # From lilavati dataset

        # For lilavati1/lilavati2/lilavati3/lilavati1_fact_only/lilavati2_fact_only: need to compute masks before predictions
        # lilavati1 now uses same format as lilavati3 (single sequence with CAR/AVY/FACT token)
        if self.dataset_mode in {"lilavati1", "lilavati2", "lilavati3", "lilavati1_fact_only", "lilavati2_fact_only"}:
            inputs = new_carry.current_data["inputs"]
            batch_size, seq_len = labels.shape
            positions = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, -1)
            
            # Detect special token: <CAR> (14) for addition and multiplication, <FACT> (15) for factorization
            # Also check for <AVY> (16) for backward compatibility with old multiplication datasets
            car_mask_inputs = (inputs == CAR_TOKEN_ID)  # <CAR> token (used for both addition and multiplication)
            fact_mask_inputs = (inputs == FACT_TOKEN_ID)  # <FACT> token (for multiplication factorization)
            avy_mask_inputs = (inputs == AVY_TOKEN_ID)  # <AVY> token (legacy, for old multiplication datasets)
            special_token_mask_inputs = car_mask_inputs | fact_mask_inputs | avy_mask_inputs
            
            car_mask_labels = (labels == CAR_TOKEN_ID)
            fact_mask_labels = (labels == FACT_TOKEN_ID)
            avy_mask_labels = (labels == AVY_TOKEN_ID)
            special_token_mask_labels = car_mask_labels | fact_mask_labels | avy_mask_labels
            
            # Create position indices (use whichever token is found: CAR, FACT, or AVY)
            # For FACT-only mode, some examples may not have a special token (vanilla format)
            # Use argmax with a sentinel value to detect when no token exists
            special_token_exists = special_token_mask_inputs.any(dim=-1, keepdim=True)  # [B, 1]
            car_positions = special_token_mask_inputs.float().argmax(dim=-1, keepdim=True)  # [B, 1]
            # If no token found, argmax returns 0, but we need to mark it as invalid
            # Set to a large value so positions > car_positions is False for all positions
            car_positions = torch.where(special_token_exists, car_positions, torch.full_like(car_positions, seq_len))
            
            # Carry/factorization mask: positions after special token (CAR, FACT, or AVY)
            # For examples without token, this will be all False
            carry_pos_mask = (positions > car_positions) & special_token_exists
            
            if self.dataset_mode in {"lilavati1", "lilavati2", "lilavati1_fact_only", "lilavati2_fact_only"}:
                # Lilavati1/lilavati2: use only lm_head for both result and carry positions
                combined_preds = torch.argmax(outputs["logits"], dim=-1)
            else:
                # Lilavati3: use lm_head for result positions, carry_head for carry positions
                lm_preds = torch.argmax(outputs["logits"], dim=-1)
                carry_preds = torch.argmax(outputs["carry_logits"], dim=-1)
                combined_preds = torch.where(carry_pos_mask, carry_preds, lm_preds)
            
            # For loss computation, use labels to find CAR mask
            car_mask = car_mask_labels
            car_positions = car_positions
            carry_pos_mask = carry_pos_mask
            # Store special_token_mask_inputs for use in loss computation (needed for FACT-only mode)
            special_token_mask_inputs_stored = special_token_mask_inputs
            # Store special_token_exists for use in loss computation
            special_token_exists_stored = special_token_exists
        else:
            combined_preds = None
            car_mask = None
            carry_pos_mask = None
            car_positions = None
            result_pos_mask = None
            eq_positions = None
            special_token_mask_inputs_stored = None
            special_token_exists_stored = None

        with torch.no_grad():
            # Preds - for lilavati2, use combined predictions
            if combined_preds is not None:
                outputs["preds"] = combined_preds
            else:
                outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)
            
            # For lilavati1/lilavati2: predictions come from lm_head only (no separate carry_preds)
            # For lilavati3: carry predictions come from carry_head
            if self.dataset_mode == "lilavati3" and "carry_logits" in outputs:
                carry_preds = torch.argmax(outputs["carry_logits"], dim=-1)
                outputs["carry_preds"] = carry_preds

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            # Metrics (halted)
            halted_for_metrics = new_carry.halted
            valid_metrics = halted_for_metrics & (loss_counts > 0)
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
                "steps":          torch.where(valid_metrics, (new_carry.steps[:len(halted_for_metrics)] if len(new_carry.steps) > len(halted_for_metrics) else new_carry.steps), 0).sum(),
            }

        # Losses
        
        # For lilavati1/lilavati1_fact_only: same format as lilavati3 but uses only lm_head for both result and carry/factorization
        if self.dataset_mode in {"lilavati1", "lilavati1_fact_only"} and car_positions is not None:
            if self.dataset_mode == "lilavati1_fact_only":
                # For FACT-only: examples without FACT token should be treated like vanilla
                # Check which examples have a special token
                has_special_token = special_token_mask_inputs_stored.any(dim=-1)  # [B]
                
                # Result mask: valid positions strictly before special token (CAR/FACT, excluding token itself)
                # For examples without token, use all valid positions (like vanilla)
                result_mask_with_token = mask & (positions < car_positions)
                result_mask_no_token = mask  # All valid positions for vanilla-format examples
                result_mask = torch.where(has_special_token.unsqueeze(-1), result_mask_with_token, result_mask_no_token)
                
                # Carry/factorization mask: valid positions after special token (CAR/FACT, excluding token itself)
                # For examples without token, this should be all False (no carry/factorization)
                # FACT token itself is excluded from loss (IGNORE_LABEL_ID in dataset)
                carry_mask = mask & carry_pos_mask
                
                # Compute result loss (using lm_head logits)
                result_counts = result_mask.sum(-1)
                result_divisor = result_counts.clamp_min(1).unsqueeze(-1)
                lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=result_mask) / result_divisor).sum()
                
                # Compute carry/factorization loss (only for examples with special token)
                carry_counts = carry_mask.sum(-1)
                carry_divisor = carry_counts.clamp_min(1).unsqueeze(-1)
                # Only compute loss for examples that have carry/factorization positions
                if carry_counts.sum() > 0:
                    carry_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=carry_mask) / carry_divisor).sum()
                else:
                    carry_loss = torch.tensor(0.0, device=labels.device, dtype=lm_loss.dtype)
                
                # Carry/factorization accuracy metrics
                with torch.no_grad():
                    if carry_counts.sum() > 0:
                        carry_correct = carry_mask & (outputs["preds"] == labels)
                        carry_accuracy = torch.where(valid_metrics, (carry_correct.to(torch.float32) / carry_divisor).sum(-1), 0).sum()
                    else:
                        carry_accuracy = torch.tensor(0.0, device=labels.device)
                    metrics["carry_accuracy"] = carry_accuracy
                    metrics["carry_loss"] = carry_loss.detach()
                
                # Total loss: L_y + λ * L_carry/factorization (both using lm_head)
                # For examples without token, carry_loss is 0, so it's just lm_loss (vanilla behavior)
                total_lm_loss = lm_loss + self.carry_loss_weight * carry_loss
            else:
                # Regular lilavati1: all examples have CAR token
                # Result mask: valid positions at or before special token (CAR/FACT, including token itself)
                result_mask = mask & (positions <= car_positions)
                # Carry/factorization mask: valid positions after special token (CAR/FACT)
                carry_mask = mask & carry_pos_mask
                
                # Compute result loss (using lm_head logits)
                result_counts = result_mask.sum(-1)
                result_divisor = result_counts.clamp_min(1).unsqueeze(-1)
                lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=result_mask) / result_divisor).sum()
                
                # Compute carry/factorization loss (using lm_head logits - KEY DIFFERENCE from lilavati3)
                carry_counts = carry_mask.sum(-1)
                carry_divisor = carry_counts.clamp_min(1).unsqueeze(-1)
                carry_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=carry_mask) / carry_divisor).sum()
                
                # Carry/factorization accuracy metrics
                with torch.no_grad():
                    carry_correct = carry_mask & (outputs["preds"] == labels)
                    carry_accuracy = torch.where(valid_metrics, (carry_correct.to(torch.float32) / carry_divisor).sum(-1), 0).sum()
                    metrics["carry_accuracy"] = carry_accuracy
                    metrics["carry_loss"] = carry_loss.detach()
                
                # Total loss: L_y + λ * L_carry/factorization (both using lm_head)
                total_lm_loss = lm_loss + self.carry_loss_weight * carry_loss
        
        # For lilavati2: single unified loss using only lm_head logits for entire sequence
        elif self.dataset_mode == "lilavati2" and car_positions is not None:
            # Compute single unified cross-entropy loss over entire sequence (result + CAR/FACT token + carry/factorization positions)
            # using only lm_head logits. Unlike lilavati1 which has separate losses, lilavati2 includes the special token
            # in the unified loss (same as addition lilavati2). The CAR/FACT token is in labels and included in mask.
            # Use same normalization as vanilla/lilavati1: divide by loss_divisor per sequence, then sum
            lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
            total_lm_loss = lm_loss  # For lilavati2, total loss is just the unified loss
            
            # For metrics: compute separate accuracies for result and carry positions
            # Result mask: positions strictly before CAR token (CAR token is excluded from result accuracy)
            result_mask = mask & (positions < car_positions)
            # Carry mask: positions after CAR token (CAR token is excluded from carry accuracy)
            carry_mask = mask & carry_pos_mask
            
            with torch.no_grad():
                # Result accuracy
                result_counts = result_mask.sum(-1)
                result_divisor = result_counts.clamp_min(1).unsqueeze(-1)
                result_correct = result_mask & (outputs["preds"] == labels)
                result_accuracy = torch.where(valid_metrics, (result_correct.to(torch.float32) / result_divisor).sum(-1), 0).sum()
                
                # Carry accuracy
                carry_counts = carry_mask.sum(-1)
                carry_divisor = carry_counts.clamp_min(1).unsqueeze(-1)
                carry_correct = carry_mask & (outputs["preds"] == labels)
                carry_accuracy = torch.where(valid_metrics, (carry_correct.to(torch.float32) / carry_divisor).sum(-1), 0).sum()
                
                metrics["carry_accuracy"] = carry_accuracy
                # Store result accuracy separately for clarity
                metrics["result_accuracy"] = result_accuracy
        
        # For lilavati2_fact_only: single unified loss using only lm_head logits for entire sequence
        # Similar to lilavati2, but handles FACT tokens and examples without FACT token (vanilla format)
        elif self.dataset_mode == "lilavati2_fact_only" and car_positions is not None:
            # For FACT-only: examples without FACT token should be treated like vanilla (unified loss on all positions)
            # Examples with FACT token: unified loss on entire sequence (result + FACT token + factorization positions)
            # Unlike lilavati1_fact_only, the FACT token is included in labels (for unified loss, like lilavati2)
            has_special_token = special_token_mask_inputs_stored.any(dim=-1)  # [B]
            
            # For examples with FACT token: unified loss on entire sequence (result + FACT token + factorization)
            # For examples without FACT token: unified loss on result positions only (like vanilla)
            # The FACT token is in labels (not IGNORE_LABEL_ID), so it's included in mask and unified loss
            unified_mask = mask  # All valid positions (including FACT token position if it exists)
            
            # Compute single unified cross-entropy loss over entire sequence
            lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=unified_mask) / loss_divisor).sum()
            total_lm_loss = lm_loss  # For lilavati2_fact_only, total loss is just the unified loss
            
            # For metrics: compute separate accuracies for result and factorization positions
            result_mask = mask & (positions < car_positions)  # Positions before FACT token (excluding FACT token)
            # For examples without FACT token, use all valid positions
            result_mask = torch.where(has_special_token.unsqueeze(-1), result_mask, mask)
            
            # Factorization mask: positions at or after FACT token (including FACT token itself)
            fact_mask = mask & (positions >= car_positions) & special_token_exists
            
            with torch.no_grad():
                # Result accuracy
                result_counts = result_mask.sum(-1)
                result_divisor = result_counts.clamp_min(1).unsqueeze(-1)
                result_correct = result_mask & (outputs["preds"] == labels)
                result_accuracy = torch.where(valid_metrics, (result_correct.to(torch.float32) / result_divisor).sum(-1), 0).sum()
                
                # Factorization accuracy (only for examples with FACT token)
                fact_counts = fact_mask.sum(-1)
                fact_divisor = fact_counts.clamp_min(1).unsqueeze(-1)
                fact_correct = fact_mask & (outputs["preds"] == labels)
                fact_accuracy = torch.where(valid_metrics & has_special_token.unsqueeze(-1), (fact_correct.to(torch.float32) / fact_divisor).sum(-1), 0).sum()
                
                # Store metrics (use carry_accuracy name for consistency, even though it's factorization)
                metrics["carry_accuracy"] = fact_accuracy
                metrics["result_accuracy"] = result_accuracy
        
        # For lilavati3: uses carry_head (separate from lilavati2 which uses only lm_head) - compute separate losses for result and carry positions
        elif self.dataset_mode == "lilavati3" and "carry_logits" in outputs:
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
            
            # Total loss: L_y + λ * L_carry
            total_lm_loss = lm_loss + self.carry_loss_weight * carry_loss
        
        else:
            # Vanilla / lilavati1 without labels_carry (evaluation): original behavior
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

        # Check if all sequences are halted
        all_finish = new_carry.halted.all()
        
        return new_carry, total_lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, all_finish

