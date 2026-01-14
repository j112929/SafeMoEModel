"""
SafeMoE Support for ORPO (Odds Ratio Preference Optimization) and SimPO (Simple Preference Optimization)
Compatible with Hugging Face TRL library.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class PreferenceConfig:
    """Config for Preference Optimization (DPO/ORPO/SimPO)."""
    method: str = "dpo"         # dpo, orpo, simpo
    beta: float = 0.1           # Beta (temperature) for DPO/SimPO
    lambda_orpo: float = 0.1    # Lambda weight for ORPO odds ratio loss
    simpo_gamma: float = 0.5    # Target margin for SimPO
    label_smoothing: float = 0.0

class PreferenceLoss(nn.Module):
    """
    Unified Preference Loss Module supporting:
    1. DPO (Direct Preference Optimization) - Rafailov et al. 2023
    2. ORPO (Odds Ratio Preference Optimization) - Hong et al. 2024
    3. SimPO (Simple Preference Optimization) - Meng et al. 2024
    """
    def __init__(self, cfg: PreferenceConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: Optional[torch.Tensor] = None,
        ref_rejected_logps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        losses = {}
        
        # 1. DPO Loss
        if self.cfg.method == "dpo":
            assert ref_chosen_logps is not None and ref_rejected_logps is not None
            
            # log(pi_theta / pi_ref)
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            
            logits = pi_logratios - ref_logratios
            
            # Loss = -log sigmoid( beta * logits )
            loss = -F.logsigmoid(self.cfg.beta * logits) * (1 - self.cfg.label_smoothing) \
                   - F.logsigmoid(-self.cfg.beta * logits) * self.cfg.label_smoothing
            
            chosen_rewards = self.cfg.beta * (policy_chosen_logps - ref_chosen_logps).detach()
            rejected_rewards = self.cfg.beta * (policy_rejected_logps - ref_rejected_logps).detach()
            
            losses["dpo_loss"] = loss.mean()
            losses["reward_margin"] = (chosen_rewards - rejected_rewards).mean()
            return loss.mean(), losses

        # 2. SimPO Loss (No Reference Model needed!)
        elif self.cfg.method == "simpo":
            # pi_chosen - pi_rejected > gamma
            logits = (policy_chosen_logps - policy_rejected_logps) - self.cfg.simpo_gamma
            
            loss = -F.logsigmoid(self.cfg.beta * logits)
            
            losses["simpo_loss"] = loss.mean()
            losses["logp_margin"] = (policy_chosen_logps - policy_rejected_logps).mean().detach()
            return loss.mean(), losses
            
        # 3. ORPO (Usually strictly auxiliary to SFT, but here implemented as log-odds term)
        # ORPO is officially: Loss_SFT + lambda * Loss_Odds
        # This module only calculates the Odds Ratio part.
        elif self.cfg.method == "orpo":
            # OR = odds_chosen / odds_rejected
            # log(OR) = log(sigmoid(logp_chosen)) - log(1-sigmoid(logp_chosen)) ... simplified:
            # log_odds = log(p/(1-p)) = logit(p). But here we work with log probs.
            
            # Official ORPO implementation:
            # log_odds = log(p) - log(1 - p)
            # log_odds_ratio = log_odds_chosen - log_odds_rejected
            
            # Helper to compute log odds from logs
            def get_log_odds(log_p):
                # log(p / (1-p)) = log_p - log(1 - exp(log_p))
                return log_p - torch.log1p(-torch.exp(log_p) + 1e-6)
            
            log_odds_chosen = get_log_odds(policy_chosen_logps)
            log_odds_rejected = get_log_odds(policy_rejected_logps)
            
            ratios = log_odds_chosen - log_odds_rejected
            
            # Loss = -log sigmoid(ratios)
            loss = -F.logsigmoid(ratios)
            
            losses["orpo_odds_loss"] = loss.mean()
            losses["log_odds_margin"] = ratios.mean().detach()
            
            # Weighted by lambda (caller must handle adding SFT loss)
            return self.cfg.lambda_orpo * loss.mean(), losses

        else:
            raise ValueError(f"Unknown preference method: {self.cfg.method}")

