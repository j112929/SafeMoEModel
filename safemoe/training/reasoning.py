"""
Reasoning & RL Module for SafeMoE.
Implements GRPO (Group Relative Policy Optimization) and Verifier utilities for Reasoning tasks (CoT).

References:
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

@dataclass
class ReasoningConfig:
    """Configuration for Reasoning/RL training."""
    method: str = "grpo"        # grpo
    group_size: int = 16        # Number of samples per prompt (G)
    clip_epsilon: float = 0.2   # PPO clip parameter
    kl_beta: float = 0.04       # KL divergence coefficient
    entropy_beta: float = 0.01  # Entropy regularization coefficient
    
class VerifierHead(nn.Module):
    """
    A scalar head for Reward Modeling (ORM) or Process Reward Modeling (PRM).
    Usually attached to the last hidden state of the Transformer.
    """
    def __init__(self, d_model: int, hidden_dim: int = 1024):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, S, D]
        Returns:
            scores: [B, S, 1] for PRM or [B, 1] for ORM (if using last token)
        """
        return self.score(hidden_states)

class GRPOLoss(nn.Module):
    """
    Group Relative Policy Optimization (GRPO) Loss.
    
    A PPO-variant that eliminates the need for a Critic model by using
    group-based normalization of rewards to estimate advantages.
    
    A_i = (r_i - mean(R_group)) / (std(R_group) + epsilon)
    """
    def __init__(self, cfg: ReasoningConfig):
        super().__init__()
        self.cfg = cfg
        
    def forward(
        self,
        policy_logprobs: torch.Tensor,      # [G, S] - Log probs of current policy
        old_logprobs: torch.Tensor,         # [G, S] - Log probs of old policy (reference)
        ref_logprobs: Optional[torch.Tensor],# [G, S] - Log probs of initial reference model (for KL)
        rewards: torch.Tensor,              # [G]    - Scalar reward for each sample in group
        mask: torch.Tensor,                 # [G, S] - Mask for generated tokens (1 for gen, 0 for prompt/pad)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the GRPO loss for a single group of samples (size G).
        
        Args:
            policy_logprobs: Log probabilities of the generated tokens under current policy.
            old_logprobs: Log probabilities under the policy used for sampling (often same as policy if 1-step).
            ref_logprobs: Log probabilities under the base reference model (for KL calculation).
            rewards: Ground truth rewards (oracle/verifier) for each sequence.
            mask: Padding mask (1 for valid generated tokens).
            
        Returns:
            loss: Scalar loss.
            stats: Dictionary of metrics.
        """
        G, S = policy_logprobs.shape
        assert rewards.size(0) == G
        
        # 1. Compute Advantages using Group statistics
        # Advantages are constant for all tokens in a sequence i
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward # [G]
        
        # Broadcast advantages to sequence length: [G, S]
        advantages = advantages.unsqueeze(1).repeat(1, S)
        
        # 2. Compute Ratio
        # ratio = exp(log_p - log_p_old)
        ratio = torch.exp(policy_logprobs - old_logprobs)
        
        # 3. PPO Clipped Objective
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon)
        pg_loss = torch.max(pg_loss1, pg_loss2) # Note: max because we minimize negative objective
        
        # Mask out non-generated tokens
        pg_loss = (pg_loss * mask).sum() / mask.sum()
        
        # 4. KL Divergence Penalty (Optional, if ref_model provided)
        # DeepSeek-R1 approximates KL as (exp(log_p - log_ref) - 1) - (log_p - log_ref)
        # Or simpler: exp(ref - p) ... standard ref-based KL: log_p - log_ref
        kl_loss = 0.0
        if ref_logprobs is not None:
            # Token-level KL: log_p - log_ref
            # D_KL = \sum p * log(p/q) approx sample average log(p/q)
            # We use the unbiased estimator in PPO formulations:
            # kl = (policy_logprobs - ref_logprobs)
            # But standard GRPO/PPO uses: http://joschu.net/blog/kl-approx.html
            log_ratio = policy_logprobs - ref_logprobs
            kl = (torch.exp(log_ratio) - 1) - log_ratio # Schulman estimator
            kl_loss = self.cfg.kl_beta * (kl * mask).sum() / mask.sum()
            
        # 5. Total Loss
        loss = pg_loss + kl_loss
        
        stats = {
            "grpo_loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else 0.0,
            "avg_reward": mean_reward.item(),
            "reward_std": std_reward.item(),
            "avg_advantage": advantages.mean().item(),
        }
        
        return loss, stats

