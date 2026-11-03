"""
Distributed SafeMoE with Expert Parallelism (EP)

Combines:
- V2's AllToAll communication for expert parallelism
- SafeMoE's safety mechanisms (fallback, capacity control)
- Auxiliary losses (z-loss, load balancing)
- Vectorized expert computation where possible

Supports both single-GPU and multi-GPU execution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import math
from .routing import plan_routes
from .communication import _all_to_all_varying, _exchange_counts, distributed_expert_forward
from ..config import DistributedMoEConfig
# Conditional import for distributed
try:
    import torch.distributed as dist
    HAS_DIST = True
except ImportError:
    HAS_DIST = False





class ExpertFFN(nn.Module):
    """Single expert FFN."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)))


class DenseFallbackFFN(nn.Module):
    """Dense FFN used as fallback for dropped/low-confidence tokens."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class DistributedSafeMoE(nn.Module):
    """
    Distributed Mixture of Experts with Safety Mechanisms.
    
    Features:
    - Expert Parallelism: Experts are sharded across ranks
    - AllToAll Communication: Tokens routed to appropriate ranks
    - Capacity Control: Per-expert token limits
    - Fallback: Dropped tokens processed by dense FFN
    - Auxiliary Losses: Z-loss and load balancing
    
    For single-GPU usage, this behaves like standard SafeMoE.
    """
    
    def __init__(self, cfg: DistributedMoEConfig):
        super().__init__()
        self.cfg = cfg
        
        # Determine distributed setup
        self.group = cfg.expert_parallel_group
        if HAS_DIST and dist.is_initialized():
            self.rank = dist.get_rank(self.group)
            self.world_size = dist.get_world_size(self.group)
        else:
            self.rank = 0
            self.world_size = 1
        
        # Validate expert count
        assert cfg.n_experts_global % self.world_size == 0, \
            f"n_experts_global ({cfg.n_experts_global}) must be divisible by world_size ({self.world_size})"
        
        self.n_local_experts = cfg.n_experts_global // self.world_size
        
        # Router (global, predicts over all experts)
        self.router = nn.Linear(cfg.d_model, cfg.n_experts_global, bias=False)
        
        # Local experts (only this rank's subset)
        self.local_experts = nn.ModuleList([
            ExpertFFN(cfg.d_model, cfg.d_ff) 
            for _ in range(self.n_local_experts)
        ])
        
        # Fallback dense FFN (for dropped/low-confidence tokens)
        if cfg.use_fallback:
            self.fallback = DenseFallbackFFN(cfg.d_model, cfg.d_ff)
        else:
            self.fallback = None
    
    def _compute_capacity(self, total_tokens: int) -> int:
        """Compute per-expert capacity."""
        # Expected tokens per expert with top_k routing
        expected = (total_tokens * self.cfg.top_k) / self.cfg.n_experts_global
        cap = int(expected * self.cfg.capacity_factor)
        return max(cap, self.cfg.min_capacity)
    
    def _route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Compute routing decisions.
        
        Returns:
            topk_experts: [T, k] - selected expert indices
            topk_scores: [T, k] - normalized routing weights
            topk_raw: [T, k] - raw probabilities (for thresholding)
            aux: dict - auxiliary losses
        """
        T, D = x.shape
        
        logits = self.router(x)  # [T, E]
        
        if self.cfg.router_dropout > 0 and self.training:
            logits = F.dropout(logits, p=self.cfg.router_dropout)
        
        probs = F.softmax(logits, dim=-1)  # [T, E]
        topk_raw, topk_experts = torch.topk(probs, k=self.cfg.top_k, dim=-1)
        
        # Normalize top-k weights
        topk_scores = topk_raw / topk_raw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        
        # Auxiliary losses
        # Z-loss: prevent logits from growing too large
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        
        # Load balancing loss
        importance = probs.sum(dim=0)  # [E]
        top1_experts = topk_experts[:, 0]
        load = torch.bincount(top1_experts, minlength=self.cfg.n_experts_global).float()
        
        importance = importance / importance.sum().clamp_min(1e-9)
        load = load / load.sum().clamp_min(1e-9)
        lb_loss = (importance * load).sum() * (self.cfg.n_experts_global ** 2)
        
        aux = {
            "router_z_loss": z_loss * self.cfg.router_z_loss,
            "load_balance_loss": lb_loss * self.cfg.load_balance_loss,
        }
        
        return topk_experts, topk_scores, topk_raw, aux
    
    def _apply_capacity_and_fallback(self, x, topk_experts, topk_scores, topk_raw, capacity):
        plan = plan_routes(topk_experts, topk_scores, topk_raw, capacity,
                           self.cfg.route_threshold, self.cfg.n_experts_global)
        return (x[plan.token_indices], plan.expert_indices, plan.scores,
                plan.token_indices, plan.fallback_mask,
                {'overflow_tokens': plan.overflow_tokens})
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Forward pass with distributed expert parallelism.
        
        Args:
            x: [B, S, D] input tensor
            
        Returns:
            y: [B, S, D] output tensor
            aux_losses: dict of auxiliary losses
            stats: dict of routing statistics
        """
        B, S, D = x.shape
        T = B * S
        x_flat = x.reshape(T, D)
        device = x.device
        
        # 1. Route
        topk_experts, topk_scores, topk_raw, aux = self._route(x_flat)
        
        capacity = self._compute_capacity(T)
        
        # 2. Capacity control and fallback identification
        kept_x, kept_experts, kept_scores, kept_token_idx, fallback_mask, cap_stats = \
            self._apply_capacity_and_fallback(x_flat, topk_experts, topk_scores, topk_raw, capacity)
        
        # 3. Initialize output
        y_flat = torch.zeros_like(x_flat)
        
        # 4. Process fallback tokens
        if self.fallback is not None and fallback_mask.any():
            y_flat[fallback_mask] = self.fallback(x_flat[fallback_mask]).to(y_flat.dtype)
        
        # 5. Distributed routing (if multi-GPU) or local routing
        if self.world_size > 1 or kept_x.numel() > 0:
            if self.world_size > 1:
                y_expert = self._distributed_expert_forward(
                    kept_x, kept_experts, kept_scores, kept_token_idx, T
                )
            else:
                y_expert = self._local_expert_forward(
                    kept_x, kept_experts, kept_scores, kept_token_idx, T
                )
            
            # Scatter expert outputs back
            # Note: y_expert is [T, D] with zeros for non-participating tokens
            y_flat = y_flat + y_expert
        
        # 6. Stats
        stats = {
            "moe_capacity": torch.tensor(capacity, device=device),
            "overflow_tokens": torch.tensor(cap_stats["overflow_tokens"], device=device),
            "overflow_rate": torch.tensor(cap_stats["overflow_tokens"] / max(1, T), device=device),
            "fallback_rate": fallback_mask.float().mean(),
            "world_size": self.world_size,
            "n_local_experts": self.n_local_experts,
        }
        
        return y_flat.reshape(B, S, D), aux, stats
    
    def _local_expert_forward(
        self,
        x: torch.Tensor,
        experts: torch.Tensor,
        scores: torch.Tensor,
        token_idx: torch.Tensor,
        total_tokens: int,
    ) -> torch.Tensor:
        """Process experts locally (single-GPU mode)."""
        device = x.device
        D = x.size(1)
        
        y_flat = torch.zeros(total_tokens, D, device=device, dtype=x.dtype)
        
        # Process each expert
        for e in range(self.cfg.n_experts_global):
            mask = (experts == e)
            if not mask.any():
                continue
            
            x_e = x[mask]
            y_e = self.local_experts[e % self.n_local_experts](x_e)
            w_e = scores[mask].unsqueeze(-1)
            
            # Accumulate weighted output
            tok_e = token_idx[mask]
            y_flat.index_add_(0, tok_e, (y_e * w_e).to(y_flat.dtype))
        
        return y_flat
    
    def _distributed_expert_forward(self, x, experts, scores, token_idx, total_tokens):
        def forward_grouped(tokens, local_ids):
            return torch.cat([
                self.local_experts[e](tokens[local_ids == e])
                for e in range(self.n_local_experts)
                if (local_ids == e).any()
            ], dim=0)
        return distributed_expert_forward(
            x, experts, scores, token_idx, total_tokens,
            self.n_local_experts, self.group, forward_grouped,
        )


class DistributedSafeMoEBlock(nn.Module):
    """
    Full Transformer block with Distributed SafeMoE.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        moe_cfg: DistributedMoEConfig,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        from ..layers.attention import MultiheadSelfAttention
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, n_heads, attn_dropout=attn_dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = DistributedSafeMoE(moe_cfg)
        
        self.resid_dropout = resid_dropout
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Forward pass.
        
        Returns:
            y: [B, S, D]
            aux_losses: dict
            stats: dict
        """
        # Attention
        h = self.attn(self.norm1(x), attn_mask=attn_mask)
        if self.resid_dropout > 0:
            h = F.dropout(h, p=self.resid_dropout, training=self.training)
        x = x + h
        
        # MoE FFN
        h, aux, stats = self.moe(self.norm2(x))
        if self.resid_dropout > 0:
            h = F.dropout(h, p=self.resid_dropout, training=self.training)
        x = x + h
        
        return x, aux, stats
