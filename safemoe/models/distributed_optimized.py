"""
Optimized Distributed SafeMoE with Vectorized Expert Computation

Combines:
- Distributed Expert Parallelism (AllToAll communication)
- Vectorized expert computation (stacked weights + einsum)
- Full safety mechanisms (fallback, capacity control)
- Auxiliary losses (z-loss, load balancing)

Key optimizations:
1. Expert weights stacked as [n_local, D, d_ff] for batched indexing
2. einsum-based forward eliminates per-expert Python loops
3. Segment-based capacity control using cumsum (GPU-friendly)
4. Grouped expert processing to maximize batch efficiency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math
from .routing import plan_routes
from .communication import _all_to_all_varying, _exchange_counts, distributed_expert_forward
from ..config import OptimizedDistributedMoEConfig

# Conditional import for distributed
try:
    import torch.distributed as dist
    HAS_DIST = True
except ImportError:
    HAS_DIST = False





class VectorizedExpertLayer(nn.Module):
    """
    Vectorized expert computation using stacked weights.
    All local experts' weights are stored as single tensors for efficient batched computation.
    """
    def __init__(
        self,
        n_experts: int,
        d_model: int,
        d_ff: int,
        activation: str = "silu",
        use_bias: bool = False,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Stacked weights: [n_experts, d_model, d_ff]
        # SwiGLU: w1(gate), w3(up), w2(down)
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        
        # SwiGLU typically has no bias
        
        # Initialize weights
        for i in range(n_experts):
            nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w3[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))
        
        # Activation function
        if activation == "silu":
            self.act = F.silu
        elif activation == "gelu":
            self.act = F.gelu
        elif activation == "relu":
            self.act = F.relu
        else:
            self.act = F.silu
    
    def forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """
        Batched forward through selected experts (SwiGLU).
        
        Args:
            x: [N, D] - input tokens
            expert_ids: [N] - local expert index for each token
            
        Returns:
            y: [N, D] - output tokens
        """
        if x.numel() == 0:
            return x
        
        # Gather weights for each token's expert
        w1 = self.w1[expert_ids]  # [N, D, d_ff]
        w3 = self.w3[expert_ids]  # [N, D, d_ff]
        w2 = self.w2[expert_ids]  # [N, d_ff, D]
        
        # SwiGLU: w2(act(x @ w1) * (x @ w3))
        # einsum: 'nd,ndf->nf'
        
        gate = torch.einsum('nd,ndf->nf', x, w1)
        up = torch.einsum('nd,ndf->nf', x, w3)
        
        hidden = self.act(gate) * up
        
        out = torch.einsum('nf,nfd->nd', hidden, w2)
        
        return out
    
    def forward_grouped(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """
        Alternative: Process by grouping tokens per expert.
        More efficient when tokens are already sorted by expert.
        
        Args:
            x: [N, D] - tokens sorted by expert_ids
            expert_ids: [N] - sorted local expert indices
        """
        if x.numel() == 0:
            return x
        
        N = x.size(0)
        device = x.device
        
        # Find segment boundaries
        if N == 1:
            # Single token
            return self.forward(x, expert_ids)
        
        seg_change = torch.ones(N, dtype=torch.bool, device=device)
        seg_change[1:] = expert_ids[1:] != expert_ids[:-1]
        
        # Get segment starts and expert IDs
        seg_starts = seg_change.nonzero(as_tuple=True)[0]
        seg_expert_ids = expert_ids[seg_starts]
        
        # Add end position
        seg_ends = torch.cat([seg_starts[1:], torch.tensor([N], device=device)])
        
        # Process each segment
        outputs = []
        for i in range(len(seg_starts)):
            start = seg_starts[i].item()
            end = seg_ends[i].item()
            e = seg_expert_ids[i].item()
            
            x_seg = x[start:end]
            
            # Single expert forward (SwiGLU)
            # w1: [D, d_ff], w3: [D, d_ff], w2: [d_ff, D]
            w1 = self.w1[e]
            w3 = self.w3[e]
            w2 = self.w2[e]
            
            # Gate & Up
            gate = x_seg @ w1
            up = x_seg @ w3
            
            # Act & Mult
            hidden = self.act(gate) * up
            
            # Down
            out = hidden @ w2
            outputs.append(out)
        
        return torch.cat(outputs, dim=0)


class DenseFallbackFFN(nn.Module):
    """Dense FFN for fallback tokens."""
    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu"):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)
        self.act = F.gelu if activation == "gelu" else F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class OptimizedDistributedSafeMoE(nn.Module):
    """
    Optimized Distributed SafeMoE with Vectorized Experts.
    
    Combines the best of:
    - VectorizedSafeMoE: Batched expert computation
    - DistributedSafeMoE: Expert parallelism across GPUs
    - SafeMoE: Fallback and capacity safety mechanisms
    
    Performance features:
    - Stacked expert weights for efficient indexing
    - einsum-based batched forward pass
    - Segment-based capacity control (no Python loops)
    - Grouped expert processing after AllToAll
    """
    
    def __init__(self, cfg: OptimizedDistributedMoEConfig):
        super().__init__()
        self.cfg = cfg
        
        # Distributed setup
        self.group = cfg.expert_parallel_group
        if HAS_DIST and dist.is_initialized():
            self.rank = dist.get_rank(self.group)
            self.world_size = dist.get_world_size(self.group)
        else:
            self.rank = 0
            self.world_size = 1
        
        assert cfg.n_experts_global % self.world_size == 0
        self.n_local_experts = cfg.n_experts_global // self.world_size
        
        # Router (predicts over all global experts)
        self.router = nn.Linear(cfg.d_model, cfg.n_experts_global, bias=False)
        
        # Vectorized local experts
        self.experts = VectorizedExpertLayer(
            n_experts=self.n_local_experts,
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            activation=cfg.activation,
            use_bias=cfg.use_bias,
        )
        
        # Fallback
        if cfg.use_fallback:
            self.fallback = DenseFallbackFFN(cfg.d_model, cfg.d_ff, activation="gelu")
        else:
            self.fallback = None
            
        # Shared Experts (Local, always on)
        if cfg.n_shared_experts > 0:
            # We treat shared experts as a local DenseFFN that runs on all tokens.
            # In distributed setting, shared experts are typically replicated on all ranks.
            # Here we implement n_shared_experts * d_ff dimension.
            from .moe_vectorized import ExpertFFN
            self.shared = ExpertFFN(cfg.d_model, cfg.d_ff * cfg.n_shared_experts, cfg.activation)
        else:
            self.shared = None
    
    def _compute_capacity(self, total_tokens: int) -> int:
        """Compute per-expert capacity."""
        expected = (total_tokens * self.cfg.top_k) / self.cfg.n_experts_global
        cap = int(expected * self.cfg.capacity_factor)
        return max(cap, self.cfg.min_capacity)
    
    def _route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Compute routing with auxiliary losses."""
        T = x.size(0)
        
        logits = self.router(x)
        if self.cfg.router_dropout > 0 and self.training:
            logits = F.dropout(logits, p=self.cfg.router_dropout)
        
        probs = F.softmax(logits, dim=-1)
        topk_raw, topk_experts = torch.topk(probs, k=self.cfg.top_k, dim=-1)
        topk_scores = topk_raw / topk_raw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        
        # Auxiliary losses
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        
        importance = probs.sum(dim=0)
        top1 = topk_experts[:, 0]
        load = torch.bincount(top1, minlength=self.cfg.n_experts_global).float()
        importance = importance / importance.sum().clamp_min(1e-9)
        load = load / load.sum().clamp_min(1e-9)
        lb_loss = (importance * load).sum() * (self.cfg.n_experts_global ** 2)
        
        aux = {
            "router_z_loss": z_loss * self.cfg.router_z_loss,
            "load_balance_loss": lb_loss * self.cfg.load_balance_loss,
        }
        
        return topk_experts, topk_scores, topk_raw, aux
    
    def _apply_capacity_vectorized(self, x, topk_experts, topk_scores, topk_raw, capacity):
        plan = plan_routes(topk_experts, topk_scores, topk_raw, capacity,
                           self.cfg.route_threshold, self.cfg.n_experts_global)
        return (x[plan.token_indices], plan.expert_indices, plan.scores,
                plan.token_indices, plan.fallback_mask, plan.overflow_tokens)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Optimized forward pass.
        
        Args:
            x: [B, S, D]
            
        Returns:
            y: [B, S, D]
            aux_losses: dict
            stats: dict
        """
        B, S, D = x.shape
        T = B * S
        x_flat = x.reshape(T, D)
        device = x.device
        
        # 1. Route
        topk_experts, topk_scores, topk_raw, aux = self._route(x_flat)
        capacity = self._compute_capacity(T)
        
        # 2. Capacity control (vectorized)
        kept_x, kept_experts, kept_scores, kept_token_idx, fallback_mask, overflow = \
            self._apply_capacity_vectorized(x_flat, topk_experts, topk_scores, topk_raw, capacity)
        
        # 3. Initialize output
        y_flat = torch.zeros_like(x_flat)
        
        # 3.5 Shared Experts (Always On, Replicated)
        if self.shared is not None:
            y_shared = self.shared(x_flat)
            y_flat = y_flat + y_shared
        
        # 4. Fallback
        if self.fallback is not None and fallback_mask.any():
            y_flat[fallback_mask] = y_flat[fallback_mask] + self.fallback(x_flat[fallback_mask])
        
        # 5. Expert computation
        if self.world_size > 1 or kept_x.numel() > 0:
            if self.world_size > 1:
                y_expert = self._distributed_expert_forward(
                    kept_x, kept_experts, kept_scores, kept_token_idx, T
                )
            else:
                y_expert = self._local_expert_forward(
                    kept_x, kept_experts, kept_scores, kept_token_idx, T
                )
            y_flat = y_flat + y_expert
        
        # 6. Stats
        stats = {
            "moe_capacity": torch.tensor(capacity, device=device),
            "overflow_tokens": torch.tensor(overflow, device=device),
            "overflow_rate": torch.tensor(overflow / max(1, T), device=device),
            "fallback_rate": fallback_mask.float().mean(),
            "world_size": self.world_size,
            "n_local_experts": self.n_local_experts,
            "shared_experts": 1 if self.shared is not None else 0,
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
        """
        Local vectorized expert computation.
        Uses sorted grouped processing for efficiency.
        """
        device = x.device
        D = x.size(1)
        
        # x and experts are already sorted by expert during capacity control
        # Use grouped forward for efficiency
        y_e = self.experts.forward_grouped(x, experts)
        
        # Weighted scatter back
        y_flat = torch.zeros(total_tokens, D, device=device, dtype=x.dtype)
        y_flat.index_add_(0, token_idx, (y_e * scores.unsqueeze(-1)).to(y_flat.dtype))
        
        return y_flat
    
    def _distributed_expert_forward(self, x, experts, scores, token_idx, total_tokens):
        return distributed_expert_forward(
            x, experts, scores, token_idx, total_tokens,
            self.n_local_experts, self.group, self.experts.forward_grouped,
        )


class OptimizedDistributedSafeMoEBlock(nn.Module):
    """Full Transformer block with Optimized Distributed SafeMoE."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        moe_cfg: OptimizedDistributedMoEConfig,
        n_kv_heads: Optional[int] = None,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        from ..layers.attention import MultiheadSelfAttention
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, n_heads, n_kv_heads=n_kv_heads, attn_dropout=attn_dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = OptimizedDistributedSafeMoE(moe_cfg)
        
        self.resid_dropout = resid_dropout
    
    def forward(self, x: torch.Tensor, attn_mask=None):
        h = self.attn(self.norm1(x), attn_mask=attn_mask)
        if self.resid_dropout > 0:
            h = F.dropout(h, p=self.resid_dropout, training=self.training)
        x = x + h
        
        h, aux, stats = self.moe(self.norm2(x))
        if self.resid_dropout > 0:
            h = F.dropout(h, p=self.resid_dropout, training=self.training)
        x = x + h
        
        return x, aux, stats
