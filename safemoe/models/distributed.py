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


def _all_to_all_varying(
    x: torch.Tensor,
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    group=None
) -> torch.Tensor:
    """
    Variable-length AllToAll communication.
    
    Args:
        x: [sum(send_counts), D] - data to send
        send_counts: [P] - number of elements to send to each rank (CPU tensor)
        recv_counts: [P] - number of elements to receive from each rank (CPU tensor)
        group: Process group
        
    Returns:
        y: [sum(recv_counts), D]
    """
    if not HAS_DIST or not dist.is_initialized():
        # Single GPU fallback
        return x
    
    assert x.is_contiguous()
    x_splits = list(x.split(send_counts.tolist(), dim=0))
    y_splits = [
        torch.empty((int(rc), x.size(1)), device=x.device, dtype=x.dtype)
        for rc in recv_counts.tolist()
    ]
    dist.all_to_all(y_splits, x_splits, group=group)
    return torch.cat(y_splits, dim=0)


def _exchange_counts(send_counts: torch.Tensor, group=None) -> torch.Tensor:
    """Exchange send counts between ranks to get receive counts."""
    if not HAS_DIST or not dist.is_initialized():
        return send_counts
    
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    return recv_counts


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
    
    def _apply_capacity_and_fallback(
        self,
        x: torch.Tensor,
        topk_experts: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_raw: torch.Tensor,
        capacity: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Apply capacity control and identify fallback tokens.
        
        Returns:
            kept_x: tokens to route to experts
            kept_experts: expert indices for kept tokens
            kept_scores: weights for kept tokens
            kept_token_idx: original token indices
            fallback_mask: [T] bool mask of tokens that go to fallback
            stats: routing statistics
        """
        T, D = x.shape
        device = x.device
        
        # Low-confidence fallback
        fallback_mask = torch.zeros(T, dtype=torch.bool, device=device)
        if self.cfg.route_threshold > 0:
            low_conf = topk_raw[:, 0] < self.cfg.route_threshold
            fallback_mask |= low_conf
        
        # Expand to per-(token, k) view
        M = T * self.cfg.top_k
        flat_experts = topk_experts.reshape(M)  # [M]
        flat_scores = topk_scores.reshape(M)    # [M]
        flat_token_idx = torch.arange(T, device=device).repeat_interleave(self.cfg.top_k)  # [M]
        
        # Remove already-fallback tokens
        valid = ~fallback_mask[flat_token_idx]
        flat_experts = flat_experts[valid]
        flat_scores = flat_scores[valid]
        flat_token_idx = flat_token_idx[valid]
        
        # Sort by expert for capacity control
        order = torch.argsort(flat_experts)
        flat_experts = flat_experts[order]
        flat_scores = flat_scores[order]
        flat_token_idx = flat_token_idx[order]
        
        # Compute position within each expert's allocation
        M_valid = flat_experts.numel()
        if M_valid == 0:
            # All tokens go to fallback
            return (
                torch.empty(0, D, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.ones(T, dtype=torch.bool, device=device),
                {"overflow_tokens": 0}
            )
        
        # Find segment boundaries
        seg_change = torch.ones(M_valid, dtype=torch.bool, device=device)
        seg_change[1:] = flat_experts[1:] != flat_experts[:-1]
        
        # Position within segment using cumsum
        seg_starts = torch.where(seg_change, torch.arange(M_valid, device=device), 
                                  torch.zeros(M_valid, dtype=torch.long, device=device))
        seg_starts = seg_starts.cummax(dim=0).values
        pos_in_seg = torch.arange(M_valid, device=device) - seg_starts
        
        # Keep only within capacity
        keep = pos_in_seg < capacity
        overflow_count = (~keep).sum().item()
        
        # Mark overflow tokens for fallback
        overflow_token_idx = flat_token_idx[~keep].unique()
        fallback_mask[overflow_token_idx] = True
        
        # Filter to kept tokens
        kept_x = x[flat_token_idx[keep]]
        kept_experts = flat_experts[keep]
        kept_scores = flat_scores[keep]
        kept_token_idx = flat_token_idx[keep]
        
        stats = {
            "overflow_tokens": overflow_count,
        }
        
        return kept_x, kept_experts, kept_scores, kept_token_idx, fallback_mask, stats
    
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
            y_flat[fallback_mask] = self.fallback(x_flat[fallback_mask])
        
        # 5. Distributed routing (if multi-GPU) or local routing
        if kept_x.numel() > 0:
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
            y_flat.index_add_(0, tok_e, y_e * w_e)
        
        return y_flat
    
    def _distributed_expert_forward(
        self,
        x: torch.Tensor,
        experts: torch.Tensor,
        scores: torch.Tensor,
        token_idx: torch.Tensor,
        total_tokens: int,
    ) -> torch.Tensor:
        """Process experts with AllToAll communication (multi-GPU mode)."""
        device = x.device
        D = x.size(1)
        P = self.world_size
        
        # Determine target rank for each token
        tgt_rank = experts // self.n_local_experts
        tgt_local = experts % self.n_local_experts
        
        # Sort by target rank for AllToAll
        order = torch.argsort(tgt_rank)
        x_sorted = x[order].contiguous()
        tgt_local_sorted = tgt_local[order].contiguous()
        token_idx_sorted = token_idx[order].contiguous()
        scores_sorted = scores[order].contiguous()
        tgt_rank_sorted = tgt_rank[order]
        
        # Compute send counts per rank
        send_counts = torch.bincount(tgt_rank_sorted, minlength=P).to(torch.int64).cpu()
        recv_counts = _exchange_counts(send_counts, self.group)
        
        # AllToAll: send tokens to appropriate ranks
        x_recv = _all_to_all_varying(x_sorted, send_counts, recv_counts, self.group)
        local_recv = _all_to_all_varying(
            tgt_local_sorted.unsqueeze(-1), 
            send_counts, recv_counts, self.group
        ).squeeze(-1)
        token_recv = _all_to_all_varying(
            token_idx_sorted.unsqueeze(-1),
            send_counts, recv_counts, self.group
        ).squeeze(-1)
        scores_recv = _all_to_all_varying(
            scores_sorted.unsqueeze(-1),
            send_counts, recv_counts, self.group
        ).squeeze(-1)
        
        # Process local experts
        R = x_recv.size(0)
        if R > 0:
            # Sort by local expert for batched processing
            le_order = torch.argsort(local_recv)
            x_e = x_recv[le_order]
            le = local_recv[le_order]
            
            y_e = torch.empty_like(x_e)
            
            # Process each local expert's batch
            start = 0
            while start < R:
                e = int(le[start].item())
                end = start + 1
                while end < R and int(le[end].item()) == e:
                    end += 1
                y_e[start:end] = self.local_experts[e](x_e[start:end])
                start = end
            
            # Undo sort
            inv_order = torch.empty_like(le_order)
            inv_order[le_order] = torch.arange(R, device=device)
            y_recv = y_e[inv_order]
            scores_recv_ordered = scores_recv
        else:
            y_recv = torch.empty(0, D, device=device, dtype=x.dtype)
            scores_recv_ordered = scores_recv
        
        # AllToAll: send results back
        send_counts_back = recv_counts
        recv_counts_back = send_counts
        
        y_back = _all_to_all_varying(y_recv.contiguous(), send_counts_back, recv_counts_back, self.group)
        token_back = _all_to_all_varying(
            token_recv.unsqueeze(-1),
            send_counts_back, recv_counts_back, self.group
        ).squeeze(-1)
        scores_back = _all_to_all_varying(
            scores_recv_ordered.unsqueeze(-1),
            send_counts_back, recv_counts_back, self.group
        ).squeeze(-1)
        
        # Combine results
        y_flat = torch.zeros(total_tokens, D, device=device, dtype=x.dtype)
        if y_back.numel() > 0:
            y_flat.index_add_(0, token_back, y_back * scores_back.unsqueeze(-1))
        
        return y_flat


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
        from .attention import MultiheadSelfAttention
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, n_heads, attn_dropout)
        
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
