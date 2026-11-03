"""
Vectorized SafeMoE Implementation
Replaces the per-expert Python loop with batched tensor operations for better GPU utilization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from ..config import MoEConfig
from .routing import plan_routes, routing_stats, apply_routes


class ExpertFFN(nn.Module):
    """
    SwiGLU Expert:
    Output = w2(Swish(w1(x)) * w3(x))
    """
    def __init__(self, d_model: int, d_ff: int, activation="silu"):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # Down
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Up
        self.act = nn.SiLU() if activation == "silu" else getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2(act(w1(x)) * w3(x))
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class DenseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation="silu"):
        super().__init__()
        self.ffn = ExpertFFN(d_model, d_ff, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class TopKRouter(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        logits = self.router(x) * self.cfg.score_scale
        if self.cfg.router_dropout > 0:
            logits = F.dropout(logits, p=self.cfg.router_dropout, training=self.training)

        if self.cfg.use_softmax_router:
            probs = F.softmax(logits, dim=-1)
        else:
            probs = torch.sigmoid(logits)

        topk_scores, topk_experts = torch.topk(probs, k=self.cfg.top_k, dim=-1)
        denom = topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topk_scores_norm = topk_scores / denom

        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        importance = probs.sum(dim=0)
        top1 = topk_experts[:, 0]
        load = torch.bincount(top1, minlength=self.cfg.n_experts).float()
        importance = importance / importance.sum().clamp_min(1e-9)
        load = load / load.sum().clamp_min(1e-9)
        lb_loss = (importance * load).sum() * (self.cfg.n_experts ** 2)

        aux = {
            "router_z_loss": z_loss * self.cfg.router_z_loss,
            "load_balance_loss": lb_loss * self.cfg.load_balance_loss,
            "router_logits_mean": logits.mean().detach(),
        }
        return topk_experts, topk_scores_norm, aux, topk_scores


class VectorizedSafeMoE(nn.Module):
    """
    Vectorized Safe MoE-FFN:
      - Uses scatter/gather operations instead of Python loops
      - Maintains all safety features (overflow, low-confidence fallback)
      - Much faster on GPU
    """
    def __init__(self, cfg: MoEConfig, activation="silu"):
        super().__init__()
        self.cfg = cfg
        self.router = TopKRouter(cfg)
        
        # Stack all expert weights: SwiGLU needs w1(gate), w3(up), w2(down)
        self.w1 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_model, cfg.d_ff))  # Gate
        self.w3 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_model, cfg.d_ff))  # Up
        self.w2 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_ff, cfg.d_model))  # Down
        
        # Shared Experts (Always Active)
        if cfg.n_shared_experts > 0:
            self.shared = ExpertFFN(cfg.d_model, cfg.d_ff * cfg.n_shared_experts, activation)
        else:
            self.shared = None
        
        # Initialize weights
        for i in range(cfg.n_experts):
            nn.init.kaiming_uniform_(self.w1[i])
            nn.init.kaiming_uniform_(self.w3[i])
            nn.init.kaiming_uniform_(self.w2[i])
        
        self.fallback = DenseFFN(cfg.d_model, cfg.d_ff, activation)
        self.act = nn.SiLU() if activation == "silu" else (getattr(F, activation) if hasattr(F, activation) else F.gelu)

    def _capacity(self, T: int) -> int:
        cap = int(self.cfg.capacity_factor * (T * self.cfg.top_k / self.cfg.n_experts))
        return max(cap, self.cfg.min_capacity)

    def _expert_forward_batched(self, x: torch.Tensor, expert_idx: torch.Tensor) -> torch.Tensor:
        """
        Batched expert computation using einsum (SwiGLU).
        x: [N, D]
        expert_idx: [N]
        Returns: [N, D]
        """
        # Gather weights
        w1 = self.w1[expert_idx]  # [N, D, d_ff]
        w3 = self.w3[expert_idx]  # [N, D, d_ff]
        w2 = self.w2[expert_idx]  # [N, d_ff, D]
        
        # SwiGLU: w2(act(x @ w1) * (x @ w3))
        # Note: einsum is slightly slower than bmm but easier for broadcasting.
        # For max perf, gather -> bmm is often better, but einsum is consistent.
        
        gate = torch.einsum('nd,ndf->nf', x, w1)
        up = torch.einsum('nd,ndf->nf', x, w3)
        
        hidden = self.act(gate) * up
        
        out = torch.einsum('nf,nfd->nd', hidden, w2)
        return out

    def forward(self, x):
        B, S, D = x.shape
        flat = x.reshape(-1, D)
        experts, scores, aux, raw = self.router(flat)
        capacity = self._capacity(flat.size(0))
        plan = plan_routes(experts, scores, raw, capacity,
                           self.cfg.route_threshold, self.cfg.n_experts)
        def expert_forward(e, tokens):
            return (self.act(tokens @ self.w1[e]) * (tokens @ self.w3[e])) @ self.w2[e]
        output = apply_routes(flat, plan, self.cfg.n_experts, expert_forward,
                              self.fallback, self.shared)
        stats = routing_stats(plan, capacity, flat.size(0))
        stats['shared_experts'] = int(self.shared is not None)
        return output.reshape(B, S, D), aux, stats
