import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from ..config import MoEConfig
from .routing import plan_routes, routing_stats, apply_routes

class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation="gelu"):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)
        self.act = getattr(F, activation) if hasattr(F, activation) else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class DenseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation="gelu"):
        super().__init__()
        self.ffn = ExpertFFN(d_model, d_ff, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class TopKRouter(nn.Module):
    """
    输出：
      - topk_experts: [T, k]
      - topk_scores:  [T, k]  (归一化后的 gating 权重)
      - aux_losses: dict
    """
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # x: [T, d_model]  (T = batch * seq)
        logits = self.router(x) * self.cfg.score_scale  # [T, E]

        if self.cfg.router_dropout > 0:
            logits = F.dropout(logits, p=self.cfg.router_dropout, training=self.training)

        if self.cfg.use_softmax_router:
            probs = F.softmax(logits, dim=-1)  # [T, E]
        else:
            # 另一种常见路由：sigmoid，然后再做 top-k 与归一（更“独立”）
            probs = torch.sigmoid(logits)

        topk_scores, topk_experts = torch.topk(probs, k=self.cfg.top_k, dim=-1)  # [T, k]

        # 归一化 top-k 权重（确保 sum=1）
        denom = topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topk_scores = topk_scores / denom  # [T, k]

        # router z-loss（抑制 logits 爆炸）
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        # load balancing loss（Switch/Token-level常用形式之一）
        # importance: 每个 expert 被分配的“概率质量”
        importance = probs.sum(dim=0)  # [E]
        # load: 每个 expert 实际 top1 命中次数（也可用 topk）
        top1 = topk_experts[:, 0]
        load = torch.bincount(top1, minlength=self.cfg.n_experts).float()  # [E]

        importance = importance / importance.sum().clamp_min(1e-9)
        load = load / load.sum().clamp_min(1e-9)

        lb_loss = (importance * load).sum() * (self.cfg.n_experts ** 2)

        aux = {
            "router_z_loss": z_loss * self.cfg.router_z_loss,
            "load_balance_loss": lb_loss * self.cfg.load_balance_loss,
            "router_logits_mean": logits.mean().detach(),
        }
        # Pick relevant raw probability for thresholding decisions.
        # We return the values corresponding to the selected indices.
        # Note: torch.topk returns values (prob) and indices.
        # We already computed topk_scores (normalized).
        # We also need original topk values from 'probs' for thresholding check.
        # Actually topk() on probs gives exactly that if we did it on probs.
        topk_raw_probs, _ = torch.topk(probs, k=self.cfg.top_k, dim=-1)

        return topk_experts, topk_scores, aux, topk_raw_probs


class SafeMoE(nn.Module):
    """
    Safe MoE-FFN:
      - capacity per expert
      - overflow -> fallback dense ffn
      - low confidence -> fallback dense ffn
      - returns output + aux losses + stats
    """
    def __init__(self, cfg: MoEConfig, activation="gelu"):
        super().__init__()
        self.cfg = cfg
        self.router = TopKRouter(cfg)
        self.experts = nn.ModuleList([ExpertFFN(cfg.d_model, cfg.d_ff, activation) for _ in range(cfg.n_experts)])
        self.fallback = DenseFFN(cfg.d_model, cfg.d_ff, activation)

    def _capacity(self, T: int) -> int:
        # capacity per expert (按 top_k 分配后，理论平均每个 expert 接到 T*k/E tokens)
        cap = int(self.cfg.capacity_factor * (T * self.cfg.top_k / self.cfg.n_experts))
        return max(cap, self.cfg.min_capacity)

    def forward(self, x):
        B, S, D = x.shape
        flat = x.reshape(-1, D)
        experts, scores, aux, raw = self.router(flat)
        capacity = self._capacity(flat.size(0))
        plan = plan_routes(experts, scores, raw, capacity,
                           self.cfg.route_threshold, self.cfg.n_experts)
        output = apply_routes(flat, plan, self.cfg.n_experts,
                              lambda e, tokens: self.experts[e](tokens), self.fallback)
        return output.reshape(B, S, D), aux, routing_stats(plan, capacity, flat.size(0))
