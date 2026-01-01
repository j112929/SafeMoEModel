import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    d_model: int
    d_ff: int
    n_experts: int
    top_k: int = 2
    capacity_factor: float = 1.25        # 安全阈：>1 提供余量
    min_capacity: int = 4                # 防止小 batch 下 capacity=0
    router_z_loss: float = 1e-3          # router logits 的 z-loss（稳定训练）
    load_balance_loss: float = 1e-2      # 负载均衡辅助损失
    router_dropout: float = 0.0          # 路由 dropout（可选）
    score_scale: float = 1.0             # router logits scale
    route_threshold: float = 0.0         # 低置信度阈值：<阈值触发fallback
    use_softmax_router: bool = True      # softmax 或者 sigmoid gating（这里默认softmax）


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
        return topk_experts, topk_scores, aux


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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        x: [B, S, D]
        returns:
          y: [B, S, D]
          aux_losses: dict
          stats: dict
        """
        B, S, D = x.shape
        T = B * S
        x_flat = x.reshape(T, D)

        topk_experts, topk_scores, aux = self.router(x_flat)  # [T,k], [T,k]

        cap = self._capacity(T)
        device = x.device

        # 记录：哪些 token 走了 fallback（低置信度或容量溢出）
        fallback_mask = torch.zeros(T, dtype=torch.bool, device=device)

        # 低置信度 fallback：如果 top1 score < threshold
        if self.cfg.route_threshold > 0:
            low_conf = topk_scores[:, 0] < self.cfg.route_threshold
            fallback_mask |= low_conf

        # 输出缓冲
        y_flat = torch.zeros_like(x_flat)

        # 先算 fallback 输出（必要时再覆盖）
        if fallback_mask.any():
            y_fb = self.fallback(x_flat[fallback_mask])
            y_flat[fallback_mask] = y_fb

        # 对每个 expert：收集 token -> 计算 -> scatter 回去
        # 注意：这是清晰实现，性能优化可用 grouped GEMM / all-to-all / fused router
        overflow_total = 0
        usage = torch.zeros(self.cfg.n_experts, device=device)

        for e in range(self.cfg.n_experts):
            # 找所有路由到 expert e 的 (token_index, which_k)
            # mask_k: [T,k]
            mask_k = (topk_experts == e)  # bool
            if not mask_k.any():
                continue

            # token indices for this expert (可能重复：同一 token 的不同 k 路由到不同 expert)
            tok_idx, k_idx = torch.where(mask_k)  # both [N]
            if tok_idx.numel() == 0:
                continue

            # 排序以稳定（可选）
            # (稳定性对复现/线上排查有帮助)
            order = torch.argsort(tok_idx)
            tok_idx = tok_idx[order]
            k_idx = k_idx[order]

            # 去掉已经 fallback 的 token（低置信度）
            keep = ~fallback_mask[tok_idx]
            tok_idx = tok_idx[keep]
            k_idx = k_idx[keep]
            if tok_idx.numel() == 0:
                continue

            # 容量截断
            if tok_idx.numel() > cap:
                overflow = tok_idx.numel() - cap
                overflow_total += overflow

                # 超出 capacity 的 token 走 fallback（如果未算过 fallback，这里补算）
                overflow_idx = tok_idx[cap:]
                # 标记 fallback
                fallback_mask[overflow_idx] = True
                # 如果这些 token 之前没写入 y_flat，就补写
                # （这里简单起见：直接用 dense 计算覆盖）
                y_flat[overflow_idx] = self.fallback(x_flat[overflow_idx])

                tok_idx = tok_idx[:cap]
                k_idx = k_idx[:cap]

            usage[e] += tok_idx.numel()

            x_e = x_flat[tok_idx]                          # [Ne, D]
            y_e = self.experts[e](x_e)                     # [Ne, D]
            w_e = topk_scores[tok_idx, k_idx].unsqueeze(-1)  # [Ne, 1]

            # 按权重累加回 token 输出（top-k mixture）
            y_flat[tok_idx] += y_e * w_e

        # stats
        stats = {
            "moe_capacity": torch.tensor(cap, device=device),
            "overflow_tokens": torch.tensor(overflow_total, device=device),
            "overflow_rate": torch.tensor(float(overflow_total) / float(max(1, T)), device=device),
            "fallback_rate": fallback_mask.float().mean(),
            "expert_usage_mean": usage.mean() / max(1.0, float(S * B)),
            "expert_usage_max": usage.max() / max(1.0, float(S * B)),
            "expert_usage_min": usage.min() / max(1.0, float(S * B)),
        }

        # 合并 aux losses
        aux_losses = aux
        # 如果你希望对 overflow 做惩罚，也可加一项
        # aux_losses["overflow_penalty"] = stats["overflow_rate"] * 0.0

        y = y_flat.reshape(B, S, D)
        return y, aux_losses, stats
