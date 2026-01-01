import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


# -----------------------
# 1) HF Config
# -----------------------
class SafeMoETransformerConfig(PretrainedConfig):
    model_type = "safe_moe_transformer"

    def __init__(
        self,
        vocab_size: int = 50257,
        max_position_embeddings: int = 2048,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 16,
        rms_norm_eps: float = 1e-5,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,

        # MoE params
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_capacity_factor: float = 1.25,
        moe_min_capacity: int = 4,
        moe_route_threshold: float = 0.0,    # top1 score < threshold => fallback
        moe_router_z_loss: float = 1e-3,
        moe_load_balance_loss: float = 1e-2,
        moe_router_dropout: float = 0.0,
        moe_score_scale: float = 1.0,

        # misc
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout

        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_capacity_factor = moe_capacity_factor
        self.moe_min_capacity = moe_min_capacity
        self.moe_route_threshold = moe_route_threshold
        self.moe_router_z_loss = moe_router_z_loss
        self.moe_load_balance_loss = moe_load_balance_loss
        self.moe_router_dropout = moe_router_dropout
        self.moe_score_scale = moe_score_scale

        self.tie_word_embeddings = tie_word_embeddings


# -----------------------
# 2) Building Blocks
# -----------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class MultiheadSelfAttention(nn.Module):
    """
    用清晰版 attention。你可以后续替换为 torch.nn.functional.scaled_dot_product_attention
    或 FlashAttention。
    """
    def __init__(self, hidden_size: int, num_heads: int, attn_dropout: float):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,S,D]
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3,B,H,S,dh]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,S,dh]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,S,S]
        if attention_mask is not None:
            # HF 常见：attention_mask 为 additive mask，shape 可广播到 [B,1,1,S] 或 [B,1,S,S]
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        if self.attn_dropout > 0:
            attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

        out = attn @ v  # [B,H,S,dh]
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.proj(out)


class ExpertFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class DenseFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.ffn = ExpertFFN(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class TopKRouter(nn.Module):
    """
    返回：
      topk_experts: [T,k]
      topk_scores : [T,k] (归一后的权重)
      aux_losses  : dict
    """
    def __init__(self, config: SafeMoETransformerConfig):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)

    def forward(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        logits = self.router(x_flat) * self.config.moe_score_scale  # [T,E]
        if self.config.moe_router_dropout > 0:
            logits = F.dropout(logits, p=self.config.moe_router_dropout, training=self.training)

        probs = F.softmax(logits, dim=-1)  # [T,E]
        topk_scores, topk_experts = torch.topk(probs, k=self.config.moe_top_k, dim=-1)  # [T,k]

        # top-k weights normalize
        denom = topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topk_scores = topk_scores / denom

        # router z-loss
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        # load balance (一种常用形式：importance * load)
        importance = probs.sum(dim=0)  # [E]
        top1 = topk_experts[:, 0]
        load = torch.bincount(top1, minlength=self.config.moe_num_experts).float()

        importance = importance / importance.sum().clamp_min(1e-9)
        load = load / load.sum().clamp_min(1e-9)
        lb_loss = (importance * load).sum() * (self.config.moe_num_experts ** 2)

        aux = {
            "router_z_loss": z_loss * self.config.moe_router_z_loss,
            "load_balance_loss": lb_loss * self.config.moe_load_balance_loss,
        }
        return topk_experts, topk_scores, aux


class SafeMoE(nn.Module):
    """
    Safe MoE-FFN:
      - capacity per expert
      - overflow -> dense fallback
      - low confidence -> dense fallback
      - 统计指标可用于日志/监控
    """
    def __init__(self, config: SafeMoETransformerConfig):
        super().__init__()
        self.config = config
        self.router = TopKRouter(config)
        self.experts = nn.ModuleList(
            [ExpertFFN(config.hidden_size, config.intermediate_size) for _ in range(config.moe_num_experts)]
        )
        self.fallback = DenseFFN(config.hidden_size, config.intermediate_size)

    def _capacity(self, T: int) -> int:
        cap = int(self.config.moe_capacity_factor * (T * self.config.moe_top_k / self.config.moe_num_experts))
        return max(cap, self.config.moe_min_capacity)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # x: [B,S,D]
        B, S, D = x.shape
        T = B * S
        x_flat = x.view(T, D)

        topk_experts, topk_scores, aux = self.router(x_flat)  # [T,k], [T,k]

        cap = self._capacity(T)
        device = x.device

        fallback_mask = torch.zeros(T, dtype=torch.bool, device=device)

        # low confidence fallback
        if self.config.moe_route_threshold > 0:
            fallback_mask |= (topk_scores[:, 0] < self.config.moe_route_threshold)

        y_flat = torch.zeros_like(x_flat)

        # precompute fallback where needed
        if fallback_mask.any():
            y_flat[fallback_mask] = self.fallback(x_flat[fallback_mask])

        overflow_total = 0
        usage = torch.zeros(self.config.moe_num_experts, device=device)

        for e in range(self.config.moe_num_experts):
            mask_k = (topk_experts == e)
            if not mask_k.any():
                continue

            tok_idx, k_idx = torch.where(mask_k)
            if tok_idx.numel() == 0:
                continue

            # sort by tok_idx for stability
            order = torch.argsort(tok_idx)
            tok_idx = tok_idx[order]
            k_idx = k_idx[order]

            # remove already-fallback tokens
            keep = ~fallback_mask[tok_idx]
            tok_idx = tok_idx[keep]
            k_idx = k_idx[keep]
            if tok_idx.numel() == 0:
                continue

            # capacity truncate => overflow fallback
            if tok_idx.numel() > cap:
                overflow = tok_idx.numel() - cap
                overflow_total += overflow

                overflow_idx = tok_idx[cap:]
                fallback_mask[overflow_idx] = True
                y_flat[overflow_idx] = self.fallback(x_flat[overflow_idx])

                tok_idx = tok_idx[:cap]
                k_idx = k_idx[:cap]

            usage[e] += tok_idx.numel()

            x_e = x_flat[tok_idx]  # [Ne,D]
            y_e = self.experts[e](x_e)
            w_e = topk_scores[tok_idx, k_idx].unsqueeze(-1)  # [Ne,1]
            y_flat[tok_idx] += y_e * w_e

        stats = {
            "moe_capacity": torch.tensor(cap, device=device),
            "overflow_tokens": torch.tensor(overflow_total, device=device),
            "overflow_rate": torch.tensor(float(overflow_total) / float(max(1, T)), device=device),
            "fallback_rate": fallback_mask.float().mean(),
            "expert_usage_mean": usage.mean() / float(max(1, T)),
            "expert_usage_max": usage.max() / float(max(1, T)),
            "expert_usage_min": usage.min() / float(max(1, T)),
        }

        return y_flat.view(B, S, D), aux, stats


class SafeMoETransformerBlock(nn.Module):
    """
    Pre-Norm: x + Attn(LN(x)), x + MoE(LN(x))
    """
    def __init__(self, config: SafeMoETransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MultiheadSelfAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            attn_dropout=config.attn_dropout,
        )
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = SafeMoE(config)
        self.resid_dropout = config.resid_dropout

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        h = self.attn(self.norm1(x), attention_mask=attention_mask)
        if self.resid_dropout > 0:
            h = F.dropout(h, p=self.resid_dropout, training=self.training)
        x = x + h

        h2, aux, stats = self.moe(self.norm2(x))
        if self.resid_dropout > 0:
            h2 = F.dropout(h2, p=self.resid_dropout, training=self.training)
        x = x + h2

        return x, aux, stats


# -----------------------
# 3) HF Model (Causal LM)
# -----------------------
class SafeMoETransformerLM(PreTrainedModel):
    config_class = SafeMoETransformerConfig
    main_input_name = "input_ids"

    def __init__(self, config: SafeMoETransformerConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.blocks = nn.ModuleList([SafeMoETransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.post_init()

    def _build_causal_mask(self, B: int, S: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # additive causal mask: [1,1,S,S] with 0 for allowed, -inf for blocked
        mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.view(1, 1, S, S)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_router_logits: bool = False,  # 是否输出 aux/stats
        **kwargs,
    ) -> CausalLMOutputWithPast:
        B, S = input_ids.shape
        device = input_ids.device

        pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = self.embed_tokens(input_ids) + self.embed_positions(pos_ids)

        # 组合 causal mask + padding mask（如果给了 attention_mask）
        # attention_mask: [B,S] where 1 means keep, 0 means pad
        causal = self._build_causal_mask(B, S, device, x.dtype)

        if attention_mask is not None:
            # padding additive mask: [B,1,1,S]
            pad = (1.0 - attention_mask.float()).to(dtype=x.dtype)  # pad positions => 1
            pad = pad.view(B, 1, 1, S) * torch.finfo(x.dtype).min
            attn_mask = causal + pad
        else:
            attn_mask = causal

        aux_losses: List[Dict[str, torch.Tensor]] = []
        stats_list: List[Dict[str, torch.Tensor]] = []

        for blk in self.blocks:
            x, aux, st = blk(x, attention_mask=attn_mask)
            aux_losses.append(aux)
            stats_list.append(st)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # [B,S,V]

        loss = None
        if labels is not None:
            # standard causal LM shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # sum aux losses across layers
            router_z = torch.stack([a["router_z_loss"] for a in aux_losses]).sum()
            lb = torch.stack([a["load_balance_loss"] for a in aux_losses]).sum()
            loss = task_loss + router_z + lb

        # 你可以把 stats / aux 输出给 Trainer callback 做 logging
        extra: Dict[str, Any] = {}
        if output_router_logits:
            extra["aux_losses"] = aux_losses
            extra["moe_stats"] = stats_list

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            **extra,
        )
