import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Config
# ----------------------------
@dataclass
class MoEEngramConfig:
    vocab_size: int = 50257
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    d_ff: int = 8192

    # MoE
    n_experts: int = 16
    top_k: int = 2
    moe_capacity_factor: float = 1.25  # simple capacity control

    # Engram
    use_engram: bool = True
    engram_layers: int = 8  # apply to first K layers
    ngram_sizes: Tuple[int, ...] = (2, 3, 4, 5)
    ngram_hashes: int = 2
    table_size: int = 2_000_000  # per n (simplified)
    engram_dropout: float = 0.0
    engram_init_gate_bias: float = -3.0

    # LM
    max_seq_len: int = 4096
    tie_word_embeddings: bool = True


# ----------------------------
# Utilities: simple rolling hash (demo)
# ----------------------------
@torch.no_grad()
def ngram_hash_addrs(
    input_ids: torch.Tensor,
    n: int,
    n_hash: int,
    table_size: int,
) -> torch.Tensor:
    """
    input_ids: [B, T] (int64)
    returns addrs: [B, T, n_hash] (int64), addr for each position (invalid for t<n-1 will be 0)
    NOTE: demo-quality hash; replace with xxhash / custom CUDA for production.
    """
    B, T = input_ids.shape
    device = input_ids.device

    # base primes for multiple hashes
    primes = torch.tensor([1315423911, 2654435761, 2246822519, 3266489917], device=device, dtype=torch.int64)
    primes = primes[:n_hash]

    addrs = torch.zeros((B, T, n_hash), device=device, dtype=torch.int64)

    if n == 1:
        # trivial
        x = input_ids.to(torch.int64)
        for h in range(n_hash):
            addrs[:, :, h] = (x * primes[h]) % table_size
        return addrs

    # compute n-gram hash per position by mixing tokens
    # h = sum_{i=0..n-1} token[t-i] * base^(i) * prime_h
    base = 911382323
    pow_base = torch.ones((n,), device=device, dtype=torch.int64)
    for i in range(1, n):
        pow_base[i] = (pow_base[i - 1] * base) % (2**63 - 1)

    ids = input_ids.to(torch.int64)
    for t in range(n - 1, T):
        window = ids[:, t - n + 1 : t + 1]  # [B, n]
        mixed = (window * pow_base).sum(dim=-1)  # [B]
        for h in range(n_hash):
            addrs[:, t, h] = (mixed * primes[h]) % table_size

    return addrs


# ----------------------------
# Engram Memory + Fusion
# ----------------------------
class EngramMemory(nn.Module):
    def __init__(self, cfg: MoEEngramConfig):
        super().__init__()
        self.cfg = cfg

        # one table per n (simplified)
        self.tables = nn.ModuleDict()
        for n in cfg.ngram_sizes:
            emb = nn.Embedding(cfg.table_size, cfg.d_model)
            # for host offload, you can move emb.weight to CPU pinned memory later
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            self.tables[str(n)] = emb

        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.gate = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        nn.init.constant_(self.gate.bias, cfg.engram_init_gate_bias)
        self.dropout = nn.Dropout(cfg.engram_dropout)
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        hidden: [B, T, D]
        input_ids: [B, T]
        """
        B, T, D = hidden.shape
        device = hidden.device

        # accumulate memory from multiple n and hashes
        mem = torch.zeros((B, T, D), device=device, dtype=hidden.dtype)

        for n in self.cfg.ngram_sizes:
            addrs = ngram_hash_addrs(input_ids, n=n, n_hash=self.cfg.ngram_hashes, table_size=self.cfg.table_size)
            # addrs: [B, T, H]
            # gather per hash then average (you can do weighted, or concat+MLP)
            m_n = 0.0
            for h in range(self.cfg.ngram_hashes):
                idx = addrs[:, :, h]  # [B, T]
                # embedding lookup expects long on same device as weights; for CPU-offload,
                # you'd do CPU gather then copy to GPU.
                m = self.tables[str(n)](idx)  # [B, T, D]
                m_n = m_n + m
            m_n = m_n / float(self.cfg.ngram_hashes)
            mem = mem + m_n

        mem = mem / float(len(self.cfg.ngram_sizes))
        mem = self.dropout(mem)
        mem = self.proj(mem)

        h = self.ln(hidden)
        g = torch.sigmoid(self.gate(h))  # [B, T, D]
        return hidden + g * mem


# ----------------------------
# Simple MoE FFN (single-device demo)
# ----------------------------
class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)))


class MoEFeedForward(nn.Module):
    def __init__(self, cfg: MoEEngramConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, T, D]
        NOTE: demo implementation (no capacity, no all-to-all). Replace with production MoE.
        """
        B, T, D = x.shape
        logits = self.router(x)  # [B, T, E]
        topk = torch.topk(logits, k=self.cfg.top_k, dim=-1)
        idx = topk.indices  # [B, T, K]
        w = F.softmax(topk.values, dim=-1).to(x.dtype)  # [B, T, K]

        y = torch.zeros_like(x)
        for k in range(self.cfg.top_k):
            e_idx = idx[:, :, k]  # [B, T]
            w_k = w[:, :, k].unsqueeze(-1)  # [B, T, 1]
            # naive per-expert masking
            for e in range(self.cfg.n_experts):
                mask = (e_idx == e).unsqueeze(-1)  # [B, T, 1]
                if mask.any():
                    y = y + w_k * mask * self.experts[e](x)
        return y


# ----------------------------
# Attention + Block
# ----------------------------
class SelfAttention(nn.Module):
    def __init__(self, cfg: MoEEngramConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln(x)
        qkv = self.qkv(h).view(B, T, 3, self.cfg.n_heads, D // self.cfg.n_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, T, H, Hd]

        q = q.transpose(1, 2)  # [B, H, T, Hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # [B, H, T, T]
        if attn_mask is not None:
            att = att + attn_mask  # broadcastable

        att = F.softmax(att, dim=-1).to(v.dtype)
        y = att @ v  # [B, H, T, Hd]
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return x + self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: MoEEngramConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        self.engram = EngramMemory(cfg) if (cfg.use_engram and layer_idx < cfg.engram_layers) else None
        self.attn = SelfAttention(cfg)
        self.ffn_ln = nn.LayerNorm(cfg.d_model)
        self.moe = MoEFeedForward(cfg)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.engram is not None:
            x = self.engram(x, input_ids)
        x = self.attn(x, attn_mask=attn_mask)
        h = self.ffn_ln(x)
        x = x + self.moe(h)
        return x


# ----------------------------
# Full LM
# ----------------------------
class MoEEngramLM(nn.Module):
    def __init__(self, cfg: MoEEngramConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

    def _causal_mask(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.view(1, 1, T, T)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]
        returns logits: [B, T, V]
        """
        B, T = input_ids.shape
        x = self.embed(input_ids)
        attn_mask = self._causal_mask(T, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, input_ids=input_ids, attn_mask=attn_mask)
        x = self.final_ln(x)
        logits = self.lm_head(x)
        return logits
