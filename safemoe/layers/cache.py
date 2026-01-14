"""
Attention modules with KV Cache support for efficient inference.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.
    Stores past keys and values to avoid recomputation.
    """
    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
    
    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new keys and values to cache.
        
        Args:
            k_new: [B, H, T_new, d_head]
            v_new: [B, H, T_new, d_head]
            
        Returns:
            Full k, v tensors with cache included
        """
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v
    
    def reset(self):
        """Clear the cache."""
        self.k = None
        self.v = None
    
    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        return 0 if self.k is None else self.k.size(2)


class MultiheadSelfAttentionWithCache(nn.Module):
    """
    Multi-head Self-Attention with KV Cache support for efficient inference.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = attn_dropout
        
        # Precompute causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
            persistent=False
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward pass with optional KV caching.
        
        Args:
            x: [B, T_new, D] input hidden states
            attn_mask: Optional additive attention mask
            kv_cache: Optional KVCache from previous forward pass
            use_cache: Whether to return updated cache
            
        Returns:
            output: [B, T_new, D]
            cache: Updated KVCache if use_cache=True, else None
        """
        B, T_new, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T_new, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T_new, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Update cache if provided
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
        elif use_cache:
            kv_cache = KVCache()
            k, v = kv_cache.update(k, v)
        
        T_total = k.size(2)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T_new, T_total]
        
        # Apply causal mask
        # For generation: q is [B, H, 1, d] and we attend to all past + current
        if T_new == 1 and T_total > 1:
            # Single token generation - no masking needed (can attend to all past)
            pass
        else:
            # Full sequence - apply causal mask
            # We need to mask positions where query at pos i attends to key at pos j > i
            # For cached case, query positions are [T_total - T_new, T_total)
            start_pos = T_total - T_new
            mask = self.causal_mask[start_pos:start_pos+T_new, :T_total]
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply custom mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        if self.attn_dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout)
        
        # Compute output
        out = torch.matmul(attn_weights, v)  # [B, H, T_new, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T_new, D)
        out = self.proj(out)
        
        return out, kv_cache if use_cache else None


class CachedTransformerBlock(nn.Module):
    """
    Transformer block with KV cache support for efficient generation.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttentionWithCache(
            d_model, n_heads, attn_dropout, max_seq_len
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        
        self.resid_dropout = resid_dropout
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward with optional caching.
        
        Args:
            x: [B, T, D]
            attn_mask: Optional attention mask
            kv_cache: Optional cache from previous call
            use_cache: Whether to return cache
            
        Returns:
            output: [B, T, D]
            cache: Updated cache if use_cache=True
        """
        # Attention with cache
        h, cache = self.attn(self.norm1(x), attn_mask, kv_cache, use_cache)
        if self.resid_dropout > 0:
            h = F.dropout(h, p=self.resid_dropout, training=self.training)
        x = x + h
        
        # FFN (no caching needed)
        h = self.ffn(self.norm2(x))
        if self.resid_dropout > 0:
            h = F.dropout(h, p=self.resid_dropout, training=self.training)
        x = x + h
        
        return x, cache


class CachedSafeMoEBlock(nn.Module):
    """
    SafeMoE Transformer block with KV cache support.
    Combines cached attention with SafeMoE FFN.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        moe_cfg,  # MoEConfig
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        from .moe import SafeMoE
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttentionWithCache(
            d_model, n_heads, attn_dropout, max_seq_len
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = SafeMoE(moe_cfg)
        
        self.resid_dropout = resid_dropout
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, dict, dict, Optional[KVCache]]:
        """
        Forward with caching and MoE stats.
        
        Returns:
            output: [B, T, D]
            aux_losses: dict
            stats: dict
            cache: Updated cache if use_cache=True
        """
        # Attention
        h, cache = self.attn(self.norm1(x), attn_mask, kv_cache, use_cache)
        if self.resid_dropout > 0:
            h = F.dropout(h, p=self.resid_dropout, training=self.training)
        x = x + h
        
        # MoE FFN
        h, aux_losses, stats = self.moe(self.norm2(x))
        if self.resid_dropout > 0:
            h = F.dropout(h, p=self.resid_dropout, training=self.training)
        x = x + h
        
        return x, aux_losses, stats, cache
