"""
Rotary Positional Embedding (RoPE) for SafeMoE Attention
Compatible with LLaMA, DeepSeek, and other modern architectures.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (RoPE).
    Returns: [max_seq_len, dim//2, 2] - cos and sin values
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # [seq_len, dim//2]
    
    # Return as [seq_len, dim//2, 2] for cos and sin
    freqs_cos = freqs.cos()
    freqs_sin = freqs.sin()
    return torch.stack([freqs_cos, freqs_sin], dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    
    xq, xk: [batch, n_heads, seq_len, head_dim]
    freqs_cis: [seq_len, head_dim//2, 2]
    
    Returns: rotated xq and xk with same shape
    """
    # Reshape x to [batch, n_heads, seq_len, head_dim//2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # Get cos and sin
    seq_len = xq.shape[2]
    freqs_cis = freqs_cis[:seq_len]  # [seq_len, dim//2, 2]
    freqs_cos = freqs_cis[..., 0].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
    freqs_sin = freqs_cis[..., 1].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
    
    # Apply rotation
    # x_rotated = x * cos + rotate_half(x) * sin
    xq_r = xq_[..., 0] * freqs_cos - xq_[..., 1] * freqs_sin
    xq_i = xq_[..., 0] * freqs_sin + xq_[..., 1] * freqs_cos
    xq_out = torch.stack([xq_r, xq_i], dim=-1).flatten(-2)
    
    xk_r = xk_[..., 0] * freqs_cos - xk_[..., 1] * freqs_sin
    xk_i = xk_[..., 0] * freqs_sin + xk_[..., 1] * freqs_cos
    xk_out = torch.stack([xk_r, xk_i], dim=-1).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPEMultiheadAttention(nn.Module):
    """
    Multi-head Self-Attention with Rotary Positional Embeddings.
    Compatible with modern LLM architectures.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 4096,
        attn_dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        
        # Separate Q, K, V projections (more flexible than combined QKV)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.attn_dropout = attn_dropout
        
        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.d_head, max_seq_len, rope_theta),
            persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, S, D]
        attn_mask: Optional [B, 1, S, S] or [1, 1, S, S] additive mask
        position_ids: Optional [B, S] for variable length sequences
        """
        B, S, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        # q, k, v: [B, n_heads, S, d_head]
        
        # Apply RoPE to Q and K
        q, k = apply_rotary_emb(q, k, self.freqs_cis)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S, S]
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        if self.attn_dropout > 0:
            attn_weights = torch.dropout(attn_weights, p=self.attn_dropout, train=self.training)
        
        out = torch.matmul(attn_weights, v)  # [B, H, S, d_head]
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        
        return self.o_proj(out)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA, DeepSeek)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight
