import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadSelfAttention(nn.Module):
    """
    Multi-Head Attention with GQA (Grouped Query Attention) support.
    If n_kv_heads is None or equal to n_heads, it behaves as MHA.
    If n_kv_heads < n_heads, it behaves as GQA.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        n_kv_heads: Optional[int] = None, 
        attn_dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.d_head = d_model // n_heads
        
        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        
        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,S,D]
        B, S, D = x.shape
        
        # Project
        xq = self.wq(x).reshape(B, S, self.n_heads, self.d_head)
        xk = self.wk(x).reshape(B, S, self.n_kv_heads, self.d_head)
        xv = self.wv(x).reshape(B, S, self.n_kv_heads, self.d_head)
        
        # Transpose for attention: [B, H, S, D_head]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Repeat KV for GQA
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=1)
            xv = xv.repeat_interleave(self.n_rep, dim=1)
        
        # Scaled dot-product attention
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.d_head) # [B,H,S,S]
        
        if attn_mask is not None:
            scores = scores + attn_mask
            
        probs = F.softmax(scores, dim=-1)
        
        if self.attn_dropout > 0:
            probs = F.dropout(probs, p=self.attn_dropout, training=self.training)
            
        output = torch.matmul(probs, xv) # [B,H,S,D_head]
        
        # Restore shape
        output = output.transpose(1, 2).reshape(B, S, D)
        return self.wo(output)
