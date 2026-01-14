from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MoEConfig
from ..layers.attention import MultiheadSelfAttention
from .moe import SafeMoE

class TransformerBlockSafeMoE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, moe_cfg: MoEConfig, n_kv_heads: Optional[int] = None, resid_dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, n_heads, n_kv_heads=n_kv_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = SafeMoE(moe_cfg)
        self.resid_dropout = resid_dropout

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # Attention
        h = self.attn(self.ln1(x), attn_mask=attn_mask)
        if self.resid_dropout > 0:
            h = F.dropout(h, p=self.resid_dropout, training=self.training)
        x = x + h

        # MoE-FFN
        h2, aux_losses, stats = self.moe(self.ln2(x))
        if self.resid_dropout > 0:
            h2 = F.dropout(h2, p=self.resid_dropout, training=self.training)
        x = x + h2
        return x, aux_losses, stats
