class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,S,D]
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,S,dh]

        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,S,S]
        if attn_mask is not None:
            attn = attn + attn_mask  # mask should be additive (-inf)

        attn = F.softmax(attn, dim=-1)
        if self.attn_dropout > 0:
            attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

        out = attn @ v  # [B,H,S,dh]
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.proj(out)


class TransformerBlockSafeMoE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, moe_cfg: MoEConfig, resid_dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, n_heads)
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
