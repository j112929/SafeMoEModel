"""
Engram Memory Module
N-gram based memory augmentation for Transformer models.
Inspired by the V1 implementation, with improvements for efficiency and safety.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional
from ..config import EngramConfig





class NGramHasher:
    """
    Vectorized N-gram hashing utility.
    Uses rolling hash for efficiency.
    """
    # Prime numbers for multiple hash functions
    PRIMES = [1315423911, 2654435761, 2246822519, 3266489917]
    BASE = 911382323
    MOD = 2**62 - 1  # Large prime-ish modulus for stability
    
    @staticmethod
    @torch.no_grad()
    def compute_hashes(
        input_ids: torch.Tensor,
        n: int,
        n_hash: int,
        table_size: int,
    ) -> torch.Tensor:
        """
        Compute n-gram hash addresses for each position.
        
        Args:
            input_ids: [B, T] token IDs
            n: n-gram size
            n_hash: number of hash functions
            table_size: size of embedding table
            
        Returns:
            addrs: [B, T, n_hash] hash addresses
        """
        B, T = input_ids.shape
        device = input_ids.device
        dtype = torch.int64
        
        primes = torch.tensor(NGramHasher.PRIMES[:n_hash], device=device, dtype=dtype)
        addrs = torch.zeros((B, T, n_hash), device=device, dtype=dtype)
        
        if T < n:
            return addrs
        
        ids = input_ids.to(dtype)
        
        # Precompute base powers: base^0, base^1, ..., base^(n-1)
        base = NGramHasher.BASE
        mod = NGramHasher.MOD
        powers = torch.ones(n, device=device, dtype=dtype)
        for i in range(1, n):
            powers[i] = (powers[i-1] * base) % mod
        
        # Vectorized rolling hash using unfold
        # Pad for positions < n-1
        padded = F.pad(ids, (n-1, 0), value=0)  # [B, T+n-1]
        
        # Create windows: [B, T, n]
        windows = padded.unfold(dimension=1, size=n, step=1)  # [B, T, n]
        
        # Compute hash: sum(token * base^i) for each window
        mixed = (windows * powers).sum(dim=-1)  # [B, T]
        
        # Apply multiple hash functions
        for h in range(n_hash):
            addrs[:, :, h] = (mixed * primes[h]) % table_size
        
        # Zero out invalid positions (first n-1 positions don't have full n-gram)
        addrs[:, :n-1, :] = 0
        
        return addrs


class EngramMemory(nn.Module):
    """
    N-gram based memory augmentation module.
    
    For each token position, looks up embeddings based on the preceding n-gram
    pattern and fuses them with the hidden state via a learned gate.
    
    This can help the model remember common phrases, idioms, and patterns
    without relying solely on attention.
    """
    def __init__(self, config: EngramConfig):
        super().__init__()
        self.config = config
        
        # Create embedding table for each n-gram size
        self.tables = nn.ModuleDict()
        for n in config.ngram_sizes:
            emb = nn.Embedding(config.table_size, config.d_model)
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            self.tables[str(n)] = emb
        
        # Projection and gating
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gate = nn.Linear(config.d_model, config.d_model, bias=True)
        nn.init.constant_(self.gate.bias, config.init_gate_bias)
        
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)
        
        self.hasher = NGramHasher()
    
    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Augment hidden states with n-gram memory.
        
        Args:
            hidden: [B, T, D] hidden states from transformer
            input_ids: [B, T] original token IDs
            
        Returns:
            augmented: [B, T, D] hidden states with memory fusion
        """
        B, T, D = hidden.shape
        device = hidden.device
        
        # Accumulate memory from all n-gram sizes
        mem_sum = torch.zeros((B, T, D), device=device, dtype=hidden.dtype)
        n_contributions = 0
        
        for n in self.config.ngram_sizes:
            # Get hash addresses
            addrs = self.hasher.compute_hashes(
                input_ids, 
                n=n, 
                n_hash=self.config.n_hashes,
                table_size=self.config.table_size
            )  # [B, T, n_hash]
            
            # Average over multiple hashes
            mem_n = torch.zeros((B, T, D), device=device, dtype=hidden.dtype)
            for h in range(self.config.n_hashes):
                idx = addrs[:, :, h]  # [B, T]
                mem_h = self.tables[str(n)](idx)  # [B, T, D]
                mem_n = mem_n + mem_h
            
            mem_n = mem_n / self.config.n_hashes
            mem_sum = mem_sum + mem_n
            n_contributions += 1
        
        # Average across n-gram sizes
        mem = mem_sum / n_contributions
        
        # Project memory
        mem = self.dropout(mem)
        mem = self.proj(mem)
        
        # Compute gate based on hidden state
        h_norm = self.norm(hidden)
        gate = torch.sigmoid(self.gate(h_norm))  # [B, T, D]
        
        # Fuse: hidden + gate * memory
        return hidden + gate * mem
    
    def get_memory_usage(self) -> dict:
        """Report memory usage of embedding tables."""
        total_params = 0
        table_info = {}
        for n, table in self.tables.items():
            params = table.weight.numel()
            total_params += params
            table_info[f"{n}-gram"] = {
                "params": params,
                "size_mb": params * 4 / (1024**2)  # Assuming float32
            }
        return {
            "tables": table_info,
            "total_params": total_params,
            "total_size_mb": total_params * 4 / (1024**2)
        }


class EngramTransformerBlock(nn.Module):
    """
    Transformer block with optional Engram memory augmentation.
    Applies Engram before attention for memory-enhanced attention.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        engram_config: Optional[EngramConfig] = None,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        from .attention import MultiheadSelfAttention
        from ..models.moe import DenseFFN
        
        self.engram = EngramMemory(engram_config) if engram_config else None
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, n_heads, attn_dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = DenseFFN(d_model, d_ff)
        
        self.resid_dropout = resid_dropout
    
    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] hidden states
            input_ids: [B, T] token IDs (required if using engram)
            attn_mask: Optional attention mask
        """
        # Engram augmentation
        if self.engram is not None:
            if input_ids is None:
                raise ValueError("input_ids required when using EngramMemory")
            x = self.engram(x, input_ids)
        
        # Attention
        h = self.attn(self.norm1(x), attn_mask=attn_mask)
        if self.resid_dropout > 0:
            h = F.dropout(h, p=self.resid_dropout, training=self.training)
        x = x + h
        
        # FFN
        h = self.ffn(self.norm2(x))
        if self.resid_dropout > 0:
            h = F.dropout(h, p=self.resid_dropout, training=self.training)
        x = x + h
        
        return x

# Alias
class EngramLayer(EngramMemory):
    """Alias for EngramMemory to match safe_moe exports."""
    pass
