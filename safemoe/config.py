from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class EngramConfig:
    """Configuration for Engram Memory."""
    d_model: int = 512
    ngram_sizes: Tuple[int, ...] = (2, 3, 4)  # Which n-grams to use
    n_hashes: int = 2                          # Number of hash functions (reduces collision)
    table_size: int = 500_000                  # Size of each embedding table
    dropout: float = 0.0
    init_gate_bias: float = -3.0               # Start with small memory contribution

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
    n_shared_experts: int = 0            # 共享专家数量 (DeepSeek/Qwen style)
    use_softmax_router: bool = True      # softmax 或者 sigmoid gating（这里默认softmax）

@dataclass
class MambaConfig:
    d_model: int = 512
    d_state: int = 64     # SSM state expansion factor (N)
    d_conv: int = 4       # Local convolution width
    expand: int = 2       # Block expansion factor
    headdim: int = 64     # Head dimension (P)
    ngroups: int = 1      # Number of groups for GQA-style behavior
    chunk_size: int = 256 # Chunk size for efficient computation

@dataclass
class OptimizedDistributedMoEConfig:
    """Configuration for Optimized Distributed SafeMoE."""
    d_model: int = 512
    d_ff: int = 2048
    n_experts_global: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25
    min_capacity: int = 4
    n_shared_experts: int = 0  # Shared experts (per device or global?) Usually local per rank if duplicated, or just local.
    
    # Safety
    route_threshold: float = 0.0
    use_fallback: bool = True
    
    # Auxiliary losses
    router_z_loss: float = 1e-3
    load_balance_loss: float = 1e-2
    router_dropout: float = 0.0
    
    # Performance
    activation: str = "silu"  # silu, gelu, relu
    use_bias: bool = False    # Bias in expert FFN
    
    # Distributed
    expert_parallel_group: Optional[object] = None

@dataclass
class DistributedMoEConfig:
    """Configuration for Distributed SafeMoE (Original)."""
    d_model: int = 512
    d_ff: int = 2048
    n_experts_global: int = 8          # Total experts across all ranks
    top_k: int = 2
    capacity_factor: float = 1.25
    min_capacity: int = 4
    
    # Safety
    route_threshold: float = 0.0       # Low-confidence fallback threshold
    use_fallback: bool = True          # Whether to use dense fallback
    
    # Auxiliary losses
    router_z_loss: float = 1e-3
    load_balance_loss: float = 1e-2
    router_dropout: float = 0.0
    
    # Distributed
    expert_parallel_group: Optional[object] = None  # Process group for EP
