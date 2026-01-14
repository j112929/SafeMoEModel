from dataclasses import dataclass

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
