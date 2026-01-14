# SafeMoEModel

A Safe, Robust Mixture-of-Experts (MoE) implementation with modern features.

## Project Description

**SafeMoE** is a cutting-edge research platform designed to replicate and extend the state-of-the-art in Large Language Model architectures. It integrates the latest advancements from 2024/2025, enabling researchers to experiment with:

1.  **SOTA Architecture**:
    *   **DeepSeek-SiGLE Style**: Hybrid architecture combining Shared Experts ("Always-On") with Routed Experts for optimal knowledge separation.
    *   **Mamba 2 SSM**: Linear-time sequence modeling via Structured State Space Models (SSD) for long-context efficiency.
    *   **SwiGLU & GQA**: Modern MLP activiations and Grouped Query Attention for inference speed.
    
2.  **High-Performance Training**:
    *   **Vectorized Einsum**: Highly optimized expert computation that completely avoids Python loops.
    *   **Expert Parallelism**: Distributed "All-to-All" communication for scaling to massive expert counts across GPUs.

3.  **Full-Stack Alignment**:
    *   **Reasoning (System 2)**: **GRPO** (Group Relative Policy Optimization) + **Verifier** for training reasoning models without a Critic (DeepSeek-R1 method).
    *   **Preference (Post-Training)**: Native support for **DPO**, **ORPO**, and **SimPO** for robust human alignment.

4.  **Agentic Capabilities**:
    *   Built-in ReAct loop and Tool abstraction for building autonomous agents.

Whether you are building the next reasoning model or an efficient long-context assistant, SafeMoE provides the modular, verified building blocks you need.

## Project Structure

```
SafeMoEModel/
├── safemoe/                    # Main Package
│   ├── __init__.py             # Public API exports
│   ├── config.py               # Centralized Configurations
│   ├── agent.py                # Agent & Tool Definitions
│   ├── inference.py            # High-level Inference Engine
│   ├── layers/                 # Foundational Layers
│   │   ├── attention.py        # MultiHead & Grouped Query Attention
│   │   ├── rope.py             # Rotary Positional Embeddings
│   │   ├── cache.py            # KV Cache for Autoregression
│   │   └── engram.py           # N-gram Memory
│   ├── models/                 # Model Architectures
│   │   ├── moe.py              # Standard SafeMoE
│   │   ├── moe_vectorized.py   # Vectorized SafeMoE
│   │   ├── distributed_optimized.py # SwiGLU + Distributed Expert Parallel
│   │   ├── ssm.py              # Mamba 2 Mixer (SSD)
│   │   └── block.py            # Transformer Blocks
│   └── training/               # Training Components
│       ├── post_train.py       # DPO/ORPO/SimPO Losses
│       └── reasoning.py        # GRPO + Verifier Logic
├── examples/                   # Usage Examples
│   ├── benchmark_moe.py        # Performance benchmarks
│   └── train_demo.py           # Training demos
├── tests/                      # Comprehensive Test Suite
│   ├── test_agent.py           # Agent capabilities
│   ├── test_reasoning.py       # GRPO/RL tests
│   ├── test_post_train.py      # Preference learning tests
│   └── ...
└── requirements.txt            # Dependencies
```

## Features

### Core Safety
- **Safe Routing**: Automatic fallback to dense FFN when:
  - Routing confidence is below threshold
  - Expert capacity overflows
- **Z-Loss**: Stabilizes router logits during training
- **Vectorized Expert Computation**: SwiGLU + Einsum based batched execution (1.5x - 2x faster)

### DeepSeek-Style Shared Experts
Implement "Always-On" experts that process every token to capture common knowledge, while routed experts focus on specialized tasks.

**Shared Experts Configuration:**
- `n_shared_experts > 0`: Enables shared experts.
- Shared experts are implemented as a separate SwiGLU block that runs in parallel to the routed experts.
- In distributed mode, shared experts are replicated on each rank (standard practice).

### Vectorized Expert Computation (Recommended)

### Modern Components
- **RoPE Support**: Rotary positional embeddings (LLaMA/DeepSeek compatible)
- **GQA (Grouped Query Attention)**: Memory-efficient attention mechanism for long sequences.
- **RMSNorm**: Efficient normalization layer
- **SwiGLU**: GLU variants for FFNs

### Performance
- **KV Cache**: Efficient autoregressive generation

### Memory Augmentation
- **Engram Memory**: N-gram based memory lookup for pattern recognition
- **Gated Fusion**: Learned gates control memory contribution

### Analysis Tools
- **RoutingAnalyzer**: Track overflow/fallback rates
- **Visualization**: ASCII and matplotlib plotting

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Training
```bash
python examples/train_demo.py
```

### SFT (Supervised Fine-Tuning)
```bash
python examples/sft_demo.py
```

### DPO (Preference Alignment)
```bash
python examples/dpo_demo.py
```

### Run All Tests
```bash
python -m pytest tests/ -v
# Or individually:
python tests/test_safety.py
python tests/test_new_modules.py
python tests/test_engram_cache.py
```

## API Reference

### Core Components

```python
from safemoe import MoEConfig, SafeMoE, TransformerBlockSafeMoE

# Configure MoE
cfg = MoEConfig(
    d_model=512,
    d_ff=2048,
    n_experts=8,
    top_k=2,
    capacity_factor=1.25,
    route_threshold=0.1
)

# Create layers
moe = SafeMoE(cfg)
block = TransformerBlockSafeMoE(d_model=512, n_heads=8, moe_cfg=cfg)
```

### Engram Memory

```python
from safemoe import EngramConfig, EngramMemory

engram_cfg = EngramConfig(
    d_model=512,
    ngram_sizes=(2, 3, 4),
    table_size=500_000
)

engram = EngramMemory(engram_cfg)
augmented_hidden = engram(hidden_states, input_ids)
```

### KV Cache for Inference

```python
from safemoe import KVCache, CachedSafeMoEBlock

# Create cached block
block = CachedSafeMoEBlock(d_model=512, n_heads=8, moe_cfg=cfg)

# Prefill
output, aux, stats, cache = block(prompt_hidden, use_cache=True)

# Generate tokens
for _ in range(max_tokens):
    output, aux, stats, cache = block(new_token, kv_cache=cache, use_cache=True)
```

### Distributed Expert Parallelism

We provide two distributed implementations:
1. `DistributedSafeMoE`: Standard implementation with Python loops for experts.
2. **`OptimizedDistributedSafeMoE`** (Recommended): Vectorized implementation with **~1.4x - 1.7x speedup**.

```python
from safemoe import OptimizedDistributedMoEConfig, OptimizedDistributedSafeMoE

# Multi-GPU configuration
cfg = OptimizedDistributedMoEConfig(
    d_model=512,
    d_ff=2048,
    n_experts_global=8,
    top_k=2,
    use_fallback=True,
    expert_parallel_group=None  # or your dist.new_group()
)

model = OptimizedDistributedSafeMoE(cfg)

# Works on Single GPU (faster) or Multi-GPU (AllToAll)
y, aux, stats = model(x)
```

### Performance Benchmarks

| Model | Batch | Seq Len | Speedup |
|-------|-------|---------|---------|
| SafeMoE (Baseline) | 1 | 64 | 1.00x |
| **OptimizedDistributed** | 1 | 64 | **1.69x** |
| OptimizedDistributed | 4 | 256 | 1.33x |
| OptimizedDistributed | 8 | 512 | 1.21x |

*Benchmark run on CPU. Expect higher speedups on GPU due to vectorized operations.*

### Reasoning & RL (GRPO + Verifier)
Train your model to reason (Chain-of-Thought) using Group Relative Policy Optimization, the method behind DeepSeekMath.

```python
from safemoe.reasoning import ReasoningConfig, GRPOLoss, VerifierHead

# 1. Config
rl_cfg = ReasoningConfig(group_size=16, kl_beta=0.04)
grpo_loss = GRPOLoss(rl_cfg)

# 2. Training Loop (Pseudocode)
# Step 1: Generate G=16 completions per prompt
# Step 2: Score completions using Rule/Oracle or VerifierHead
rewards = oracle_verifier(completions)

# Step 3: Compute Loss (No Critic needed!)
loss, stats = grpo_loss(
    policy_logprobs, old_logprobs, ref_logprobs, 
    rewards, mask
)
loss.backward()
```

### Agentic Capabilities (ReAct & Tool Use)
Build autonomous agents that can use tools and reason.

```python
from safemoe.agent import SafeMoEAgent, Tool

def get_weather(location):
    return "Sunny in " + location

tools = [
    Tool(
        name="get_weather",
        description="Get current weather",
        parameters={"location": "string"},
        func=get_weather
    )
]

agent = SafeMoEAgent(model, tokenizer, tools)
# The agent will automatically inject tool definitions into the system prompt
# and parse model outputs to execute tools.
messages = agent.run("What's the weather in Tokyo?")
```

### Post-Training (DPO, ORPO, SimPO)
We provide a unified loss module for preference alignment, supporting state-of-the-art methods.

```python
from safemoe.post_train import PreferenceConfig, PreferenceLoss

# 1. DPO (Direct Preference Optimization)
cfg = PreferenceConfig(method="dpo", beta=0.1)
loss_fn = PreferenceLoss(cfg)
loss, metrics = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

# 2. SimPO (Simple PO - Reference Free)
cfg = PreferenceConfig(method="simpo", beta=2.0, simpo_gamma=1.0)
loss_fn = PreferenceLoss(cfg)
loss, metrics = loss_fn(policy_chosen, policy_rejected)

# 3. ORPO (Odds Ratio PO - Auxiliary Loss for SFT)
cfg = PreferenceConfig(method="orpo", lambda_orpo=0.1)
loss_fn = PreferenceLoss(cfg)
# Add this loss to your SFT loss
loss, metrics = loss_fn(policy_chosen, policy_rejected)
```

### Mamba 2 Hybrid Architecture

Combine the linear-time efficiency of State Space Models (SSMs) with the capacity of SafeMoE.

```python
from safemoe import MambaConfig, MambaSafeMoEBlock, OptimizedDistributedMoEConfig

# Configuration
mamba_cfg = MambaConfig(
    d_model=512,
    d_state=64,
    d_conv=4,
    expand=2,
    headdim=64
)

moe_cfg = OptimizedDistributedMoEConfig(
    d_model=512,
    n_experts_global=8,
    use_fallback=True
)

# Hybrid Block: Mamba Mixer (Time) + SafeMoE (Channel)
block = MambaSafeMoEBlock(mamba_cfg, moe_cfg)

x = torch.randn(1, 128, 512)
y, aux, stats = block(x)
```

### Generation

```python
from safemoe import GenerationConfig, InferenceEngine

engine = InferenceEngine(model, tokenizer, device="cuda")
text = engine.generate(
    "Once upon a time",
    max_new_tokens=100,
    temperature=0.7,
    top_k=50
)
```

## Architecture Diagram

```
Input Tokens
     │
     ▼
┌─────────────────┐
│ Engram Memory   │  ◀── Optional: N-gram pattern augmentation
│ (if enabled)    │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Layer Norm 1   │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Self-Attention │  ◀── RoPE or Standard, with KV Cache
│     + Residual  │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Layer Norm 2   │
└─────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│           SafeMoE               │
│  ┌───────────┐                  │
│  │  Router   │──▶ Top-K Select  │
│  └───────────┘                  │
│       │                         │
│  ┌────┴────┐    ┌───────────┐   │
│  │ Experts │    │  Fallback │   │  ◀── Dense FFN as safety net
│  └────┬────┘    │   (Dense) │   │
│       │         └─────┬─────┘   │
│       └───────────────┤         │
│                       ▼         │
│              Weighted Sum       │
│                + Residual       │
└─────────────────────────────────┘
     │
     ▼
Output Tokens
```

## Performance Tips

1. **Use VectorizedSafeMoE** for better GPU utilization
2. **Enable KV Cache** during inference for 10x+ speedup
3. **Monitor overflow_rate** - if consistently high, increase `capacity_factor`
4. **Tune route_threshold** based on your task's safety requirements

## License

MIT
