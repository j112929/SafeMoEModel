"""
Benchmark comparing different MoE implementations.
Run with: python examples/benchmark_moe.py
"""
import os
import sys
import time
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safemoe import MoEConfig, SafeMoE
from safemoe.moe_vectorized import VectorizedSafeMoE
from safemoe.distributed_optimized import OptimizedDistributedMoEConfig, OptimizedDistributedSafeMoE


def benchmark_model(model, x, warmup=5, iterations=20, name="Model"):
    """Benchmark a model's forward pass."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    avg_time = (end - start) / iterations * 1000  # ms
    throughput = x.shape[0] * x.shape[1] / (avg_time / 1000)  # tokens/sec
    
    return avg_time, throughput


def main():
    print("=" * 60)
    print("SafeMoE Performance Benchmark")
    print("=" * 60)
    
    # Configuration
    batch_sizes = [1, 4, 8]
    seq_lengths = [64, 256, 512]
    
    d_model = 512
    d_ff = 2048
    n_experts = 8
    top_k = 2
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Config: d_model={d_model}, d_ff={d_ff}, n_experts={n_experts}, top_k={top_k}")
    
    # Create models
    cfg_basic = MoEConfig(
        d_model=d_model, d_ff=d_ff, n_experts=n_experts, top_k=top_k
    )
    
    cfg_optimized = OptimizedDistributedMoEConfig(
        d_model=d_model, d_ff=d_ff, n_experts_global=n_experts, top_k=top_k
    )
    
    model_basic = SafeMoE(cfg_basic).to(device)
    model_vectorized = VectorizedSafeMoE(cfg_basic).to(device)
    model_optimized = OptimizedDistributedSafeMoE(cfg_optimized).to(device)
    
    print(f"\nParameters:")
    print(f"  SafeMoE:              {sum(p.numel() for p in model_basic.parameters()):,}")
    print(f"  VectorizedSafeMoE:    {sum(p.numel() for p in model_vectorized.parameters()):,}")
    print(f"  OptimizedDistributed: {sum(p.numel() for p in model_optimized.parameters()):,}")
    
    results = []
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            x = torch.randn(batch_size, seq_len, d_model, device=device)
            tokens = batch_size * seq_len
            
            print(f"\n--- Batch={batch_size}, Seq={seq_len}, Tokens={tokens} ---")
            
            # Basic SafeMoE
            time_basic, tp_basic = benchmark_model(model_basic, x, name="SafeMoE")
            print(f"SafeMoE:              {time_basic:.2f}ms, {tp_basic:.0f} tok/s")
            
            # Vectorized
            time_vec, tp_vec = benchmark_model(model_vectorized, x, name="Vectorized")
            speedup_vec = time_basic / time_vec
            print(f"Vectorized:           {time_vec:.2f}ms, {tp_vec:.0f} tok/s ({speedup_vec:.2f}x)")
            
            # Optimized Distributed
            time_opt, tp_opt = benchmark_model(model_optimized, x, name="OptimizedDist")
            speedup_opt = time_basic / time_opt
            print(f"OptimizedDistributed: {time_opt:.2f}ms, {tp_opt:.0f} tok/s ({speedup_opt:.2f}x)")
            
            results.append({
                "batch": batch_size,
                "seq": seq_len,
                "tokens": tokens,
                "basic_ms": time_basic,
                "vec_ms": time_vec,
                "opt_ms": time_opt,
                "vec_speedup": speedup_vec,
                "opt_speedup": speedup_opt,
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    avg_vec_speedup = sum(r["vec_speedup"] for r in results) / len(results)
    avg_opt_speedup = sum(r["opt_speedup"] for r in results) / len(results)
    
    print(f"\nAverage speedup vs basic SafeMoE:")
    print(f"  Vectorized:           {avg_vec_speedup:.2f}x")
    print(f"  OptimizedDistributed: {avg_opt_speedup:.2f}x")
    
    # Best speedup
    best = max(results, key=lambda r: r["opt_speedup"])
    print(f"\nBest speedup: {best['opt_speedup']:.2f}x at batch={best['batch']}, seq={best['seq']}")


if __name__ == "__main__":
    main()
