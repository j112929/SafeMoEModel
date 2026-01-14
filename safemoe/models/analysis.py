"""
Routing Analysis and Visualization Tools for SafeMoE
"""
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json


@dataclass
class RoutingStats:
    """Accumulated routing statistics across batches."""
    expert_counts: List[int] = field(default_factory=lambda: [])
    overflow_counts: List[int] = field(default_factory=list)
    fallback_rates: List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    
    def update(self, stats: Dict[str, torch.Tensor], n_experts: int):
        """Update with stats from a single forward pass."""
        self.overflow_counts.append(stats.get("overflow_tokens", torch.tensor(0)).item())
        self.fallback_rates.append(stats.get("fallback_rate", torch.tensor(0.0)).item())
        
        if not self.expert_counts:
            self.expert_counts = [0] * n_experts
            
    def summary(self) -> Dict:
        """Generate summary report."""
        return {
            "total_overflows": sum(self.overflow_counts),
            "mean_fallback_rate": sum(self.fallback_rates) / max(1, len(self.fallback_rates)),
            "max_fallback_rate": max(self.fallback_rates) if self.fallback_rates else 0,
            "samples": len(self.overflow_counts)
        }


class RoutingAnalyzer:
    """
    Analyzer for SafeMoE routing patterns.
    Attach to model to collect and visualize routing decisions.
    """
    def __init__(self, n_experts: int, n_layers: int = 1):
        self.n_experts = n_experts
        self.n_layers = n_layers
        self.layer_stats: List[RoutingStats] = [RoutingStats() for _ in range(n_layers)]
        self.step = 0
        
    def log_routing(self, layer_idx: int, stats: Dict[str, torch.Tensor]):
        """Log routing stats for a layer."""
        if layer_idx < len(self.layer_stats):
            self.layer_stats[layer_idx].update(stats, self.n_experts)
        self.step += 1
    
    def get_summary(self) -> Dict:
        """Get overall summary across all layers."""
        return {
            f"layer_{i}": stats.summary() 
            for i, stats in enumerate(self.layer_stats)
        }
    
    def print_ascii_report(self):
        """Print a simple ASCII visualization of routing health."""
        print("\n" + "=" * 50)
        print("SafeMoE Routing Analysis Report")
        print("=" * 50)
        
        for i, stats in enumerate(self.layer_stats):
            summary = stats.summary()
            print(f"\nLayer {i}:")
            print(f"  Samples analyzed: {summary['samples']}")
            print(f"  Total overflows: {summary['total_overflows']}")
            print(f"  Mean fallback rate: {summary['mean_fallback_rate']:.2%}")
            print(f"  Max fallback rate: {summary['max_fallback_rate']:.2%}")
            
            # Health indicator
            if summary['mean_fallback_rate'] < 0.1:
                health = "✓ HEALTHY"
            elif summary['mean_fallback_rate'] < 0.3:
                health = "⚠ MODERATE"
            else:
                health = "✗ HIGH FALLBACK"
            print(f"  Status: {health}")
        
        print("\n" + "=" * 50)
    
    def to_json(self) -> str:
        """Export stats to JSON."""
        return json.dumps(self.get_summary(), indent=2)
    
    def reset(self):
        """Reset all collected stats."""
        self.layer_stats = [RoutingStats() for _ in range(self.n_layers)]
        self.step = 0


def analyze_expert_affinity(
    model,
    dataloader,
    tokenizer=None,
    max_batches: int = 10,
    device: str = "cpu"
) -> Dict:
    """
    Analyze which types of tokens prefer which experts.
    Returns statistics on expert preferences.
    """
    from collections import defaultdict
    
    expert_token_counts = defaultdict(lambda: defaultdict(int))
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            
            # Get routing decisions (need to hook into model)
            # This is a simplified version - full implementation would
            # register forward hooks on MoE layers
            
            # Placeholder for actual implementation
            pass
    
    return dict(expert_token_counts)


def plot_routing_heatmap(stats_history: List[Dict], save_path: Optional[str] = None):
    """
    Plot expert usage over time as a heatmap.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract expert usage over steps
        n_steps = len(stats_history)
        if n_steps == 0:
            print("No stats to plot")
            return
            
        # Assume we track usage_mean, usage_max, usage_min
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ["expert_usage_mean", "expert_usage_max", "expert_usage_min"]
        titles = ["Mean Expert Usage", "Max Expert Usage", "Min Expert Usage"]
        
        for ax, metric, title in zip(axes, metrics, titles):
            values = [s.get(metric, 0) for s in stats_history]
            ax.plot(values)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel("Usage Rate")
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved routing heatmap to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("matplotlib required for plotting. Install with: pip install matplotlib")
