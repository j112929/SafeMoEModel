"""
Tests for Optimized Distributed SafeMoE.
"""
import sys
import os
import unittest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safemoe.distributed_optimized import (
    OptimizedDistributedMoEConfig,
    OptimizedDistributedSafeMoE,
    OptimizedDistributedSafeMoEBlock,
    VectorizedExpertLayer,
    DenseFallbackFFN,
)


class TestVectorizedExpertLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        
    def test_forward_shape(self):
        """Test basic forward pass shape."""
        experts = VectorizedExpertLayer(
            n_experts=4, d_model=64, d_ff=128
        )
        x = torch.randn(100, 64)
        expert_ids = torch.randint(0, 4, (100,))
        
        y = experts(x, expert_ids)
        self.assertEqual(y.shape, x.shape)
        
    def test_forward_grouped_shape(self):
        """Test grouped forward pass shape."""
        experts = VectorizedExpertLayer(
            n_experts=4, d_model=64, d_ff=128
        )
        
        # Pre-sorted by expert
        expert_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
        x = torch.randn(10, 64)
        
        y = experts.forward_grouped(x, expert_ids)
        self.assertEqual(y.shape, x.shape)
        
    def test_forward_equivalence(self):
        """Test that forward and forward_grouped give same results."""
        experts = VectorizedExpertLayer(
            n_experts=4, d_model=64, d_ff=128
        )
        
        expert_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        x = torch.randn(8, 64)
        
        y1 = experts(x, expert_ids)
        y2 = experts.forward_grouped(x, expert_ids)
        
        self.assertTrue(torch.allclose(y1, y2, atol=1e-5))
        
    def test_with_bias(self):
        """Test with bias enabled."""
        experts = VectorizedExpertLayer(
            n_experts=4, d_model=64, d_ff=128, use_bias=True
        )
        x = torch.randn(10, 64)
        expert_ids = torch.randint(0, 4, (10,))
        
        y = experts(x, expert_ids)
        self.assertEqual(y.shape, x.shape)
        
    def test_gradient_flow(self):
        """Test gradients flow through experts."""
        experts = VectorizedExpertLayer(
            n_experts=4, d_model=64, d_ff=128
        )
        x = torch.randn(10, 64, requires_grad=True)
        expert_ids = torch.randint(0, 4, (10,))
        
        y = experts(x, expert_ids)
        y.sum().backward()
        
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(experts.w1.grad)
        self.assertIsNotNone(experts.w2.grad)


class TestOptimizedDistributedSafeMoE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = OptimizedDistributedMoEConfig(
            d_model=64,
            d_ff=128,
            n_experts_global=4,
            top_k=2,
            capacity_factor=1.5,
            use_fallback=True,
        )
        
    def test_forward_shape(self):
        """Test output shape matches input."""
        model = OptimizedDistributedSafeMoE(self.cfg)
        x = torch.randn(2, 16, 64)
        
        y, aux, stats = model(x)
        
        self.assertEqual(y.shape, x.shape)
        
    def test_aux_losses(self):
        """Test auxiliary losses are computed."""
        model = OptimizedDistributedSafeMoE(self.cfg)
        x = torch.randn(2, 16, 64)
        
        _, aux, _ = model(x)
        
        self.assertIn("router_z_loss", aux)
        self.assertIn("load_balance_loss", aux)
        
    def test_stats(self):
        """Test stats are computed."""
        model = OptimizedDistributedSafeMoE(self.cfg)
        x = torch.randn(2, 16, 64)
        
        _, _, stats = model(x)
        
        self.assertIn("moe_capacity", stats)
        self.assertIn("overflow_tokens", stats)
        self.assertIn("fallback_rate", stats)
        self.assertIn("n_local_experts", stats)
        
    def test_gradient_flow(self):
        """Test gradients flow through model."""
        model = OptimizedDistributedSafeMoE(self.cfg)
        x = torch.randn(2, 16, 64, requires_grad=True)
        
        y, aux, _ = model(x)
        loss = y.sum() + aux["router_z_loss"] + aux["load_balance_loss"]
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(model.router.weight.grad)
        self.assertIsNotNone(model.experts.w1.grad)
        
    def test_fallback_threshold(self):
        """Test high threshold triggers fallback."""
        cfg = OptimizedDistributedMoEConfig(
            d_model=64, d_ff=128, n_experts_global=4,
            top_k=1, route_threshold=1.0, use_fallback=True,
        )
        model = OptimizedDistributedSafeMoE(cfg)
        x = torch.randn(2, 16, 64)
        
        _, _, stats = model(x)
        
        self.assertGreater(stats["fallback_rate"].item(), 0.9)
        
    def test_capacity_overflow(self):
        """Test capacity overflow handling."""
        cfg = OptimizedDistributedMoEConfig(
            d_model=64, d_ff=128, n_experts_global=2,
            top_k=1, capacity_factor=0.1, min_capacity=1,
            use_fallback=True,
        )
        model = OptimizedDistributedSafeMoE(cfg)
        x = torch.randn(2, 16, 64)
        
        _, _, stats = model(x)
        
        self.assertGreater(stats["overflow_tokens"].item(), 0)
        
    def test_single_token(self):
        """Test single token input."""
        model = OptimizedDistributedSafeMoE(self.cfg)
        x = torch.randn(1, 1, 64)
        
        y, _, _ = model(x)
        self.assertEqual(y.shape, x.shape)
        
    def test_large_batch(self):
        """Test large batch."""
        model = OptimizedDistributedSafeMoE(self.cfg)
        x = torch.randn(8, 128, 64)
        
        y, _, _ = model(x)
        self.assertEqual(y.shape, x.shape)


class TestOptimizedDistributedSafeMoEBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = OptimizedDistributedMoEConfig(
            d_model=64, d_ff=128, n_experts_global=4, top_k=2,
        )
        
    def test_forward(self):
        """Test full block forward."""
        block = OptimizedDistributedSafeMoEBlock(
            d_model=64, n_heads=4, moe_cfg=self.cfg,
        )
        x = torch.randn(2, 16, 64)
        
        y, aux, stats = block(x)
        
        self.assertEqual(y.shape, x.shape)
        self.assertIn("router_z_loss", aux)
        
    def test_with_mask(self):
        """Test with attention mask."""
        block = OptimizedDistributedSafeMoEBlock(
            d_model=64, n_heads=4, moe_cfg=self.cfg,
        )
        x = torch.randn(2, 16, 64)
        
        mask = torch.triu(torch.ones(16, 16), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf'))
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        y, _, _ = block(x, attn_mask=mask)
        self.assertEqual(y.shape, x.shape)


class TestPerformanceComparison(unittest.TestCase):
    """Compare performance between implementations."""
    
    def test_vectorized_faster_than_loop(self):
        """Verify vectorized is at least as fast (hard to test properly without CUDA)."""
        cfg = OptimizedDistributedMoEConfig(
            d_model=64, d_ff=128, n_experts_global=4, top_k=2,
        )
        model = OptimizedDistributedSafeMoE(cfg)
        x = torch.randn(4, 32, 64)
        
        # Just verify it runs without error
        for _ in range(3):
            y, _, _ = model(x)
            
        self.assertEqual(y.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
