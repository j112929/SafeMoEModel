"""
Tests for Distributed SafeMoE.
Tests both single-GPU and simulated multi-GPU behavior.
"""
import sys
import os
import unittest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safemoe.distributed import (
    DistributedMoEConfig,
    DistributedSafeMoE,
    DistributedSafeMoEBlock,
    ExpertFFN,
    DenseFallbackFFN,
)


class TestDistributedMoEConfig(unittest.TestCase):
    def test_default_config(self):
        """Test default configuration values."""
        cfg = DistributedMoEConfig()
        self.assertEqual(cfg.n_experts_global, 8)
        self.assertEqual(cfg.top_k, 2)
        self.assertTrue(cfg.use_fallback)


class TestExpertFFN(unittest.TestCase):
    def test_forward(self):
        """Test ExpertFFN forward pass."""
        expert = ExpertFFN(d_model=64, d_ff=128)
        x = torch.randn(10, 64)
        y = expert(x)
        self.assertEqual(y.shape, x.shape)


class TestDistributedSafeMoE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = DistributedMoEConfig(
            d_model=64,
            d_ff=128,
            n_experts_global=4,
            top_k=2,
            capacity_factor=1.5,
            use_fallback=True,
        )
    
    def test_forward_shape(self):
        """Test output shape matches input."""
        model = DistributedSafeMoE(self.cfg)
        x = torch.randn(2, 16, 64)
        
        y, aux, stats = model(x)
        
        self.assertEqual(y.shape, x.shape)
        
    def test_aux_losses_present(self):
        """Test auxiliary losses are computed."""
        model = DistributedSafeMoE(self.cfg)
        x = torch.randn(2, 16, 64)
        
        _, aux, _ = model(x)
        
        self.assertIn("router_z_loss", aux)
        self.assertIn("load_balance_loss", aux)
        
    def test_stats_present(self):
        """Test routing stats are computed."""
        model = DistributedSafeMoE(self.cfg)
        x = torch.randn(2, 16, 64)
        
        _, _, stats = model(x)
        
        self.assertIn("moe_capacity", stats)
        self.assertIn("overflow_tokens", stats)
        self.assertIn("fallback_rate", stats)
        self.assertIn("world_size", stats)
        
    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        model = DistributedSafeMoE(self.cfg)
        x = torch.randn(2, 16, 64, requires_grad=True)
        
        y, aux, _ = model(x)
        loss = y.sum() + aux["router_z_loss"] + aux["load_balance_loss"]
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(model.router.weight.grad)
        
    def test_fallback_triggered_by_threshold(self):
        """Test that high threshold triggers fallback."""
        cfg = DistributedMoEConfig(
            d_model=64,
            d_ff=128,
            n_experts_global=4,
            top_k=1,
            route_threshold=1.0,  # Impossible to reach
            use_fallback=True,
        )
        model = DistributedSafeMoE(cfg)
        x = torch.randn(2, 16, 64)
        
        _, _, stats = model(x)
        
        # All tokens should go to fallback
        self.assertGreater(stats["fallback_rate"].item(), 0.9)
        
    def test_capacity_overflow(self):
        """Test capacity overflow handling."""
        cfg = DistributedMoEConfig(
            d_model=64,
            d_ff=128,
            n_experts_global=2,
            top_k=1,
            capacity_factor=0.1,  # Very low capacity
            min_capacity=1,
            use_fallback=True,
        )
        model = DistributedSafeMoE(cfg)
        x = torch.randn(2, 16, 64)
        
        _, _, stats = model(x)
        
        # Should have overflow
        self.assertGreater(stats["overflow_tokens"].item(), 0)
        
    def test_no_fallback_mode(self):
        """Test model works without fallback."""
        cfg = DistributedMoEConfig(
            d_model=64,
            d_ff=128,
            n_experts_global=4,
            top_k=2,
            use_fallback=False,
        )
        model = DistributedSafeMoE(cfg)
        x = torch.randn(2, 16, 64)
        
        y, _, _ = model(x)
        self.assertEqual(y.shape, x.shape)
        
    def test_single_gpu_mode(self):
        """Test that single GPU mode uses local expert forward."""
        model = DistributedSafeMoE(self.cfg)
        
        self.assertEqual(model.world_size, 1)
        self.assertEqual(model.rank, 0)
        self.assertEqual(model.n_local_experts, 4)  # All experts are local


class TestDistributedSafeMoEBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = DistributedMoEConfig(
            d_model=64,
            d_ff=128,
            n_experts_global=4,
            top_k=2,
        )
    
    def test_forward(self):
        """Test full block forward."""
        block = DistributedSafeMoEBlock(
            d_model=64,
            n_heads=4,
            moe_cfg=self.cfg,
        )
        x = torch.randn(2, 16, 64)
        
        y, aux, stats = block(x)
        
        self.assertEqual(y.shape, x.shape)
        self.assertIn("router_z_loss", aux)
        
    def test_with_attn_mask(self):
        """Test block with attention mask."""
        block = DistributedSafeMoEBlock(
            d_model=64,
            n_heads=4,
            moe_cfg=self.cfg,
        )
        x = torch.randn(2, 16, 64)
        
        # Causal mask
        mask = torch.triu(torch.ones(16, 16), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf'))
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        y, _, _ = block(x, attn_mask=mask)
        self.assertEqual(y.shape, x.shape)


class TestEdgeCases(unittest.TestCase):
    def test_empty_batch(self):
        """Test handling of very small inputs."""
        cfg = DistributedMoEConfig(
            d_model=64,
            d_ff=128,
            n_experts_global=4,
            top_k=2,
        )
        model = DistributedSafeMoE(cfg)
        
        # Single token
        x = torch.randn(1, 1, 64)
        y, _, _ = model(x)
        self.assertEqual(y.shape, x.shape)
        
    def test_large_batch(self):
        """Test with larger batch."""
        cfg = DistributedMoEConfig(
            d_model=64,
            d_ff=128,
            n_experts_global=4,
            top_k=2,
        )
        model = DistributedSafeMoE(cfg)
        
        x = torch.randn(8, 128, 64)
        y, _, _ = model(x)
        self.assertEqual(y.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
