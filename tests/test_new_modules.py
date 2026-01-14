"""
Test suite for new SafeMoE modules:
- VectorizedSafeMoE
- RoPE Attention
- Routing Analysis
"""
import sys
import os
import unittest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safemoe import (
    MoEConfig, 
    VectorizedSafeMoE,
    RoPEMultiheadAttention,
    RMSNorm,
    RoutingAnalyzer
)


class TestVectorizedSafeMoE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = MoEConfig(
            d_model=64,
            d_ff=128,
            n_experts=4,
            top_k=2,
        )
        
    def test_forward_shape(self):
        """Test output shape matches input."""
        model = VectorizedSafeMoE(self.cfg)
        x = torch.randn(2, 10, 64)
        y, aux, stats = model(x)
        
        self.assertEqual(y.shape, x.shape)
        self.assertIn("router_z_loss", aux)
        self.assertIn("overflow_rate", stats)
        
    def test_gradient_flow(self):
        """Ensure gradients flow through vectorized implementation."""
        model = VectorizedSafeMoE(self.cfg)
        x = torch.randn(2, 10, 64, requires_grad=True)
        y, _, _ = model(x)
        loss = y.sum()
        loss.backward()
        
        self.assertIsNotNone(model.w1.grad)
        self.assertIsNotNone(model.w2.grad)


class TestRoPEAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        
    def test_forward_shape(self):
        """Test RoPE attention output shape."""
        attn = RoPEMultiheadAttention(
            d_model=64,
            n_heads=4,
            max_seq_len=128
        )
        x = torch.randn(2, 32, 64)
        out = attn(x)
        
        self.assertEqual(out.shape, x.shape)
        
    def test_with_mask(self):
        """Test with causal attention mask."""
        attn = RoPEMultiheadAttention(d_model=64, n_heads=4, max_seq_len=128)
        x = torch.randn(2, 16, 64)
        
        # Create causal mask
        S = 16
        mask = torch.full((S, S), float("-inf"))
        mask = torch.triu(mask, diagonal=1).view(1, 1, S, S)
        
        out = attn(x, attn_mask=mask)
        self.assertEqual(out.shape, x.shape)


class TestRMSNorm(unittest.TestCase):
    def test_normalization(self):
        """Test RMSNorm normalizes correctly."""
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64) * 10  # Large values
        
        out = norm(x)
        
        # RMS should be approximately 1 after normalization
        rms = out.pow(2).mean(dim=-1).sqrt()
        # With learned weight=1, output RMS should be close to 1
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=0.1))


class TestRoutingAnalyzer(unittest.TestCase):
    def test_logging(self):
        """Test that RoutingAnalyzer collects stats."""
        analyzer = RoutingAnalyzer(n_experts=4, n_layers=2)
        
        # Simulate some stats
        stats = {
            "overflow_tokens": torch.tensor(5),
            "fallback_rate": torch.tensor(0.1),
        }
        
        analyzer.log_routing(0, stats)
        analyzer.log_routing(1, stats)
        
        summary = analyzer.get_summary()
        self.assertIn("layer_0", summary)
        self.assertIn("layer_1", summary)
        
        self.assertEqual(summary["layer_0"]["samples"], 1)


if __name__ == '__main__':
    unittest.main()
