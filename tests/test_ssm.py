"""
Tests for Mamba SSM and MambaSafeMoEBlock
"""
import sys
import os
import unittest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safemoe.ssm import MambaConfig, Mamba2Mixer, MambaSafeMoEBlock
from safemoe.distributed_optimized import OptimizedDistributedMoEConfig

class TestMambaMixer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = MambaConfig(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=32,
            ngroups=1
        )
        
    def test_mixer_shape(self):
        """Test Mamba mixer input/output shapes."""
        model = Mamba2Mixer(self.cfg)
        x = torch.randn(2, 128, 64) # [B, S, D]
        
        y = model(x)
        self.assertEqual(y.shape, x.shape)
        
    def test_recurrence_logic(self):
        """Test basic causal property (future shouldn't affect past)."""
        model = Mamba2Mixer(self.cfg)
        model.eval()
        
        # Seq 1
        x1 = torch.randn(1, 10, 64)
        y1 = model(x1)
        
        # Seq 2 (First 10 same as Seq 1, then new tokens)
        x2 = torch.cat([x1, torch.randn(1, 5, 64)], dim=1)
        y2 = model(x2)
        
        # First 10 outputs should be identical
        self.assertTrue(torch.allclose(y1, y2[:, :10, :], atol=1e-5))

class TestMambaSafeMoEBlock(unittest.TestCase):
    def setUp(self):
        self.mamba_cfg = MambaConfig(d_model=64, d_state=16, headdim=16)
        self.moe_cfg = OptimizedDistributedMoEConfig(
            d_model=64, d_ff=128, n_experts_global=4, top_k=2
        )
        
    def test_block_forward(self):
        """Test hybrid block forward pass."""
        block = MambaSafeMoEBlock(self.mamba_cfg, self.moe_cfg)
        x = torch.randn(2, 32, 64)
        
        y, aux, stats = block(x)
        
        self.assertEqual(y.shape, x.shape)
        self.assertIn("router_z_loss", aux)
        self.assertIn("overflow_tokens", stats)
        
    def test_gradient_flow(self):
        """Test gradients through both Mamba and MoE parts."""
        block = MambaSafeMoEBlock(self.mamba_cfg, self.moe_cfg)
        x = torch.randn(2, 16, 64, requires_grad=True)
        
        y, aux, _ = block(x)
        loss = y.sum() + aux["router_z_loss"]
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(block.mamba.in_proj.weight.grad)
        self.assertIsNotNone(block.moe.router.weight.grad)

if __name__ == '__main__':
    unittest.main()
