
import sys
import os
import unittest
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safemoe import MoEConfig, SafeMoE

class TestSafeMoE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        
    def test_overflow_mechanism(self):
        """
        Test that overflow actually triggers when we force it.
        We set top_k=1, but capacity very low (0).
        """
        print("\n=== Testing Overflow Mechanism ===")
        
        # 1. Config with minimal capacity
        cfg = MoEConfig(
            d_model=32,
            d_ff=64,
            n_experts=2,
            top_k=1,
            capacity_factor=0.0, # Force minimal capacity
            min_capacity=0       # Allow 0 capacity to force overflow
        )
        model = SafeMoE(cfg)
        
        # 2. Input: 10 tokens
        B, S, D = 1, 10, 32
        x = torch.randn(B, S, D)
        
        # 3. Forward
        # Since capacity is 0 (or near 0), tokens should overflow
        y, aux, stats = model(x)
        
        print(f"Stats: {stats}")
        
        # 4. Assertions
        # Checks if overflow tokens is greater than 0
        self.assertGreater(stats['overflow_tokens'].item(), 0, "Should have overflow tokens with 0 capacity factor")
        self.assertGreater(stats['fallback_rate'].item(), 0.0, "Fallback rate should be > 0")
        
    def test_low_confidence_fallback(self):
        """
        Test that low confidence tokens are routed to fallback.
        We set route_threshold to a high value (e.g., 0.99) and random logits likely won't meet it.
        """
        print("\n=== Testing Low Confidence Fallback ===")
        cfg = MoEConfig(
            d_model=32,
            d_ff=64,
            n_experts=4,
            top_k=1,
            route_threshold=1.0 # Impossible to reach with softmax unless perfect one-hot
        )
        model = SafeMoE(cfg)
        
        x = torch.randn(1, 10, 32)
        y, aux, stats = model(x)
        
        print(f"Stats: {stats}")
        
        # Almost all should fall back
        self.assertGreater(stats['fallback_rate'].item(), 0.8, "Most tokens should fallback due to high threshold")

    def test_gradient_flow(self):
        """
        Ensure gradients flow through both Experts and Fallback.
        """
        print("\n=== Testing Gradient Flow ===")
        cfg = MoEConfig(
            d_model=32, 
            d_ff=64, 
            n_experts=2, 
            top_k=1,
            capacity_factor=0.0, # Force fallback
            min_capacity=0
        )
        model = SafeMoE(cfg)
        
        x = torch.randn(1, 5, 32, requires_grad=True)
        y, _, _ = model(x)
        loss = y.sum()
        loss.backward()
        
        self.assertIsNotNone(model.router.router.weight.grad, "Router should get grads")
        self.assertIsNotNone(model.fallback.ffn.w1.weight.grad, "Fallback should get grads (since we forced overflow)")

if __name__ == '__main__':
    unittest.main()
