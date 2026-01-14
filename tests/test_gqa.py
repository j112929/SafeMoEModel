"""
Tests for GQA attention module.
"""
import sys
import os
import unittest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safemoe.attention import MultiheadSelfAttention

class TestGQA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        
    def test_mha_equivalence(self):
        """Standard MHA (n_kv_heads=n_heads) should work as expected."""
        B, S, D = 2, 16, 64
        attn = MultiheadSelfAttention(d_model=D, n_heads=8, n_kv_heads=8)
        x = torch.randn(B, S, D)
        y = attn(x)
        self.assertEqual(y.shape, x.shape)
        
    def test_gqa_shape(self):
        """GQA (n_kv_heads < n_heads) should output correct shape."""
        B, S, D = 2, 16, 64
        # 8 Query heads, 2 KV heads (GQA-4)
        attn = MultiheadSelfAttention(d_model=D, n_heads=8, n_kv_heads=2)
        x = torch.randn(B, S, D)
        y = attn(x)
        self.assertEqual(y.shape, x.shape)
        
    def test_mqa_shape(self):
        """MQA (Multi-Query Attention, n_kv_heads=1)."""
        B, S, D = 2, 16, 64
        attn = MultiheadSelfAttention(d_model=D, n_heads=8, n_kv_heads=1)
        x = torch.randn(B, S, D)
        y = attn(x)
        self.assertEqual(y.shape, x.shape)
        
    def test_invalid_heads(self):
        """Should raise error if n_heads not divisible by n_kv_heads."""
        with self.assertRaises(AssertionError):
            MultiheadSelfAttention(d_model=64, n_heads=8, n_kv_heads=3)

if __name__ == '__main__':
    unittest.main()
