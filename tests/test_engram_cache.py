"""
Tests for Engram Memory and KV Cache modules.
"""
import sys
import os
import unittest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safemoe import (
    EngramConfig,
    EngramMemory,
    NGramHasher,
    KVCache,
    MultiheadSelfAttentionWithCache,
    CachedTransformerBlock,
    MoEConfig,
    CachedSafeMoEBlock,
)


class TestNGramHasher(unittest.TestCase):
    def test_hash_shape(self):
        """Test output shape of n-gram hashing."""
        input_ids = torch.randint(0, 1000, (2, 32))
        
        addrs = NGramHasher.compute_hashes(
            input_ids, n=3, n_hash=2, table_size=10000
        )
        
        self.assertEqual(addrs.shape, (2, 32, 2))
        
    def test_hash_determinism(self):
        """Test that hashing is deterministic."""
        input_ids = torch.randint(0, 1000, (2, 32))
        
        addrs1 = NGramHasher.compute_hashes(input_ids, n=3, n_hash=2, table_size=10000)
        addrs2 = NGramHasher.compute_hashes(input_ids, n=3, n_hash=2, table_size=10000)
        
        self.assertTrue(torch.equal(addrs1, addrs2))
        
    def test_invalid_positions_zeroed(self):
        """Test that positions without full n-gram have zero addresses."""
        input_ids = torch.randint(0, 1000, (2, 10))
        
        addrs = NGramHasher.compute_hashes(input_ids, n=4, n_hash=2, table_size=10000)
        
        # First 3 positions should be zero (need 4-gram but don't have enough history)
        self.assertTrue((addrs[:, :3, :] == 0).all())


class TestEngramMemory(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = EngramConfig(
            d_model=64,
            ngram_sizes=(2, 3),
            n_hashes=2,
            table_size=1000,
        )
        
    def test_forward_shape(self):
        """Test output shape matches input."""
        engram = EngramMemory(self.config)
        
        hidden = torch.randn(2, 16, 64)
        input_ids = torch.randint(0, 1000, (2, 16))
        
        out = engram(hidden, input_ids)
        self.assertEqual(out.shape, hidden.shape)
        
    def test_gradient_flow(self):
        """Test gradients flow through engram."""
        engram = EngramMemory(self.config)
        
        hidden = torch.randn(2, 16, 64, requires_grad=True)
        input_ids = torch.randint(0, 1000, (2, 16))
        
        out = engram(hidden, input_ids)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(engram.proj.weight.grad)
        
    def test_memory_usage_report(self):
        """Test memory usage reporting."""
        engram = EngramMemory(self.config)
        usage = engram.get_memory_usage()
        
        self.assertIn("tables", usage)
        self.assertIn("total_params", usage)


class TestKVCache(unittest.TestCase):
    def test_initial_state(self):
        """Test initial cache is empty."""
        cache = KVCache()
        self.assertEqual(cache.seq_len, 0)
        
    def test_update(self):
        """Test cache update accumulates correctly."""
        cache = KVCache()
        
        k1 = torch.randn(2, 4, 5, 16)  # [B, H, T, d]
        v1 = torch.randn(2, 4, 5, 16)
        
        k_out, v_out = cache.update(k1, v1)
        self.assertEqual(cache.seq_len, 5)
        
        k2 = torch.randn(2, 4, 3, 16)
        v2 = torch.randn(2, 4, 3, 16)
        
        k_out, v_out = cache.update(k2, v2)
        self.assertEqual(cache.seq_len, 8)
        self.assertEqual(k_out.shape, (2, 4, 8, 16))
        
    def test_reset(self):
        """Test cache reset."""
        cache = KVCache()
        cache.update(torch.randn(2, 4, 5, 16), torch.randn(2, 4, 5, 16))
        cache.reset()
        self.assertEqual(cache.seq_len, 0)


class TestCachedAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        
    def test_no_cache(self):
        """Test attention works without cache."""
        attn = MultiheadSelfAttentionWithCache(d_model=64, n_heads=4)
        x = torch.randn(2, 16, 64)
        
        out, cache = attn(x, use_cache=False)
        self.assertEqual(out.shape, x.shape)
        self.assertIsNone(cache)
        
    def test_with_cache(self):
        """Test attention with cache returns cache."""
        attn = MultiheadSelfAttentionWithCache(d_model=64, n_heads=4)
        x = torch.randn(2, 16, 64)
        
        out, cache = attn(x, use_cache=True)
        self.assertEqual(out.shape, x.shape)
        self.assertIsNotNone(cache)
        self.assertEqual(cache.seq_len, 16)
        
    def test_incremental_generation(self):
        """Test incremental token generation with cache."""
        attn = MultiheadSelfAttentionWithCache(d_model=64, n_heads=4)
        
        # Process initial sequence
        x0 = torch.randn(2, 10, 64)
        out0, cache = attn(x0, use_cache=True)
        self.assertEqual(cache.seq_len, 10)
        
        # Generate one token at a time
        x1 = torch.randn(2, 1, 64)
        out1, cache = attn(x1, kv_cache=cache, use_cache=True)
        self.assertEqual(out1.shape, (2, 1, 64))
        self.assertEqual(cache.seq_len, 11)
        
        # Another token
        x2 = torch.randn(2, 1, 64)
        out2, cache = attn(x2, kv_cache=cache, use_cache=True)
        self.assertEqual(cache.seq_len, 12)


class TestCachedSafeMoEBlock(unittest.TestCase):
    def test_forward_with_cache(self):
        """Test SafeMoE block with cache."""
        moe_cfg = MoEConfig(d_model=64, d_ff=128, n_experts=4, top_k=2)
        block = CachedSafeMoEBlock(d_model=64, n_heads=4, moe_cfg=moe_cfg)
        
        x = torch.randn(2, 10, 64)
        out, aux, stats, cache = block(x, use_cache=True)
        
        self.assertEqual(out.shape, x.shape)
        self.assertIn("router_z_loss", aux)
        self.assertEqual(cache.seq_len, 10)


if __name__ == '__main__':
    unittest.main()
