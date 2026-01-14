"""
Tests for Reasoning/RL module (GRPO).
"""
import sys
import os
import unittest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safemoe.reasoning import ReasoningConfig, GRPOLoss, VerifierHead

class TestReasoning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = ReasoningConfig(group_size=4)
        
    def test_verifier_head(self):
        """Test score head output shape."""
        d_model = 64
        head = VerifierHead(d_model)
        
        x = torch.randn(2, 10, d_model)
        score = head(x)
        
        self.assertEqual(score.shape, (2, 10, 1))
        
    def test_grpo_loss_computation(self):
        """Test GRPO loss calculation logic."""
        loss_fn = GRPOLoss(self.cfg)
        
        G, S = 4, 10
        # Mock rollouts
        # 4 sequences in a group
        policy_logprobs = torch.randn(G, S, requires_grad=True)
        old_logprobs = policy_logprobs.detach() # Ratio = 1 initially
        ref_logprobs = policy_logprobs.detach() - 0.1 # Small KL
        
        # Rewards: group statistics
        rewards = torch.tensor([1.0, 0.5, 2.0, -1.0])
        # Mean=0.625, Std approx 1.25
        
        mask = torch.ones(G, S)
        
        loss, stats = loss_fn(
            policy_logprobs, old_logprobs, ref_logprobs,
            rewards, mask
        )
        
        # Check backward
        loss.backward()
        
        self.assertTrue(torch.is_tensor(loss))
        self.assertIsNotNone(policy_logprobs.grad)
        self.assertIn("avg_advantage", stats)
        
        # Check advantage signs:
        # High reward (2.0) should correspond to negative gradient (maximize prob)
        # Low reward (-1.0) should correspond to positive gradient (suppress prob)
        # Note: Optimization minimizes loss, so loss = -Adv * log_prob
        # grad(loss) w.r.t log_prob = -Adv
        
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        # Index 2 is max reward (2.0) -> Pos Adv -> Neg Grad
        grad_high_reward = policy_logprobs.grad[2].mean().item()
        # Index 3 is min reward (-1.0) -> Neg Adv -> Pos Grad
        grad_low_reward = policy_logprobs.grad[3].mean().item()
        
        self.assertLess(grad_high_reward, 0, "High reward should have negative grad (maximize)")
        self.assertGreater(grad_low_reward, 0, "Low reward should have positive grad (minimize)")

if __name__ == '__main__':
    unittest.main()
