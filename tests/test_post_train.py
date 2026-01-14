"""
Tests for Preference Optimization Loss module (DPO, SimPO, ORPO).
"""
import sys
import os
import unittest
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safemoe.post_train import PreferenceConfig, PreferenceLoss

class TestPreferenceLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        
        # Mock inputs: Batch=2
        # Chosen > Rejected usually means higher logp
        self.policy_chosen = torch.tensor([-5.0, -2.0])
        self.policy_rejected = torch.tensor([-6.0, -5.0]) # Gap: +1.0, +3.0
        
        self.ref_chosen = torch.tensor([-5.0, -2.0])
        self.ref_rejected = torch.tensor([-6.0, -5.0])
        
    def test_dpo_loss(self):
        """Test DPO computation."""
        cfg = PreferenceConfig(method="dpo", beta=0.1)
        loss_fn = PreferenceLoss(cfg)
        
        loss, metrics = loss_fn(
            self.policy_chosen, self.policy_rejected,
            self.ref_chosen, self.ref_rejected
        )
        
        # Check output
        self.assertTrue(torch.is_tensor(loss))
        self.assertIn("dpo_loss", metrics)
        self.assertIn("reward_margin", metrics)
        
        # Rewards should be positive since policy matches ref perfectly here
        # Actually reward = beta * (pi - ref). Here pi=ref, so reward=0.
        self.assertAlmostEqual(metrics["reward_margin"].item(), 0.0)

    def test_simpo_loss(self):
        """Test SimPO computation (no ref)."""
        cfg = PreferenceConfig(method="simpo", beta=1.0, simpo_gamma=0.5)
        loss_fn = PreferenceLoss(cfg)
        
        # Gap is 1.0 and 3.0. Gamma is 0.5.
        # Logits: (1.0 - 0.5) = 0.5, (3.0 - 0.5) = 2.5
        # Loss: -logsigmoid(0.5) and -logsigmoid(2.5)
        
        loss, metrics = loss_fn(self.policy_chosen, self.policy_rejected)
        
        expected_Loss = (-F.logsigmoid(torch.tensor(0.5)) - F.logsigmoid(torch.tensor(2.5))) / 2
        self.assertTrue(torch.allclose(loss, expected_Loss))
        self.assertIn("simpo_loss", metrics)
        
    def test_orpo_loss(self):
        """Test ORPO logic."""
        cfg = PreferenceConfig(method="orpo", lambda_orpo=1.0)
        loss_fn = PreferenceLoss(cfg)
        
        loss, metrics = loss_fn(self.policy_chosen, self.policy_rejected)
        
        self.assertTrue(torch.is_tensor(loss))
        self.assertIn("orpo_odds_loss", metrics)

if __name__ == '__main__':
    unittest.main()
