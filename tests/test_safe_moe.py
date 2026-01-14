import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from safemoe import MoEConfig, TransformerBlockSafeMoE

def test_training_loop():
    print("Initializing MoE Config...")
    cfg = MoEConfig(
        d_model=128,
        d_ff=512,
        n_experts=4,
        top_k=2,
        capacity_factor=1.5
    )
    
    # Create the block
    print("Creating TransformerBlockSafeMoE...")
    block = TransformerBlockSafeMoE(
        d_model=cfg.d_model, 
        n_heads=4, 
        moe_cfg=cfg, 
        resid_dropout=0.1
    )
    
    # Create dummy input: [Batch=2, Seq=10, D=128]
    x = torch.randn(2, 10, cfg.d_model)
    
    print("Forward pass...")
    output, aux_losses, stats = block(x)
    
    print("Output shape:", output.shape)
    print("Aux losses:", aux_losses)
    print("Stats:", stats)
    
    # Simple backward check
    loss = output.mean() + aux_losses['router_z_loss'] + aux_losses['load_balance_loss']
    loss.backward()
    print("Backward pass successful!")

if __name__ == "__main__":
    test_training_loop()
