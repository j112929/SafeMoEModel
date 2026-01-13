import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from SafeMOE import MoEConfig
from MultiheadSelfAttention import TransformerBlockSafeMoE

def run_training_demo():
    # 1. Configuration
    print("Setting up config...")
    cfg = MoEConfig(
        d_model=256,
        d_ff=1024,
        n_experts=4,
        top_k=2,
        capacity_factor=1.25,
        router_z_loss=0.001,
        load_balance_loss=0.01
    )
    
    # 2. Model: A simple sequence model with one SafeMoE Block
    print("Building model...")
    class SimpleMoEModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.embed = nn.Embedding(1000, cfg.d_model) # vocab size 1000
            self.block = TransformerBlockSafeMoE(cfg.d_model, n_heads=4, moe_cfg=cfg)
            self.head = nn.Linear(cfg.d_model, 1000)
        
        def forward(self, x):
            x = self.embed(x)
            x, aux, stats = self.block(x)
            logits = self.head(x)
            return logits, aux, stats

    model = SimpleMoEModel(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # 3. Dummy Data
    print("Generating dummy data...")
    batch_size = 8
    seq_len = 32
    # 10 batches
    input_data = torch.randint(0, 1000, (80, seq_len))
    target_data = torch.randint(0, 1000, (80, seq_len))
    
    dataset = TensorDataset(input_data, target_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 4. Training Loop
    print("Starting training loop...")
    model.train()
    for epoch in range(2):
        total_loss = 0
        for i, (xb, yb) in enumerate(loader):
            optimizer.zero_grad()
            
            logits, aux, stats = model(xb)
            
            # CE Loss
            loss_task = nn.CrossEntropyLoss()(logits.view(-1, 1000), yb.view(-1))
            
            # Aux Losses
            loss_aux = aux["router_z_loss"] + aux["load_balance_loss"]
            
            loss = loss_task + loss_aux
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 5 == 0:
                print(f"Epoch {epoch} Step {i} | Task Loss: {loss_task.item():.4f} | Aux Loss: {loss_aux.item():.4f}")
                print(f"  -> Stats: Overflow Rate={stats['overflow_rate']:.2f}, Expert Usage Min/Max={stats['expert_usage_min']:.2f}/{stats['expert_usage_max']:.2f}")

    print("Training demo completed successfully.")

if __name__ == "__main__":
    run_training_demo()
