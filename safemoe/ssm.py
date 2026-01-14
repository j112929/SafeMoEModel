"""
Mamba 2-style State Space Model (SSM) implementation in pure PyTorch.
Implements the Structured State Space Duality (SSD) mechanism.

Reference: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
(Dao et al., 2024)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MambaConfig:
    d_model: int = 512
    d_state: int = 64     # SSM state expansion factor (N)
    d_conv: int = 4       # Local convolution width
    expand: int = 2       # Block expansion factor
    headdim: int = 64     # Head dimension (P)
    ngroups: int = 1      # Number of groups for GQA-style behavior
    chunk_size: int = 256 # Chunk size for efficient computation

class Mamba2Mixer(nn.Module):
    """
    Mamba 2 Mixer Layer (Simplified PyTorch Implementation).
    
    Uses the "SSD" (State Space Duality) formulation which connects
    Linear Attention and SSMs.
    """
    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.d_inner = cfg.expand * cfg.d_model
        self.headdim = cfg.headdim
        self.nheads = self.d_inner // self.headdim
        assert self.d_inner % self.headdim == 0
        
        # Order: z, x, B, C, dt
        # z: gate
        # x: input to SSM
        # B, C: dynamics
        # dt: timescale
        d_in_proj = 2 * self.d_inner + 2 * self.cfg.ngroups * self.cfg.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)

        # 1D Conv for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner + 2 * self.cfg.ngroups * self.cfg.d_state,
            out_channels=self.d_inner + 2 * self.cfg.ngroups * self.cfg.d_state,
            bias=True,
            kernel_size=cfg.d_conv,
            groups=self.d_inner + 2 * self.cfg.ngroups * self.cfg.d_state,
            padding=cfg.d_conv - 1,
        )

        # A parameter (decay rate), structure: [nheads]
        self.A_log = nn.Parameter(torch.log(torch.randn(self.nheads).abs() + 1))
        
        # D parameter (skip connection), structure: [nheads]
        self.D = nn.Parameter(torch.ones(self.nheads))
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.norm = nn.RMSNorm(self.d_inner)


    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: [B, S, D]
        Returns:
            out: [B, S, D]
        """
        B, S, D = u.shape
        
        # 1. Project inputs
        zxbcdt = self.in_proj(u)  # [B, S, d_in_proj]
        
        # Split projections
        # z: [B, S, d_inner] (Gate)
        # x: [B, S, d_inner] (Signal)
        # B_state: [B, S, ngroups * d_state]
        # C_state: [B, S, ngroups * d_state]
        # dt: [B, S, nheads]
        
        A_dim = self.d_inner + 2 * self.cfg.ngroups * self.cfg.d_state
        z = zxbcdt[:, :, :self.d_inner]
        x_bc_seq = zxbcdt[:, :, self.d_inner:self.d_inner+A_dim]
        dt = zxbcdt[:, :, -self.nheads:]
        
        # 2. Convolution (Time-mix normalization)
        # Transpose for Conv1d: [B, C, S]
        x_bc_conv = self.conv1d(x_bc_seq.transpose(1, 2))[:, :, :S]
        x_bc_conv = x_bc_conv.transpose(1, 2)
        x_bc_conv = F.silu(x_bc_conv) # Activation after conv
        
        # Split convolution output
        x = x_bc_conv[:, :, :self.d_inner]
        B_state = x_bc_conv[:, :, self.d_inner:self.d_inner + self.cfg.ngroups * self.cfg.d_state]
        C_state = x_bc_conv[:, :, self.d_inner + self.cfg.ngroups * self.cfg.d_state:]

        # 3. SSM / Linear Attention Chunking
        # Reshape for multi-head
        x = x.view(B, S, self.nheads, self.headdim)
        z = z.view(B, S, self.nheads, self.headdim)
        dt = dt.view(B, S, self.nheads) # [B, S, H]
        dt = F.softplus(dt)  # Ensure positive timescale
        
        # Broadcast B, C to heads (GQA logic)
        # B_state: [B, S, G, N] -> [B, S, H, N]
        ratio = self.nheads // self.cfg.ngroups
        B_state = B_state.view(B, S, self.cfg.ngroups, self.cfg.d_state)
        C_state = C_state.view(B, S, self.cfg.ngroups, self.cfg.d_state)
        B_state = B_state.repeat_interleave(ratio, dim=2)
        C_state = C_state.repeat_interleave(ratio, dim=2)

        # Discretize A
        A = -torch.exp(self.A_log)  # [H] (Force negative for interaction decay)
        
        # Core SSD (Structured State Space Duality) Computation
        # y = SSD(x, A, B, C, dt)
        y = self.ssd_minimal(x, A, B_state, C_state, dt)
        
        # 4. Gating and Output
        y = y * F.silu(z)  # Gated
        y = y.reshape(B, S, self.d_inner)
        out = self.out_proj(self.norm(y))
        
        return out

    def ssd_minimal(self, x, A, B_param, C_param, dt):
        """
        Minimal pure-PyTorch implementation of SSD.
        Simulates the "Semi-Separable Matrix" multiplication.
        
        Args:
            x: [B, S, H, P]  (Input)
            A: [H]           (Decay)
            B_param: [B, S, H, N]  (State B)
            C_param: [B, S, H, N]  (State C)
            dt: [B, S, H]    (Timescale)
            
        Returns:
            y: [B, S, H, P]
        """
        # Note: This is a simplified recurrence loop (O(S)) for demonstration and portability.
        # A real optimized Mamba 2 kernel uses chunked parallel scan (O(log S)).
        
        bsz, seqlen, nheads, headdim = x.shape
        d_state = B_param.shape[-1]
        
        # Discretize Loop
        # h_t = (1 - dt * A) * h_{t-1} + dt * B * x_t
        # y_t = C * h_t + D * x_t
        # But Mamba 2 uses a slightly cleaner projection form.
        
        # We'll use the "Diagonal SSM" recurrence for correctness in pure PyTorch
        # approx: A_bar = exp(A * dt)
        dA = torch.exp(A * dt)  # [B, S, H]
        dA = dA.unsqueeze(-1)   # [B, S, H, 1] decay factor
        
        # x_projs = B * x
        # This is strictly "scalar" SSM per head channel if we view x as [H, P]
        # But Mamba 2 usually maps (H, N) state.
        # Let's align with standard SSM recurrence:
        # State H: [B, H, N, P]
        
        h = torch.zeros(bsz, nheads, d_state, headdim, device=x.device)
        ys = []
        
        # B_param: [B, S, H, N]
        # x: [B, S, H, P]
        
        for t in range(seqlen):
            # 1. Update State
            # h[t] = dA[t] * h[t-1] + B[t]' * x[t]
            
            decay = dA[:, t]            # [B, H, 1]
            b_t = B_param[:, t]         # [B, H, N]
            x_t = x[:, t]               # [B, H, P]
            
            # term: [B, H, N, 1] * [B, H, 1, P] -> [B, H, N, P]
            update = torch.einsum('bhn,bhp->bhnp', b_t, x_t)
            
            # Recurrence
            h = h * decay.unsqueeze(-1) + update
            
            # 2. Compute Output
            # y[t] = C[t] * h[t] + D * x[t]
            c_t = C_param[:, t]         # [B, H, N]
            
            # [B, H, N] * [B, H, N, P] -> [B, H, P] (sum over N)
            y_t = torch.einsum('bhn,bhnp->bhp', c_t, h)
            
            # Add Residual D (Skip connection)
            # y_t += x_t * D
            y_t = y_t + x_t * self.D.view(1, nheads, 1)
            
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1) # [B, S, H, P]
        return y


class MambaSafeMoEBlock(nn.Module):
    """
    Hybrid Block: Mamba 2 Mixer + Optimized Distributed SafeMoE.
    
    Structure:
    Input -> Norm -> Mamba2Mixer -> Norm -> SafeMoE -> Output
    """
    def __init__(
        self,
        mamba_cfg: MambaConfig,
        moe_cfg: 'OptimizedDistributedMoEConfig',
    ):
        super().__init__()
        from .distributed_optimized import OptimizedDistributedSafeMoE
        
        self.norm1 = nn.RMSNorm(mamba_cfg.d_model)
        self.mamba = Mamba2Mixer(mamba_cfg)
        
        self.norm2 = nn.RMSNorm(mamba_cfg.d_model)
        self.moe = OptimizedDistributedSafeMoE(moe_cfg)
        
    def forward(self, x, **kwargs):
        # Mamba Mixer (Time Mixing)
        h = self.mamba(self.norm1(x))
        x = x + h
        
        # MoE (Channel Mixing)
        h, aux, stats = self.moe(self.norm2(x))
        x = x + h
        
        return x, aux, stats

