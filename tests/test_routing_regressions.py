import pytest
import torch
from torch import nn

from safemoe import MoEConfig, SafeMoE, VectorizedSafeMoE
from safemoe.config import DistributedMoEConfig, OptimizedDistributedMoEConfig
from safemoe.models.distributed import DistributedSafeMoE
from safemoe.models.distributed_optimized import OptimizedDistributedSafeMoE


class Scale(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return x * self.weight


class FixedRouter(nn.Module):
    def __init__(self, ids, raw=None):
        super().__init__()
        self.register_buffer('ids', torch.tensor(ids))
        self.logits = nn.Parameter(torch.zeros(self.ids.shape))
        self.register_buffer('raw', torch.tensor(raw) if raw is not None
                             else torch.full(self.ids.shape, .5))

    def forward(self, x):
        return self.ids, self.logits.softmax(-1), {}, self.raw


class Ones(nn.Module):
    def forward(self, x):
        return torch.ones_like(x)


def make_layer(kind, routes, raw=None, threshold=0, shared=False):
    cfg = MoEConfig(d_model=2, d_ff=4, n_experts=2, top_k=2,
                    capacity_factor=0, min_capacity=1, route_threshold=threshold)
    if kind == 'core':
        layer = SafeMoE(cfg)
        layer.experts = nn.ModuleList([Scale(1), Scale(2)])
    elif kind == 'vectorized':
        layer = VectorizedSafeMoE(cfg)
        # Choose weights whose SwiGLU output is exactly x times expert scalar
        # by replacing the activation; the production grouped path is exercised.
        layer.act = Ones()
        with torch.no_grad():
            layer.w3.zero_()
            layer.w2.zero_()
            for e in range(2):
                layer.w3[e, :, :2].copy_(torch.eye(2))
                layer.w2[e, :2, :].copy_((e + 1) * torch.eye(2))
        layer.shared = Scale(100) if shared else None
    else:
        pytest.importorskip('transformers')
        from hf_integration.safe_moe_hf import SafeMoE as HFMoE, SafeMoETransformerConfig
        layer = HFMoE(SafeMoETransformerConfig(
            hidden_size=2, intermediate_size=4, moe_num_experts=2, moe_top_k=2,
            moe_capacity_factor=0, moe_min_capacity=1, moe_route_threshold=threshold,
        ))
        layer.experts = nn.ModuleList([Scale(1), Scale(2)])
    layer.router = FixedRouter(routes, raw)
    layer.fallback = Scale(10)
    return layer


@pytest.mark.parametrize('kind', ['core', 'vectorized', 'hf'])
def test_capacity_is_shared_across_slots_and_fallback_runs_once(kind):
    layer = make_layer(kind, [[0, 1], [0, 1], [1, 0]])
    x = torch.ones(1, 3, 2, requires_grad=True)
    y, _, stats = layer(x)
    torch.testing.assert_close(y[0, :, 0], torch.tensor([1.5, 10., 10.]))
    assert stats['overflow_tokens'].item() == 2
    assert stats['fallback_rate'].item() == pytest.approx(2 / 3)
    assert stats['expert_usage_max'].item() <= 1 / 3 + 1e-7
    y.sum().backward()
    torch.testing.assert_close(layer.fallback.weight.grad, torch.tensor(4.))
    assert x.grad.isfinite().all()


@pytest.mark.parametrize('kind', ['core', 'vectorized', 'hf'])
def test_low_confidence_and_overflow_can_backpropagate_together(kind):
    layer = make_layer(kind, [[0, 1], [0, 1], [0, 1]],
                       [[.4, .3], [.9, .8], [.9, .8]], threshold=.5)
    y, _, stats = layer(torch.ones(1, 3, 2, requires_grad=True))
    torch.testing.assert_close(y[0, :, 0], torch.tensor([10., 1.5, 10.]))
    y.sum().backward()
    torch.testing.assert_close(layer.fallback.weight.grad, torch.tensor(4.))
    assert stats['overflow_tokens'].item() == 1
    assert layer.router.logits.grad is not None


def test_shared_expert_survives_whole_token_fallback():
    layer = make_layer('vectorized', [[0, 1], [0, 1]], shared=True)
    y, _, _ = layer(torch.ones(1, 2, 2))
    torch.testing.assert_close(y[0, :, 0], torch.tensor([101.5, 110.]))


@pytest.mark.parametrize('optimized', [False, True])
def test_distributed_local_fallback_excludes_every_slot(optimized):
    config_class = OptimizedDistributedMoEConfig if optimized else DistributedMoEConfig
    model_class = OptimizedDistributedSafeMoE if optimized else DistributedSafeMoE
    layer = model_class(config_class(d_model=2, d_ff=4, n_experts_global=3,
                                    capacity_factor=0, min_capacity=1))
    ids = torch.tensor([[0, 1], [1, 2]])
    scores = torch.full((2, 2), .5)
    layer._route = lambda x: (ids, scores, scores, {})
    layer.fallback = Scale(10)
    if optimized:
        layer.experts.forward_grouped = lambda x, e: x * (e + 1).unsqueeze(-1)
    else:
        layer.local_experts = nn.ModuleList([Scale(1), Scale(2), Scale(3)])
    y, _, stats = layer(torch.ones(1, 2, 2))
    torch.testing.assert_close(y[0, :, 0], torch.tensor([1.5, 10.]))
    assert stats['fallback_rate'].item() == .5


def test_public_exports_and_moved_constructors():
    import safemoe
    namespace = {}
    exec('from safemoe import *', namespace)
    assert all(name in namespace for name in safemoe.__all__)
    cfg = MoEConfig(d_model=8, d_ff=16, n_experts=2)
    cached = safemoe.CachedSafeMoEBlock(8, 2, cfg, max_seq_len=8)
    assert cached(torch.randn(1, 3, 8), use_cache=True)[0].shape == (1, 3, 8)
    engram = safemoe.EngramTransformerBlock(8, 2, 16, attn_dropout=.1)
    assert engram.attn.n_kv_heads == 2
    distributed = safemoe.DistributedSafeMoEBlock(
        8, 2, DistributedMoEConfig(d_model=8, d_ff=16, n_experts_global=2),
        attn_dropout=.1,
    )
    assert distributed(torch.randn(1, 3, 8))[0].shape == (1, 3, 8)


@pytest.mark.parametrize('kind', ['core', 'vectorized', 'hf', 'distributed', 'optimized'])
@pytest.mark.parametrize('fallback', [False, True])
def test_autocast_dispatch_preserves_residual_dtype(kind, fallback):
    capacity = 0 if fallback else 100
    if kind in ['core', 'vectorized']:
        cfg = MoEConfig(d_model=8, d_ff=16, n_experts=2,
                        capacity_factor=capacity, min_capacity=0)
        layer = (SafeMoE if kind == 'core' else VectorizedSafeMoE)(cfg)
    elif kind == 'hf':
        pytest.importorskip('transformers')
        from hf_integration.safe_moe_hf import SafeMoE as HFMoE, SafeMoETransformerConfig
        layer = HFMoE(SafeMoETransformerConfig(
            hidden_size=8, intermediate_size=16, moe_num_experts=2,
            moe_capacity_factor=capacity, moe_min_capacity=0,
        ))
    else:
        cfg_type = DistributedMoEConfig if kind == 'distributed' else OptimizedDistributedMoEConfig
        layer_type = DistributedSafeMoE if kind == 'distributed' else OptimizedDistributedSafeMoE
        layer = layer_type(cfg_type(d_model=8, d_ff=16, n_experts_global=2,
                                    capacity_factor=capacity, min_capacity=0))
    x = torch.randn(1, 3, 8, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y, _, _ = layer(x)
    assert y.dtype == x.dtype
    y.square().sum().backward()
    assert x.grad.isfinite().all()
