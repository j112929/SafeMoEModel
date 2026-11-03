"""Real two-process transport checks, including empty-rank backward paths."""
from datetime import timedelta
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from safemoe.config import DistributedMoEConfig, OptimizedDistributedMoEConfig
from safemoe.models.distributed import DistributedSafeMoE
from safemoe.models.distributed_optimized import OptimizedDistributedSafeMoE


def _copy_shard(model, reference, rank, optimized):
    model.router.load_state_dict(reference.router.state_dict())
    model.fallback.load_state_dict(reference.fallback.state_dict())
    begin = rank * model.n_local_experts
    end = begin + model.n_local_experts
    if optimized:
        with torch.no_grad():
            for name in ['w1', 'w2', 'w3']:
                getattr(model.experts, name).copy_(getattr(reference.experts, name)[begin:end])
    else:
        for local, expert in enumerate(reference.local_experts[begin:end]):
            model.local_experts[local].load_state_dict(expert.state_dict())


def _gradient(parameter):
    return torch.zeros_like(parameter) if parameter.grad is None else parameter.grad


def _worker(rank, rendezvous, optimized, backend):
    torch.set_num_threads(1)
    device = torch.device('cuda', rank) if backend == 'nccl' else torch.device('cpu')
    if backend == 'nccl':
        torch.cuda.set_device(device)
    config_type = OptimizedDistributedMoEConfig if optimized else DistributedMoEConfig
    model_type = OptimizedDistributedSafeMoE if optimized else DistributedSafeMoE
    cases = ['normal', 'zero_send', 'zero_receive', 'all_fallback', 'raw_features', 'autocast_zero_receive']
    references = []
    # Build unsharded references before initializing the process group.
    for index, case in enumerate(cases):
        torch.manual_seed(100 + index)
        threshold = 1.0 if case == 'all_fallback' or (case == 'zero_send' and rank == 0) else 0.0
        cfg = config_type(d_model=4, d_ff=8, n_experts_global=4, top_k=2,
                          capacity_factor=100, route_threshold=threshold)
        reference = model_type(cfg).to(device)
        if case in ['zero_receive', 'autocast_zero_receive']:
            with torch.no_grad():
                reference.router.weight.zero_()
                reference.router.weight[:, 0].copy_(torch.tensor([2., 1., -1., -2.], device=device))
        references.append(reference)

    dist.init_process_group(backend, init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=25))
    try:
        for index, (case, reference) in enumerate(zip(cases, references)):
            model = model_type(reference.cfg).to(device)
            _copy_shard(model, reference, rank, optimized)
            torch.manual_seed(1000 + rank)
            # Uneven local token counts exercise variable split sizes.
            x = torch.randn(1, 3 - rank, 4, device=device)
            if case in ['zero_receive', 'autocast_zero_receive']:
                x[..., 0] = x[..., 0].abs() + 1
            x.requires_grad_(case != 'raw_features')
            ref_x = x.detach().clone().requires_grad_(case != 'raw_features')
            amp = case == 'autocast_zero_receive'
            context = torch.autocast(device.type, dtype=torch.bfloat16) if amp else nullcontext()
            with context:
                actual, _, _ = model(x)
                expected, _, _ = reference(ref_x)
            tolerance = dict(rtol=3e-2, atol=3e-3) if amp else dict(rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(actual, expected, **tolerance)
            actual.square().sum().backward()
            expected.square().sum().backward()
            if x.requires_grad:
                torch.testing.assert_close(x.grad, ref_x.grad, **tolerance)
            torch.testing.assert_close(_gradient(model.router.weight), _gradient(reference.router.weight),
                                       **tolerance)
            for parameter, ref_parameter in zip(model.fallback.parameters(), reference.fallback.parameters()):
                torch.testing.assert_close(_gradient(parameter), _gradient(ref_parameter), **tolerance)
            # Each expert receives task gradients from every source rank.
            ref_experts = reference.experts if optimized else reference.local_experts
            global_gradients = {}
            for name, parameter in ref_experts.named_parameters():
                expected_grad = _gradient(parameter).clone()
                dist.all_reduce(expected_grad)
                global_gradients[name] = expected_grad
            if optimized:
                for name, parameter in model.experts.named_parameters():
                    expected_grad = global_gradients[name][rank * 2:rank * 2 + 2]
                    torch.testing.assert_close(_gradient(parameter), expected_grad, **tolerance)
            else:
                for local, expert in enumerate(model.local_experts):
                    for name, parameter in expert.named_parameters():
                        expected_grad = global_gradients[f'{rank * 2 + local}.{name}']
                        torch.testing.assert_close(_gradient(parameter), expected_grad, **tolerance)
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize('optimized', [False, True])
def test_two_rank_cpu_forward_and_backward(tmp_path, optimized):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('CPU Gloo distributed support is unavailable')
    mp.spawn(_worker, args=((tmp_path / 'rendezvous').as_uri(), optimized, 'gloo'),
             nprocs=2, join=True)


@pytest.mark.distributed
@pytest.mark.parametrize('optimized', [False, True])
def test_two_rank_nccl_forward_and_backward(tmp_path, optimized):
    if torch.cuda.device_count() < 2 or not dist.is_nccl_available():
        pytest.skip('requires two CUDA devices and NCCL')
    mp.spawn(_worker, args=((tmp_path / 'rendezvous').as_uri(), optimized, 'nccl'),
             nprocs=2, join=True)
