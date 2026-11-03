"""Variable-split expert-parallel transport with an inverse backward exchange."""
import torch
import torch.distributed as dist


def _exchange_counts(send_counts, group=None):
    """Counts stay on the payload device (CUDA for NCCL, CPU for CPU Gloo)."""
    if not dist.is_initialized():
        return send_counts
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    return recv_counts


def _exchange(x, send_splits, recv_splits, group):
    result = x.new_empty((sum(recv_splits), *x.shape[1:]))
    dist.all_to_all_single(
        result, x.contiguous(), output_split_sizes=recv_splits,
        input_split_sizes=send_splits, group=group,
    )
    return result


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, send_splits, recv_splits, group):
        ctx.send_splits, ctx.recv_splits, ctx.group = send_splits, recv_splits, group
        return _exchange(x, send_splits, recv_splits, group)

    @staticmethod
    def backward(ctx, grad_output):
        return (_exchange(grad_output, ctx.recv_splits, ctx.send_splits, ctx.group),
                None, None, None)


def _all_to_all_varying(x, send_counts, recv_counts, group=None):
    if not dist.is_initialized():
        return x
    send_splits = send_counts.tolist()
    recv_splits = recv_counts.tolist()
    if x.is_floating_point() and torch.is_grad_enabled():
        # Even a rank with no outgoing activations must execute both backwards.
        # A local leaf also covers raw feature inputs that do not require grad.
        if not x.requires_grad:
            x = x.detach().requires_grad_()
        return _AllToAll.apply(x, send_splits, recv_splits, group)
    return _exchange(x, send_splits, recv_splits, group)


def distributed_expert_forward(x, experts, scores, token_idx, total_tokens,
                               n_local_experts, group, forward_grouped):
    """Every EP rank participates, including ranks with zero sends/receives.

    Return exchange preserves sender order, so gate scores and token indices
    remain local and retain their original autograd history.
    """
    target_rank = experts // n_local_experts
    order = torch.argsort(target_rank, stable=True)
    send_counts = torch.bincount(
        target_rank, minlength=dist.get_world_size(group)
    ).to(dtype=torch.int64)
    recv_counts = _exchange_counts(send_counts, group)
    received_x = _all_to_all_varying(x[order], send_counts, recv_counts, group)
    received_ids = _all_to_all_varying(
        (experts[order] % n_local_experts).unsqueeze(-1),
        send_counts, recv_counts, group,
    ).squeeze(-1)

    local_order = torch.argsort(received_ids, stable=True)
    if received_x.size(0):
        local_y = forward_grouped(received_x[local_order], received_ids[local_order])
        # Empty receiver ranks retain the input dtype, so autocast outputs must
        # use that same transport dtype on every rank before the return exchange.
        received_y = local_y[torch.argsort(local_order)].to(received_x.dtype)
    else:
        # Preserve the autograd edge on empty ranks, not a fresh empty buffer.
        received_y = received_x
    returned_y = _all_to_all_varying(received_y, recv_counts, send_counts, group)
    # An empty index_add is intentional: it preserves collective participation
    # in backward on ranks whose local tokens all used fallback.
    return x.new_zeros((total_tokens, x.size(-1))).index_add(
        0, token_idx[order], (returned_y * scores[order].unsqueeze(-1)).to(x.dtype)
    )
