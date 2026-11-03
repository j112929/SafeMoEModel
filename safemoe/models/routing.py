"""Shared capacity planning, completed before differentiable dispatch begins."""
from dataclasses import dataclass

import torch


@dataclass
class RoutePlan:
    token_indices: torch.Tensor
    expert_indices: torch.Tensor
    scores: torch.Tensor
    fallback_mask: torch.Tensor
    overflow_tokens: int
    usage: torch.Tensor


def plan_routes(experts, scores, raw_scores, capacity, threshold, n_experts):
    """Enforce capacity across all slots and use whole-token fallback.

    Assignments are admitted in token order within each expert. If any slot
    overflows, all slots for that token are removed. Vacated slots are not
    refilled, so every expert stays within capacity. Masks are finalized before
    they are used in autograd-tracked indexing and must not be mutated later.
    """
    if capacity < 0:
        raise ValueError("capacity must be nonnegative")
    tokens, slots = experts.shape
    device = experts.device
    with torch.no_grad():
        fallback = torch.zeros(tokens, dtype=torch.bool, device=device)
        if threshold > 0:
            fallback = raw_scores[:, 0] < threshold
        token_ids = torch.arange(tokens, device=device).repeat_interleave(slots)
        assignment_ids = (~fallback[token_ids]).nonzero(as_tuple=True)[0]
        expert_ids = experts.reshape(-1)[assignment_ids]
        order = torch.argsort(expert_ids, stable=True)
        assignment_ids = assignment_ids[order]
        expert_ids = expert_ids[order]
        token_ids = token_ids[assignment_ids]

        positions = torch.arange(expert_ids.numel(), device=device)
        starts = torch.ones_like(expert_ids, dtype=torch.bool)
        starts[1:] = expert_ids[1:] != expert_ids[:-1]
        segment_starts = torch.where(starts, positions, 0).cummax(0).values
        accepted = positions - segment_starts < capacity
        overflow_ids = token_ids[~accepted].unique()
        fallback[overflow_ids] = True
        accepted = accepted & ~fallback[token_ids]
        assignment_ids = assignment_ids[accepted]
        token_ids = token_ids[accepted]
        expert_ids = expert_ids[accepted]
        usage = torch.bincount(expert_ids, minlength=n_experts).float()

    # Index differentiable scores only after all routing decisions are final.
    return RoutePlan(token_ids, expert_ids, scores.reshape(-1)[assignment_ids],
                     fallback, overflow_ids.numel(), usage)


def routing_stats(plan, capacity, tokens):
    device = plan.fallback_mask.device
    denom = max(1, tokens)
    return {
        "moe_capacity": torch.tensor(capacity, device=device),
        "overflow_tokens": torch.tensor(plan.overflow_tokens, device=device),
        "overflow_rate": torch.tensor(plan.overflow_tokens / denom, device=device),
        "fallback_rate": plan.fallback_mask.float().sum() / denom,
        "expert_usage_mean": plan.usage.mean() / denom,
        "expert_usage_max": plan.usage.max() / denom,
        "expert_usage_min": plan.usage.min() / denom,
    }


def apply_routes(x, plan, n_experts, expert_forward, fallback, shared=None):
    """Apply an immutable plan; fallback replaces all routed slots of a token."""
    output = torch.zeros_like(x) if shared is None else shared(x).to(x.dtype)
    fallback_ids = plan.fallback_mask.nonzero(as_tuple=True)[0]
    if fallback is not None and fallback_ids.numel():
        values = fallback(x[fallback_ids]).to(output.dtype)
        output = output.index_add(0, fallback_ids, values)
    for expert in range(n_experts):
        selected = plan.expert_indices == expert
        token_ids = plan.token_indices[selected]
        if token_ids.numel():
            values = expert_forward(expert, x[token_ids])
            output = output.index_add(
                0, token_ids, (values * plan.scores[selected].unsqueeze(-1)).to(output.dtype)
            )
    return output
