import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def _all_to_all_varying(x, send_counts, recv_counts, group=None):
    """
    x: [sum(send_counts), D]
    send_counts/recv_counts: 1D int64 cpu tensors length P
    Return: y [sum(recv_counts), D]
    """
    P = len(send_counts)
    assert x.is_contiguous()
    # split input
    x_splits = list(x.split(send_counts.tolist(), dim=0))
    y_splits = [torch.empty((int(rc), x.size(1)), device=x.device, dtype=x.dtype) for rc in recv_counts.tolist()]
    dist.all_to_all(y_splits, x_splits, group=group)
    return torch.cat(y_splits, dim=0)

class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

class EPMoE(nn.Module):
    """
    Expert Parallel MoE with AllToAll (toy but correct).
    Assumes experts are sharded across ranks (EP).
    """
    def __init__(self, d_model, d_ff, n_experts_global, top_k=2,
                 capacity_factor=1.25, group=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts_global
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.group = group

        self.rank = dist.get_rank(group) if dist.is_initialized() else 0
        self.world = dist.get_world_size(group) if dist.is_initialized() else 1
        assert self.n_experts % self.world == 0
        self.n_local = self.n_experts // self.world

        self.router = nn.Linear(d_model, n_experts_global, bias=False)
        self.local_experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(self.n_local)])

    def forward(self, x):
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)

        # 1) route
        logits = self.router(x_flat)                                # [N, E]
        topk = torch.topk(logits, k=self.top_k, dim=-1)             # values/indices [N, K]
        topk_idx = topk.indices                                     # [N, K] global expert id
        topk_w = F.softmax(topk.values, dim=-1).to(x.dtype)         # [N, K]

        # 2) build (token-copy) list: total M = N*K
        M = N * self.top_k
        flat_expert = topk_idx.reshape(M)                           # [M]
        flat_w = topk_w.reshape(M)                                  # [M]
        flat_token = torch.arange(N, device=x.device).repeat_interleave(self.top_k)  # [M]

        # target rank / local expert
        tgt_rank = flat_expert // self.n_local                      # [M]
        tgt_local = flat_expert % self.n_local                      # [M]

        # 3) capacity control per global expert (optional but recommended)
        # Here implement simple per-(rank, local expert) cap by dropping overflow.
        # For production: do this on GPU with sort+prefixsum.
        cap = int(((M / self.n_experts) * self.capacity_factor) + 1)

        # group by (tgt_rank, tgt_local)
        # sort by tgt_rank then tgt_local
        key = tgt_rank * self.n_local + tgt_local                   # [M]
        order = torch.argsort(key)                                  # [M]
        tgt_rank_s = tgt_rank[order]
        tgt_local_s = tgt_local[order]
        flat_token_s = flat_token[order]
        flat_w_s = flat_w[order]
        x_s = x_flat[flat_token_s]                                  # [M, D]

        # apply capacity per expert
        # compute positions within each expert segment
        # segment id = key_sorted
        key_s = key[order]
        # find segment starts
        seg_change = torch.ones_like(key_s, dtype=torch.bool)
        seg_change[1:] = key_s[1:] != key_s[:-1]
        seg_id = torch.cumsum(seg_change.to(torch.int32), dim=0) - 1  # [M]
        # position within segment
        pos_in_seg = torch.arange(M, device=x.device) - torch.where(seg_change, torch.arange(M, device=x.device), torch.tensor(0, device=x.device)).cummax(dim=0).values
        keep = pos_in_seg < cap
        x_s = x_s[keep]
        tgt_rank_s = tgt_rank_s[keep]
        tgt_local_s = tgt_local_s[keep]
        flat_token_s = flat_token_s[keep]
        flat_w_s = flat_w_s[keep]

        # 4) bucket by rank for AllToAll
        P = self.world
        send_counts = torch.bincount(tgt_rank_s, minlength=P).to(torch.int64).cpu()
        # exchange counts
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)  # 1D counts

        # permute x_s already grouped by tgt_rank because key sort includes tgt_rank major,
        # but capacity pruning broke contiguity. Re-sort by tgt_rank only.
        rank_order = torch.argsort(tgt_rank_s)
        x_send = x_s[rank_order].contiguous()
        local_send = tgt_local_s[rank_order].contiguous()
        token_send = flat_token_s[rank_order].contiguous()
        w_send = flat_w_s[rank_order].contiguous()

        # AllToAll: send token features
        x_recv = _all_to_all_varying(x_send, send_counts, recv_counts, group=self.group)  # [R, D]
        local_recv = _all_to_all_varying(local_send.unsqueeze(-1),
                                         send_counts, recv_counts, group=self.group).squeeze(-1)
        token_recv = _all_to_all_varying(token_send.unsqueeze(-1),
                                         send_counts, recv_counts, group=self.group).squeeze(-1)
        w_recv = _all_to_all_varying(w_send.unsqueeze(-1),
                                     send_counts, recv_counts, group=self.group).squeeze(-1)

        # 5) expert compute on recv side: group by local expert
        R = x_recv.size(0)
        # sort by local expert to batch
        le_order = torch.argsort(local_recv)
        x_e = x_recv[le_order]
        le = local_recv[le_order]
        tok = token_recv[le_order]
        ww = w_recv[le_order]

        y_e = torch.empty_like(x_e)
        # process each local expert contiguous block
        # (for production: grouped GEMM / fused kernels)
        start = 0
        while start < R:
            e = int(le[start].item())
            end = start + 1
            while end < R and int(le[end].item()) == e:
                end += 1
            y_e[start:end] = self.local_experts[e](x_e[start:end])
            start = end

        # undo local-expert sort
        inv_le = torch.empty_like(le_order)
        inv_le[le_order] = torch.arange(R, device=x.device)
        y_recv = y_e[inv_le]
        tok_recv = tok[inv_le]
        ww_recv = ww[inv_le]

        # 6) send back results to source ranks (reverse alltoall)
        # Need original source rank for each received item; for correctness we can reconstruct:
        # source rank is "the rank that sent it", which in AllToAll corresponds to slot structure.
        # easiest: do symmetric AllToAll with same counts; here we reuse recv_counts as send_counts_back.
        send_counts_back = recv_counts
        recv_counts_back = send_counts

        y_sendback = _all_to_all_varying(y_recv.contiguous(), send_counts_back, recv_counts_back, group=self.group)
        tok_sendback = _all_to_all_varying(tok_recv.unsqueeze(-1),
                                           send_counts_back, recv_counts_back, group=self.group).squeeze(-1)
        w_sendback = _all_to_all_varying(ww_recv.unsqueeze(-1),
                                         send_counts_back, recv_counts_back, group=self.group).squeeze(-1)

        # 7) combine on source
        y_flat = torch.zeros((N, D), device=x.device, dtype=x.dtype)
        y_flat.index_add_(0, tok_sendback, y_sendback * w_sendback.unsqueeze(-1))

        return y_flat.view(B, T, D)
