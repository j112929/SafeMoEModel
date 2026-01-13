@torch.no_grad()
def rolling_ngram_hash(input_ids: torch.Tensor, n: int, base: int, mod: int):
    """
    input_ids: [B, T] int64
    returns: [B, T] int64, positions < n-1 are 0
    """
    B, T = input_ids.shape
    ids = input_ids.to(torch.int64)

    # precompute base^n % mod
    base_n = pow(base, n, mod)

    h = torch.zeros((B,), device=ids.device, dtype=torch.int64)
    out = torch.zeros((B, T), device=ids.device, dtype=torch.int64)

    for t in range(T):
        h = (h * base + ids[:, t]) % mod
        if t >= n:
            h = (h - (ids[:, t - n] * base_n) % mod) % mod
        if t >= n - 1:
            out[:, t] = h
    return out
