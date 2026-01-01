def aggregate_aux(aux_list):
    out = {}
    for aux in aux_list:
        for k, v in aux.items():
            out[k] = out.get(k, 0.0) + v
    return out

# forward loop
x = ...
aux_all = []
stats_all = []
for blk in blocks:
    x, aux, st = blk(x, attn_mask=None)
    aux_all.append(aux)
    stats_all.append(st)

task_loss = ...
aux_sum = aggregate_aux(aux_all)
loss = task_loss + aux_sum["router_z_loss"] + aux_sum["load_balance_loss"]
