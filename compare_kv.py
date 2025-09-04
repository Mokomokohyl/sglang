import torch

a_path = "/tmp/kv_dumps/clusterfusion_layer0.pt"
b_path = "/tmp/kv_dumps/flashinfer_layer0.pt"
a = torch.load(a_path)
b = torch.load(b_path)


def _calc_stats(ta: torch.Tensor, tb: torch.Tensor):
    diff = (ta - tb)
    abs_diff = diff.abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    denom = tb.abs().max().item()
    rel_max = max_abs / (denom if denom > 0 else 1.0)
    return max_abs, mean_abs, rel_max


def compare_tensor(name_a, name_b, max_slices_print=200):
    try:
        ta = a[name_a].float()
    except KeyError:
        print(f"Key {name_a!r} not found in {a_path}")
        return
    try:
        tb = b[name_b].float()
    except KeyError:
        print(f"Key {name_b!r} not found in {b_path}")
        return

    if ta.shape != tb.shape:
        tb_orig_shape = tuple(tb.shape)
        # 如果元素总数一致，可以直接 view 成 ta 的 shape
        if tb.numel() == ta.numel():
            tb = tb.view(ta.shape)
            print(f"shape mismatch: reshaped {name_b} from {tb_orig_shape} -> {tuple(ta.shape)}")
        else:
            print(f"shape mismatch and different element counts: {name_a} {tuple(ta.shape)} vs {name_b} {tb_orig_shape}; cannot reshape, aborting compare for this key.")
            return

    # 输出原始元素均值
    mean_a = ta.mean().item() if ta.numel() > 0 else float("nan")
    mean_b = tb.mean().item() if tb.numel() > 0 else float("nan")
    print(f"[Mean] {name_a}: {mean_a:.6e}, {name_b}: {mean_b:.6e}")

    # 总体比较
    max_abs, mean_abs, rel_max = _calc_stats(ta.view(-1), tb.view(-1))
    print(f"[Overall] {name_a} vs {name_b}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, rel_max={rel_max:.6e}")

    # 如果第0维可以切分，则逐 slice 比较
    if ta.ndim >= 1 and ta.shape[0] > 1:
        n = ta.shape[0]
        n_print = min(n, max_slices_print)
        print(f"[Per-slice] splitting along dim0, total slices={n}, printing first {n_print}")
        for i in range(n_print):
            sa = ta[i].view(-1)
            sb = tb[i].view(-1)
            max_abs_i, mean_abs_i, rel_max_i = _calc_stats(sa, sb)
            print(f" slice[{i}]: max_abs={max_abs_i:.6e}, mean_abs={mean_abs_i:.6e}, rel_max={rel_max_i:.6e}")
        if n > max_slices_print:
            print(f"... skipped remaining {n - max_slices_print} slices")

    # 对每个 (i, j) 做比较（覆盖原来按列的逻辑）
    if ta.ndim >= 2:
        D0, D1 = ta.shape[0], ta.shape[1]
        total_pairs = D0 * D1
        print(f"[Per-(i,j) pairs] comparing all (i,j) over dim0 x dim1 = {D0} x {D1} = {total_pairs} pairs")

        # 累加用于平均（对所有 pairs 计算平均误差）
        sum_mean_abs = 0.0
        sum_max_abs = 0.0
        sum_rel = 0.0
        cnt = 0

        printed = 0
        for i in range(D0):
            for j in range(D1):
                sa = ta[i, j].contiguous().view(-1)
                sb = tb[i, j].contiguous().view(-1)
                # 若子张量为空则跳过
                if sa.numel() == 0 or sb.numel() == 0:
                    continue
                max_abs_ij, mean_abs_ij, rel_max_ij = _calc_stats(sa, sb)
                sum_mean_abs += mean_abs_ij
                sum_max_abs += max_abs_ij
                sum_rel += rel_max_ij
                cnt += 1
                if printed < max_slices_print:
                    print(f" pair[{i},{j}]: max_abs={max_abs_ij:.6e}, mean_abs={mean_abs_ij:.6e}, rel_max={rel_max_ij:.6e}")
                    printed += 1

        if cnt > 0:
            avg_max = sum_max_abs / cnt
            avg_mean = sum_mean_abs / cnt
            avg_rel = sum_rel / cnt
            print(f"[Pairs Avg over {cnt} pairs] avg_max_abs={avg_max:.6e}, avg_mean_abs={avg_mean:.6e}, avg_rel_max={avg_rel:.6e}")
        else:
            print("[Pairs Avg] no valid (i,j) pairs to compare")

        if total_pairs > max_slices_print:
            print(f"... printed first {max_slices_print} pairs, skipped remaining {total_pairs - max_pairs_print} pairs")


# 常见比较（根据实际键名修改）
compare_tensor("k_new", "k_new")
compare_tensor("v_new", "v_new")
compare_tensor("output", "output")