"""Benchmark the Triton attention kernel vs PyTorch SDPA (flash backend).

    python -m bench.bench_attention                       # full sweep -> results/attention.csv
    python -m bench.bench_attention --seq 4096 --once     # single launch, for ncu / nsys

Profile:
    ncu --set full -k regex:_attn_fwd python -m bench.bench_attention --seq 4096 --once
"""
import argparse
import csv
import os

import torch
import torch.nn.functional as F
import triton

from kernels import flash_attention


def flops(B, H, S, D, causal):
    f = 4 * B * H * S * S * D          # QK^T + PV, 2 FLOP/MAC each
    return f / 2 if causal else f


def run(seq, causal, dtype, H=32, D=128, tokens=16384, once=False):
    B = max(1, tokens // seq)
    q, k, v = (torch.randn(B, H, seq, D, device="cuda", dtype=dtype) for _ in range(3))
    f_tri = lambda: flash_attention(q, k, v, causal=causal)
    f_ref = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    if once:
        f_tri(); torch.cuda.synchronize()
        return None
    t_tri, t_ref = triton.testing.do_bench(f_tri), triton.testing.do_bench(f_ref)
    fl = flops(B, H, seq, D, causal)
    return {
        "gpu": torch.cuda.get_device_name(0), "dtype": str(dtype).split(".")[-1], "causal": causal,
        "B": B, "H": H, "seq": seq, "D": D,
        "triton_ms": round(t_tri, 4), "sdpa_ms": round(t_ref, 4),
        "triton_TFLOPs": round(fl / t_tri / 1e9, 1), "sdpa_TFLOPs": round(fl / t_ref / 1e9, 1),
        "pct_of_sdpa": round(100 * t_ref / t_tri, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--out", default="results/attention.csv")
    a = ap.parse_args()

    seqs = [a.seq] if a.seq else [512, 1024, 2048, 4096, 8192]
    rows = []
    for seq in seqs:
        for causal in (False, True):
            r = run(seq, causal, torch.bfloat16, once=a.once)
            if r:
                rows.append(r)
                print(r)
    if rows:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
