"""Benchmark fused softmax vs torch.softmax.   python -m bench.bench_softmax"""
import csv
import os

import torch
import triton

from kernels import softmax


def main(out="results/softmax.csv"):
    props = torch.cuda.get_device_properties(0)
    rows = []
    for n_cols in (512, 1024, 2048, 4096, 8192, 16384, 32768):
        x = torch.randn(8192, n_cols, device="cuda", dtype=torch.float16)
        nbytes = 2 * x.numel() * x.element_size()      # 1 read + 1 write
        row = {"gpu": props.name, "n_rows": 8192, "n_cols": n_cols}
        for name, fn in (("triton", lambda: softmax(x)), ("torch", lambda: torch.softmax(x, dim=-1))):
            ms = triton.testing.do_bench(fn)
            row[f"{name}_ms"] = round(ms, 4)
            row[f"{name}_GBps"] = round(nbytes / ms / 1e6, 1)
        rows.append(row)
        print(row)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
