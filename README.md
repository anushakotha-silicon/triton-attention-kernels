# triton-attention-kernels

Fused attention kernels in Triton, benchmarked against PyTorch SDPA and profiled with Nsight Compute.

- `kernels/softmax.py`: fused row softmax (1 read + 1 write per element)
- `kernels/flash_attention.py`: FlashAttention-style forward, online softmax, causal + GQA

**Status: written but not yet run on a GPU.** The tests and benchmarks below are the first thing to do on a rented H100; expect to fix small issues (Triton API names vary by version; the code targets Triton 3.x with `tl.exp2`). No performance number in this repo is real until `results/*.csv` is committed from a run.

## Run (needs a CUDA GPU)

```bash
pip install -r requirements.txt
pytest -q                                   # correctness vs fp32 reference (MHA + GQA, causal, ragged lengths)
python -m bench.bench_softmax               # -> results/softmax.csv
python -m bench.bench_attention             # -> results/attention.csv
ncu --set full -k regex:_attn_fwd python -m bench.bench_attention --seq 4096 --once
```

## What to record for the resume

From `results/attention.csv`: `pct_of_sdpa` at seq 4096/8192 (BF16, D=128), causal and non-causal. From `ncu`: tensor-core utilization, achieved occupancy, and the top stall reason before and after each change. Log every optimization with its before/after in `NOTES.md`; "removed X, tensor-core utilization went from A% to B%" is the bullet.

## Optimization ladder (one commit each, with numbers)

1. Baseline autotuned kernel (this commit)
2. Tune tile sizes and `num_stages` for D=64 and D=128 separately
3. Move the causal mask to the diagonal tiles only (skip mask math on fully-visible tiles)
4. Swap the K load to a pre-transposed layout or block pointers; check for shared-memory bank conflicts in ncu
5. Split-K decode kernel for Sq = 1 (current kernel wastes most of a 64-row tile); compare with the roofline model's decode prediction
6. FP8 variant (see the low-precision repo)

## Known limits

Forward only. `head_dim` power of two. Causal needs Sq == Sk. No dropout, no sliding window, no variable-length batching.

Companion: [roofline-model](../roofline-model), the predicted ceiling these kernels are compared against.
