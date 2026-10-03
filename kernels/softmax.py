"""Fused row-wise softmax in Triton.

One program per row: load once, max/exp/sum/normalize in registers, store once.
PyTorch's unfused path reads/writes the row several times; the fused kernel moves
2 * numel * itemsize bytes (1 read + 1 write), which is the memory-bound ideal.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim of a 2D CUDA tensor. Row must fit one block (n_cols <= ~64K)."""
    assert x.is_cuda and x.ndim == 2
    n_rows, n_cols = x.shape
    block = triton.next_power_of_2(n_cols)
    num_warps = 4 if block <= 2048 else 8 if block <= 8192 else 16
    y = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK=block, num_warps=num_warps)
    return y
