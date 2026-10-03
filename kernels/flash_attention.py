"""FlashAttention-style fused forward kernel in Triton.

Online softmax over K/V tiles: the S x S score matrix never touches HBM, so HBM traffic is
O(S * D) per head instead of O(S^2). Supports causal masking and GQA (n_kv_heads < n_heads).

Layout: q (B, H, Sq, D), k/v (B, H_kv, Sk, D); any strides, last-dim contiguous preferred.
Limits (deliberate, for a first version):
  * D must be a power of two >= 16 (64 / 128 typical).
  * causal=True requires Sq == Sk (prefill). Decode (Sq == 1) works non-causal but wastes
    most of a BLOCK_M tile; a split-K decode kernel is the planned follow-up.
  * Forward only.
"""
import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634


def _configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
        for bm in (64, 128)
        for bn in (32, 64)
        for w in (4, 8)
        for s in (2, 3)
    ]


@triton.autotune(configs=_configs(), key=["N_CTX_Q", "N_CTX_K", "HEAD_DIM", "CAUSAL"])
@triton.jit
def _attn_fwd(
    Q, K, V, O, sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    H, H_KV, N_CTX_Q, N_CTX_K,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    b = off_bh // H
    h = off_bh % H
    kvh = h // (H // H_KV)

    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + kvh * stride_kh
    v_base = V + b * stride_vb + kvh * stride_vh
    o_base = O + b * stride_ob + h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=offs_m[:, None] < N_CTX_Q, other=0.0,
    )

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * _LOG2E          # fold log2(e) so we can use exp2

    hi = N_CTX_K
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX_K)

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(
            k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk,
            mask=offs_n[None, :] < N_CTX_K, other=0.0,
        )                                                   # (D, BLOCK_N)
        qk = tl.dot(q, k) * qk_scale                        # (BLOCK_M, BLOCK_N)

        valid = offs_n[None, :] < N_CTX_K
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(valid, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
            mask=offs_n[:, None] < N_CTX_K, other=0.0,
        )                                                   # (BLOCK_N, D)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
        acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX_Q,
    )


def flash_attention(q, k, v, causal: bool = False, sm_scale: float | None = None):
    B, H, Sq, D = q.shape
    _, H_kv, Sk, _ = k.shape
    assert q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
    assert k.shape == v.shape and k.shape[0] == B and k.shape[3] == D
    assert H % H_kv == 0, "n_heads must be a multiple of n_kv_heads"
    assert D >= 16 and (D & (D - 1)) == 0, "head_dim must be a power of two >= 16"
    if causal:
        assert Sq == Sk, "causal requires Sq == Sk"
    if sm_scale is None:
        sm_scale = D ** -0.5

    o = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(Sq, meta["BLOCK_M"]), B * H)
    _attn_fwd[grid](
        q, k, v, o, sm_scale,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        H, H_kv, Sq, Sk,
        HEAD_DIM=D, CAUSAL=causal,
    )
    return o
