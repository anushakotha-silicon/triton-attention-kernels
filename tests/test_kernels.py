import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def ref_attention(q, k, v, causal):
    """fp32 reference with explicit GQA expansion."""
    q, k, v = q.float(), k.float(), v.float()
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    if causal:
        m = torch.ones(s.shape[-2:], dtype=torch.bool, device=q.device).tril()
        s = s.masked_fill(~m, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


@cuda
@pytest.mark.parametrize("shape", [(4096, 1024), (1000, 777), (64, 32768)])
def test_softmax_matches_torch(shape):
    from kernels import softmax
    x = torch.randn(*shape, device="cuda", dtype=torch.float32) * 3
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-5)


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seq", [128, 1000, 2048])      # 1000 exercises the tail masks
@pytest.mark.parametrize("heads,kv_heads", [(8, 8), (32, 8)])   # MHA and GQA
def test_flash_attention_matches_reference(dtype, causal, seq, heads, kv_heads):
    from kernels import flash_attention
    torch.manual_seed(0)
    B, D = 2, 128
    q = torch.randn(B, heads, seq, D, device="cuda", dtype=dtype)
    k = torch.randn(B, kv_heads, seq, D, device="cuda", dtype=dtype)
    v = torch.randn(B, kv_heads, seq, D, device="cuda", dtype=dtype)
    out = flash_attention(q, k, v, causal=causal)
    ref = ref_attention(q, k, v, causal)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@cuda
def test_flash_attention_decode_shape():
    from kernels import flash_attention
    q = torch.randn(4, 32, 1, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(4, 8, 3000, 128, device="cuda", dtype=torch.float16)
    v = torch.randn(4, 8, 3000, 128, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(flash_attention(q, k, v).float(), ref_attention(q, k, v, False),
                               atol=2e-2, rtol=2e-2)
