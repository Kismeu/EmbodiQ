"""INT4 packing tests, including the empirical probe that pinned the layout.

The probe is kept as a test, not deleted after use: if a future torch changes
the tinygemm nibble order or the scales_and_zeros axis order, every quantized
model would still RUN and would silently produce transposed weights. This test
is what turns that into a red build.
"""

from __future__ import annotations

import pytest
import torch

from embodiq.quant.pack import (
    dequantize_int4, dequantize_int8_per_channel, group_quantize_int4,
    quantize_int8_per_channel, to_tinygemm, verify_roundtrip,
)
from embodiq.quant.spec import QuantSpec

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SHAPES = [(4096, 4096), (11008, 4096), (896, 4864), (3072, 3072), (1152, 1152), (32064, 4096)]


def test_group_quantize_is_exact_on_representable_values():
    """A weight built from the 16 grid points must quantize with zero error."""
    g = 128
    scale, zero = 0.01, 0.0
    q_true = torch.randint(0, 16, (8, g * 3))
    w = (q_true.float() - 8.0) * scale + zero
    q, s, z = group_quantize_int4(w, g)
    w_hat = dequantize_int4(q, s, z, g)
    assert torch.allclose(w_hat, w, atol=1e-5), (w_hat - w).abs().max()


def test_quantization_error_is_bounded_by_half_a_step():
    g = 128
    w = torch.randn(64, g * 4) * 0.02
    q, s, z = group_quantize_int4(w, g)
    err = (dequantize_int4(q, s, z, g) - w).abs()
    step = s.repeat_interleave(g, dim=1)
    assert (err <= step / 2 + 1e-6).all()


@cuda
@pytest.mark.parametrize("n,k", SHAPES)
def test_packed_kernel_matches_dequantized_matmul(n, k):
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    r = verify_roundtrip(w)
    assert r["kernel_matches"], r
    # measured footprint must equal what the report claims
    assert abs(8 * r["bytes_q"] / (n * k) - QuantSpec().bits_per_param(128)) < 1e-3


@cuda
def test_probe_documents_tinygemm_layout():
    """Pins the two conventions that cannot be inferred from the signature:
    even-K index in the HIGH nibble, and scales_and_zeros shaped [K/g, N, 2]."""
    dev, N, K, g = "cuda", 32, 128, 128
    q = (torch.arange(N * K, device=dev) % 16).reshape(N, K).to(torch.uint8)
    s = torch.ones(N, K // g, device=dev)
    z = torch.zeros(N, K // g, device=dev)
    packed, sz = to_tinygemm(q, s, z, g)
    assert tuple(sz.shape) == (K // g, N, 2), sz.shape

    eye = torch.eye(K, device=dev, dtype=torch.bfloat16)
    w_eff = torch.ops.aten._weight_int4pack_mm(eye, packed, g, sz).float().T
    expect = q.float() - 8.0  # (q - 8) * 1 + 0
    assert torch.allclose(w_eff, expect, atol=1e-3), \
        f"layout drift: got {w_eff[0, :8].tolist()} expected {expect[0, :8].tolist()}"


def test_int8_per_channel_is_the_siglip_fallback():
    """SigLIP mlp.fc2 has in_features=4304 = 2^4 * 269, divisible by no allowed
    group size. Per-channel INT8 has no divisibility constraint at all."""
    w = torch.randn(1152, 4304) * 0.02
    assert all(4304 % g for g in (128, 64, 32))
    q, s = quantize_int8_per_channel(w)
    rel = float((dequantize_int8_per_channel(q, s) - w).norm() / w.norm())
    assert rel < 0.01, rel
