"""The FastWAM correctness invariant, tested against a bf16 control.

FastWAM runs ``torch.compile(fullgraph=True, mode="reduce-overhead")`` by default,
so graph-safety is a Stage B correctness gate, not a Stage F optimization.

The gate is DIFFERENTIAL, not absolute. Measured on this machine, a plain bf16
LayerNorm+residual block already diverges 6.165e-03 between eager and compiled
(inductor keeps the fused residual+norm in fp32 where eager rounds to bf16), and
an INT4 block diverges by exactly the same 6.165e-03. An absolute "<= 2e-4"
threshold would therefore fail on FP16 itself and send someone hunting a
non-existent quantization bug. What must hold is that quantization adds NO
divergence beyond what inductor already introduces.
"""

from __future__ import annotations

import pytest
import torch
import torch._dynamo as dynamo
import torch.nn as nn

from embodiq.quant.qlinear import QuantLinear

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _block(quant: bool) -> nn.Module:
    class Blk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n = nn.LayerNorm(3072, dtype=torch.bfloat16, device="cuda")
            lin = lambda i, o: nn.Linear(i, o, bias=False).cuda().to(torch.bfloat16)
            wrap = QuantLinear.from_linear if quant else (lambda L: L)
            self.q, self.o = wrap(lin(3072, 3072)), wrap(lin(3072, 3072))
            self.f0, self.f2 = wrap(lin(3072, 14336)), wrap(lin(14336, 3072))

        def forward(self, x):
            h = self.o(self.q(self.n(x)))
            x = x + h
            return x + self.f2(torch.nn.functional.gelu(self.f0(self.n(x))))

    torch.manual_seed(0)
    return Blk().eval()


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def _eager_vs_compiled(m, x):
    with torch.no_grad():
        ye = m(x).clone()
        dynamo.reset()
        c = torch.compile(m, fullgraph=True, mode="reduce-overhead")
        for _ in range(5):
            yc = c(x).clone()
        torch.cuda.synchronize()
    return _rel(ye, yc), c


@cuda
def test_no_graph_break_with_fullgraph():
    m = _block(quant=True)
    x = torch.randn(1, 64, 3072, device="cuda", dtype=torch.bfloat16)
    dynamo.reset()
    exp = dynamo.explain(m)(x)
    assert exp.graph_break_count == 0, exp.break_reasons
    assert exp.graph_count == 1


@cuda
def test_quantization_adds_no_compile_divergence():
    """The differential gate."""
    x = torch.randn(1, 64, 3072, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(0)
    ctrl, _ = _eager_vs_compiled(_block(quant=False), x)
    torch.manual_seed(0)
    quant, _ = _eager_vs_compiled(_block(quant=True), x)
    assert quant <= max(ctrl * 1.5, 2e-4), (
        f"INT4 diverges {quant:.3e} under compile vs a bf16 control of {ctrl:.3e}"
    )


@cuda
def test_isolated_quantlinear_is_bit_identical_under_compile():
    """With no norm/residual in the graph there is nothing for inductor to
    re-associate, so the op itself must be exactly reproducible."""
    torch.manual_seed(0)
    lin = lambda i, o: nn.Linear(i, o, bias=False).cuda().to(torch.bfloat16)
    m = nn.Sequential(
        QuantLinear.from_linear(lin(3072, 14336)),
        nn.GELU(),
        QuantLinear.from_linear(lin(14336, 3072)),
    ).eval()
    x = torch.randn(1, 64, 3072, device="cuda", dtype=torch.bfloat16)
    r, _ = _eager_vs_compiled(m, x)
    assert r == 0.0, r


@cuda
def test_cudagraph_replays_are_stable_and_require_clone():
    """reduce-overhead reuses a static output buffer. Callers must clone; the
    repo does the same for its KV cache (fastwam.py cache.clone())."""
    m = _block(quant=True)
    x = torch.randn(1, 64, 3072, device="cuda", dtype=torch.bfloat16)
    _, c = _eager_vs_compiled(m, x)
    with torch.no_grad():
        cloned = [c(x).clone() for _ in range(3)]
        raw = [c(x) for _ in range(3)]
    torch.cuda.synchronize()
    assert all(torch.equal(cloned[0], t) for t in cloned[1:]), "replay is nondeterministic"
    assert raw[0].data_ptr() == raw[1].data_ptr(), "expected a reused static buffer"
