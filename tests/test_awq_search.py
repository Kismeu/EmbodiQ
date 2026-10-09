import torch

from embodiq.quant.awq_search import search_input_scale
from embodiq.quant.pack import dequantize_int4, group_quantize_int4


def test_awq_scale_search_is_finite_and_beats_rtn_on_calibration_batch():
    torch.manual_seed(42)
    weight = torch.randn(64, 128)
    samples = torch.randn(32, 128)
    result = search_input_scale(
        weight,
        {"samples": samples, "mean_abs": samples.abs().mean(0)},
        group_size=128,
        n_grid=20,
    )
    assert result["implementation"] == "awq_linear_scale_search_v1"
    assert torch.isfinite(result["scale"]).all()
    assert result["loss"] <= result["rtn_reference_loss"]

    q, scales, zeros = group_quantize_int4(weight * result["scale"][None], 128)
    effective = dequantize_int4(q, scales.bfloat16(), zeros.bfloat16(), 128)
    assert torch.isfinite(effective).all()
