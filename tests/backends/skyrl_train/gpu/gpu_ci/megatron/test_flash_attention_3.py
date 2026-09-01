from importlib.metadata import version

import pytest
import torch

pytestmark = [pytest.mark.megatron, pytest.mark.h100]


def test_transformer_engine_detects_working_flash_attention_3():
    from flash_attn_interface import flash_attn_func
    from transformer_engine.pytorch.attention.dot_product_attention import backends

    assert version("flash-attn-3") == "3.0.0"
    assert backends.fa_utils.v3_is_installed

    q, k, v = [
        torch.randn(
            2, 128, heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for heads in (8, 2, 2)
    ]
    output = flash_attn_func(q, k, v, causal=True)
    output.float().square().mean().backward()

    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
