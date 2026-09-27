import pytest
import torch

from skyrl.backends.skyrl_train.patches.megatron.patch_dsa_masked_softmax import (
    _masked_softmax,
)


def _reference_masked_softmax(
    logits: torch.Tensor, valid_mask: torch.Tensor, dim: int = -1, eps: float = 1e-10
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
    row_has_valid = valid_mask.any(dim=dim, keepdim=True)
    row_max = masked_logits.max(dim=dim, keepdim=True).values
    row_max = torch.where(row_has_valid, row_max, torch.zeros_like(row_max))
    probabilities = torch.exp(masked_logits - row_max)
    probabilities = probabilities.masked_fill(~valid_mask, 0.0)
    probabilities = probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(eps)
    return probabilities.masked_fill(~valid_mask, 0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_memory_efficient_masked_softmax_matches_forward_and_backward(
    dtype: torch.dtype,
) -> None:
    logits = torch.randn(2, 3, 7, dtype=dtype, requires_grad=True)
    valid_mask = torch.rand_like(logits) > 0.3
    valid_mask[0, 0] = False
    weights = torch.randn_like(logits)

    reference = _reference_masked_softmax(logits * 1.0, valid_mask)
    (reference_gradient,) = torch.autograd.grad((reference * weights).sum(), logits)

    actual_logits = logits.detach().clone().requires_grad_()
    actual = _masked_softmax(actual_logits * 1.0, valid_mask)
    (actual_gradient,) = torch.autograd.grad((actual * weights).sum(), actual_logits)

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual_gradient, reference_gradient)


def test_memory_efficient_masked_softmax_validates_inputs() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        _masked_softmax(torch.ones(2, dtype=torch.int64), torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="same shape"):
        _masked_softmax(torch.ones(2), torch.ones(3, dtype=torch.bool))
