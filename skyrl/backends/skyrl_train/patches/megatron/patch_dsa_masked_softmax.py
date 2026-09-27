"""Memory-efficient autograd implementation for megatron-core's DSA softmax."""

from __future__ import annotations

import torch


class _MemoryEfficientMaskedSoftmax(torch.autograd.Function):
    """Compute masked softmax in place and reconstruct its gradient from the output."""

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
        dim: int,
        eps: float,
    ) -> torch.Tensor:
        ctx.dim = dim
        ctx.mark_dirty(logits)

        logits.masked_fill_(~valid_mask, torch.finfo(logits.dtype).min)
        row_has_valid = valid_mask.any(dim=dim, keepdim=True)
        row_max = logits.max(dim=dim, keepdim=True).values
        row_max = torch.where(row_has_valid, row_max, torch.zeros_like(row_max))

        logits.sub_(row_max)
        logits.exp_()
        logits.masked_fill_(~valid_mask, 0.0)
        logits.div_(logits.sum(dim=dim, keepdim=True).clamp_min(eps))
        logits.masked_fill_(~valid_mask, 0.0)

        ctx.save_for_backward(logits, valid_mask)
        return logits

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        probabilities, valid_mask = ctx.saved_tensors
        projection = (grad_output * probabilities).sum(dim=ctx.dim, keepdim=True)
        grad_logits = probabilities * (grad_output - projection)
        grad_logits.masked_fill_(~valid_mask, 0.0)
        return grad_logits, None, None, None


def _masked_softmax(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-10,
) -> torch.Tensor:
    if not logits.is_floating_point():
        raise TypeError("masked_softmax expects a floating-point tensor")
    if logits.shape != valid_mask.shape:
        raise ValueError("logits and valid_mask must have the same shape")
    return _MemoryEfficientMaskedSoftmax.apply(logits, valid_mask, dim, eps)


def apply_dsa_masked_softmax_patch() -> None:
    """Replace DSA's quadratic-copy softmax with an autograd-safe in-place version."""
    from megatron.core.transformer.experimental_attention_variant import dsa_masking

    if getattr(dsa_masking.masked_softmax, "_skyrl_memory_efficient", False):
        return
    _masked_softmax._skyrl_memory_efficient = True
    dsa_masking.masked_softmax = _masked_softmax
