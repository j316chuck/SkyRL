from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from skyrl.train.looped_lora import (
    LayerExecution,
    blend_hidden_states,
    build_looped_lora_schedule,
    validate_looped_lora_config,
)

_adapter_only: ContextVar[bool] = ContextVar("looped_lora_adapter_only", default=False)
_loop_schedule_active: ContextVar[bool] = ContextVar("looped_lora_schedule_active", default=False)
_loop_gamma: ContextVar[float | None] = ContextVar("looped_lora_gamma", default=None)


@contextmanager
def _set_context(variable: ContextVar, value: Any) -> Iterator[None]:
    token = variable.set(value)
    try:
        yield
    finally:
        variable.reset(token)


def _gather_sequence_parallel_input(inputs: torch.Tensor, group: Any) -> torch.Tensor:
    from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

    return gather_from_sequence_parallel_region(inputs, group=group)


def _is_megatron_checkpointing() -> bool:
    try:
        from megatron.core.tensor_parallel.random import is_checkpointing
    except ModuleNotFoundError:
        return False
    return is_checkpointing()


def _schedule_is_active() -> bool:
    return _loop_schedule_active.get() or _is_megatron_checkpointing()


def _normalize_adapter_input(linear: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    base = linear.to_wrap
    weight = getattr(base, "layer_norm_weight", None)
    if weight is None:
        return inputs

    normalization = getattr(base, "normalization", None)
    eps = base.eps
    if getattr(base, "zero_centered_gamma", False):
        weight = weight + 1
    if normalization == "RMSNorm":
        normalized = F.rms_norm(inputs, (inputs.shape[-1],), weight, eps)
    elif normalization == "LayerNorm":
        normalized = F.layer_norm(
            inputs,
            (inputs.shape[-1],),
            weight,
            getattr(base, "layer_norm_bias", None),
            eps,
        )
    else:
        raise ValueError(f"Unsupported looped LoRA normalization: {normalization!r}")

    if getattr(base, "return_layernorm_output_gathered", False):
        normalized = _gather_sequence_parallel_input(normalized, linear.adapter.tp_group)
    return normalized


def _looped_linear_forward(linear: nn.Module, inputs: torch.Tensor, *args: Any, **kwargs: Any):
    if not _adapter_only.get():
        return linear._looped_lora_original_forward(inputs, *args, **kwargs)
    if not linear._adapter_enabled:
        raise RuntimeError("Looped LoRA extra passes require enabled LoRA adapters")

    adapter_inputs = _normalize_adapter_input(linear, inputs).contiguous()
    output = linear.adapter_forward(linear.adapter, adapter_inputs, *args, **kwargs)
    if linear._base_returns_tuple:
        return output, None
    return output


def _looped_layer_forward(layer: nn.Module, *args: Any, **kwargs: Any):
    output = layer._looped_lora_original_forward(*args, **kwargs)
    gamma = _loop_gamma.get()
    if gamma is None:
        return output

    input_hidden_states = args[0] if args else kwargs["hidden_states"]
    output_hidden_states, context = output
    return blend_hidden_states(input_hidden_states, output_hidden_states, gamma), context


class _LoopedModuleList(nn.ModuleList):
    def __init__(
        self,
        layers: Sequence[nn.Module],
        schedule: Sequence[LayerExecution],
        *,
        mode: str = "lora_only",
        gamma: float = 0.25,
    ) -> None:
        super().__init__(layers)
        validate_looped_lora_config(mode, gamma)
        self._executions = tuple(
            (
                super(_LoopedModuleList, self).__getitem__(execution.physical_layer),
                execution.lora_only if mode == "lora_only" else False,
                gamma if mode == "gated_full_block" and execution.lora_only else None,
            )
            for execution in schedule
        )

    def __iter__(self):
        if _schedule_is_active():
            return self._iter_executions()
        return super().__iter__()

    def _iter_executions(self):
        for layer, lora_only, gamma in self._executions:
            with (
                _set_context(_adapter_only, lora_only),
                _set_context(_loop_gamma, gamma),
            ):
                yield layer

    def __len__(self) -> int:
        if _schedule_is_active():
            return len(self._executions)
        return super().__len__()

    def __getitem__(self, index):
        if _schedule_is_active():
            if isinstance(index, slice):
                return tuple(layer for layer, _, _ in self._executions[index])
            layer, lora_only, gamma = self._executions[index]
            _adapter_only.set(lora_only)
            _loop_gamma.set(gamma)
            return layer
        return super().__getitem__(index)


def _looped_block_forward(block: nn.Module, *args: Any, **kwargs: Any):
    with (
        _set_context(_loop_schedule_active, True),
        _set_context(_adapter_only, False),
        _set_context(_loop_gamma, None),
    ):
        physical_layer_count = block.num_layers_per_pipeline_rank
        block.num_layers_per_pipeline_rank = len(block.layers)
        try:
            return block._looped_lora_original_forward(*args, **kwargs)
        finally:
            block.num_layers_per_pipeline_rank = physical_layer_count


def install_looped_lora(
    model: nn.Module | Sequence[nn.Module],
    sections: Sequence[dict[str, int]],
    mode: str,
    gamma: float,
) -> None:
    """Install the Megatron forward schedule matching SkyRL's vLLM loop semantics."""
    validate_looped_lora_config(mode, gamma)

    from megatron.bridge.peft.lora_layers import LoRALinear, TEFusedLoRALinear

    roots = (model,) if isinstance(model, nn.Module) else tuple(model)
    modules = tuple(module for root in roots for module in root.modules())
    blocks = [module for module in modules if hasattr(module, "layers") and module.layers]
    blocks = [block for block in blocks if all(hasattr(layer, "layer_number") for layer in block.layers)]
    if len(blocks) != 1:
        raise ValueError("Looped LoRA training currently requires one non-empty Megatron transformer block (PP=1)")

    block = blocks[0]
    physical_layers = list(block.layers)
    num_hidden_layers = block.config.num_layers
    if len(physical_layers) != num_hidden_layers:
        raise ValueError("Looped LoRA training currently requires every transformer layer on one pipeline stage (PP=1)")
    if any(layer.layer_number != index + 1 for index, layer in enumerate(physical_layers)):
        raise ValueError("Looped LoRA requires contiguous, one-based Megatron layer numbers")
    if getattr(block.config, "enable_mhc_connections", False):
        raise ValueError("Looped LoRA does not support Megatron hyper-connections")

    schedule = build_looped_lora_schedule(num_hidden_layers, sections)

    for module in modules:
        if isinstance(module, TEFusedLoRALinear):
            raise TypeError("Looped LoRA does not support fused Transformer Engine LoRA")
        if isinstance(module, LoRALinear):
            module._looped_lora_original_forward = module.forward
            module.forward = MethodType(_looped_linear_forward, module)

    if mode == "gated_full_block":
        for layer in physical_layers:
            layer._looped_lora_original_forward = layer.forward
            layer.forward = MethodType(_looped_layer_forward, layer)

    block.layers = _LoopedModuleList(physical_layers, schedule, mode=mode, gamma=gamma)
    block._looped_lora_original_forward = block.forward
    block.forward = MethodType(_looped_block_forward, block)
