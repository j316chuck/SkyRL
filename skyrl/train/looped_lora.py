from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

SUPPORTED_LOOPED_LORA_MODES = frozenset({"lora_only", "full_block", "gated_full_block"})


@dataclass(frozen=True)
class LoopedLoraSection:
    start_layer: int
    end_layer: int
    repeat_count: int


@dataclass(frozen=True)
class LayerExecution:
    physical_layer: int
    lora_only: bool


def validate_looped_lora_config(mode: str, gamma: float) -> None:
    if mode not in SUPPORTED_LOOPED_LORA_MODES:
        raise ValueError(f"Unsupported looped LoRA mode: {mode!r}")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"Looped LoRA gamma must be in (0, 1], got {gamma!r}")


def blend_hidden_states(
    input_hidden_states: torch.Tensor,
    output_hidden_states: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Blend a recurrent block output with the block input."""
    return torch.lerp(input_hidden_states, output_hidden_states, gamma)


def blend_qwen_residual_stream(
    input_hidden_states: torch.Tensor,
    input_residual: torch.Tensor | None,
    output_hidden_states: torch.Tensor,
    output_residual: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend a Qwen block while retaining vLLM's fused residual representation."""
    input_stream = input_hidden_states if input_residual is None else input_hidden_states + input_residual
    return output_hidden_states * gamma, torch.lerp(input_stream, output_residual, gamma)


def parse_looped_lora_sections(
    sections: Sequence[Mapping[str, Any] | LoopedLoraSection],
    num_hidden_layers: int | None = None,
) -> tuple[LoopedLoraSection, ...]:
    parsed = tuple(
        (
            section
            if isinstance(section, LoopedLoraSection)
            else LoopedLoraSection(
                start_layer=section["start_layer"],
                end_layer=section["end_layer"],
                repeat_count=section["repeat_count"],
            )
        )
        for section in sections
    )

    previous_end = 0
    for section in parsed:
        if section.start_layer < previous_end:
            raise ValueError("Looped LoRA sections must be ordered and non-overlapping")
        if section.start_layer < 0 or section.end_layer <= section.start_layer:
            raise ValueError("Looped LoRA sections must use a non-empty half-open layer range")
        if section.repeat_count < 1:
            raise ValueError("Looped LoRA repeat_count must be at least 1")
        if num_hidden_layers is not None and section.end_layer > num_hidden_layers:
            raise ValueError(
                f"Looped LoRA section ends at layer {section.end_layer}, "
                f"but the model has {num_hidden_layers} layers"
            )
        previous_end = section.end_layer

    return parsed


def build_looped_lora_schedule(
    num_hidden_layers: int,
    sections: Sequence[Mapping[str, Any] | LoopedLoraSection],
) -> tuple[LayerExecution, ...]:
    parsed = parse_looped_lora_sections(sections, num_hidden_layers)
    schedule: list[LayerExecution] = []
    next_layer = 0

    for section in parsed:
        schedule.extend(LayerExecution(layer, False) for layer in range(next_layer, section.end_layer))
        for _ in range(section.repeat_count - 1):
            schedule.extend(LayerExecution(layer, True) for layer in range(section.start_layer, section.end_layer))
        next_layer = section.end_layer

    schedule.extend(LayerExecution(layer, False) for layer in range(next_layer, num_hidden_layers))
    return tuple(schedule)


def get_lora_only_executions_by_physical_layer(
    num_hidden_layers: int,
    schedule: Sequence[LayerExecution],
) -> tuple[tuple[int, ...], ...]:
    executions: list[list[int]] = [[] for _ in range(num_hidden_layers)]
    for execution_index, execution in enumerate(schedule):
        if execution.lora_only:
            executions[execution.physical_layer].append(execution_index)
    return tuple(tuple(indices) for indices in executions)
