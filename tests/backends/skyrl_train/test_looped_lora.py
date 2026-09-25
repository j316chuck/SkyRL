from types import MethodType

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from skyrl.backends.skyrl_train.patches.megatron import looped_lora
from skyrl.backends.skyrl_train.patches.megatron.looped_lora import (
    _adapter_only,
    _looped_block_forward,
    _looped_linear_forward,
    _LoopedModuleList,
    _set_context,
)
from skyrl.train.looped_lora import (
    LayerExecution,
    build_looped_lora_schedule,
    get_lora_only_executions_by_physical_layer,
)


class _RecordingLayer(nn.Module):
    def __init__(self, layer_number: int, calls: list[tuple[int, bool]]) -> None:
        super().__init__()
        self.layer_number = layer_number
        self.calls = calls

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls.append((self.layer_number - 1, _adapter_only.get()))
        return hidden_states + self.layer_number


class _RecordingBlock(nn.Module):
    def __init__(self, num_layers: int, schedule) -> None:
        super().__init__()
        self.calls: list[tuple[int, bool]] = []
        layers = [_RecordingLayer(index + 1, self.calls) for index in range(num_layers)]
        self.layers = _LoopedModuleList(layers, schedule)
        self.num_layers_per_pipeline_rank = num_layers
        self._looped_lora_original_forward = self.forward
        self.forward = MethodType(_looped_block_forward, self)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _FakeRMSNormLinear(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layer_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.weight = nn.Parameter(torch.eye(hidden_size))
        self.normalization = "RMSNorm"
        self.eps = 1e-6
        self.zero_centered_gamma = False


class _FakeLoraLinear(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.to_wrap = _FakeRMSNormLinear(hidden_size)
        self.adapter = nn.Linear(hidden_size, hidden_size, bias=False)
        self._adapter_enabled = True
        self._base_returns_tuple = True
        self.base_calls = 0

    def _looped_lora_original_forward(self, inputs: torch.Tensor):
        self.base_calls += 1
        return self.to_wrap.weight @ inputs, None

    def adapter_forward(self, adapter: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        return adapter(inputs)

    def base_linear_forward(self, inputs: torch.Tensor):
        self.base_calls += 1
        return F.linear(inputs, self.to_wrap.weight), None, inputs


class _FactorizedAdapter(nn.Module):
    def __init__(self, input_size: int, output_size: int, rank: int = 2) -> None:
        super().__init__()
        self.linear_in = nn.Linear(input_size, rank, bias=False)
        self.linear_out = nn.Linear(rank, output_size, bias=False)
        nn.init.normal_(self.linear_in.weight)
        nn.init.zeros_(self.linear_out.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear_out(self.linear_in(inputs))


class _FactorizedLoraLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, *, behavior: str = "output_adapter") -> None:
        super().__init__()
        self.to_wrap = nn.Linear(input_size, output_size, bias=False)
        self.to_wrap.weight.requires_grad_(False)
        self.adapter = _FactorizedAdapter(input_size, output_size)
        self._adapter_enabled = True
        self._base_returns_tuple = True
        self._looped_lora_extra_behavior = behavior

    def _looped_lora_original_forward(self, inputs: torch.Tensor):
        return self.to_wrap(inputs) + self.adapter(inputs), None

    def base_linear_forward(self, inputs: torch.Tensor):
        return self.to_wrap(inputs), None, inputs

    def adapter_forward(self, adapter: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        return adapter(inputs)


def _run_base_output_adapter_layer(
    inputs: torch.Tensor,
    qkv: _FactorizedLoraLinear,
    output: _FactorizedLoraLinear,
    gate_up: _FactorizedLoraLinear,
    down: _FactorizedLoraLinear,
) -> torch.Tensor:
    with _set_context(_adapter_only, True):
        attention_features, _ = _looped_linear_forward(qkv, inputs)
        attention_update, _ = _looped_linear_forward(output, torch.tanh(attention_features))
        hidden_states = inputs + attention_update
        gate_up_features, _ = _looped_linear_forward(gate_up, hidden_states)
        gate, up = gate_up_features.chunk(2, dim=-1)
        mlp_update, _ = _looped_linear_forward(down, F.silu(gate) * up)
    return hidden_states + mlp_update


def test_megatron_extra_pass_skips_frozen_linear_and_backpropagates_through_lora() -> None:
    linear = _FakeLoraLinear(hidden_size=3)
    inputs = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    with _set_context(_adapter_only, True):
        output, bias = _looped_linear_forward(linear, inputs)
    output.sum().backward()

    assert bias is None
    assert linear.base_calls == 0
    assert linear.adapter.weight.grad is not None
    assert linear.to_wrap.weight.grad is None
    assert inputs.grad is not None


def test_base_output_adapter_is_identity_at_zero_b_with_first_order_b_gradients() -> None:
    torch.manual_seed(0)
    qkv = _FactorizedLoraLinear(4, 4, behavior="base_feature")
    output = _FactorizedLoraLinear(4, 4, behavior="output_adapter")
    gate_up = _FactorizedLoraLinear(4, 8, behavior="base_feature")
    down = _FactorizedLoraLinear(4, 4, behavior="output_adapter")
    inputs = torch.randn(3, 4, requires_grad=True)

    result = _run_base_output_adapter_layer(inputs, qkv, output, gate_up, down)

    assert torch.equal(result, inputs)
    result.square().sum().backward()
    assert output.adapter.linear_out.weight.grad is not None
    assert output.adapter.linear_out.weight.grad.abs().sum() > 0
    assert down.adapter.linear_out.weight.grad is not None
    assert down.adapter.linear_out.weight.grad.abs().sum() > 0
    assert qkv.adapter.linear_out.weight.grad is None
    assert gate_up.adapter.linear_out.weight.grad is None
    assert qkv.to_wrap.weight.grad is None
    assert gate_up.to_wrap.weight.grad is None
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_base_output_adapter_has_finite_nonlinear_forward_and_backward() -> None:
    torch.manual_seed(1)
    qkv = _FactorizedLoraLinear(4, 4, behavior="base_feature")
    output = _FactorizedLoraLinear(4, 4, behavior="output_adapter")
    gate_up = _FactorizedLoraLinear(4, 8, behavior="base_feature")
    down = _FactorizedLoraLinear(4, 4, behavior="output_adapter")
    with torch.no_grad():
        output.adapter.linear_out.weight.normal_(std=1e-2)
        down.adapter.linear_out.weight.normal_(std=1e-2)
    inputs = torch.randn(3, 4, requires_grad=True)

    result = _run_base_output_adapter_layer(inputs, qkv, output, gate_up, down)
    result.sum().backward()

    assert not torch.equal(result, inputs)
    assert torch.isfinite(result).all()
    for parameter in (*output.parameters(), *down.parameters()):
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_install_base_output_adapter_routes_repeated_layer_projections(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_qkv = _FactorizedLoraLinear(4, 4)
            self.linear_proj = _FactorizedLoraLinear(4, 4)

    class _Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_fc1 = _FactorizedLoraLinear(4, 8)
            self.linear_fc2 = _FactorizedLoraLinear(4, 4)

    class _Layer(nn.Module):
        def __init__(self, layer_number: int) -> None:
            super().__init__()
            self.layer_number = layer_number
            self.self_attention = _Attention()
            self.mlp = _Mlp()

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_Layer(1), _Layer(2)])
            self.config = type("Config", (), {"num_layers": 2, "enable_mhc_connections": False})()
            self.num_layers_per_pipeline_rank = 2

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = _Block()

    class _FakeFusedLora(nn.Module):
        pass

    import megatron.bridge.peft.lora_layers as lora_layers

    monkeypatch.setattr(lora_layers, "LoRALinear", _FactorizedLoraLinear)
    monkeypatch.setattr(lora_layers, "TEFusedLoRALinear", _FakeFusedLora)
    model = _Model()

    looped_lora.install_looped_lora(
        model,
        [{"start_layer": 0, "end_layer": 1, "repeat_count": 2}],
        "base_output_adapter",
    )

    repeated = model.decoder.layers[0]
    assert repeated.self_attention.linear_qkv._looped_lora_extra_behavior == "base_feature"
    assert repeated.mlp.linear_fc1._looped_lora_extra_behavior == "base_feature"
    assert repeated.self_attention.linear_proj._looped_lora_extra_behavior == "output_adapter"
    assert repeated.mlp.linear_fc2._looped_lora_extra_behavior == "output_adapter"


def test_megatron_extra_pass_repeats_fused_layernorm_sequence_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    linear = _FakeLoraLinear(hidden_size=3)
    linear.to_wrap.return_layernorm_output_gathered = True
    linear.adapter.tp_group = object()
    gathered_group = None

    def gather(inputs: torch.Tensor, group: object) -> torch.Tensor:
        nonlocal gathered_group
        gathered_group = group
        return torch.cat((inputs, inputs), dim=0)

    monkeypatch.setattr(looped_lora, "_gather_sequence_parallel_input", gather)

    with _set_context(_adapter_only, True):
        output, _ = _looped_linear_forward(linear, torch.ones(2, 3))

    assert output.shape == (4, 3)
    assert gathered_group is linear.adapter.tp_group


@pytest.mark.parametrize("repeat_count", [1, 2, 4])
def test_megatron_block_uses_vllm_section_order(repeat_count: int) -> None:
    schedule = build_looped_lora_schedule(
        6,
        [{"start_layer": 1, "end_layer": 4, "repeat_count": repeat_count}],
    )
    block = _RecordingBlock(6, schedule)

    output = block(torch.zeros(1))

    physical_calls = [0, 1, 2, 3] + [1, 2, 3] * (repeat_count - 1) + [4, 5]
    expected_calls = [
        (physical_layer, execution.lora_only) for physical_layer, execution in zip(physical_calls, schedule)
    ]
    assert block.calls == expected_calls
    assert output.item() == sum(index + 1 for index in physical_calls)
    assert [layer.layer_number for layer in block.layers] == list(range(1, 7))


def test_megatron_checkpoint_recompute_uses_logical_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    schedule = build_looped_lora_schedule(
        6,
        [{"start_layer": 2, "end_layer": 4, "repeat_count": 2}],
    )
    block = _RecordingBlock(6, schedule)
    monkeypatch.setattr(looped_lora, "_is_megatron_checkpointing", lambda: True)

    replayed_layer = block.layers[4]

    assert len(block.layers) == 8
    assert replayed_layer.layer_number == 3
    assert _adapter_only.get()
    _adapter_only.set(False)


@pytest.mark.parametrize(
    ("repeat_count", "expected_length", "expected_lora_only"),
    [(1, 36, 0), (2, 44, 8), (4, 60, 24)],
)
def test_middle_section_repeats_only_lora_after_first_pass(
    repeat_count: int,
    expected_length: int,
    expected_lora_only: int,
) -> None:
    schedule = build_looped_lora_schedule(
        36,
        [{"start_layer": 14, "end_layer": 22, "repeat_count": repeat_count}],
    )

    assert len(schedule) == expected_length
    assert sum(execution.lora_only for execution in schedule) == expected_lora_only
    assert [execution.physical_layer for execution in schedule if not execution.lora_only] == list(range(36))


def test_sections_can_have_independent_repeat_counts() -> None:
    schedule = build_looped_lora_schedule(
        36,
        [
            {"start_layer": 0, "end_layer": 4, "repeat_count": 2},
            {"start_layer": 4, "end_layer": 20, "repeat_count": 3},
            {"start_layer": 20, "end_layer": 36, "repeat_count": 1},
        ],
    )

    assert len(schedule) == 72
    assert schedule[:4] == tuple(LayerExecution(layer, False) for layer in range(4))
    assert schedule[4:8] == tuple(LayerExecution(layer, True) for layer in range(4))
    assert schedule[8:24] == tuple(LayerExecution(layer, False) for layer in range(4, 20))
    assert schedule[24:40] == tuple(LayerExecution(layer, True) for layer in range(4, 20))
    assert schedule[40:56] == tuple(LayerExecution(layer, True) for layer in range(4, 20))
    assert schedule[56:] == tuple(LayerExecution(layer, False) for layer in range(20, 36))


def test_unconfigured_gaps_execute_once() -> None:
    schedule = build_looped_lora_schedule(
        8,
        [{"start_layer": 2, "end_layer": 4, "repeat_count": 2}],
    )

    assert schedule == (
        LayerExecution(0, False),
        LayerExecution(1, False),
        LayerExecution(2, False),
        LayerExecution(3, False),
        LayerExecution(2, True),
        LayerExecution(3, True),
        LayerExecution(4, False),
        LayerExecution(5, False),
        LayerExecution(6, False),
        LayerExecution(7, False),
    )


def test_lora_only_executions_get_distinct_logical_indices() -> None:
    schedule = build_looped_lora_schedule(
        6,
        [{"start_layer": 2, "end_layer": 4, "repeat_count": 3}],
    )

    assert get_lora_only_executions_by_physical_layer(6, schedule) == (
        (),
        (),
        (4, 6),
        (5, 7),
        (),
        (),
    )


@pytest.mark.parametrize(
    "sections",
    [
        [{"start_layer": 4, "end_layer": 4, "repeat_count": 2}],
        [{"start_layer": -1, "end_layer": 4, "repeat_count": 2}],
        [{"start_layer": 2, "end_layer": 4, "repeat_count": 0}],
        [{"start_layer": 2, "end_layer": 7, "repeat_count": 2}],
        [
            {"start_layer": 2, "end_layer": 5, "repeat_count": 2},
            {"start_layer": 4, "end_layer": 6, "repeat_count": 2},
        ],
    ],
)
def test_invalid_sections_fail_before_model_start(
    sections: list[dict[str, int]],
) -> None:
    with pytest.raises(ValueError):
        build_looped_lora_schedule(6, sections)
