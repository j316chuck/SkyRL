"""Unit tests for SkyRLTrainBackend._normalize_policy_loss_request.

No Ray runtime or GPUs are needed — the method is pure. Requires the
SkyRL-Train backend deps (ray/vllm) to be importable. Run:
  uv run --extra dev --extra fsdp pytest tests/tinker/skyrl_train/test_loss_normalization.py
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

# Skip if skyrl_train_backend.py cannot be imported
skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

_normalize = skyrl_train_backend.SkyRLTrainBackend._normalize_policy_loss_request


def test_ppo_thresholds_map_to_eps_clip():
    loss_fn, config = _normalize(None, "policy", "ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28})
    assert loss_fn == "regular"
    assert config == pytest.approx({"eps_clip_low": 0.2, "eps_clip_high": 0.28})


def test_gspo_thresholds_map_to_native_sequence_mean():
    loss_fn, config = _normalize(
        None, "policy", "gspo", {"clip_low_threshold": 0.98, "clip_high_threshold": 1.03}
    )
    assert loss_fn == "gspo"
    assert config.pop("loss_reduction") == "sequence_mean"
    assert config == pytest.approx({"eps_clip_low": 0.02, "eps_clip_high": 0.03})


def test_gspo_without_threshold_overrides_still_uses_sequence_mean():
    loss_fn, config = _normalize(None, "policy", "gspo", None)
    assert loss_fn == "gspo"
    assert config == {"loss_reduction": "sequence_mean"}


def test_gspo_sequence_mean_weights_datums_equally_across_lengths():
    batch = skyrl_train_backend.TrainingInputBatch(
        {
            "loss_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
            "advantages": torch.full((2, 4), 2.0),
        }
    )

    skyrl_train_backend._apply_gspo_sequence_mean(batch)

    contributions = (batch["advantages"] * batch["loss_mask"]).sum(dim=-1)
    assert contributions.tolist() == pytest.approx([1.0, 1.0])
    assert contributions.sum().item() == pytest.approx(2.0)


def test_gspo_dispatches_native_loss_with_scaled_advantages_and_clip_metric():
    batch = skyrl_train_backend.TrainingInputBatch(
        {
            "loss_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
            "advantages": torch.full((2, 4), 2.0),
        }
    )
    captured = {}

    class FakeDispatch:
        def forward_backward(self, role, dispatched_batch, **kwargs):
            captured.update(role=role, batch=dispatched_batch, **kwargs)
            return SimpleNamespace(
                loss_fn_output_type="policy",
                loss_fn_outputs=[{"logprobs": [0.0]}, {"logprobs": [0.0]}],
                metrics={"loss_metrics/clip_ratio": 0.25},
            )

    backend = SimpleNamespace(
        _cfg=SimpleNamespace(
            trainer=SimpleNamespace(strategy="megatron", micro_train_batch_size_per_gpu=1)
        ),
        _dispatch=FakeDispatch(),
        _get_batch_role=lambda _: "policy",
        _validate_batch_role_and_loss=lambda *_: None,
        _to_training_batch=lambda *_: batch,
        _pad_batch=lambda dispatched_batch, micro_batch_size: (dispatched_batch, 0),
        _extract_metrics=lambda _: {},
    )
    backend._normalize_policy_loss_request = MethodType(
        skyrl_train_backend.SkyRLTrainBackend._normalize_policy_loss_request, backend
    )
    prepared_batch = skyrl_train_backend.types.PreparedModelPassBatch(
        all_model_inputs=[
            skyrl_train_backend.types.ModelInput(
                chunks=[skyrl_train_backend.types.EncodedTextChunk(tokens=[1])]
            )
        ]
        * 2,
        all_targets=[[2], [2]],
        all_token_weights=[[1.0], [1.0]],
        all_sampling_logprobs=[[-1.0], [-1.0]],
        all_advantages=[[2.0], [2.0]],
        all_values=[[], []],
        all_returns=[[], []],
        all_model_ids=["model", "model"],
        all_loss_fns=["gspo", "gspo"],
        all_loss_fn_configs=[
            {"clip_low_threshold": 0.98, "clip_high_threshold": 1.03},
            {"clip_low_threshold": 0.98, "clip_high_threshold": 1.03},
        ],
        request_batch_slices=[("request", "model", 0, 2)],
    )

    results = skyrl_train_backend.SkyRLTrainBackend._forward_backward_single_model_batch(
        backend, prepared_batch
    )

    assert captured["role"] == "policy"
    assert captured["loss_fn"] == "gspo"
    assert captured["loss_fn_config"].pop("loss_reduction") == "sequence_mean"
    assert captured["loss_fn_config"] == pytest.approx({"eps_clip_low": 0.02, "eps_clip_high": 0.03})
    contributions = (captured["batch"]["advantages"] * captured["batch"]["loss_mask"]).sum(dim=-1)
    assert contributions.tolist() == pytest.approx([1.0, 1.0])
    assert results["request"].metrics == {"gspo/clipped_frac:mean": 0.25}


def test_dppo_deltas_are_nested_under_dppo():
    loss_fn, config = _normalize(None, "policy", "dppo", {"delta_low": 0.2, "delta_high": 0.3})
    assert loss_fn == "dppo"
    assert config == {"dppo": {"delta_low": 0.2, "delta_high": 0.3}}


def test_dppo_partial_deltas():
    loss_fn, config = _normalize(None, "policy", "dppo", {"delta_high": 0.05})
    assert loss_fn == "dppo"
    assert config == {"dppo": {"delta_high": 0.05}}


def test_dppo_without_config_passes_through():
    loss_fn, config = _normalize(None, "policy", "dppo", None)
    assert loss_fn == "dppo"
    assert config is None


def test_critic_config_passes_through_unchanged():
    loss_fn, config = _normalize(None, "critic", "ppo", {"value_clip": 0.2})
    assert loss_fn == "ppo"
    assert config == {"value_clip": 0.2}
