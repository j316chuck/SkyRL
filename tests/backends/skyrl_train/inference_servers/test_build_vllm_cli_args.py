"""Tests for build_vllm_cli_args on GPU-less hosts."""

import pytest

from skyrl.backends.skyrl_train.inference_servers.utils import (
    build_vllm_cli_args,
    resolve_policy_model_name,
)
from skyrl.train.config import SkyRLTrainConfig


@pytest.mark.vllm
def test_build_vllm_cli_args_succeeds_on_gpu_less_host(monkeypatch):
    import vllm.platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    # Simulate the GPU-less Ray head-node case: vLLM resolves current_platform
    # to UnspecifiedPlatform (device_type == ""), so AsyncEngineArgs.add_cli_args
    # walks VllmConfig defaults, instantiates DeviceConfig() and its
    # __post_init__ raises "Failed to infer device type" during arg parsing.
    # With the fix in build_vllm_cli_args, current_platform.device_type is
    # pinned to "cuda" before add_cli_args runs.
    monkeypatch.setattr(vllm.platforms, "_current_platform", UnspecifiedPlatform())

    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.served_model_name = "served-alias"
    cfg.generator.inference_engine.engine_init_kwargs = {
        "hf_overrides": {"rope_parameters": {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}}
    }
    args = build_vllm_cli_args(cfg)

    assert args is not None
    assert args.model == cfg.trainer.policy.model.path
    assert args.served_model_name == ["served-alias"]
    assert args.tensor_parallel_size == cfg.generator.inference_engine.tensor_parallel_size
    assert args.hf_overrides["rope_parameters"] == {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}
    assert vllm.platforms.current_platform.device_type == "cuda"

    # NOTE: the MTP speculative_config wiring test lives in
    # tests/backends/skyrl_train/mtp/test_build_vllm_cli_args_mtp.py


@pytest.mark.vllm
def test_nemotron_35_lightning_uses_bf16_kv_cache(monkeypatch):
    import vllm.platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    monkeypatch.setattr(vllm.platforms, "_current_platform", UnspecifiedPlatform())

    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    cfg.generator.inference_engine.engine_init_kwargs = {"kv_cache_dtype": "fp8"}

    args = build_vllm_cli_args(cfg)

    assert args.kv_cache_dtype == "bfloat16"


@pytest.mark.vllm
def test_other_models_preserve_fp8_kv_cache(monkeypatch):
    import vllm.platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    monkeypatch.setattr(vllm.platforms, "_current_platform", UnspecifiedPlatform())

    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "Qwen/Qwen3.6-27B"
    cfg.generator.inference_engine.engine_init_kwargs = {"kv_cache_dtype": "fp8"}

    args = build_vllm_cli_args(cfg)

    assert args.kv_cache_dtype == "fp8"


def test_resolve_policy_model_name_uses_served_model_name():
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "base-model"
    cfg.generator.inference_engine.served_model_name = "served-alias"

    assert resolve_policy_model_name(cfg) == "served-alias"
