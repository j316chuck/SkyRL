from skyrl.backends.skyrl_train_backend import (
    FSDPBackendOverrides,
    MegatronBackendOverrides,
    _build_skyrl_train_config,
)

NEMOTRON_35_LIGHTNING_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


def test_nemotron_35_megatron_enables_router_replay_without_overrides():
    cfg = _build_skyrl_train_config(NEMOTRON_35_LIGHTNING_MODEL, MegatronBackendOverrides())

    assert cfg.generator.inference_engine.enable_return_routed_experts is True
    assert cfg.trainer.policy.megatron_config.moe_enable_routing_replay is True
    assert cfg.trainer.policy.megatron_config.transformer_config_kwargs["moe_router_fusion"] is False


def test_other_megatron_model_keeps_router_replay_defaults():
    cfg = _build_skyrl_train_config("Qwen/Qwen3-4B", MegatronBackendOverrides())

    assert cfg.generator.inference_engine.enable_return_routed_experts is False
    assert cfg.trainer.policy.megatron_config.moe_enable_routing_replay is False
    assert "moe_router_fusion" not in cfg.trainer.policy.megatron_config.transformer_config_kwargs


def test_nemotron_35_fsdp_keeps_router_replay_defaults():
    cfg = _build_skyrl_train_config(NEMOTRON_35_LIGHTNING_MODEL, FSDPBackendOverrides())

    assert cfg.generator.inference_engine.enable_return_routed_experts is False
    assert cfg.trainer.policy.megatron_config.moe_enable_routing_replay is False
    assert "moe_router_fusion" not in cfg.trainer.policy.megatron_config.transformer_config_kwargs
