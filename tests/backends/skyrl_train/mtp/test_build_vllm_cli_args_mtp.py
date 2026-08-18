"""MTP speculative-decoding wiring into the vLLM CLI args (runs on a GPU-less host).

uv run --isolated --extra dev pytest tests/backends/skyrl_train/mtp/test_build_vllm_cli_args_mtp.py
"""

import pytest

from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl.train.config import SkyRLTrainConfig


@pytest.mark.vllm
def test_build_vllm_cli_args_speculative_config_mtp(monkeypatch):
    """speculative_config is passed through to vLLM for MTP draft decoding."""
    pytest.importorskip("vllm")
    import vllm.platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    # Pin the platform so add_cli_args works on a GPU-less host (see the sibling
    # test_build_vllm_cli_args.py for the rationale).
    monkeypatch.setattr(vllm.platforms, "_current_platform", UnspecifiedPlatform())

    cfg = SkyRLTrainConfig()
    # Default: no speculative decoding.
    args = build_vllm_cli_args(cfg)
    assert getattr(args, "speculative_config", None) is None

    # Enable MTP speculative decoding.
    spec = {"method": "mtp", "num_speculative_tokens": 1}
    cfg.generator.inference_engine.speculative_config = spec
    args = build_vllm_cli_args(cfg)
    assert args.speculative_config == spec


@pytest.mark.vllm
def test_nemotron_35_lightning_enables_one_mtp_draft_token(monkeypatch):
    pytest.importorskip("vllm")
    import vllm.platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    monkeypatch.setattr(vllm.platforms, "_current_platform", UnspecifiedPlatform())

    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

    args = build_vllm_cli_args(cfg)

    assert args.speculative_config == {"method": "mtp", "num_speculative_tokens": 1}


@pytest.mark.vllm
def test_nemotron_35_lightning_preserves_explicit_speculative_config(monkeypatch):
    pytest.importorskip("vllm")
    import vllm.platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    monkeypatch.setattr(vllm.platforms, "_current_platform", UnspecifiedPlatform())

    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    cfg.generator.inference_engine.speculative_config = {
        "method": "mtp",
        "num_speculative_tokens": 2,
    }

    args = build_vllm_cli_args(cfg)

    assert args.speculative_config == {"method": "mtp", "num_speculative_tokens": 2}
