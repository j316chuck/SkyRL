"""Run the GLM-5.3-Flash trainer-to-vLLM LoRA publication check."""

import hashlib
import math
from typing import Any

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.utils.utils import validate_cfg
from skyrl.utils.tok import get_tokenizer
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL = "zai-org/GLM-5.3-Flash"
MEAN_ATOL = 0.1


class Glm5NextLoraWorker(MegatronPolicyWorkerBase):
    @torch.no_grad()
    def perturb_lora(self):
        changed = 0
        for chunk_index, chunk in enumerate(self.actor_module):
            for name, parameter in chunk.named_parameters():
                if not parameter.requires_grad or name.endswith(".linear_in.weight"):
                    continue
                assert "adapter" in name and name.endswith(".linear_out.weight")
                assert torch.count_nonzero(parameter) == 0
                seed_name = f"chunk{chunk_index}.{name}"
                seed = int.from_bytes(
                    hashlib.sha256(seed_name.encode()).digest()[:8], "little"
                )
                generator = torch.Generator(device=parameter.device).manual_seed(
                    seed % (2**63)
                )
                parameter.add_(
                    torch.randn(
                        parameter.shape,
                        generator=generator,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    ),
                    alpha=1e-2,
                )
                changed += parameter.numel()
        assert changed > 0


def _build_config():
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.critic.model.path = None
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(rank=32, alpha=32)
    cfg.trainer.policy.model.lora.target_modules = ["linear_proj"]  # type: ignore
    cfg.trainer.policy.language_model_only = True
    cfg.trainer.ref.language_model_only = True
    cfg.trainer.remove_microbatch_padding = True
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 8

    megatron = cfg.trainer.policy.megatron_config
    megatron.tensor_model_parallel_size = 2
    megatron.pipeline_model_parallel_size = 1
    megatron.context_parallel_size = 1
    megatron.expert_model_parallel_size = 8
    megatron.expert_tensor_parallel_size = 1
    megatron.lora_config.merge_lora = False

    inference = cfg.generator.inference_engine
    inference.run_engines_locally = True
    inference.distributed_executor_backend = "ray"
    inference.tensor_parallel_size = 8
    inference.num_engines = 1
    inference.max_num_seqs = 512
    inference.gpu_memory_utilization = 0.8
    inference.engine_init_kwargs = {"max_model_len": 4096}
    inference.language_model_only = True
    validate_cfg(cfg)
    return cfg


def _build_batch(sequences, pad_token_id):
    responses = [tokens[1:] for tokens in sequences]
    rewards_by_token: list[list[float] | torch.Tensor] = [
        [1.0] * len(tokens) for tokens in responses
    ]
    loss_masks = [[1] * len(tokens) for tokens in responses]
    tokens, attention, response, rewards, loss_mask, _, _ = (
        convert_prompts_responses_to_batch_tensors(
            pad_token_id,
            [[tokens[0]] for tokens in sequences],
            responses,
            rewards_by_token,
            loss_masks,
        )
    )
    batch = TrainingInputBatch(
        {
            "sequences": tokens,
            "attention_mask": attention,
            "response_mask": response,
            "rewards": rewards,
            "loss_mask": loss_mask,
            "rollout_expert_indices": None,
            "rollout_logprobs": torch.zeros_like(loss_mask),
            "action_log_probs": torch.zeros_like(loss_mask),
            "base_action_log_probs": torch.zeros_like(loss_mask),
            "advantages": torch.zeros_like(loss_mask),
        }
    )
    batch.metadata = {"response_length": response.shape[1]}
    return batch


def _score_trainer(policy, batch):
    results = ray.get(
        policy.async_run_ray_method(
            "mesh", "forward", data=batch, loss_fn="cross_entropy"
        )
    )
    output = WorkerOutput.cat(policy.actor_infos, results)
    scores = [score for row in output.loss_fn_outputs for score in row["logprobs"]]
    assert scores and all(map(math.isfinite, scores))
    return torch.tensor(scores, dtype=torch.float64)


async def _score_sampler(client, sequences, model):
    await client.reset_prefix_cache()
    scores = []
    for tokens in sequences:
        result = await client.sample(
            {
                "json": {
                    "model": model,
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
                    "sampling_params": {"max_tokens": 1, "temperature": 1.0},
                    "num_samples": 1,
                    "prompt_logprobs": True,
                }
            }
        )
        values = result["prompt_logprobs"]
        assert values is not None and len(values) == len(tokens)
        assert values[0] is None and all(value is not None for value in values[1:])
        scores.extend(values[1:])
    assert scores and all(map(math.isfinite, scores))
    return torch.tensor(scores, dtype=torch.float64)


async def _publish(policy, client, cfg):
    await client.pause_generation()
    try:
        ray.get(
            policy.async_run_ray_method(
                "pass_through",
                "broadcast_to_inference_engines",
                client,
                cfg.generator.inference_engine,
            )
        )
    finally:
        await client.resume_generation()


@pytest.mark.asyncio
@pytest.mark.b300
@pytest.mark.megatron_models
async def test_glm5_next_lora_publication_changes_sampler_scores():
    cfg = _build_config()
    tokenizer: Any = get_tokenizer(MODEL)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    sequences = [
        tokenizer.encode(
            "A river flows beneath a bridge. " * 8, add_special_tokens=False
        ),
        tokenizer.encode(
            "Calculate seven times eight. " * 16, add_special_tokens=False
        ),
        tokenizer.encode(
            "The quick brown fox jumps over the lazy dog. " * 12,
            add_special_tokens=False,
        ),
        tokenizer.encode(
            "List the first five prime numbers. " * 20, add_special_tokens=False
        ),
    ]
    batch = _build_batch(sequences, pad_token_id)

    with ray_init():
        async with InferenceEngineState.create(
            cfg=cfg,
            model=MODEL,
            use_local=True,
            tp_size=8,
            colocate_all=False,
            backend="vllm",
            enable_lora=True,
            gpu_memory_utilization=0.8,
            max_num_seqs=512,
            engine_init_kwargs={"max_model_len": 4096},
            language_model_only=True,
        ) as engines:
            policy = PPORayActorGroup(
                cfg.trainer,
                num_nodes=1,
                num_gpus_per_node=8,
                ray_actor_type=ray.remote(Glm5NextLoraWorker),  # type: ignore
                num_gpus_per_actor=0.75,
                colocate_all=False,
                sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
                record_memory=cfg.trainer.policy.record_memory,
            )
            ray.get(policy.async_init_model(MODEL))
            ray.get(
                policy.async_run_ray_method(
                    "pass_through",
                    "init_weight_sync_state",
                    engines.client,
                    cfg.generator.inference_engine,
                )
            )

            await _publish(policy, engines.client, cfg)
            adapter = resolve_policy_model_name(cfg)
            trainer_zero = _score_trainer(policy, batch)
            sampler_zero = await _score_sampler(engines.client, sequences, adapter)
            assert (trainer_zero - sampler_zero).abs().mean().item() < MEAN_ATOL

            ray.get(policy.async_run_ray_method("pass_through", "perturb_lora"))
            trainer_updated = _score_trainer(policy, batch)
            assert (trainer_updated - trainer_zero).abs().mean().item() >= MEAN_ATOL

            await _publish(policy, engines.client, cfg)
            sampler_updated = await _score_sampler(engines.client, sequences, adapter)
            assert (trainer_updated - sampler_updated).abs().mean().item() < MEAN_ATOL
            assert (sampler_updated - sampler_zero).abs().mean().item() >= MEAN_ATOL
