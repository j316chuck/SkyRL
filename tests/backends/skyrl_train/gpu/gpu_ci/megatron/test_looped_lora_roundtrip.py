"""Megatron/vLLM looped-LoRA parity on Hermes tool prompts across one update.

Run K=1, 2, and 4 separately on four H100/H200 GPUs:

LOOPED_LORA_K=1 uv run --isolated --extra dev --extra megatron pytest -s \
  tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_looped_lora_roundtrip.py
"""

import math
import os

import pytest
import ray
import torch
from transformers import AutoConfig, AutoTokenizer

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SamplingParams, SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    init_worker_with_type,
)

MODEL_NAME = os.environ.get("LOOPED_LORA_MODEL", "Qwen/Qwen3-4B-Thinking-2507")
REPEAT_COUNT = int(os.environ.get("LOOPED_LORA_K", "4"))
LOOPED_LORA_MODE = os.environ.get("LOOPED_LORA_MODE", "base_output_adapter")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the current status of a customer's order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]
CONVERSATIONS = [
    [
        {"role": "system", "content": "Help the user with retail orders. Use tools when needed."},
        {"role": "user", "content": "Where is order A-1042?"},
    ],
    [
        {"role": "system", "content": "Help the user with retail orders. Use tools when needed."},
        {"role": "user", "content": "Check whether order B-77 has shipped."},
    ],
    [
        {"role": "system", "content": "Help the user with retail orders. Use tools when needed."},
        {"role": "user", "content": "Please look up order C-9."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_order_status", "arguments": {"order_id": "C-9"}},
                }
            ],
        },
        {"role": "tool", "name": "get_order_status", "content": '{"status":"delivered"}'},
        {"role": "user", "content": "Summarize that result."},
    ],
]


def _build_hermes_prompt_ids(tokenizer) -> list[list[int]]:
    prompt_ids = []
    for conversation in CONVERSATIONS:
        prompt = tokenizer.apply_chat_template(
            conversation,
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert "<tools>" in prompt
        prompt_ids.append(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt_ids


def _get_config(num_hidden_layers: int) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.policy.language_model_only = True
    cfg.trainer.ref.language_model_only = True
    cfg.generator.inference_engine.language_model_only = True
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 4
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.policy.optimizer_config.lr = 1e-3

    lora = cfg.trainer.policy.model.lora
    lora.rank = 16
    lora.alpha = 32
    cfg.trainer.policy.megatron_config.lora_config.merge_lora = False

    middle = num_hidden_layers // 2
    cfg.trainer.policy.model.looped_lora.sections = [
        {
            "start_layer": middle - 4,
            "end_layer": middle + 4,
            "repeat_count": REPEAT_COUNT,
        }
    ]
    cfg.trainer.policy.model.looped_lora.mode = LOOPED_LORA_MODE
    cfg.generator.inference_engine.num_engines = 4
    cfg.generator.inference_engine.tensor_parallel_size = 1
    cfg.generator.inference_engine.distributed_executor_backend = "mp"
    cfg.generator.inference_engine.max_num_seqs = len(CONVERSATIONS)
    cfg.generator.inference_engine.max_num_batched_tokens = 2048
    cfg.generator.inference_engine.engine_init_kwargs = {"max_model_len": 2048}
    validate_cfg(cfg)
    return cfg


def _build_batch(tokenizer, prompt_ids, result) -> TrainingInputBatch:
    responses = result["response_ids"]
    loss_masks = [[1] * len(response) for response in responses]
    rewards = [[0.0] * len(response) for response in responses]
    sequences, attention_mask, response_mask, rewards_t, loss_mask_t, logprobs_t, _, _ = (
        convert_prompts_responses_to_batch_tensors(
            pad_token_id=tokenizer.pad_token_id,
            prompts=prompt_ids,
            responses=responses,
            rewards=rewards,
            loss_masks=loss_masks,
            logprobs=result["response_logprobs"],
        )
    )
    assert logprobs_t is not None
    num_actions = response_mask.shape[1]
    batch_size = sequences.shape[0]
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "rewards": rewards_t,
            "loss_mask": loss_mask_t,
            "rollout_logprobs": logprobs_t,
            "action_log_probs": torch.zeros((batch_size, num_actions)),
            "base_action_log_probs": torch.zeros((batch_size, num_actions)),
            "advantages": torch.zeros((batch_size, num_actions)),
        }
    )
    batch.metadata = {"response_length": num_actions}
    return batch


def _score(policy, batch: TrainingInputBatch) -> torch.Tensor:
    results = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
    output = WorkerOutput.cat(policy.actor_infos, results)
    return loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")


async def _generate(client, prompt_ids, cfg: SkyRLTrainConfig):
    sampling_params = get_sampling_params_for_backend(
        "vllm",
        SamplingParams(temperature=0.0, max_generate_length=12, logprobs=1),
    )
    return await client.generate(
        {"prompt_token_ids": prompt_ids, "sampling_params": sampling_params},
        model=resolve_policy_model_name(cfg),
    )


async def _sync(policy, client, cfg: SkyRLTrainConfig) -> None:
    policy.offload_to_cpu(offload_optimizer=True, offload_model=False)
    await client.wake_up(tags=["weights"])
    ray.get(
        policy.async_run_ray_method(
            "pass_through",
            "broadcast_to_inference_engines",
            client,
            cfg.generator.inference_engine,
        )
    )
    policy.offload_to_cpu(offload_optimizer=False, offload_model=True)
    await client.wake_up(tags=["kv_cache"])
    await client.reset_prefix_cache()


@pytest.mark.asyncio
@pytest.mark.h100
async def test_base_output_adapter_hermes_logprobs_roundtrip() -> None:
    assert REPEAT_COUNT in {1, 2, 4, 8}
    assert LOOPED_LORA_MODE == "base_output_adapter"
    hf_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    cfg = _get_config(hf_config.num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    prompt_ids = _build_hermes_prompt_ids(tokenizer)

    with ray_init():
        async with InferenceEngineState.create(
            cfg=cfg,
            model=MODEL_NAME,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=1,
            gpu_memory_utilization=0.7,
        ) as engines:
            client, placement_group = engines.client, engines.pg
            await client.sleep()
            policy = init_worker_with_type(
                "policy",
                shared_pg=placement_group,
                colocate_all=True,
                num_gpus_per_node=4,
                cfg=cfg,
            )
            ray.get(
                policy.async_run_ray_method(
                    "pass_through",
                    "init_weight_sync_state",
                    client,
                    cfg.generator.inference_engine,
                )
            )

            await _sync(policy, client, cfg)
            initial_vllm = await _generate(client, prompt_ids, cfg)
            initial_batch = _build_batch(tokenizer, prompt_ids, initial_vllm)
            await client.sleep()
            policy.backload_to_gpu(backload_optimizer=True, backload_model=True)
            initial_megatron = _score(policy, initial_batch)
            mask = initial_batch["response_mask"].bool()
            initial_vllm_logprobs = initial_batch["rollout_logprobs"]
            assert torch.isfinite(initial_vllm_logprobs[mask]).all()
            assert torch.isfinite(initial_megatron[mask]).all()
            initial_diff = (initial_vllm_logprobs[mask] - initial_megatron[mask]).abs().mean().item()
            assert math.isfinite(initial_diff)
            assert initial_diff < 0.05

            generator = torch.Generator().manual_seed(0)
            initial_batch["advantages"] = torch.randn(initial_batch["advantages"].shape, generator=generator)
            ray.get(policy.async_run_ray_method("mesh", "forward_backward", data=initial_batch))
            ray.get(policy.async_run_ray_method("pass_through", "optim_step"))
            updated_megatron = _score(policy, initial_batch)
            assert torch.isfinite(updated_megatron[mask]).all()
            model_movement = (updated_megatron[mask] - initial_megatron[mask]).abs().max().item()
            stale_diff = (initial_vllm_logprobs[mask] - updated_megatron[mask]).abs().mean().item()
            assert model_movement > 1e-4
            assert stale_diff > initial_diff

            policy.offload_to_cpu(offload_optimizer=True, offload_model=True)
            await client.wake_up(tags=["weights", "kv_cache"])
            stale_vllm = await _generate(client, prompt_ids, cfg)
            assert stale_vllm["response_ids"] == initial_vllm["response_ids"]
            stale_engine_delta = max(
                abs(stale - initial)
                for stale_row, initial_row in zip(stale_vllm["response_logprobs"], initial_vllm["response_logprobs"])
                for stale, initial in zip(stale_row, initial_row)
            )
            assert stale_engine_delta < 1e-6

            await client.sleep()
            policy.backload_to_gpu(backload_optimizer=True, backload_model=True)
            await _sync(policy, client, cfg)
            synced_vllm = await _generate(client, prompt_ids, cfg)
            synced_batch = _build_batch(tokenizer, prompt_ids, synced_vllm)
            await client.sleep()
            policy.backload_to_gpu(backload_optimizer=False, backload_model=True)
            synced_megatron = _score(policy, synced_batch)
            synced_mask = synced_batch["response_mask"].bool()
            assert torch.isfinite(synced_batch["rollout_logprobs"][synced_mask]).all()
            assert torch.isfinite(synced_megatron[synced_mask]).all()
            synced_diff = (
                (synced_batch["rollout_logprobs"][synced_mask] - synced_megatron[synced_mask]).abs().mean().item()
            )
            assert math.isfinite(synced_diff)

            print(
                f"K={REPEAT_COUNT}: initial={initial_diff:.6f}, stale={stale_diff:.6f}, "
                f"movement={model_movement:.6f}, synced={synced_diff:.6f}"
            )
            assert synced_diff < 0.05
            assert synced_diff < stale_diff
