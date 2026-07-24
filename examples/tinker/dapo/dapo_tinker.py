"""Matched five-step DAPO driver for hosted Tinker and the SkyRL Qwen3.6-27B recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq
import tinker
import torch
import wandb
from tinker import types
from torchdata.stateful_dataloader import StatefulDataLoader

MODEL_NAME = "Qwen/Qwen3.6-27B"
TRAIN_BATCH_SIZE = 128
POLICY_MINI_BATCH_SIZE = 32
N_SAMPLES_PER_PROMPT = 16
EVAL_N_SAMPLES_PER_PROMPT = 32
MAX_PROMPT_LENGTH = 2_048
MAX_RESPONSE_LENGTH = 8_192
TEMPERATURE = 1.0
TRAIN_TOP_P = 1.0
EVAL_TOP_P = 0.7
CLIP_RATIO_LOW = 0.2
CLIP_RATIO_HIGH = 0.28
CLIP_RATIO_C = 10.0
TIS_RATIO_CAP = 2.0
OVERLONG_BUFFER_LENGTH = 4_096
LEARNING_RATE = 1.0e-5
WARMUP_MINIBATCH_STEPS = 40
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
LORA_RANK = 32

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]
logger = logging.getLogger(__name__)


@dataclass
class PromptRecord:
    prompt_tokens: list[int]
    ground_truth: str
    dataset_index: int
    data_source: str


@dataclass
class Rollout:
    prompt_tokens: list[int]
    response_tokens: list[int]
    rollout_logprobs: list[float]
    loss_mask: list[float]
    reward: float
    advantage: float
    prompt_group: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url")
    parser.add_argument("--project", default="qwen3_6_dapo_lora")
    parser.add_argument("--run-name", default="qwen3_6_27b_dapo_tinker_5step")
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expression in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expression, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def aime_reward(solution: str, ground_truth: str) -> float:
    matches = re.findall(r"(?i)Answer\s*:\s*([^\n<]+)", solution[-300:])
    prediction = normalize_final_answer(matches[-1] if matches else "[INVALID]")
    return 1.0 if prediction == normalize_final_answer(ground_truth) else -1.0


def soft_overlong_reward(reward: float, response_length: int) -> float:
    ramp_start = MAX_RESPONSE_LENGTH - OVERLONG_BUFFER_LENGTH
    if response_length > MAX_RESPONSE_LENGTH:
        return 0.0
    if response_length > ramp_start:
        penalty = (response_length - ramp_start) / OVERLONG_BUFFER_LENGTH
        return reward - penalty
    return reward


def load_records(path: str, tokenizer: Any) -> list[PromptRecord]:
    rows = pq.read_table(path).to_pylist()
    records: list[PromptRecord] = []
    for dataset_index, row in enumerate(rows):
        prompt_tokens = list(
            tokenizer.apply_chat_template(
                row["prompt"],
                add_generation_prompt=True,
                return_dict=False,
                tokenize=True,
            )
        )
        if len(prompt_tokens) <= MAX_PROMPT_LENGTH:
            records.append(
                PromptRecord(
                    prompt_tokens=prompt_tokens,
                    ground_truth=str(row["reward_model"]["ground_truth"]),
                    dataset_index=dataset_index,
                    data_source=str(row["data_source"]),
                )
            )
    return records


def shuffled_indices(length: int, seed: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(length, generator=generator).tolist()


def training_batches(length: int, seed: int, steps: int) -> list[list[int]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = StatefulDataLoader(
        list(range(length)),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
        collate_fn=list,
    )
    batches: list[list[int]] = []
    while len(batches) < steps:
        for batch in dataloader:
            batches.append(batch)
            if len(batches) == steps:
                break
    return batches


def group_advantages(rewards: Sequence[float]) -> list[float]:
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    if len(rewards) == 1:
        return [0.0]
    normalized = (reward_tensor - reward_tensor.mean()) / (
        reward_tensor.std(unbiased=True) + 1.0e-6
    )
    return normalized.tolist()


def collect_rollouts(
    sampling_client: tinker.SamplingClient,
    records: Sequence[PromptRecord],
    tokenizer: Any,
    seed: int,
    global_step: int,
    n_samples: int,
    top_p: float,
) -> tuple[list[Rollout], dict[str, float]]:
    pending: list[tuple[PromptRecord, Any]] = []
    for offset, record in enumerate(records):
        future = sampling_client.sample(
            prompt=types.ModelInput.from_ints(record.prompt_tokens),
            num_samples=n_samples,
            sampling_params=types.SamplingParams(
                max_tokens=MAX_RESPONSE_LENGTH,
                seed=seed + global_step * TRAIN_BATCH_SIZE + offset,
                temperature=TEMPERATURE,
                top_p=top_p,
                top_k=-1,
            ),
        )
        pending.append((record, future))

    rollouts: list[Rollout] = []
    raw_rewards: list[float] = []
    shaped_rewards: list[float] = []
    pass_count = 0
    truncated_count = 0
    for prompt_group, (record, future) in enumerate(pending):
        result = future.result()
        group_rows: list[tuple[list[int], list[float], list[float], float]] = []
        for sequence in result.sequences:
            response_tokens = list(sequence.tokens)
            rollout_logprobs = list(sequence.logprobs or [])
            if not response_tokens or len(rollout_logprobs) != len(response_tokens):
                raise ValueError(
                    f"Invalid Tinker rollout lengths: tokens={len(response_tokens)}, "
                    f"logprobs={len(rollout_logprobs)}"
                )
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            raw_reward = aime_reward(response, record.ground_truth)
            reward = soft_overlong_reward(raw_reward, len(response_tokens))
            stopped = str(sequence.stop_reason).lower().endswith("stop")
            loss_mask = [1.0 if stopped else 0.0] * len(response_tokens)
            if not stopped:
                truncated_count += 1
            group_rows.append((response_tokens, rollout_logprobs, loss_mask, reward))
            raw_rewards.append(raw_reward)
            shaped_rewards.append(reward)
        if len(group_rows) != n_samples:
            raise ValueError(
                f"Expected {n_samples} samples, received {len(group_rows)}"
            )
        advantages = group_advantages([row[3] for row in group_rows])
        pass_count += int(any(row[3] > 0 for row in group_rows))
        for (response_tokens, rollout_logprobs, loss_mask, reward), advantage in zip(
            group_rows, advantages, strict=True
        ):
            rollouts.append(
                Rollout(
                    prompt_tokens=record.prompt_tokens,
                    response_tokens=response_tokens,
                    rollout_logprobs=rollout_logprobs,
                    loss_mask=loss_mask,
                    reward=reward,
                    advantage=advantage,
                    prompt_group=prompt_group,
                )
            )

    return rollouts, {
        "reward/aime_unshaped_reward": sum(raw_rewards) / len(raw_rewards),
        "reward/avg_raw_reward": sum(shaped_rewards) / len(shaped_rewards),
        f"reward/avg_pass_at_{n_samples}": pass_count / len(records),
        "rollout/num_trajectories": float(len(rollouts)),
        "rollout/truncated_fraction": truncated_count / len(rollouts),
    }


def to_datum(rollout: Rollout) -> types.Datum:
    tokens = rollout.prompt_tokens + rollout.response_tokens
    target_tokens = tokens[1:]
    weights = [0.0] * (len(rollout.prompt_tokens) - 1) + rollout.loss_mask
    assert len(tokens[:-1]) == len(target_tokens) == len(weights)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[len(target_tokens)],
            ),
            "weights": types.TensorData(
                data=weights,
                dtype="float32",
                shape=[len(weights)],
            ),
        },
    )


def dapo_loss(
    rollouts: Sequence[Rollout],
    logprobs_list: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = torch.stack([logprobs.sum() * 0.0 for logprobs in logprobs_list]).sum()
    total_tokens = sum(sum(rollout.loss_mask) for rollout in rollouts)
    if total_tokens == 0:
        return zero, {"policy/loss": 0.0, "policy/valid_tokens": 0.0}

    total_loss = zero
    diff_sum = 0.0
    diff_sq_sum = 0.0
    diff_max = 0.0
    diff_min = math.inf
    clipped_tokens = 0.0
    tis_clipped_tokens = 0.0
    for rollout, new_logprobs in zip(rollouts, logprobs_list, strict=True):
        response_logprobs = new_logprobs[len(rollout.prompt_tokens) - 1 :]
        assert len(response_logprobs) == len(rollout.response_tokens)
        rollout_logprobs = torch.tensor(
            rollout.rollout_logprobs,
            dtype=response_logprobs.dtype,
            device=response_logprobs.device,
        )
        loss_mask = torch.tensor(
            rollout.loss_mask,
            dtype=response_logprobs.dtype,
            device=response_logprobs.device,
        )
        advantages = (
            torch.full_like(response_logprobs, rollout.advantage)
            * loss_mask
            / max(total_tokens, 1.0)
        )
        old_logprobs = response_logprobs.detach()
        ratio = torch.exp(torch.clamp(response_logprobs - old_logprobs, -20.0, 20.0))
        clipped_ratio = ratio.clamp(1.0 - CLIP_RATIO_LOW, 1.0 + CLIP_RATIO_HIGH)
        surrogate = -torch.min(ratio * advantages, clipped_ratio * advantages)
        dual_clip = torch.minimum(-advantages * CLIP_RATIO_C, surrogate)
        token_loss = torch.where(advantages < 0, dual_clip, surrogate)
        tis_ratio = torch.exp(torch.clamp(old_logprobs - rollout_logprobs, -20.0, 20.0))
        tis_clipped_tokens += float(
            ((tis_ratio > TIS_RATIO_CAP) * loss_mask).sum().item()
        )
        token_loss = token_loss * tis_ratio.clamp(max=TIS_RATIO_CAP).detach()
        total_loss = total_loss + (token_loss * loss_mask).sum()

        clipped_tokens += float(((ratio != clipped_ratio) * loss_mask).sum().item())
        valid_diff = (old_logprobs - rollout_logprobs).abs()[loss_mask.bool()]
        if valid_diff.numel():
            diff_sum += float(valid_diff.sum().item())
            diff_sq_sum += float(valid_diff.square().sum().item())
            diff_max = max(diff_max, float(valid_diff.max().item()))
            diff_min = min(diff_min, float(valid_diff.min().item()))

    diff_mean = diff_sum / total_tokens
    diff_variance = (
        max(
            (diff_sq_sum - total_tokens * diff_mean * diff_mean) / (total_tokens - 1.0),
            0.0,
        )
        if total_tokens > 1
        else 0.0
    )
    metrics = {
        "policy/loss": float(total_loss.detach().item()),
        "policy/clip_ratio": clipped_tokens / total_tokens,
        "policy/tis_token_clip_high_ratio": tis_clipped_tokens / total_tokens,
        "policy/valid_tokens": float(total_tokens),
        "policy/rollout_train_logprobs_abs_diff_mean": diff_mean,
        "policy/rollout_train_logprobs_abs_diff_std": math.sqrt(diff_variance),
        "policy/rollout_train_logprobs_abs_diff_max": diff_max,
        "policy/rollout_train_logprobs_abs_diff_min": diff_min,
    }
    return total_loss, metrics


def train_step(
    training_client: tinker.TrainingClient,
    rollouts: Sequence[Rollout],
    optimizer_step: int,
) -> tuple[dict[str, float], int]:
    minibatch_metrics: list[dict[str, float]] = []
    trainable_trajectories = 0
    sequences_per_minibatch = POLICY_MINI_BATCH_SIZE * N_SAMPLES_PER_PROMPT
    for minibatch in chunked(list(rollouts), sequences_per_minibatch):
        minibatch = [rollout for rollout in minibatch if any(rollout.loss_mask)]
        if not minibatch:
            continue
        trainable_trajectories += len(minibatch)
        data = [to_datum(rollout) for rollout in minibatch]

        def loss_fn(
            _data: list[types.Datum],
            logprobs_list: list[torch.Tensor],
            current_minibatch: Sequence[Rollout] = minibatch,
        ) -> tuple[torch.Tensor, dict[str, float]]:
            return dapo_loss(current_minibatch, logprobs_list)

        forward_backward = training_client.forward_backward_custom(
            data, loss_fn=loss_fn
        ).result()
        learning_rate = LEARNING_RATE * min(
            optimizer_step / WARMUP_MINIBATCH_STEPS, 1.0
        )
        optimizer = training_client.optim_step(
            types.AdamParams(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.999,
                eps=1.0e-8,
                weight_decay=WEIGHT_DECAY,
                grad_clip_norm=MAX_GRAD_NORM,
            )
        ).result()
        metrics = dict(forward_backward.metrics or {})
        metrics.update(optimizer.metrics or {})
        metrics["policy/lr"] = learning_rate
        minibatch_metrics.append({key: float(value) for key, value in metrics.items()})
        optimizer_step += 1

    averaged: dict[str, float] = {}
    for key in {key for metrics in minibatch_metrics for key in metrics}:
        values = [metrics[key] for metrics in minibatch_metrics if key in metrics]
        averaged[key] = sum(values) / len(values)
    averaged["policy/trainable_trajectories"] = float(trainable_trajectories)
    averaged["policy/trainable_trajectory_fraction"] = trainable_trajectories / len(
        rollouts
    )
    return averaged, optimizer_step


class WandbLogger:
    def __init__(self, args: argparse.Namespace, config: dict[str, Any]) -> None:
        self.run = None
        self.wandb = wandb
        self.run = wandb.init(
            project=args.project,
            name=args.run_name,
            dir=args.output_dir,
            config=config,
            tags=["qwen3.6-27b", "dapo", "tinker", "parity"],
        )
        self.run.define_metric("step")
        self.run.define_metric("*", step_metric="step")

    def log(self, metrics: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(metrics)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def append_metrics(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")


def experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "policy_mini_batch_size": POLICY_MINI_BATCH_SIZE,
        "n_samples_per_prompt": N_SAMPLES_PER_PROMPT,
        "eval_n_samples_per_prompt": EVAL_N_SAMPLES_PER_PROMPT,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_response_length": MAX_RESPONSE_LENGTH,
        "temperature": TEMPERATURE,
        "train_top_p": TRAIN_TOP_P,
        "eval_top_p": EVAL_TOP_P,
        "clip_ratio_low": CLIP_RATIO_LOW,
        "clip_ratio_high": CLIP_RATIO_HIGH,
        "clip_ratio_c": CLIP_RATIO_C,
        "tis_ratio_cap": TIS_RATIO_CAP,
        "overlong_buffer_length": OVERLONG_BUFFER_LENGTH,
        "learning_rate": LEARNING_RATE,
        "warmup_minibatch_steps": WARMUP_MINIBATCH_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "lora_rank": LORA_RANK,
        "seed": args.seed,
        "steps": args.steps,
        "eval_interval": args.eval_interval,
        "train_file_sha256": sha256(args.train_file),
        "eval_file_sha256": {path: sha256(path) for path in args.eval_file},
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    config = experiment_config(args)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    wandb_logger = WandbLogger(args, config)
    kwargs: dict[str, Any] = {
        "user_metadata": {
            "experiment": args.run_name,
            "frontend": "matched-dapo-driver",
        }
    }
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY must be set")
    service_client = tinker.ServiceClient(**kwargs)
    try:
        training_client = service_client.create_lora_training_client(
            base_model=MODEL_NAME,
            rank=LORA_RANK,
            seed=args.seed,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
            user_metadata={"experiment": args.run_name},
        )
        tokenizer = training_client.get_tokenizer()
        train_records = load_records(args.train_file, tokenizer)
        eval_records_by_source: dict[str, list[PromptRecord]] = {}
        for eval_file in args.eval_file:
            for record in load_records(eval_file, tokenizer):
                eval_records_by_source.setdefault(record.data_source, []).append(record)
        batches = training_batches(len(train_records), args.seed, args.steps)
        selected_indices = [
            train_records[index].dataset_index for batch in batches for index in batch
        ]
        (output_dir / "train_indices.json").write_text(
            json.dumps(selected_indices, indent=2) + "\n"
        )
        logger.info(
            "Loaded %s train and %s eval records; selected first %s shuffled prompts",
            len(train_records),
            sum(len(records) for records in eval_records_by_source.values()),
            len(selected_indices),
        )

        optimizer_step = 0
        run_start = time.monotonic()
        for global_step in range(args.steps):
            step_start = time.monotonic()
            sampling_client = training_client.save_weights_and_get_sampling_client()
            batch = [train_records[index] for index in batches[global_step]]
            rollout_start = time.monotonic()
            rollouts, rollout_metrics = collect_rollouts(
                sampling_client,
                batch,
                tokenizer,
                seed=args.seed,
                global_step=global_step,
                n_samples=N_SAMPLES_PER_PROMPT,
                top_p=TRAIN_TOP_P,
            )
            rollout_seconds = time.monotonic() - rollout_start
            train_start = time.monotonic()
            update_metrics, optimizer_step = train_step(
                training_client, rollouts, optimizer_step
            )
            train_seconds = time.monotonic() - train_start
            payload: dict[str, Any] = {
                "step": global_step + 1,
                "time/step_seconds": time.monotonic() - step_start,
                "time/elapsed_seconds": time.monotonic() - run_start,
                "time/rollout_seconds": rollout_seconds,
                "time/train_seconds": train_seconds,
                **rollout_metrics,
                **update_metrics,
            }
            if any(
                not math.isfinite(value)
                for value in payload.values()
                if isinstance(value, float)
            ):
                raise FloatingPointError(
                    f"Non-finite metric at step {global_step + 1}: {payload}"
                )
            append_metrics(metrics_path, payload)
            wandb_logger.log(payload)
            logger.info("Step %s metrics: %s", global_step + 1, payload)

            if not args.skip_eval and (global_step + 1) % args.eval_interval == 0:
                eval_start = time.monotonic()
                eval_payload: dict[str, Any] = {"step": global_step + 1}
                sampling_client = training_client.save_weights_and_get_sampling_client()
                for data_source, eval_records in eval_records_by_source.items():
                    eval_rollouts, eval_metrics = collect_rollouts(
                        sampling_client,
                        eval_records,
                        tokenizer,
                        seed=args.seed,
                        global_step=global_step + 1,
                        n_samples=EVAL_N_SAMPLES_PER_PROMPT,
                        top_p=EVAL_TOP_P,
                    )
                    del eval_rollouts
                    eval_payload[f"eval/{data_source}/avg_score"] = eval_metrics[
                        "reward/aime_unshaped_reward"
                    ]
                    eval_payload[
                        f"eval/{data_source}/pass_at_{EVAL_N_SAMPLES_PER_PROMPT}"
                    ] = eval_metrics[f"reward/avg_pass_at_{EVAL_N_SAMPLES_PER_PROMPT}"]
                eval_payload["time/eval_seconds"] = time.monotonic() - eval_start
                append_metrics(metrics_path, eval_payload)
                wandb_logger.log(eval_payload)
    finally:
        service_client.holder.close()
        wandb_logger.finish()


def self_test() -> None:
    assert normalize_final_answer(r"$\boxed{042}$") == "042"
    assert aime_reward("Answer: 42", "42") == 1.0
    assert aime_reward("Answer: 41", "42") == -1.0
    advantages = group_advantages([-1.0, 1.0])
    assert math.isclose(sum(advantages), 0.0, abs_tol=1.0e-6)
    assert soft_overlong_reward(1.0, 4_096) == 1.0
    assert soft_overlong_reward(1.0, 8_192) == 0.0
    print("self-test passed")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.self_test:
        self_test()
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
