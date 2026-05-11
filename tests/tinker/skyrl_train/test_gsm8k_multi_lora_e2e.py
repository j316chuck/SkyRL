"""Opt-in GSM8K-style multi-LoRA TCLI e2e timing test."""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pytest

tinker = pytest.importorskip("tinker")
from tinker import types as tinker_types  # noqa: E402

RUN_ENV = "SKYRL_RUN_GSM8K_MULTI_LORA_E2E"
BASE_URL_ENV = "SKYRL_TCLI_BASE_URL"

NUM_ADAPTERS = 4
DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_API_KEY = "tml-dummy"
DEFAULT_LORA_RANK = 8
DEFAULT_TOTAL_STEPS = 10
DEFAULT_EVAL_INTERVAL = 5
DEFAULT_MIN_ACCURACY = 0.9
DEFAULT_MAX_JOB_SECONDS = 3600
GSM8K_LR = 2e-2


@dataclass(frozen=True)
class Gsm8kExample:
    question: str
    answer: str


@dataclass(frozen=True)
class Gsm8kJob:
    label: str
    seed: int


@dataclass(frozen=True)
class Gsm8kJobResult:
    label: str
    model_id: str
    wall_clock_s: float
    final_accuracy: float
    metrics: list[dict[str, float]]


@dataclass(frozen=True)
class TimingResult:
    wall_clock_s: float
    job_results: list[Gsm8kJobResult]


GSM8K_EXAMPLES = [
    Gsm8kExample(
        "Mia has 1 marble and buys 1 more. How many marbles does Mia have?", "2"
    ),
    Gsm8kExample(
        "A box has 3 red balls and 4 blue balls. How many balls are in the box?", "7"
    ),
    Gsm8kExample(
        "Tom reads 5 pages on Monday and 6 pages on Tuesday. How many pages did he read?",
        "11",
    ),
    Gsm8kExample("There are 12 cookies. Sam eats 5. How many cookies are left?", "7"),
]

GSM8K_JOBS = [
    Gsm8kJob(label="adapter_0", seed=200),
    Gsm8kJob(label="adapter_1", seed=201),
    Gsm8kJob(label="adapter_2", seed=202),
    Gsm8kJob(label="adapter_3", seed=203),
]


def _env_enabled(name: str) -> bool:
    return os.environ.get(name) == "1"


def _base_url() -> str:
    base_url = os.environ.get(BASE_URL_ENV) or os.environ.get("TCLI_BASE_URL")
    if not base_url:
        pytest.skip(
            f"set {BASE_URL_ENV} or TCLI_BASE_URL to one running Tinker endpoint"
        )
    return base_url.rstrip("/") + "/"


def _api_key() -> str:
    return os.environ.get("TINKER_API_KEY", DEFAULT_API_KEY)


def _model_name() -> str:
    return os.environ.get("SKYRL_TCLI_MULTI_LORA_MODEL", DEFAULT_MODEL)


def _lora_rank() -> int:
    return int(os.environ.get("SKYRL_TCLI_MULTI_LORA_RANK", str(DEFAULT_LORA_RANK)))


def _total_steps() -> int:
    return int(os.environ.get("SKYRL_GSM8K_MULTI_LORA_STEPS", str(DEFAULT_TOTAL_STEPS)))


def _eval_interval() -> int:
    return int(
        os.environ.get(
            "SKYRL_GSM8K_MULTI_LORA_EVAL_INTERVAL", str(DEFAULT_EVAL_INTERVAL)
        )
    )


def _min_accuracy() -> float:
    return float(
        os.environ.get("SKYRL_GSM8K_MULTI_LORA_MIN_ACCURACY", str(DEFAULT_MIN_ACCURACY))
    )


def _prompt(question: str) -> str:
    return (
        "Solve the grade-school math problem. Show brief work, then finish with a final "
        'answer line exactly like "#### 7".\n\n'
        f"Question: {question}\nAnswer:"
    )


def _completion(answer: str) -> str:
    return f" The answer is {answer}.\n#### {answer}\n"


def _make_weighted_sft_datum(
    tokenizer, example: Gsm8kExample, answer: str | None = None
) -> tuple[tinker_types.Datum, list[float]]:
    answer = answer or example.answer
    prompt_tokens = tokenizer.encode(_prompt(example.question), add_special_tokens=True)
    completion_tokens = tokenizer.encode(_completion(answer), add_special_tokens=False)
    all_tokens = prompt_tokens + completion_tokens
    target_tokens = all_tokens[1:] + [tokenizer.eos_token_id]
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    shifted_weights = weights[1:] + [1.0]
    datum = tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(all_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": shifted_weights},
    )
    return datum, shifted_weights


def _make_sft_datum(tokenizer, example: Gsm8kExample) -> tinker_types.Datum:
    return _make_weighted_sft_datum(tokenizer, example)[0]


def _wrong_answer(answer: str) -> str:
    return str(int(answer) + 1)


def _weighted_nll(output, weights: list[float]) -> float:
    logprobs = output.loss_fn_outputs[0]["logprobs"].data
    return float(sum(-logprob * weight for logprob, weight in zip(logprobs, weights)))


async def _unload_model(base_url: str, model_id: str) -> None:
    async with tinker._client.AsyncTinker(
        api_key=_api_key(), base_url=base_url
    ) as client:  # type: ignore[attr-defined]
        future = await client.models.unload(
            request=tinker_types.UnloadModelRequest(model_id=model_id)
        )
        while True:
            result = await client.futures.retrieve(
                request=tinker_types.FutureRetrieveRequest(request_id=future.request_id)
            )
            if isinstance(result, tinker_types.RequestFailedResponse):
                raise RuntimeError(result.error)
            if isinstance(result, tinker_types.UnloadModelResponse):
                return
            await asyncio.sleep(0.2)


def _evaluate_accuracy(
    training_client, tokenizer, examples: list[Gsm8kExample], job: Gsm8kJob
) -> float:
    correct = 0
    for example in examples:
        good_datum, good_weights = _make_weighted_sft_datum(tokenizer, example)
        bad_datum, bad_weights = _make_weighted_sft_datum(
            tokenizer, example, answer=_wrong_answer(example.answer)
        )
        good_loss = _weighted_nll(
            training_client.forward([good_datum], "cross_entropy").result(),
            good_weights,
        )
        bad_loss = _weighted_nll(
            training_client.forward([bad_datum], "cross_entropy").result(), bad_weights
        )
        correct += int(good_loss < bad_loss)
    return correct / len(examples)


def _run_gsm8k_job(base_url: str, job: Gsm8kJob, *, unload: bool) -> Gsm8kJobResult:
    started_at = time.perf_counter()
    deadline = started_at + DEFAULT_MAX_JOB_SECONDS
    service_client = tinker.ServiceClient(base_url=base_url, api_key=_api_key())
    model_id = ""
    try:
        training_client = service_client.create_lora_training_client(
            base_model=_model_name(),
            rank=_lora_rank(),
            seed=job.seed,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
            user_metadata={"test": "gsm8k_multi_lora_e2e", "adapter": job.label},
        )
        model_id = training_client.model_id
        tokenizer = training_client.get_tokenizer()
        data = [
            _make_sft_datum(tokenizer, example)
            for example in GSM8K_EXAMPLES
            for _ in range(4)
        ]

        metrics: list[dict[str, float]] = []
        for step in range(1, _total_steps() + 1):
            if time.perf_counter() > deadline:
                raise TimeoutError(f"{job.label} exceeded {DEFAULT_MAX_JOB_SECONDS}s")
            training_client.forward_backward(data, "cross_entropy").result()
            training_client.optim_step(
                tinker_types.AdamParams(learning_rate=GSM8K_LR)
            ).result()
            if step % _eval_interval() == 0:
                accuracy = _evaluate_accuracy(
                    training_client, tokenizer, GSM8K_EXAMPLES, job
                )
                metrics.append(
                    {"step": float(step), "train/accuracy_unfiltered": accuracy}
                )

        final_accuracy = metrics[-1]["train/accuracy_unfiltered"]
        return Gsm8kJobResult(
            label=job.label,
            model_id=model_id,
            wall_clock_s=time.perf_counter() - started_at,
            final_accuracy=final_accuracy,
            metrics=metrics,
        )
    finally:
        if unload and model_id:
            asyncio.run(_unload_model(base_url, model_id))
        service_client.holder.close()


def _run_serial(base_url: str) -> TimingResult:
    started_at = time.perf_counter()
    results: list[Gsm8kJobResult] = []
    try:
        for job in GSM8K_JOBS:
            results.append(_run_gsm8k_job(base_url, job, unload=False))
    finally:
        for result in results:
            asyncio.run(_unload_model(base_url, result.model_id))
    return TimingResult(time.perf_counter() - started_at, results)


def _run_concurrent(base_url: str) -> TimingResult:
    started_at = time.perf_counter()
    results: list[Gsm8kJobResult] = []
    with ThreadPoolExecutor(max_workers=NUM_ADAPTERS) as executor:
        futures = [
            executor.submit(_run_gsm8k_job, base_url, job, unload=False)
            for job in GSM8K_JOBS
        ]
        for future in as_completed(futures, timeout=DEFAULT_MAX_JOB_SECONDS + 300):
            results.append(future.result())

    for result in results:
        asyncio.run(_unload_model(base_url, result.model_id))
    return TimingResult(
        time.perf_counter() - started_at, sorted(results, key=lambda r: r.label)
    )


def _print_timing(label: str, result: TimingResult) -> None:
    print(f"\n[gsm8k_multi_lora] {label}_wall_clock_s={result.wall_clock_s:.3f}")
    for job in result.job_results:
        print(
            f"[gsm8k_multi_lora] {label} {job.label} model_id={job.model_id} "
            f"wall_clock_s={job.wall_clock_s:.3f} final_accuracy={job.final_accuracy:.3f} "
            f"metrics={job.metrics}"
        )


def _assert_gsm8k_results(results: list[Gsm8kJobResult]) -> None:
    assert len(results) == NUM_ADAPTERS
    assert len({result.model_id for result in results}) == len(results)
    for result in results:
        assert result.final_accuracy > _min_accuracy(), (
            f"{result.label} final train/accuracy_unfiltered={result.final_accuracy:.3f} "
            f"did not exceed {_min_accuracy():.3f}"
        )


@pytest.mark.integrations
@pytest.mark.slow
@pytest.mark.skipif(not _env_enabled(RUN_ENV), reason=f"set {RUN_ENV}=1")
def test_gsm8k_multi_lora_sequential_vs_concurrent() -> None:
    base_url = _base_url()
    sequential = _run_serial(base_url)
    concurrent = _run_concurrent(base_url)
    speedup = sequential.wall_clock_s / concurrent.wall_clock_s

    _print_timing("sequential", sequential)
    _print_timing("concurrent", concurrent)
    print(f"\n[gsm8k_multi_lora] speedup_ratio={speedup:.3f}")

    _assert_gsm8k_results(sequential.job_results)
    _assert_gsm8k_results(concurrent.job_results)
