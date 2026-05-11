"""Opt-in multi-LoRA TCLI timing tests against one shared Tinker endpoint."""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pytest

tinker = pytest.importorskip("tinker")
from tinker import types as tinker_types  # noqa: E402

RUN_FAST_ENV = "SKYRL_RUN_TCLI_MULTI_LORA_INTEGRATION"
RUN_MEDIUM_ENV = "SKYRL_RUN_TCLI_MULTI_LORA_MEDIUM_E2E"
BASE_URL_ENV = "SKYRL_TCLI_BASE_URL"

NUM_ADAPTERS = 4
DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_API_KEY = "tml-dummy"
DEFAULT_LORA_RANK = 8
DEFAULT_MAX_JOB_SECONDS = 600
PIG_LATIN_LR = 5e-2


@dataclass(frozen=True)
class PigLatinJob:
    label: str
    word: str
    expected: str
    seed: int


@dataclass(frozen=True)
class AdapterRunResult:
    label: str
    model_id: str
    wall_clock_s: float
    pre_loss: float
    post_loss: float
    sample_text: str


@dataclass(frozen=True)
class TimingResult:
    wall_clock_s: float
    adapter_results: list[AdapterRunResult]


PIG_LATIN_JOBS = [
    PigLatinJob(label="adapter_0", word="cat", expected="atcay", seed=100),
    PigLatinJob(label="adapter_1", word="dog", expected="ogday", seed=101),
    PigLatinJob(label="adapter_2", word="fish", expected="ishfay", seed=102),
    PigLatinJob(label="adapter_3", word="bird", expected="irdbay", seed=103),
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


def _make_sft_datum(tokenizer, prompt: str, completion: str) -> tinker_types.Datum:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    all_tokens = prompt_tokens + completion_tokens
    target_tokens = all_tokens[1:] + [tokenizer.eos_token_id]
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(all_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights[1:] + [1.0]},
    )


def _loss_sum(output) -> float:
    return float(
        sum(sum(item["elementwise_loss"].data) for item in output.loss_fn_outputs)
    )


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


def _run_pig_latin_job(
    base_url: str,
    job: PigLatinJob,
    *,
    train_steps: int,
    unload: bool,
) -> AdapterRunResult:
    started_at = time.perf_counter()
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
            user_metadata={"test": "multi_lora_vs_sequential", "adapter": job.label},
        )
        model_id = training_client.model_id
        tokenizer = training_client.get_tokenizer()
        prompt = (
            f"Translate the word {job.word} to Pig Latin. "
            "Answer with only the translated word.\nAnswer:"
        )
        data = [
            _make_sft_datum(tokenizer, prompt, f" {job.expected}\n") for _ in range(8)
        ]

        pre_loss = _loss_sum(
            training_client.forward_backward(data, "cross_entropy").result()
        )
        training_client.optim_step(
            tinker_types.AdamParams(learning_rate=PIG_LATIN_LR)
        ).result()
        for _ in range(train_steps - 1):
            training_client.forward_backward(data, "cross_entropy").result()
            training_client.optim_step(
                tinker_types.AdamParams(learning_rate=PIG_LATIN_LR)
            ).result()
        post_loss = _loss_sum(
            training_client.forward_backward(data, "cross_entropy").result()
        )

        sampler = training_client.save_weights_and_get_sampling_client(
            name=f"sample_{job.label}"
        )
        sample = sampler.sample(
            prompt=tinker_types.ModelInput.from_ints(
                tokenizer.encode(prompt, add_special_tokens=True)
            ),
            num_samples=1,
            sampling_params=tinker_types.SamplingParams(
                max_tokens=8,
                temperature=0.0,
                top_k=1,
                seed=job.seed,
            ),
        ).result()
        sample_text = tokenizer.decode(
            sample.sequences[0].tokens, skip_special_tokens=True
        ).strip()
        return AdapterRunResult(
            label=job.label,
            model_id=model_id,
            wall_clock_s=time.perf_counter() - started_at,
            pre_loss=pre_loss,
            post_loss=post_loss,
            sample_text=sample_text,
        )
    finally:
        if unload and model_id:
            asyncio.run(_unload_model(base_url, model_id))
        service_client.holder.close()


def _run_serial(base_url: str, train_steps: int) -> TimingResult:
    started_at = time.perf_counter()
    results: list[AdapterRunResult] = []
    try:
        for job in PIG_LATIN_JOBS:
            results.append(
                _run_pig_latin_job(base_url, job, train_steps=train_steps, unload=False)
            )
    finally:
        for result in results:
            asyncio.run(_unload_model(base_url, result.model_id))
    return TimingResult(time.perf_counter() - started_at, results)


def _run_concurrent(base_url: str, train_steps: int) -> TimingResult:
    started_at = time.perf_counter()
    results: list[AdapterRunResult] = []
    with ThreadPoolExecutor(max_workers=NUM_ADAPTERS) as executor:
        futures = [
            executor.submit(
                _run_pig_latin_job, base_url, job, train_steps=train_steps, unload=False
            )
            for job in PIG_LATIN_JOBS
        ]
        for future in as_completed(futures, timeout=DEFAULT_MAX_JOB_SECONDS):
            results.append(future.result())

    for result in results:
        asyncio.run(_unload_model(base_url, result.model_id))
    return TimingResult(
        time.perf_counter() - started_at, sorted(results, key=lambda r: r.label)
    )


def _print_timing(label: str, result: TimingResult) -> None:
    print(f"\n[tcli_multi_lora] {label}_wall_clock_s={result.wall_clock_s:.3f}")
    for adapter in result.adapter_results:
        print(
            f"[tcli_multi_lora] {label} {adapter.label} "
            f"model_id={adapter.model_id} wall_clock_s={adapter.wall_clock_s:.3f} "
            f"pre_loss={adapter.pre_loss:.6f} post_loss={adapter.post_loss:.6f} "
            f"sample={adapter.sample_text!r}"
        )


def _assert_adapter_results(results: list[AdapterRunResult]) -> None:
    assert len(results) == NUM_ADAPTERS
    assert len({result.model_id for result in results}) == len(results)
    for result in results:
        assert result.post_loss <= result.pre_loss + 1e-3, (
            f"{result.label} loss did not decrease: pre={result.pre_loss}, post={result.post_loss}"
        )
        assert result.sample_text, f"{result.label} returned an empty sample"


@pytest.mark.integrations
@pytest.mark.skipif(not _env_enabled(RUN_FAST_ENV), reason=f"set {RUN_FAST_ENV}=1")
def test_tcli_multi_lora_pig_latin_sequential_vs_concurrent() -> None:
    base_url = _base_url()
    sequential = _run_serial(base_url, train_steps=1)
    concurrent = _run_concurrent(base_url, train_steps=1)
    speedup = sequential.wall_clock_s / concurrent.wall_clock_s

    _print_timing("sequential", sequential)
    _print_timing("concurrent", concurrent)
    print(f"\n[tcli_multi_lora] speedup_ratio={speedup:.3f}")

    _assert_adapter_results(sequential.adapter_results)
    _assert_adapter_results(concurrent.adapter_results)


@pytest.mark.integrations
@pytest.mark.slow
@pytest.mark.skipif(not _env_enabled(RUN_MEDIUM_ENV), reason=f"set {RUN_MEDIUM_ENV}=1")
def test_tcli_multi_lora_training_sampling_sequential_vs_concurrent() -> None:
    base_url = _base_url()
    sequential = _run_serial(base_url, train_steps=2)
    concurrent = _run_concurrent(base_url, train_steps=2)
    speedup = sequential.wall_clock_s / concurrent.wall_clock_s

    _print_timing("sequential_medium", sequential)
    _print_timing("concurrent_medium", concurrent)
    print(f"\n[tcli_multi_lora] medium_speedup_ratio={speedup:.3f}")

    _assert_adapter_results(sequential.adapter_results)
    _assert_adapter_results(concurrent.adapter_results)
