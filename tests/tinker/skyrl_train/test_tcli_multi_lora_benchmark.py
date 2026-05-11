"""TCLI multi-LoRA timing smoke test against one SkyRL-Train Tinker endpoint.

Run with:
  SKYRL_RUN_TCLI_MULTI_LORA_INTEGRATION=1 \
  uv run --extra tinker --extra megatron --with pytest --with pytest-timeout \
    pytest -s tests/tinker/skyrl_train/test_tcli_multi_lora_benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import pytest

cuda_available = False
try:  # pragma: no cover - import guard
    import torch

    cuda_available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
except Exception:
    cuda_available = False

pytestmark = [
    pytest.mark.integrations,
    pytest.mark.skipif(
        os.environ.get("SKYRL_RUN_TCLI_MULTI_LORA_INTEGRATION") != "1",
        reason="set SKYRL_RUN_TCLI_MULTI_LORA_INTEGRATION=1 to run this GPU integration test",
    ),
    pytest.mark.skipif(not cuda_available, reason="TCLI multi-LoRA integration requires a CUDA GPU"),
]

tinker = pytest.importorskip("tinker")
pytest.importorskip("megatron.core", reason="TCLI multi-LoRA integration requires the megatron extra")
from tinker import types as tinker_types  # noqa: E402

from tests.tinker.conftest import wait_for_condition  # noqa: E402

BASE_MODEL = os.environ.get("SKYRL_TCLI_MULTI_LORA_BASE_MODEL", "Qwen/Qwen3-4B")
TINKER_API_KEY = "tml-dummy"
TEST_PORT = int(os.environ.get("SKYRL_TCLI_MULTI_LORA_PORT", "8021"))
LORA_RANK = int(os.environ.get("SKYRL_TCLI_MULTI_LORA_RANK", "8"))
TRAIN_REPEATS = int(os.environ.get("SKYRL_TCLI_MULTI_LORA_TRAIN_REPEATS", "8"))

BACKEND_CONFIG = {
    "strategy": "megatron",
    "trainer.placement.policy_num_gpus_per_node": 1,
    "trainer.placement.policy_num_nodes": 1,
    "trainer.placement.colocate_all": False,
    "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
    "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
    "trainer.policy.megatron_config.lora_config.merge_lora": False,
    "trainer.policy.model.lora.max_loras": 4,
    "trainer.policy.model.lora.max_cpu_loras": 4,
    "generator.inference_engine.num_engines": 1,
    "generator.inference_engine.tensor_parallel_size": 1,
    "generator.inference_engine.backend": "vllm",
    "generator.inference_engine.run_engines_locally": True,
}


@dataclass(frozen=True)
class PigLatinJob:
    label: str
    word: str
    expected: str
    seed: int


@dataclass(frozen=True)
class JobResult:
    label: str
    model_id: str
    pre_loss: float
    post_loss: float
    sample_text: str
    passed: bool


PIG_LATIN_JOBS = [
    PigLatinJob(label="adapter_0", word="cat", expected="atcay", seed=100),
    PigLatinJob(label="adapter_1", word="dog", expected="ogday", seed=101),
    PigLatinJob(label="adapter_2", word="fish", expected="ishfay", seed=102),
    PigLatinJob(label="adapter_3", word="bird", expected="irdbay", seed=103),
]


@contextmanager
def _api_server(port: int):
    if shutil.which("uv") is None:
        pytest.skip("uv is required to launch the Tinker API server")

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "server.log")
        db_path = os.path.join(tmp_dir, "server.db")
        cmd = [
            "uv",
            "run",
            "--extra",
            "tinker",
            "--extra",
            "megatron",
            "-m",
            "skyrl.tinker.api",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--base-model",
            BASE_MODEL,
            "--backend",
            "megatron",
            "--backend-config",
            json.dumps(BACKEND_CONFIG),
            "--database-url",
            f"sqlite:///{db_path}",
        ]
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            try:
                ok = wait_for_condition(
                    lambda: _server_is_up(port) or proc.poll() is not None,
                    timeout_sec=180,
                    poll_interval_sec=2,
                )
                if not ok or proc.poll() is not None:
                    with open(log_path) as f:
                        print(f"=== Tinker API server log ({log_path}) ===\n{f.read()}")
                    pytest.fail("Tinker API server did not come up")
                yield f"http://127.0.0.1:{port}/"
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()


def _server_is_up(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/healthz", timeout=2).read()
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
        return False


def _make_datum(tokenizer, prompt: str, completion: str):
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f"{completion}\n\n", add_special_tokens=False)
    all_tokens = prompt_tokens + completion_tokens
    target_tokens = all_tokens[1:] + [tokenizer.eos_token_id]
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(all_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights[1:] + [1.0]},
    )


def _loss_sum(output) -> float:
    return float(sum(sum(item["elementwise_loss"].data) for item in output.loss_fn_outputs))


async def _unload_model(base_url: str, model_id: str) -> None:
    async with tinker._client.AsyncTinker(api_key=TINKER_API_KEY, base_url=base_url) as client:  # type: ignore[attr-defined]
        future = await client.models.unload(request=tinker_types.UnloadModelRequest(model_id=model_id))
        while True:
            result = await client.futures.retrieve(
                request=tinker_types.FutureRetrieveRequest(request_id=future.request_id)
            )
            if isinstance(result, tinker_types.UnloadModelResponse):
                return
            await asyncio.sleep(0.1)


def _run_pig_latin_job(base_url: str, job: PigLatinJob, unload: bool) -> JobResult:
    service_client = tinker.ServiceClient(base_url=base_url, api_key=TINKER_API_KEY)
    model_id = ""
    try:
        training_client = service_client.create_lora_training_client(
            base_model=BASE_MODEL,
            rank=LORA_RANK,
            seed=job.seed,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
        model_id = training_client.model_id
        tokenizer = training_client.get_tokenizer()
        prompt = f"Translate the word {job.word} to Pig Latin. Answer with only the translated word.\nAnswer:"
        data = [_make_datum(tokenizer, prompt, f" {job.expected}") for _ in range(TRAIN_REPEATS)]

        pre_loss = _loss_sum(training_client.forward_backward(data, "cross_entropy").result())
        training_client.optim_step(tinker_types.AdamParams(learning_rate=5e-2)).result()
        post_loss = _loss_sum(training_client.forward_backward(data, "cross_entropy").result())

        sampler = training_client.save_weights_and_get_sampling_client()
        sample = sampler.sample(
            prompt=tinker_types.ModelInput.from_ints(tokenizer.encode(prompt, add_special_tokens=True)),
            num_samples=1,
            sampling_params=tinker_types.SamplingParams(
                max_tokens=8,
                temperature=0.0,
                top_k=1,
                seed=job.seed,
            ),
        ).result()
        sample_text = tokenizer.decode(sample.sequences[0].tokens).lower()
        passed = post_loss <= pre_loss + 1e-3 and job.expected in sample_text
        return JobResult(
            label=job.label,
            model_id=model_id,
            pre_loss=pre_loss,
            post_loss=post_loss,
            sample_text=sample_text,
            passed=passed,
        )
    finally:
        if unload and model_id:
            asyncio.run(_unload_model(base_url, model_id))
        service_client.holder.close()


def _print_results(name: str, wall_time: float, results: list[JobResult]) -> None:
    print(f"\n[tcli_multi_lora] {name}_wall_clock_s={wall_time:.3f}")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"[tcli_multi_lora] {name} {result.label} {status} "
            f"model_id={result.model_id} pre_loss={result.pre_loss:.6f} "
            f"post_loss={result.post_loss:.6f} sample={result.sample_text!r}"
        )


def test_tcli_multi_lora_sequential_vs_concurrent():
    with _api_server(TEST_PORT) as base_url:
        start = time.perf_counter()
        sequential_results = [_run_pig_latin_job(base_url, job, unload=True) for job in PIG_LATIN_JOBS]
        sequential_time = time.perf_counter() - start

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(PIG_LATIN_JOBS)) as executor:
            concurrent_results = list(
                executor.map(lambda job: _run_pig_latin_job(base_url, job, unload=False), PIG_LATIN_JOBS)
            )
        concurrent_time = time.perf_counter() - start

        for result in concurrent_results:
            asyncio.run(_unload_model(base_url, result.model_id))

    speedup = sequential_time / concurrent_time if concurrent_time > 0 else float("inf")
    print(f"\n[tcli_multi_lora] speedup_ratio={speedup:.3f}")
    _print_results("sequential", sequential_time, sequential_results)
    _print_results("concurrent", concurrent_time, concurrent_results)

    all_results = sequential_results + concurrent_results
    assert len({result.model_id for result in all_results}) == len(all_results)
    assert all(result.passed for result in sequential_results)
    assert all(result.passed for result in concurrent_results)
