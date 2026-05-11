"""Opt-in slow GSM8K multi-LoRA TCLI e2e timing suite.

Run only from a GPU job with a command template that launches one GSM8K TCLI
client run and writes metrics.jsonl under {output_dir}.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integrations,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("SKYRL_RUN_GSM8K_MULTI_LORA_E2E") != "1",
        reason="set SKYRL_RUN_GSM8K_MULTI_LORA_E2E=1 to run this slow GPU e2e suite",
    ),
]

NUM_JOBS = 4
MAX_JOB_SECONDS = 3600
ACCURACY_KEY = "train/accuracy_unfiltered"
MIN_ACCURACY = 0.9


def _job_command(base_url: str, output_dir: Path, job_idx: int) -> list[str]:
    template = os.environ.get("SKYRL_GSM8K_MULTI_LORA_JOB_CMD")
    if not template:
        pytest.skip("set SKYRL_GSM8K_MULTI_LORA_JOB_CMD to run GSM8K TCLI jobs")
    return shlex.split(
        template.format(
            adapter_idx=job_idx,
            base_url=base_url,
            eval_interval=5,
            max_steps=10,
            output_dir=str(output_dir),
        )
    )


def _latest_accuracy(metrics_path: Path) -> float:
    if not metrics_path.exists():
        raise AssertionError(f"missing metrics file: {metrics_path}")
    latest = None
    for line in metrics_path.read_text().splitlines():
        payload = json.loads(line)
        if ACCURACY_KEY in payload:
            latest = float(payload[ACCURACY_KEY])
    if latest is None:
        raise AssertionError(f"{ACCURACY_KEY} not found in {metrics_path}")
    return latest


def _run_gsm8k_job(base_url: str, root: Path, job_idx: int) -> tuple[int, float]:
    output_dir = root / f"job_{job_idx}"
    output_dir.mkdir(parents=True)
    cmd = _job_command(base_url, output_dir, job_idx)
    subprocess.run(cmd, check=True, timeout=MAX_JOB_SECONDS)
    return job_idx, _latest_accuracy(output_dir / "metrics.jsonl")


def test_gsm8k_multi_lora_sequential_vs_concurrent():
    base_url = os.environ.get("SKYRL_GSM8K_MULTI_LORA_BASE_URL")
    if not base_url:
        pytest.skip("set SKYRL_GSM8K_MULTI_LORA_BASE_URL to one running Tinker endpoint")

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)

        start = time.perf_counter()
        sequential = [_run_gsm8k_job(base_url, root / "sequential", idx) for idx in range(NUM_JOBS)]
        sequential_time = time.perf_counter() - start

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=NUM_JOBS) as executor:
            concurrent = list(
                executor.map(lambda idx: _run_gsm8k_job(base_url, root / "concurrent", idx), range(NUM_JOBS))
            )
        concurrent_time = time.perf_counter() - start

    speedup = sequential_time / concurrent_time if concurrent_time > 0 else float("inf")
    print(f"\n[gsm8k_multi_lora] sequential_wall_clock_s={sequential_time:.3f}")
    print(f"[gsm8k_multi_lora] concurrent_wall_clock_s={concurrent_time:.3f}")
    print(f"[gsm8k_multi_lora] speedup_ratio={speedup:.3f}")
    print(f"[gsm8k_multi_lora] sequential={sequential}")
    print(f"[gsm8k_multi_lora] concurrent={concurrent}")

    assert all(accuracy > MIN_ACCURACY for _, accuracy in sequential)
    assert all(accuracy > MIN_ACCURACY for _, accuracy in concurrent)
