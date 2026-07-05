"""Repro: tinker-API DB pool exhaustion under concurrent rollout traffic.

The tinker API's SQLAlchemy engine defaults to pool_size=5, max_overflow=10,
pool_timeout=30. Every API request (retrieve_future, telemetry, session
heartbeats) takes a connection; ~15+ sustained concurrent requests exhaust the
pool and every later request blocks 30s then raises
`sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached`.
Observed live: a 256-trajectory GSM8K rollout burst wedged the API ~5 minutes
(session heartbeats failed 120-300s). At --concurrency 512 this also
demonstrates the fan-out saturation cascade (issue #5).

Repro (against a DISPOSABLE service; this intentionally degrades it):

  TINKER_API_KEY=... uv run python issues/repro_db_pool_exhaustion.py \
    --base-url http://<svc>:8000 --concurrency 256 --seconds 60

It hammers concurrent sampling requests and reports healthz latency from a
side channel. Broken service: healthz p95 blows past seconds/timeouts while
the burst runs. Fixed service (pool 50/100 + inmem futures + semaphore):
healthz stays <1s.

Fix shipped: create_async_engine(pool_size=50, max_overflow=100,
pool_timeout=120) + in-memory futures + Semaphore(256) workload patches.
Production fix: size the pool to the concurrency ceiling and make it config.
"""

import argparse
import asyncio
import time

import httpx
import tinker
from tinker import types
from transformers import AutoTokenizer


async def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--base-url", required=True)
  parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
  parser.add_argument("--concurrency", type=int, default=256)
  parser.add_argument("--seconds", type=int, default=60)
  args = parser.parse_args()

  service_client = tinker.ServiceClient(base_url=args.base_url)
  training_client = await service_client.create_lora_training_client_async(
    base_model=args.model, rank=32
  )
  sampling_client = await training_client.save_weights_and_get_sampling_client_async()
  tokenizer = AutoTokenizer.from_pretrained(args.model)
  prompt = types.ModelInput.from_ints(
    tokenizer.apply_chat_template(
      [{"role": "user", "content": "Write a long story about a lighthouse."}],
      add_generation_prompt=True,
    )
  )
  params = types.SamplingParams(max_tokens=512, temperature=1.0)
  deadline = time.time() + args.seconds

  async def hammer() -> None:
    while time.time() < deadline:
      try:
        await sampling_client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
      except Exception as e:  # noqa: BLE001 - we are measuring degradation, not correctness
        print(f"sample error: {type(e).__name__}: {str(e)[:120]}")

  async def watch_health() -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
      while time.time() < deadline:
        t0 = time.time()
        try:
          r = await client.get(f"{args.base_url}/api/v1/healthz")
          print(f"healthz {r.status_code} in {time.time() - t0:.2f}s")
        except Exception as e:  # noqa: BLE001
          print(f"REPRODUCED: healthz failed after {time.time() - t0:.1f}s: {type(e).__name__}")
        await asyncio.sleep(5)

  await asyncio.gather(watch_health(), *[hammer() for _ in range(args.concurrency)])


if __name__ == "__main__":
  asyncio.run(main())
