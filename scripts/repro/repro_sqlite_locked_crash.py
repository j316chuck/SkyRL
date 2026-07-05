"""Repro: SQLite `database is locked` crashes the tinker API process.

The tinker API stores sessions and futures in a single-writer SQLite DB with
`PRAGMA busy_timeout=30000`. Under sustained write pressure (futures
create/update ~4.5KB payloads + session heartbeats every few seconds), a
heartbeat UPDATE can wait >30s for the writer lock, raise
`sqlite3.OperationalError: database is locked`, and take the whole API process
down (observed: svc bc6c385b exit 1 at 13:58Z, killing tau run 1035997 and
every model hosted on the service).

Repro (against a DISPOSABLE service; this can crash it — that is the point):

  TINKER_API_KEY=... uv run python issues/repro_sqlite_locked_crash.py \
    --base-url http://<svc>:8000 --writers 64 --seconds 300

Drives many concurrent training sessions (each heartbeating) plus a sampling
firehose that generates futures writes. Broken service: process exits within
minutes (watch `kubectl get pod` RESTARTS / lastState.terminated). Fixed
service (busy_timeout 300s + inmem futures): survives.

Fix shipped: `PRAGMA busy_timeout=300000` + in-memory futures workload
patches. Production fix: sessions/futures belong in a real multi-writer store
(Postgres) or at minimum WAL mode; a lock timeout must not be process-fatal.
"""

import argparse
import asyncio
import time

import tinker
from tinker import types
from transformers import AutoTokenizer


async def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--base-url", required=True)
  parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
  parser.add_argument("--writers", type=int, default=64)
  parser.add_argument("--seconds", type=int, default=300)
  args = parser.parse_args()

  tokenizer = AutoTokenizer.from_pretrained(args.model)
  prompt = types.ModelInput.from_ints(
    tokenizer.apply_chat_template(
      [{"role": "user", "content": "List 200 animals."}], add_generation_prompt=True
    )
  )
  params = types.SamplingParams(max_tokens=256, temperature=1.0)
  deadline = time.time() + args.seconds

  async def writer(idx: int) -> None:
    # Each writer holds its own session (heartbeat stream) and creates futures.
    service_client = tinker.ServiceClient(base_url=args.base_url)
    training_client = await service_client.create_lora_training_client_async(
      base_model=args.model, rank=32
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    while time.time() < deadline:
      try:
        await sampling_client.sample_async(prompt=prompt, num_samples=2, sampling_params=params)
      except Exception as e:  # noqa: BLE001 - service death is the signal we watch for
        print(f"writer {idx}: {type(e).__name__}: {str(e)[:120]}")
        await asyncio.sleep(2)

  results = await asyncio.gather(*[writer(i) for i in range(args.writers)], return_exceptions=True)
  errs = [r for r in results if isinstance(r, Exception)]
  print(f"done: {len(errs)}/{args.writers} writers ended in error")
  print("check the service pod for restarts: kubectl get pod <svc> (lastState.terminated)")


if __name__ == "__main__":
  asyncio.run(main())
