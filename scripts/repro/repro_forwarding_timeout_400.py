"""Repro: slow vLLM completions exceed the forwarding read timeout -> empty 400s.

`skyrl_train_inference_forwarding.py` posts to the vLLM router with
`httpx.Timeout(300.0, connect=10.0)`. Any completion slower than the read
timeout raises ReadTimeout, the future is stored as failed, and the client's
retrieve_future gets `400 {'detail': ''}` — indistinguishable from a bad
request. Long-decode workloads (tau: multi-turn, 262k context) lose every
trajectory once the service is under load; short-decode ones (GSM8K) only lose
a slow tail.

Repro: deploy a service with the timeout patch DISABLED (or set
SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC=15 on a patched one), then request a
generation that decodes longer than the timeout:

  TINKER_API_KEY=... uv run python issues/repro_forwarding_timeout_400.py \
    --base-url http://<svc>:8000 --max-tokens 8192

Expected on a broken service: tinker.BadRequestError with empty detail after
~timeout seconds. Expected on a fixed (1800s) service: sequence returned.

Fix shipped: SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC env (default 1800) via
workload startup patch. Production fix: config-driven timeout + surface
timeouts as 504 with a descriptive detail, never an empty 400.
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
  parser.add_argument("--max-tokens", type=int, default=8192)
  args = parser.parse_args()

  service_client = tinker.ServiceClient(base_url=args.base_url)
  training_client = await service_client.create_lora_training_client_async(
    base_model=args.model, rank=32
  )
  sampling_client = await training_client.save_weights_and_get_sampling_client_async()
  tokenizer = AutoTokenizer.from_pretrained(args.model)
  prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Count from 1 to 5000, one number per line."}],
    add_generation_prompt=True,
  )

  start = time.time()
  try:
    result = await sampling_client.sample_async(
      prompt=types.ModelInput.from_ints(prompt),
      num_samples=1,
      sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0),
    )
    n = len(result.sequences[0].tokens)
    print(f"OK: {n} tokens in {time.time() - start:.0f}s (service is fixed)")
  except tinker.BadRequestError as e:
    print(f"REPRODUCED after {time.time() - start:.0f}s: empty-detail 400: {e}")


if __name__ == "__main__":
  asyncio.run(main())
