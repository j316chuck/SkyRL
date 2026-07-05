"""Deterministic repro for issue #11: LoRA adapter silently not applied in vLLM sampling.

Self-contained: creates a fresh rank-32 LoRA model on the target service, inflates it
with one large deterministic optim step, then greedy-samples (temp=0) the same prompt
twice — once through the adapter's model_path, once through the raw base model — and
byte-compares the returned logprobs.

  BROKEN service (remap emits keys vLLM doesn't match): the two logprob series are
  IDENTICAL — vLLM listed and "loaded" the adapter but attached 0 tensors, so every
  rollout samples the frozen base model. Exit 1, "BUG REPRODUCED".

  FIXED service (remap emits base_model.model.language_model.model.*): the series
  diverge. Exit 0, "ADAPTER APPLIED".

Run: TINKER_API_KEY=... uv run python issues/repro_lora_remap_prefix_noop.py \
  --base-url http://<svc>:8000

NOTE: this catches the NO-OP failure mode (0 tensors attached -> sampling frozen).
A remap can also map tensors WRONGLY (sampling changes, but disagrees with the
trainer) — that variant passes this test but fails the K3 probe
(scripts/dev/tcli_logprob_mismatch_probe.py), which compares against the trainer
recompute and is the authoritative gate. Observed 2026-07-04: new-image services
pre-fix were no-op (this test fails); an older-image service applied wrongly
(this test passes, probe explodes).
"""

import argparse
import asyncio

import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer

INFLATE_TEXT = "The quick brown fox jumps over the lazy dog. " * 40
SAMPLE_PROMPT = "The capital of France is"


async def greedy_logprobs(sampling_client, prompt_tokens: list[int]) -> list[float]:
  params = types.SamplingParams(max_tokens=24, temperature=0.0)
  result = await sampling_client.sample_async(
    prompt=types.ModelInput.from_ints(prompt_tokens),
    num_samples=1,
    sampling_params=params,
  )
  seq = result.sequences[0]
  if seq.logprobs is None:
    raise ValueError("service did not return per-token logprobs")
  return [round(float(x), 6) for x in seq.logprobs]


async def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--base-url", required=True)
  parser.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
  parser.add_argument("--lora-rank", type=int, default=32)
  parser.add_argument("--lr", type=float, default=0.01)
  args = parser.parse_args()

  service_client = tinker.ServiceClient(base_url=args.base_url)
  training_client = await service_client.create_lora_training_client_async(
    base_model=args.base_model, rank=args.lora_rank
  )
  tokenizer = AutoTokenizer.from_pretrained(args.base_model)

  # One deterministic datum -> forward_backward -> one large optim step.
  tokens = tokenizer.encode(INFLATE_TEXT)[:512]
  target = tokens[1:]
  n = len(target)
  datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens[:-1]),
    loss_fn_inputs={
      "target_tokens": TensorData(data=target, dtype="int64", shape=[n]),
      "logprobs": TensorData(data=[0.0] * n, dtype="float32", shape=[n]),
      "advantages": TensorData(data=[1.0] * n, dtype="float32", shape=[n]),
    },
  )
  # Plain encode (no chat template): older service images reject the newest
  # template special tokens ("Token id ... out of vocabulary").
  prompt_tokens = tokenizer.encode(SAMPLE_PROMPT)

  # BEFORE: adapter is zero-init -> sampling == base model behavior.
  client_before = await training_client.save_weights_and_get_sampling_client_async()
  lps_before = await greedy_logprobs(client_before, prompt_tokens)

  # Inflate: one deterministic forward_backward + one large optim step.
  await training_client.forward_backward([datum], loss_fn="importance_sampling").result_async(
    timeout=1800
  )
  await training_client.optim_step(types.AdamParams(learning_rate=args.lr)).result_async(
    timeout=600
  )

  # AFTER: adapter is large -> sampling MUST change if weight sync works.
  client_after = await training_client.save_weights_and_get_sampling_client_async()
  lps_after = await greedy_logprobs(client_after, prompt_tokens)

  print("before-inflation logprobs:", lps_before[:8])
  print("after-inflation  logprobs:", lps_after[:8])
  if lps_before == lps_after:
    print("BUG REPRODUCED: large weight update had ZERO effect on sampling — sampler frozen")
    return 1
  print("WEIGHTS APPLIED: sampling changed after the weight update, as expected")
  return 0


if __name__ == "__main__":
  raise SystemExit(asyncio.run(main()))
