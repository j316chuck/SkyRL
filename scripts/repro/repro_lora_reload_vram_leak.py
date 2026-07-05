"""Production repro for issue #15: LoRA reload leaks ~1 adapter of VRAM per weight sync.

Drives N inflate->save_weights_for_sampler cycles (each triggers an adapter reload on
the inference engines) and samples inference-GPU VRAM between cycles. On a broken
service VRAM grows ~170MB/cycle (rank-32 bf16 adapter on a 27B model) until engines
OOM; on a fixed service (bounded id rotation) readings are byte-flat.

VRAM sampling: pass --pod to sample via `kubectl exec <pod> nvidia-smi` (requires
KUBECONFIG + context access); without it, the script prints the manual command and
still reports cycle timings.

Run: TINKER_API_KEY=... uv run python issues/repro_lora_reload_vram_leak.py \
  --base-url http://<svc>:8000 --pod <service-pod-name> --cycles 10
PASS: max-min VRAM spread < 300MB across cycles. FAIL: monotone growth ~170MB/cycle.
"""

import argparse
import asyncio
import subprocess

import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer

INFLATE_TEXT = "The quick brown fox jumps over the lazy dog. " * 40


def sample_vram(pod: str, context: str | None) -> list[int]:
  cmd = ["kubectl"]
  if context:
    cmd += ["--context", context]
  cmd += [
    "exec",
    pod,
    "-n",
    "default",
    "--",
    "nvidia-smi",
    "--query-gpu=memory.used",
    "--format=csv,noheader,nounits",
  ]
  out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
  return [int(x) for x in out.split() if x.strip().isdigit()]


async def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--base-url", required=True)
  parser.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
  parser.add_argument("--pod", default=None, help="service pod name for VRAM sampling")
  parser.add_argument("--kube-context", default=None)
  parser.add_argument("--cycles", type=int, default=10)
  parser.add_argument("--lr", type=float, default=1e-3)
  args = parser.parse_args()

  service_client = tinker.ServiceClient(base_url=args.base_url)
  training_client = await service_client.create_lora_training_client_async(
    base_model=args.base_model, rank=32
  )
  tokenizer = AutoTokenizer.from_pretrained(args.base_model)
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

  readings: list[list[int]] = []
  for cycle in range(args.cycles):
    await training_client.forward_backward([datum], loss_fn="importance_sampling").result_async(
      timeout=1800
    )
    await training_client.optim_step(types.AdamParams(learning_rate=args.lr)).result_async(
      timeout=600
    )
    # save_weights_for_sampler triggers the export + engine-side adapter reload
    await training_client.save_weights_and_get_sampling_client_async()
    if args.pod:
      vram = sample_vram(args.pod, args.kube_context)
      readings.append(vram)
      print(f"cycle {cycle}: inference-GPU VRAM MiB = {vram}")
    else:
      print(f"cycle {cycle}: sync done (sample VRAM manually: kubectl exec <pod> -- nvidia-smi)")

  if readings:
    # inference GPUs = the max-loaded half; compare per-GPU spread across cycles
    per_gpu = list(zip(*readings))
    worst_spread = max(max(g) - min(g) for g in per_gpu)
    print(f"worst per-GPU spread across {args.cycles} cycles: {worst_spread} MiB")
    if worst_spread < 300:
      print("PASS: VRAM flat across reloads (leak fixed)")
      return 0
    print("FAIL: VRAM grows across reloads (adapter leak)")
    return 1
  print("no VRAM sampling (no --pod); inspect manually")
  return 0


if __name__ == "__main__":
  raise SystemExit(asyncio.run(main()))
