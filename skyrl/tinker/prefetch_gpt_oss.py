"""One-shot helper: dequantize gpt-oss MXFP4 weights to bf16 once and save the
result to a local directory so all subsequent FSDP / vLLM loads can mmap the
bf16 tensors directly. Avoids the 30+ minute per-rank CPU dequant on cold
start of gpt-oss-120b.

Usage:
  python -m skyrl.tinker.prefetch_gpt_oss \
    --model openai/gpt-oss-120b --dst /dev/shm/gpt-oss-120b-bf16

The script is idempotent. If <dst>/.bf16-ready exists it returns immediately.
The output directory contains a config.json with `quantization_config` removed
so HF's accelerate / vLLM treat it as a plain bf16 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

READY_MARKER = ".bf16-ready"


def _parse_shard_size(spec: str) -> int:
  """Parse strings like '20GB' / '500MB' / '1500000000' into bytes."""
  s = spec.strip().upper()
  units = {"KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
  for suf, mult in units.items():
    if s.endswith(suf):
      return int(float(s[: -len(suf)]) * mult)
  return int(s)


def _save_bf16_state_dict_as_safetensors(model, dst_dir: str, max_shard_size: str) -> None:
  """Write `model.state_dict()` to dst_dir as sharded safetensors files plus an index.json.

  Avoids transformers' save_pretrained which can't round-trip a dequantized
  MXFP4 model.
  """
  import os

  from safetensors.torch import save_file

  state_dict = model.state_dict()
  shard_budget = _parse_shard_size(max_shard_size)

  shards: list[dict[str, "torch.Tensor"]] = [{}]
  shard_sizes: list[int] = [0]
  for name, tensor in state_dict.items():
    tensor = tensor.detach().contiguous()
    nbytes = int(tensor.numel()) * tensor.element_size()
    if shards[-1] and shard_sizes[-1] + nbytes > shard_budget:
      shards.append({})
      shard_sizes.append(0)
    shards[-1][name] = tensor
    shard_sizes[-1] += nbytes

  total_size = sum(shard_sizes)
  total_shards = len(shards)
  weight_map: dict[str, str] = {}
  for i, shard in enumerate(shards, start=1):
    fname = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
    path = os.path.join(dst_dir, fname)
    save_file(shard, path, metadata={"format": "pt"})
    for name in shard:
      weight_map[name] = fname
    print(
      f"[prefetch_gpt_oss]   wrote shard {i}/{total_shards} -> {fname} "
      f"({shard_sizes[i - 1] / 1e9:.2f} GB, {len(shard)} tensors)",
      flush=True,
    )

  index = {"metadata": {"total_size": int(total_size)}, "weight_map": weight_map}
  with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)
  print(
    f"[prefetch_gpt_oss]   wrote model.safetensors.index.json (total_size={total_size / 1e9:.2f} GB)",
    flush=True,
  )


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True, help="HF id or local path of the MXFP4 source model")
  ap.add_argument("--dst", required=True, help="Local directory to write the bf16 copy into")
  ap.add_argument(
    "--max-shard-size",
    default="20GB",
    help="HF save_pretrained shard size; the default keeps each safetensor under "
    "20 GB which is friendly to /dev/shm tmpfs sharding",
  )
  args = ap.parse_args()

  marker = os.path.join(args.dst, READY_MARKER)
  if os.path.isfile(marker):
    print(f"[prefetch_gpt_oss] {args.dst} already ready (marker {marker} exists), nothing to do")
    return 0

  if os.path.isdir(args.dst):
    print(f"[prefetch_gpt_oss] {args.dst} exists but no ready marker — clearing")
    shutil.rmtree(args.dst)
  os.makedirs(args.dst, exist_ok=True)

  print(f"[prefetch_gpt_oss] dequantizing {args.model} -> {args.dst}", flush=True)
  start = time.time()

  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

  quant_config = Mxfp4Config(dequantize=True)
  print("[prefetch_gpt_oss] loading source weights with Mxfp4Config(dequantize=True)...", flush=True)
  model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
  )
  load_end = time.time()
  print(f"[prefetch_gpt_oss] dequant + load took {load_end - start:.1f}s", flush=True)

  print(f"[prefetch_gpt_oss] saving bf16 to {args.dst} (max_shard_size={args.max_shard_size})...", flush=True)
  # transformers' model.save_pretrained() refuses to write a dequantized MXFP4
  # model back through its weight-conversion pipeline (raises
  # NotImplementedError from revert_weight_conversion). Bypass that by writing
  # raw safetensors shards from the live state_dict, plus a model.safetensors
  # .index.json. We then write a plain config.json (without
  # quantization_config) and a generation_config.json so HF / vLLM treat the
  # output dir as a regular bf16 checkpoint.
  _save_bf16_state_dict_as_safetensors(model, args.dst, max_shard_size=args.max_shard_size)
  save_end = time.time()
  print(f"[prefetch_gpt_oss] safetensors write took {save_end - load_end:.1f}s", flush=True)

  # Save a plain config.json (with quantization_config removed) and the
  # generation_config that the loaded model knows about.
  plain_config = model.config.to_dict()
  plain_config.pop("quantization_config", None)
  with open(os.path.join(args.dst, "config.json"), "w") as f:
    json.dump(plain_config, f, indent=2)
  print("[prefetch_gpt_oss] wrote plain config.json (no quantization_config)", flush=True)
  if model.generation_config is not None:
    try:
      model.generation_config.save_pretrained(args.dst)
    except Exception as exc:
      print(f"[prefetch_gpt_oss] generation_config save failed (non-fatal): {exc}", flush=True)

  # Mirror the tokenizer files so callers can point at the local dir end-to-
  # end. AutoTokenizer.from_pretrained handles the copy via save_pretrained.
  try:
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.save_pretrained(args.dst)
    print("[prefetch_gpt_oss] copied tokenizer files", flush=True)
  except Exception as exc:
    print(f"[prefetch_gpt_oss] tokenizer copy failed (non-fatal): {exc}", flush=True)

  with open(marker, "w") as f:
    f.write("ok\n")
  total = time.time() - start
  print(f"[prefetch_gpt_oss] done in {total:.1f}s ({args.dst})", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
