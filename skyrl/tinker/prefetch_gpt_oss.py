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
  model.save_pretrained(args.dst, max_shard_size=args.max_shard_size, safe_serialization=True)
  save_end = time.time()
  print(f"[prefetch_gpt_oss] save_pretrained took {save_end - load_end:.1f}s", flush=True)

  # Drop the MXFP4 quantization marker from the on-disk config.json so HF /
  # vLLM treat the new copy as a plain bf16 checkpoint.
  cfg_path = os.path.join(args.dst, "config.json")
  if os.path.isfile(cfg_path):
    with open(cfg_path) as f:
      cfg = json.load(f)
    if "quantization_config" in cfg:
      del cfg["quantization_config"]
      with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
      print("[prefetch_gpt_oss] stripped quantization_config from config.json", flush=True)

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
