"""60-second probe: does the service honor SamplingParams.stop? (issue #16)

Samples greedy with a stop token that the model will certainly emit
(newline id) and checks whether generation halts there vs running to
max_tokens. PASS = stops early; FAIL = stop ignored.
"""

import asyncio
import sys

import tinker
from tinker import types
from transformers import AutoTokenizer

BASE = "Qwen/Qwen3.6-27B"


async def main():
  url = sys.argv[1]
  tok = AutoTokenizer.from_pretrained(BASE)
  nl_id = tok.encode("\n", add_special_tokens=False)[-1]
  sc = tinker.ServiceClient(base_url=url).create_sampling_client(base_model=BASE)
  prompt = tok.encode("List five colors, one per line:", add_special_tokens=False)

  async def sample(stop):
    r = await sc.sample_async(
      prompt=types.ModelInput.from_ints(prompt),
      num_samples=1,
      sampling_params=types.SamplingParams(max_tokens=120, temperature=0.0, stop=stop),
    )
    return len(r.sequences[0].tokens)

  n_nostop = await sample(None)
  n_stop = await sample([nl_id])
  print(f"no-stop len={n_nostop}  with-stop len={n_stop}")
  if n_stop < min(n_nostop, 120) and n_stop <= 25:
    print("PASS: stop tokens honored (#16 fixed on this service)")
    return 0
  print("FAIL: stop ignored — generation ran past the stop token")
  return 1


sys.exit(asyncio.run(main()))
