import asyncio
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    _load_fresh_lora_generation,
)


class _Counter:
    def __init__(self) -> None:
        self.value = 0

    def inc(self, amount: int) -> int:
        self.value += amount
        return self.value


def _models() -> SimpleNamespace:
    return SimpleNamespace(
        lora_resolver_lock=defaultdict(asyncio.Lock),
        lora_id_counter=_Counter(),
        engine_client=SimpleNamespace(add_lora=AsyncMock()),
        lora_requests={},
    )


@pytest.mark.asyncio
async def test_reload_uses_globally_unique_generation_ids() -> None:
    models = _models()

    first_a = await _load_fresh_lora_generation(models, "adapter-a", "/a/step-0")
    first_b = await _load_fresh_lora_generation(models, "adapter-b", "/b/step-0")
    next_a = await _load_fresh_lora_generation(models, "adapter-a", "/a/step-1")

    assert [first_a.lora_int_id, first_b.lora_int_id, next_a.lora_int_id] == [1, 2, 3]
    assert models.lora_requests == {"adapter-a": next_a, "adapter-b": first_b}
    assert models.engine_client.add_lora.await_count == 3


@pytest.mark.asyncio
async def test_failed_generation_is_not_published() -> None:
    models = _models()
    models.engine_client.add_lora.side_effect = RuntimeError("load failed")

    with pytest.raises(RuntimeError, match="load failed"):
        await _load_fresh_lora_generation(models, "adapter-a", "/a/step-0")

    assert models.lora_requests == {}
