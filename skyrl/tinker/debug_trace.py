"""Opt-in structured diagnostics for Tinker request routing and execution."""

import json
import os
from hashlib import sha256
from typing import Any

from pydantic import BaseModel

from skyrl.utils.log import logger

DEBUG_TRACE_ENV = "SKYRL_TINKER_DEBUG_TRACE"
_DEBUG_MODEL_IDS: set[str] = set()


def debug_trace_enabled(metadata: dict[str, Any] | None = None) -> bool:
    configured = os.environ.get(DEBUG_TRACE_ENV, "").lower() in {"1", "true", "yes"}
    if metadata is None:
        return configured
    return configured or str(metadata.get("debug_trace", "")).lower() in {"1", "true", "yes"}


def register_debug_model(model_id: str) -> None:
    _DEBUG_MODEL_IDS.add(model_id)


def unregister_debug_model(model_id: str) -> None:
    _DEBUG_MODEL_IDS.discard(model_id)


def model_debug_trace_enabled(model_id: str | None) -> bool:
    return debug_trace_enabled() or (model_id is not None and model_id in _DEBUG_MODEL_IDS)


def log_debug_trace(event: str, *, enabled: bool | None = None, **fields: Any) -> None:
    if enabled is None:
        enabled = debug_trace_enabled()
    if not enabled:
        return
    logger.info("tinker-debug-trace %s", json.dumps({"event": event, **fields}, sort_keys=True, default=str))


def fingerprint_models(values: list[BaseModel]) -> str:
    digest = sha256()
    for value in values:
        encoded = json.dumps(
            value.model_dump(mode="json", exclude_none=False),
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()
