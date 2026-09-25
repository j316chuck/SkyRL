from unittest.mock import patch

from pydantic import BaseModel

from skyrl.tinker.debug_trace import (
    debug_trace_enabled,
    fingerprint_models,
    log_debug_trace,
    model_debug_trace_enabled,
    register_debug_model,
    unregister_debug_model,
)


class _Payload(BaseModel):
    value: int


def test_debug_trace_is_opt_in_and_fingerprint_is_stable(monkeypatch):
    monkeypatch.delenv("SKYRL_TINKER_DEBUG_TRACE", raising=False)
    assert not debug_trace_enabled()
    assert debug_trace_enabled({"debug_trace": "true"})
    assert fingerprint_models([_Payload(value=1)]) == fingerprint_models([_Payload(value=1)])
    assert fingerprint_models([_Payload(value=1)]) != fingerprint_models([_Payload(value=2)])

    with patch("skyrl.tinker.debug_trace.logger.info") as info:
        log_debug_trace("ignored", enabled=False, private="value")
        info.assert_not_called()
        log_debug_trace("observed", enabled=True, model_id="model_test")

    payload = info.call_args.args[1]
    assert '"event": "observed"' in payload
    assert '"model_id": "model_test"' in payload


def test_debug_model_registry_scopes_deep_tracing(monkeypatch):
    monkeypatch.delenv("SKYRL_TINKER_DEBUG_TRACE", raising=False)
    register_debug_model("model_traced")
    try:
        assert model_debug_trace_enabled("model_traced")
        assert not model_debug_trace_enabled("model_other")
    finally:
        unregister_debug_model("model_traced")
    assert not model_debug_trace_enabled("model_traced")
