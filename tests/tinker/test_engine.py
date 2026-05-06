import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from cloudpathlib import AnyPath
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Session, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import api, types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import (
    CheckpointDB,
    CheckpointStatus,
    ModelDB,
    SessionDB,
)
from skyrl.tinker.engine import TinkerEngine, prepare_model_pass_batch

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def test_process_unload_model():
    """Test that process_unload_model removes model from backend."""
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend_config={"max_lora_adapters": 4, "max_lora_rank": 32},
    )
    engine = TinkerEngine(config)
    SQLModel.metadata.create_all(engine.db_engine)

    model_id = "test_model"
    _ = engine.process_single_request(
        types.RequestType.CREATE_MODEL, model_id, {"lora_config": {"rank": 8, "alpha": 16, "seed": 0}}
    )
    assert engine.backend.has_model(model_id)

    result = engine.process_unload_model(model_id, types.UnloadModelInput())
    assert result.status == "unloaded"
    assert not engine.backend.has_model(model_id)


def test_cleanup_stale_sessions():
    """Test that cleanup_stale_sessions unloads models from expired sessions."""
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend_config={"max_lora_adapters": 4, "max_lora_rank": 32},
        session_timeout_sec=60,
        database_url="sqlite:///:memory:",  # Use in-memory DB for test isolation
    )
    engine = TinkerEngine(config)
    SQLModel.metadata.create_all(engine.db_engine)

    model_id = "stale_model"
    session_id = "stale_session"

    # Create model in backend
    _ = engine.process_single_request(
        types.RequestType.CREATE_MODEL, model_id, {"lora_config": {"rank": 8, "alpha": 16, "seed": 0}}
    )
    assert engine.backend.has_model(model_id)

    # Insert stale session and model into DB
    stale_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=120)
    with Session(engine.db_engine) as session:
        session.add(
            SessionDB(
                session_id=session_id,
                sdk_version="test",
                status="active",
                last_heartbeat_at=stale_heartbeat,
            )
        )
        session.add(
            ModelDB(
                model_id=model_id,
                base_model=BASE_MODEL,
                lora_config=types.LoraConfig(rank=8, alpha=16, seed=0).model_dump(),
                status="ready",
                request_id=1,
                session_id=session_id,
            )
        )
        session.commit()

    # Run cleanup and assert one model was unloaded
    assert engine.cleanup_stale_sessions() == 1
    assert not engine.backend.has_model(model_id)


@pytest.mark.parametrize(
    ("loss_fn", "loss_fn_config", "advantages", "logprobs"),
    [
        pytest.param(
            "ppo",
            {"clip_low_threshold": 0.7, "clip_high_threshold": 1.3},
            [],
            [],
            id="ppo_with_loss_fn_config",
        ),
        pytest.param("cross_entropy", None, [], [], id="cross_entropy_default_config"),
        pytest.param(
            "cispo",
            {"clip_low_threshold": 0.7, "clip_high_threshold": 1.3},
            [0.1, 0.2, 0.3],
            [-1.1, -1.0, -0.9],
            id="cispo",
        ),
    ],
)
def test_prepare_model_pass_batch_loss_fn_and_config(
    loss_fn: str,
    loss_fn_config: dict[str, float] | None,
    advantages: list[float],
    logprobs: list[float],
):
    """Test that prepare_model_pass_batch preserves loss_fn and loss_fn_config values."""
    datum = types.Datum(
        model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
        loss_fn_inputs=types.LossFnInputs(
            target_tokens=types.TensorData(data=[2, 3, 4]),
            weights=types.TensorData(data=[1.0, 1.0, 1.0]),
            advantages=types.TensorData(data=advantages),
            logprobs=types.TensorData(data=logprobs),
        ),
    )

    requests = {
        "req1": (
            "model1",
            types.ForwardBackwardInput(
                data=[datum],
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            ),
        ),
    }

    batch = prepare_model_pass_batch(requests)
    assert batch.all_loss_fns == [loss_fn]
    assert batch.all_loss_fn_configs == [loss_fn_config]
    assert batch.all_model_inputs == [datum.model_input]


# ---------------------------------------------------------------------------
# api.create_checkpoint: idempotent retry on FAILED rows.
#
# Regression coverage for the orphan-row poison failure mode where a prior
# save_weights attempt left a FAILED row in CheckpointDB with the same composite
# PK, blocking all subsequent retries with 409.
# ---------------------------------------------------------------------------


async def _make_async_session_engine():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return eng


async def _seed_model(session: AsyncSession, model_id: str = "m1"):
    session.add(
        ModelDB(
            model_id=model_id,
            base_model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            lora_config={"rank": 8, "alpha": 16, "seed": 0},
            status="ready",
            request_id=1,
            session_id="s1",
        )
    )
    await session.commit()


def test_create_checkpoint_resets_failed_row():
    """A FAILED row must be reset to PENDING so retries can succeed."""

    async def _impl():
        eng = await _make_async_session_engine()
        async with AsyncSession(eng) as session:
            await _seed_model(session)
            session.add(
                CheckpointDB(
                    model_id="m1",
                    checkpoint_id="ckpt0",
                    checkpoint_type=types.CheckpointType.TRAINING,
                    status=CheckpointStatus.FAILED,
                    error_message="boom",
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()

            await api.create_checkpoint(
                session, "m1", "ckpt0", types.CheckpointType.TRAINING
            )
            await session.commit()

            row = await session.get(
                CheckpointDB, ("m1", "ckpt0", types.CheckpointType.TRAINING)
            )
            assert row is not None
            assert row.status == CheckpointStatus.PENDING
            assert row.error_message is None
            assert row.completed_at is None

    asyncio.run(_impl())


def test_create_checkpoint_completed_returns_409():
    """A COMPLETED row must NOT be clobbered."""

    async def _impl():
        eng = await _make_async_session_engine()
        async with AsyncSession(eng) as session:
            await _seed_model(session)
            session.add(
                CheckpointDB(
                    model_id="m1",
                    checkpoint_id="ckpt0",
                    checkpoint_type=types.CheckpointType.TRAINING,
                    status=CheckpointStatus.COMPLETED,
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()

            with pytest.raises(HTTPException) as exc:
                await api.create_checkpoint(
                    session, "m1", "ckpt0", types.CheckpointType.TRAINING
                )
            assert exc.value.status_code == 409

    asyncio.run(_impl())


def test_create_checkpoint_pending_returns_409():
    """An in-flight PENDING row must return 409 (do not double-start)."""

    async def _impl():
        eng = await _make_async_session_engine()
        async with AsyncSession(eng) as session:
            await _seed_model(session)
            session.add(
                CheckpointDB(
                    model_id="m1",
                    checkpoint_id="ckpt0",
                    checkpoint_type=types.CheckpointType.TRAINING,
                    status=CheckpointStatus.PENDING,
                )
            )
            await session.commit()

            with pytest.raises(HTTPException) as exc:
                await api.create_checkpoint(
                    session, "m1", "ckpt0", types.CheckpointType.TRAINING
                )
            assert exc.value.status_code == 409

    asyncio.run(_impl())


def test_create_checkpoint_missing_model_returns_404():
    """A non-existent model_id must return 404, not 409."""

    async def _impl():
        eng = await _make_async_session_engine()
        async with AsyncSession(eng) as session:
            with pytest.raises(HTTPException) as exc:
                await api.create_checkpoint(
                    session, "no_such_model", "ckpt0", types.CheckpointType.TRAINING
                )
            assert exc.value.status_code == 404

    asyncio.run(_impl())
