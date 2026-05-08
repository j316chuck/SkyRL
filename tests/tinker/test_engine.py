import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from cloudpathlib import AnyPath
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Session, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import api, types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import CheckpointDB, CheckpointStatus, ModelDB, SessionDB
from skyrl.tinker.engine import TinkerEngine, prepare_model_pass_batch

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
MODEL_ID = "model_for_checkpoint_tests"
CHECKPOINT_ID = "global_step_0"


async def _create_async_checkpoint_session(tmp_path) -> AsyncSession:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return AsyncSession(engine)


async def _insert_checkpoint_model(session: AsyncSession) -> None:
    session.add(SessionDB(session_id="session_for_checkpoint_tests", tags=[], sdk_version="test"))
    session.add(
        ModelDB(
            model_id=MODEL_ID,
            base_model=BASE_MODEL,
            lora_config={},
            status="ready",
            request_id=1,
            session_id="session_for_checkpoint_tests",
        )
    )
    await session.commit()


async def _insert_checkpoint(session: AsyncSession, status: CheckpointStatus) -> None:
    session.add(
        CheckpointDB(
            model_id=MODEL_ID,
            checkpoint_id=CHECKPOINT_ID,
            checkpoint_type=types.CheckpointType.TRAINING,
            status=status,
            completed_at=datetime.now(timezone.utc),
            error_message="old failure",
        )
    )
    await session.commit()


async def _get_checkpoint(session: AsyncSession) -> CheckpointDB:
    result = await session.exec(
        select(CheckpointDB).where(
            CheckpointDB.model_id == MODEL_ID,
            CheckpointDB.checkpoint_id == CHECKPOINT_ID,
            CheckpointDB.checkpoint_type == types.CheckpointType.TRAINING,
        )
    )
    checkpoint = result.one()
    return checkpoint


def test_create_checkpoint_failed_row_returns_409_without_overwrite(tmp_path):
    async def run_test():
        async with await _create_async_checkpoint_session(tmp_path) as session:
            await _insert_checkpoint_model(session)
            await _insert_checkpoint(session, CheckpointStatus.FAILED)

            with pytest.raises(HTTPException) as exc_info:
                await api.create_checkpoint(
                    session=session,
                    model_id=MODEL_ID,
                    checkpoint_id=CHECKPOINT_ID,
                    checkpoint_type=types.CheckpointType.TRAINING,
                )

            assert exc_info.value.status_code == 409

    asyncio.run(run_test())


@pytest.mark.parametrize("status", [CheckpointStatus.FAILED, CheckpointStatus.COMPLETED])
def test_create_checkpoint_overwrite_resets_terminal_row(tmp_path, status: CheckpointStatus):
    async def run_test():
        async with await _create_async_checkpoint_session(tmp_path) as session:
            await _insert_checkpoint_model(session)
            await _insert_checkpoint(session, status)

            await api.create_checkpoint(
                session=session,
                model_id=MODEL_ID,
                checkpoint_id=CHECKPOINT_ID,
                checkpoint_type=types.CheckpointType.TRAINING,
                overwrite=True,
            )

            checkpoint = await _get_checkpoint(session)
            assert checkpoint.status == CheckpointStatus.PENDING
            assert checkpoint.completed_at is None
            assert checkpoint.error_message is None

    asyncio.run(run_test())


@pytest.mark.parametrize(
    ("status", "overwrite"),
    [
        (CheckpointStatus.PENDING, False),
        (CheckpointStatus.PENDING, True),
        (CheckpointStatus.COMPLETED, False),
    ],
)
def test_create_checkpoint_existing_row_returns_409(tmp_path, status: CheckpointStatus, overwrite: bool):
    async def run_test():
        async with await _create_async_checkpoint_session(tmp_path) as session:
            await _insert_checkpoint_model(session)
            await _insert_checkpoint(session, status)

            with pytest.raises(HTTPException) as exc_info:
                await api.create_checkpoint(
                    session=session,
                    model_id=MODEL_ID,
                    checkpoint_id=CHECKPOINT_ID,
                    checkpoint_type=types.CheckpointType.TRAINING,
                    overwrite=overwrite,
                )

            assert exc_info.value.status_code == 409

    asyncio.run(run_test())


def test_create_checkpoint_missing_model_returns_404(tmp_path):
    async def run_test():
        async with await _create_async_checkpoint_session(tmp_path) as session:
            with pytest.raises(HTTPException) as exc_info:
                await api.create_checkpoint(
                    session=session,
                    model_id=MODEL_ID,
                    checkpoint_id=CHECKPOINT_ID,
                    checkpoint_type=types.CheckpointType.TRAINING,
                )

            assert exc_info.value.status_code == 404

    asyncio.run(run_test())


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
