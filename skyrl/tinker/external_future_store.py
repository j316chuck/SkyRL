import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel

from skyrl.tinker import types
from skyrl.tinker.db_models import RequestStatus
from skyrl.utils.log import logger


@dataclass
class ExternalFuture:
    request_id: int
    model_id: str | None
    request_data: dict
    status: RequestStatus = RequestStatus.PENDING
    result_data: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    retrieved_at: datetime | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)


class ExternalFutureStore:
    """Holds forwarded sample futures purely in memory — they never touch the database.

    Sample results are transient rollout data: a crash kills the API server and
    the engine together and the run restarts, so a persisted result would never
    be read back. Keeping the futures in memory removes sampling from the SQLite
    write path entirely. Entries are reclaimed by TTL sweeps instead of by a
    persistence hand-off.
    """

    _SWEEP_INTERVAL_SECONDS = 30.0
    # Keep terminal entries for the full client request window so an SDK retry
    # after a lost HTTP response still finds its result.
    _RETRIEVED_TTL_SECONDS = 2048.0
    # Completed entries whose client has not retrieved them yet.
    _COMPLETED_TTL_SECONDS = 2048.0
    # Pending entries whose forwarding task died without completing them.
    _PENDING_TTL_SECONDS = 3600.0

    def __init__(self):
        self._entries: dict[int, ExternalFuture] = {}
        self._request_ids_by_sequence: dict[tuple[str | None, str | None, int], int] = {}
        # Boot-epoch id space: each server process starts below every id an
        # earlier process could plausibly have handed out (2^20 ids per
        # millisecond of uptime), so a client polling a pre-restart id gets an
        # honest 404 instead of another request's result.
        self._next_request_id = -(int(time.time() * 1000) << 20) - 1
        self._sweeper: asyncio.Task | None = None

    async def start(self) -> None:
        self._sweeper = asyncio.create_task(self._sweep_loop())

    def create(self, model_id: str | None, request_data: BaseModel) -> int:
        request_id, _ = self.get_or_create(model_id, request_data)
        return request_id

    def get_or_create(self, model_id: str | None, request_data: BaseModel) -> tuple[int, bool]:
        """Create a future, or return the original future for an SDK retry."""
        serialized_request = request_data.model_dump(mode="json")
        seq_id = getattr(request_data, "seq_id", None)
        sampling_session_id = getattr(request_data, "sampling_session_id", None)
        sequence_key = (model_id, sampling_session_id, seq_id) if seq_id is not None else None
        if sequence_key is not None and (request_id := self._request_ids_by_sequence.get(sequence_key)) is not None:
            entry = self._entries[request_id]
            if entry.request_data != serialized_request:
                raise ValueError("Sampling request sequence number was reused")
            return request_id, False

        request_id = self._next_request_id
        self._next_request_id -= 1
        self._entries[request_id] = ExternalFuture(
            request_id=request_id,
            model_id=model_id,
            request_data=serialized_request,
        )
        if sequence_key is not None:
            self._request_ids_by_sequence[sequence_key] = request_id
        return request_id, True

    async def wait(self, request_id: int, timeout: float) -> tuple[RequestStatus, types.RequestType, str | None] | None:
        entry = self._entries.get(request_id)
        if entry is None:
            raise KeyError(request_id)
        try:
            await asyncio.wait_for(entry.event.wait(), timeout)
        except asyncio.TimeoutError:
            return None
        entry.retrieved_at = datetime.now(timezone.utc)
        return entry.status, types.RequestType.EXTERNAL, entry.result_data

    async def complete(self, request_id: int, result_data: BaseModel, status: RequestStatus) -> None:
        entry = self._entries.get(request_id)
        if entry is None:
            # Swept as abandoned before the forwarding task finished.
            logger.warning("External future %s was evicted before its result arrived — dropping", request_id)
            return
        entry.result_data = result_data.model_dump_json()
        entry.status = status
        entry.completed_at = datetime.now(timezone.utc)
        entry.event.set()

    async def close(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            await asyncio.gather(self._sweeper, return_exceptions=True)

    def _sweep(self, now: datetime) -> None:
        def expired(entry: ExternalFuture) -> bool:
            if entry.retrieved_at is not None:
                return (now - entry.retrieved_at).total_seconds() > self._RETRIEVED_TTL_SECONDS
            if entry.completed_at is not None:
                return (now - entry.completed_at).total_seconds() > self._COMPLETED_TTL_SECONDS
            return (now - entry.created_at).total_seconds() > self._PENDING_TTL_SECONDS

        expired_ids = [request_id for request_id, entry in self._entries.items() if expired(entry)]
        for request_id in expired_ids:
            entry = self._entries.pop(request_id)
            seq_id = entry.request_data.get("seq_id")
            sampling_session_id = entry.request_data.get("sampling_session_id")
            sequence_key = (entry.model_id, sampling_session_id, seq_id) if seq_id is not None else None
            if sequence_key is not None:
                self._request_ids_by_sequence.pop(sequence_key, None)
        if expired_ids:
            logger.info("Evicted %d expired external futures", len(expired_ids))

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self._SWEEP_INTERVAL_SECONDS)
            self._sweep(datetime.now(timezone.utc))
