"""Tests for the group-commit write path (external_future_store) and submit_future.

The contract under test: hot-path writes stay durable (committed before the
caller gets its id/result back) while no longer costing one write transaction
and one pool connection each -- concurrent bursts collapse into batched
commits, and a failing write is isolated to its own caller.
"""

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, create_engine, select
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.api import poll_futures, submit_future, wait_for_future
from skyrl.tinker.db_models import (
    FutureDB,
    RequestStatus,
    enable_sqlite_wal,
    get_async_database_url,
)
from skyrl.tinker.external_future_store import ExternalFutureStore, GroupCommitWriter


@pytest.fixture()
def db_url(tmp_path):
    """A file-backed SQLite database with the schema created."""
    url = f"sqlite:///{tmp_path / 'tinker.db'}"
    sync_engine = create_engine(url)
    enable_sqlite_wal(sync_engine)
    SQLModel.metadata.create_all(sync_engine)
    sync_engine.dispose()
    return url


@pytest_asyncio.fixture()
async def async_engine(db_url):
    # pool_size=2 on purpose: the write path must not need pool headroom to
    # survive a burst, since queued writes hold no connection.
    engine = create_async_engine(get_async_database_url(db_url), pool_size=2, max_overflow=0)
    enable_sqlite_wal(engine.sync_engine)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def writer(async_engine):
    writer = GroupCommitWriter(async_engine)
    writer.start()
    yield writer
    with suppress(Exception):
        await writer.close()


@pytest_asyncio.fixture()
async def store(async_engine, writer):
    return ExternalFutureStore(async_engine, writer)


def insert_future_op(model_id: str = "model_a", request_type=types.RequestType.EXTERNAL, seq_id: int | None = None):
    async def op(session: AsyncSession) -> int:
        row = FutureDB(
            request_type=request_type,
            model_id=model_id,
            seq_id=seq_id,
            request_data={"checkpoint_id": ""},
            status=RequestStatus.PENDING,
        )
        session.add(row)
        await session.flush()
        return row.request_id

    return op


async def fetch_row(async_engine, request_id: int) -> FutureDB | None:
    async with AsyncSession(async_engine) as session:
        return await session.get(FutureDB, request_id)


SAMPLE_RESULT = types.SampleOutput(sequences=[types.GeneratedSequence(stop_reason="stop", tokens=[1], logprobs=[-0.5])])


@pytest.mark.asyncio
async def test_submit_returns_only_after_commit(async_engine, writer):
    request_id = await writer.submit(insert_future_op())

    # A fresh session (fresh snapshot) must already see the row.
    row = await fetch_row(async_engine, request_id)
    assert row is not None and row.status == RequestStatus.PENDING


@pytest.mark.asyncio
async def test_concurrent_burst_collapses_into_batched_commits(async_engine, writer):
    commits = []

    @event.listens_for(async_engine.sync_engine, "commit")
    def _count(conn):
        commits.append(1)

    ids = await asyncio.gather(*(writer.submit(insert_future_op()) for _ in range(256)))

    assert len(set(ids)) == 256
    # 256 individual transactions would be 256 commits; group commit makes it
    # at most ceil(256 / batch) plus a couple of stragglers from batch timing.
    assert len(commits) <= 32


@pytest.mark.asyncio
async def test_failing_op_is_isolated_from_its_batch(async_engine, writer):
    # Two inserts violating the (model_id, seq_id) unique constraint, queued
    # into the same batch alongside innocent writes.
    results = await asyncio.gather(
        writer.submit(insert_future_op(seq_id=7)),
        writer.submit(insert_future_op(seq_id=7)),
        writer.submit(insert_future_op(model_id="model_b")),
        writer.submit(insert_future_op(model_id="model_c")),
        return_exceptions=True,
    )

    failures = [r for r in results if isinstance(r, Exception)]
    successes = [r for r in results if not isinstance(r, Exception)]
    assert len(failures) == 1  # exactly one of the duplicates loses
    assert len(successes) == 3
    for request_id in successes:
        assert await fetch_row(async_engine, request_id) is not None


@pytest.mark.asyncio
async def test_close_drains_accepted_writes_and_rejects_new_ones(async_engine):
    writer = GroupCommitWriter(async_engine)
    writer.start()
    pending = [asyncio.create_task(writer.submit(insert_future_op())) for _ in range(20)]
    await asyncio.sleep(0.01)  # let every submission enqueue before closing
    await writer.close()

    ids = await asyncio.gather(*pending)
    assert len(set(ids)) == 20
    with pytest.raises(RuntimeError):
        await writer.submit(insert_future_op())


@pytest.mark.asyncio
async def test_complete_resolves_a_polling_waiter(async_engine, writer, store):
    """End-to-end through the real retrieval machinery: a forwarded sample's
    completion lands durably and wakes its retrieve_future waiter."""
    request_id = await writer.submit(insert_future_op())

    waiters: dict[int, set[asyncio.Future]] = {}
    poller = asyncio.create_task(poll_futures(async_engine, waiters, poll_interval_sec=0.01))
    try:
        waiter = asyncio.create_task(wait_for_future(waiters, request_id, timeout=5))
        await store.complete(request_id, SAMPLE_RESULT, RequestStatus.COMPLETED)
        status, request_type, result_data = await waiter
    finally:
        poller.cancel()
        with suppress(asyncio.CancelledError):
            await poller

    assert (status, request_type) == (RequestStatus.COMPLETED, types.RequestType.EXTERNAL)
    assert types.SampleOutput.model_validate_json(result_data) == SAMPLE_RESULT


@pytest.mark.asyncio
async def test_complete_ignores_missing_and_already_terminal_rows(async_engine, writer, store):
    # Missing row: no exception, nothing created.
    await store.complete(424242, SAMPLE_RESULT, RequestStatus.COMPLETED)
    assert await fetch_row(async_engine, 424242) is None

    # Already terminal: the first result stands.
    request_id = await writer.submit(insert_future_op())
    await store.complete(request_id, SAMPLE_RESULT, RequestStatus.COMPLETED)
    late_error = types.ErrorResponse(error="late duplicate", status="failed")
    await store.complete(request_id, late_error, RequestStatus.FAILED)

    row = await fetch_row(async_engine, request_id)
    assert row.status == RequestStatus.COMPLETED
    assert types.SampleOutput.model_validate_json(row.result_data) == SAMPLE_RESULT


@pytest.mark.asyncio
async def test_fail_orphaned_only_touches_pending_external_rows(async_engine, writer, store):
    orphaned = await writer.submit(insert_future_op())
    completed = await writer.submit(insert_future_op())
    await store.complete(completed, SAMPLE_RESULT, RequestStatus.COMPLETED)
    engine_owned = await writer.submit(insert_future_op(request_type=types.RequestType.SAMPLE))

    assert await store.fail_orphaned() == 1

    row = await fetch_row(async_engine, orphaned)
    assert row.status == RequestStatus.FAILED
    # The stored payload must decode as the ErrorResponse retrieve_future 400s with.
    assert "restarted" in types.ErrorResponse.model_validate_json(row.result_data).error
    assert (await fetch_row(async_engine, completed)).status == RequestStatus.COMPLETED
    assert (await fetch_row(async_engine, engine_owned)).status == RequestStatus.PENDING


def _stub_request(async_engine, writer):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(db_engine=async_engine, future_writer=writer)))


def optim_input(learning_rate: float = 1e-4) -> types.OptimStepInput:
    return types.OptimStepInput(
        adam_params=types.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0)
    )


@pytest.mark.asyncio
async def test_submit_future_is_idempotent_for_retried_seq_ids(async_engine, writer):
    request = _stub_request(async_engine, writer)
    request_data = optim_input()

    async with AsyncSession(async_engine) as session:
        first = await submit_future(request, session, types.RequestType.OPTIM_STEP, "model_1", request_data, seq_id=7)
    async with AsyncSession(async_engine) as session:
        retry = await submit_future(request, session, types.RequestType.OPTIM_STEP, "model_1", request_data, seq_id=7)
    assert first == retry

    # Reusing the seq_id with a different payload is a client bug -> 409.
    other = optim_input(learning_rate=2e-4)
    async with AsyncSession(async_engine) as session:
        with pytest.raises(HTTPException) as excinfo:
            await submit_future(request, session, types.RequestType.OPTIM_STEP, "model_1", other, seq_id=7)
    assert excinfo.value.status_code == 409


@pytest.mark.asyncio
async def test_submit_future_concurrent_duplicates_converge_on_one_id(async_engine, writer):
    """Simultaneous SDK retries of one request race past the pre-insert lookup;
    the unique constraint plus the fresh-session retry must converge them."""
    request = _stub_request(async_engine, writer)
    request_data = optim_input()

    async def one_submission() -> int:
        async with AsyncSession(async_engine) as session:
            return await submit_future(
                request, session, types.RequestType.OPTIM_STEP, "model_1", request_data, seq_id=3
            )

    ids = await asyncio.gather(*(one_submission() for _ in range(8)))
    assert len(set(ids)) == 1


@pytest.mark.asyncio
async def test_1024_concurrent_requests_on_a_two_connection_pool(async_engine, writer, store):
    """The headline scenario: a 1024-way submit+complete burst against a pool
    of two connections finishes with every future durable and terminal."""

    async def one_request(i: int) -> int:
        request_id = await writer.submit(insert_future_op(model_id=f"model_{i % 4}"))
        await store.complete(request_id, SAMPLE_RESULT, RequestStatus.COMPLETED)
        return request_id

    ids = await asyncio.gather(*(one_request(i) for i in range(1024)))

    assert len(set(ids)) == 1024
    async with AsyncSession(async_engine) as session:
        completed = (
            await session.exec(select(FutureDB.request_id).where(FutureDB.status == RequestStatus.COMPLETED))
        ).all()
    assert set(completed) == set(ids)
