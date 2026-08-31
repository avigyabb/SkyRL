"""Durable, low-contention persistence for Tinker futures.

``FutureDB`` is the message bus between the API and engine processes, and under
rollout load (hundreds to thousands of concurrent ``/asample`` and
``/forward_backward`` submissions, each a single-row write transaction)
SQLite's single-writer lock serializes every write while each waiting write
holds a checked-out pool connection. The pool exhausts and unrelated requests
(heartbeats, retrieves) fail with 500s long before SQLite runs out of raw
write throughput.

This module keeps the database as the store -- every accepted request is a
committed row before its id reaches the client, so state survives an API
restart and stays visible to the engine process -- and removes the throttling
instead of the durability:

- :class:`GroupCommitWriter` funnels hot-path write transactions through one
  dedicated task that commits them in batches (group commit). N concurrent
  writes cost N/batch_size commits, and a queued write holds no pool
  connection, so readers and the future poller always find the pool free.
- :class:`ExternalFutureStore` sits on top of the writer for the sample
  forwarding path: terminal-result persistence for ``RequestType.EXTERNAL``
  futures and crash recovery for forwards that died with a previous server
  process.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from pydantic import BaseModel
from sqlmodel import update
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.db_models import FutureDB, RequestStatus
from skyrl.utils.log import logger

# One batch is one database transaction. Large enough to collapse a 1024-way
# submission burst into a handful of commits, small enough that a batch of
# multi-MB sample results stays a bounded allocation.
GROUP_COMMIT_MAX_BATCH_SIZE = 64

# Submissions past this depth wait for queue space instead of stacking up
# unboundedly; the queue is the only buffering between accept and commit.
GROUP_COMMIT_MAX_QUEUE_SIZE = 4096

# A write op stages changes on the session it is handed (it may flush, e.g. to
# obtain an autoincrement id) but must not commit; the writer owns the commit.
WriteOp = Callable[[AsyncSession], Awaitable[Any]]


@dataclass
class _PendingWrite:
    op: WriteOp
    result: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())


class GroupCommitWriter:
    """Commits write transactions in batches through one dedicated task.

    ``submit`` resolves only after the op's changes are committed, so callers
    get the same durability as committing themselves -- what changes is the
    cost: contended single-row transactions become shared batch commits, and
    waiting happens on an asyncio queue rather than on a checked-out
    connection blocked behind the database's write lock.

    Each op runs inside its own savepoint, so one failing op (e.g. a unique
    constraint hit by a concurrent SDK retry) is rolled back and reported to
    its caller alone; the rest of the batch still commits. If the batch
    commit itself fails, every op is retried once in its own transaction so a
    transient error cannot fail unrelated writes.
    """

    def __init__(
        self,
        db_engine,
        *,
        max_batch_size: int = GROUP_COMMIT_MAX_BATCH_SIZE,
        max_queue_size: int = GROUP_COMMIT_MAX_QUEUE_SIZE,
    ):
        self.db_engine = db_engine
        self._max_batch_size = max_batch_size
        self._queue: asyncio.Queue[_PendingWrite] = asyncio.Queue(maxsize=max_queue_size)
        self._task: asyncio.Task | None = None
        self._closed = False

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        """Commit everything already accepted, then stop."""
        self._closed = True
        await self._queue.join()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def submit(self, op: WriteOp) -> Any:
        """Run ``op`` in a batched transaction, returning its value once committed."""
        if self._closed:
            raise RuntimeError("GroupCommitWriter is closed")
        pending = _PendingWrite(op=op)
        await self._queue.put(pending)
        return await pending.result

    async def _run(self) -> None:
        while True:
            batch = [await self._queue.get()]
            while len(batch) < self._max_batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                await self._commit_batch(batch)
            except asyncio.CancelledError:
                raise
            except Exception:
                # _commit_batch resolves per-op outcomes itself; anything that
                # escapes is a bug, and the writer must outlive it. Ops left
                # unresolved fall back on their callers' timeouts.
                logger.exception("Group-commit batch failed unexpectedly")
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def _commit_batch(self, batch: list[_PendingWrite]) -> None:
        outcomes: list[Any] = []
        try:
            async with AsyncSession(self.db_engine) as session:
                for pending in batch:
                    try:
                        # Savepoint per op: a failing op rolls back only itself.
                        async with session.begin_nested():
                            outcomes.append(await pending.op(session))
                    except Exception as error:
                        outcomes.append(error)
                await session.commit()
        except Exception:
            # The shared commit failed (transient database error, lock timeout):
            # nothing persisted, so retry each op alone to keep failures scoped.
            await self._commit_individually(batch)
            return
        for pending, outcome in zip(batch, outcomes):
            self._resolve(pending, outcome)

    async def _commit_individually(self, batch: list[_PendingWrite]) -> None:
        for pending in batch:
            try:
                async with AsyncSession(self.db_engine) as session:
                    value = await pending.op(session)
                    await session.commit()
            except Exception as error:
                self._resolve(pending, error)
            else:
                self._resolve(pending, value)

    @staticmethod
    def _resolve(pending: _PendingWrite, outcome: Any) -> None:
        # The caller may have given up (client disconnect); its future is then
        # cancelled but the write is already committed, which is fine -- the
        # row exists for the SDK's retry to find.
        if pending.result.done():
            return
        if isinstance(outcome, BaseException):
            pending.result.set_exception(outcome)
        else:
            pending.result.set_result(outcome)


class ExternalFutureStore:
    """Persistence for forwarded (``RequestType.EXTERNAL``) sample futures.

    Submission rows are created through the caller's :class:`GroupCommitWriter`
    path; this class owns the other end of the lifecycle: terminal-result
    writes from the inference forwarding tasks, and recovery for rows whose
    forwarding task died with a previous server process.
    """

    def __init__(self, db_engine, writer: GroupCommitWriter):
        self.db_engine = db_engine
        self.writer = writer

    async def complete(self, request_id: int, result: BaseModel, status: RequestStatus) -> None:
        """Persist a terminal result, leaving already-terminal rows untouched."""
        # Serialize outside the writer so the batch transaction stays short.
        result_json = result.model_dump_json()
        completed_at = datetime.now(timezone.utc)

        async def op(session: AsyncSession) -> int:
            outcome = await session.exec(
                update(FutureDB)
                .where(FutureDB.request_id == request_id)
                .where(FutureDB.status == RequestStatus.PENDING)
                .values(result_data=result_json, status=status, completed_at=completed_at)
            )
            return outcome.rowcount

        if not await self.writer.submit(op):
            # Row deleted between scheduling and completion (cancelled request,
            # stale-session GC) or already terminal (duplicate completion).
            logger.warning("FutureDB row %s not pending on completion write — skipping", request_id)

    async def fail_orphaned(self) -> int:
        """Fail EXTERNAL rows whose forwarding task died with a previous server.

        A forwarded sample lives as a PENDING row plus an in-flight HTTP task in
        the API process; if the process dies, the row would stay PENDING forever
        and its retrieve_future callers would 408-loop. Run once at startup,
        before requests are served, so every PENDING EXTERNAL row is known to be
        orphaned (this assumes the single-API-server deployment the engine
        subprocess model already implies).
        """
        error = types.ErrorResponse(
            error="API server restarted before the forwarded sample completed; resubmit the request",
            status="failed",
        )
        async with AsyncSession(self.db_engine) as session:
            outcome = await session.exec(
                update(FutureDB)
                .where(FutureDB.request_type == types.RequestType.EXTERNAL)
                .where(FutureDB.status == RequestStatus.PENDING)
                .values(
                    result_data=error.model_dump_json(),
                    status=RequestStatus.FAILED,
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
        if outcome.rowcount:
            logger.warning("Failed %d orphaned forwarded sample futures from a previous server run", outcome.rowcount)
        return outcome.rowcount
