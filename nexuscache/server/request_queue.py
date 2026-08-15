"""
Asynchronous Priority Request Queue with Backpressure and TTL Management
Provides thread-safe and coroutine-safe queuing for incoming LLM inference requests
before they are processed by the scheduler engine.

Features:
1. Multi-Tier SLA Priority Queuing (Min-Heap based on Priority Tier + Arrival Time).
2. Backpressure Management (Immediate rejection / QueueFullError -> HTTP 429).
3. Timeout & TTL Eviction (Drops stale requests before scheduling).
4. Async Cancellation (Handles client disconnects cleanly without ghost execution).
5. Queue Telemetry & Monitoring Metrics.
"""

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger("nexuscache.server.request_queue")


class PriorityLevel(IntEnum):
    """Client priority tiers for scheduling precedence."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class QueueFullError(Exception):
    """Exception raised when queue capacity is reached (mapped to HTTP 429)."""

    pass


class RequestTimeoutError(Exception):
    """Exception raised when a request spends too long in the queue."""

    pass


@dataclass
class QueuedRequest:
    """
    Wrapper for an incoming LLM inference request waiting in the queue.
    """

    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    priority: PriorityLevel = PriorityLevel.NORMAL
    sla_target_ttft_ms: float = 100.0
    sla_target_tpot_ms: float = 20.0

    # Timing & TTL
    arrival_time: float = field(default_factory=time.perf_counter)
    timeout_s: float = 30.0  # Client timeout threshold in queue

    # Event loop response future for returning completion stream/tokens
    response_future: asyncio.Future | None = None

    @property
    def is_expired(self) -> bool:
        """Checks whether the request has exceeded its allowed queue waiting TTL."""
        return (time.perf_counter() - self.arrival_time) > self.timeout_s

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)


@dataclass(order=True)
class PriorityHeapEntry:
    """
    Dataclass entry stored in the min-heap.
    Comparison order:
    1. Priority Level (0 = CRITICAL comes before 3 = LOW)
    2. Arrival Time (earlier timestamps come first -> FCFS within same tier)
    3. Sequence Counter (tie breaker)
    """

    priority: int
    arrival_time: float
    sequence_id: int
    request: QueuedRequest = field(compare=False)


@dataclass
class QueueMetrics:
    """Telemetry counters for monitoring request queue health."""

    total_enqueued: int = 0
    total_dequeued: int = 0
    total_rejected_429: int = 0
    total_timed_out: int = 0
    total_cancelled: int = 0


class RequestQueue:
    """
    Thread-safe, asynchronous priority request queue for LLM batching engines.
    """

    def __init__(
        self,
        max_queue_size: int = 1024,
        max_token_capacity: int = 262144,  # Max total queued prompt tokens allowed
        default_timeout_s: float = 30.0,
    ):
        self.max_queue_size = max_queue_size
        self.max_token_capacity = max_token_capacity
        self.default_timeout_s = default_timeout_s

        # Internal storage
        self._heap: list[PriorityHeapEntry] = []
        self._requests_map: dict[str, QueuedRequest] = {}
        self._cancelled_request_ids: set[str] = set()

        # State tracking
        self._current_token_count: int = 0
        self._sequence_counter: int = 0  # Tie-breaker for heap comparison
        self._metrics = QueueMetrics()

        # Async synchronization primitive
        self._condition = asyncio.Condition()

    # =========================================================================
    # Public Async API
    # =========================================================================

    async def put(self, request: QueuedRequest) -> None:
        """
        Enqueues an incoming request into the priority queue.

        Raises:
            QueueFullError: If queue size or prompt token budget exceeds capacity (HTTP 429 trigger).
        """
        async with self._condition:
            # 1. Backpressure Check
            if len(self._requests_map) >= self.max_queue_size:
                self._metrics.total_rejected_429 += 1
                logger.warning(
                    f"[RequestQueue] Rejection 429: Max queue capacity ({self.max_queue_size}) "
                    f"reached for request {request.request_id}."
                )
                raise QueueFullError(
                    f"Server queue is full ({self.max_queue_size} requests pending). Retry later."
                )

            if (
                self._current_token_count + request.num_prompt_tokens
                > self.max_token_capacity
            ):
                self._metrics.total_rejected_429 += 1
                logger.warning(
                    f"[RequestQueue] Rejection 429: Token memory budget exceeded "
                    f"({self._current_token_count}/{self.max_token_capacity}) for request {request.request_id}."
                )
                raise QueueFullError(
                    "Server token processing capacity exhausted. Retry later."
                )

            # 2. Assign default timeout if not set
            if request.timeout_s <= 0:
                request.timeout_s = self.default_timeout_s

            # 3. Heap Push
            self._sequence_counter += 1
            entry = PriorityHeapEntry(
                priority=request.priority.value,
                arrival_time=request.arrival_time,
                sequence_id=self._sequence_counter,
                request=request,
            )

            heapq.heappush(self._heap, entry)
            self._requests_map[request.request_id] = request
            self._current_token_count += request.num_prompt_tokens
            self._metrics.total_enqueued += 1

            logger.debug(
                f"[RequestQueue] Enqueued req_id={request.request_id}, priority={request.priority.name}, "
                f"tokens={request.num_prompt_tokens}. Queue depth={len(self._requests_map)}"
            )

            # Signal waiting scheduler worker loop
            self._condition.notify()

    async def get_batch(
        self, max_num_seqs: int, max_num_tokens: int
    ) -> list[QueuedRequest]:
        """
        Pulls a batch of valid requests ordered by priority and arrival time.
        Filters out expired and cancelled requests automatically.
        """
        async with self._condition:
            # Wait until at least one non-expired item is available
            while True:
                self._purge_expired_and_cancelled()

                if self._heap:
                    break  # Valid items available

                # Wait for new incoming requests
                await self._condition.wait()

            batch: list[QueuedRequest] = []
            batched_tokens = 0

            while self._heap and len(batch) < max_num_seqs:
                top_entry = self._heap[0]
                req = top_entry.request

                # Check if top item is cancelled or expired during batch extraction
                if req.request_id in self._cancelled_request_ids or req.is_expired:
                    self._pop_top_heap_entry()
                    continue

                # Check token batch budget
                if batched_tokens + req.num_prompt_tokens > max_num_tokens and batch:
                    # Token limit reached for this batch pass (unless batch is empty)
                    break

                # Valid candidate -> Extract
                popped_req = self._pop_top_heap_entry()
                batch.append(popped_req)
                batched_tokens += popped_req.num_prompt_tokens
                self._metrics.total_dequeued += 1

            return batch

    async def abort(self, request_id: str) -> bool:
        """
        Cancels a request (e.g., when client disconnects).
        Marks ID as cancelled for lazy removal during dequeue.
        """
        async with self._condition:
            if (
                request_id in self._requests_map
                and request_id not in self._cancelled_request_ids
            ):
                self._cancelled_request_ids.add(request_id)
                self._metrics.total_cancelled += 1
                logger.info(f"[RequestQueue] Request {request_id} aborted by client.")
                return True
            return False

    # =========================================================================
    # Internal Utility Methods
    # =========================================================================

    def _pop_top_heap_entry(self) -> QueuedRequest:
        """Pops and removes the top item from internal heap and index map."""
        entry = heapq.heappop(self._heap)
        req = entry.request

        self._requests_map.pop(req.request_id, None)
        self._cancelled_request_ids.discard(req.request_id)
        self._current_token_count = max(
            0, self._current_token_count - req.num_prompt_tokens
        )
        return req

    def _purge_expired_and_cancelled(self) -> None:
        """Cleans up stale / cancelled requests at the top of the heap."""
        while self._heap:
            top_req = self._heap[0].request

            if top_req.request_id in self._cancelled_request_ids:
                logger.debug(
                    f"[RequestQueue] Purging cancelled request: {top_req.request_id}"
                )
                self._pop_top_heap_entry()
                continue

            if top_req.is_expired:
                self._metrics.total_timed_out += 1
                logger.warning(
                    f"[RequestQueue] Dropping timed-out request {top_req.request_id} "
                    f"(waited {time.perf_counter() - top_req.arrival_time:.2f}s > {top_req.timeout_s}s)"
                )

                # Exception hook for client response task if future exists
                if top_req.response_future and not top_req.response_future.done():
                    top_req.response_future.set_exception(
                        RequestTimeoutError(
                            "Request timed out waiting in scheduler queue."
                        )
                    )

                self._pop_top_heap_entry()
                continue

            # Top item is valid
            break

    # =========================================================================
    # Telemetry & Properties
    # =========================================================================

    @property
    def size(self) -> int:
        """Current number of requests in queue."""
        return len(self._requests_map)

    @property
    def pending_tokens(self) -> int:
        """Total prompt tokens pending in queue."""
        return self._current_token_count

    def get_metrics(self) -> dict[str, int]:
        """Returns current telemetry metrics snapshot."""
        return {
            "current_queue_depth": len(self._requests_map),
            "pending_tokens": self._current_token_count,
            "total_enqueued": self._metrics.total_enqueued,
            "total_dequeued": self._metrics.total_dequeued,
            "total_rejected_429": self._metrics.total_rejected_429,
            "total_timed_out": self._metrics.total_timed_out,
            "total_cancelled": self._metrics.total_cancelled,
        }
