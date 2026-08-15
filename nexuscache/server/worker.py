"""
High-throughput asynchronous Ray Actor bound to dedicated GPU resources.
Integrates PyTorch CUDA execution streams with C++ BlockManager/PageTable core memory engine.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

import ray
import torch

# Native C++ extension imported from build
import nexuscache._C as _C

logger = logging.getLogger("nexuscache.worker")


class RequestStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


@dataclass
class SequenceRequest:
    # Internal state wrapper for an incoming inference generation sequence.
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0

    # Dynamic runtime tracking
    generated_token_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    arrival_time: float = field(default_factory=time.perf_counter)
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    @property
    def total_length(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)


@dataclass
class WorkerStats:
    # Telemetry snapshot for Ray actor cluster status monitoring.
    worker_id: int
    gpu_id: int
    num_active_requests: int
    num_free_blocks: int
    num_allocated_blocks: int
    total_blocks: int
    vram_used_bytes: int
    vram_total_bytes: int


@ray.remote(num_gpus=1)
class InferenceWorker:

    def __init__(
        self,
        worker_id: int,
        num_blocks: int = 128,
        block_size: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float16,
        device_id: int = 0,
    ):
        self.worker_id = worker_id
        self.device_id = device_id

        # Create C++ Config directly inside the Ray Actor process scope
        self.config = _C.BlockAllocatorConfig()
        self.config.num_blocks = num_blocks
        self.config.block_size = block_size
        self.config.num_layers = num_layers
        self.config.num_heads = num_heads
        self.config.head_dim = head_dim
        self.config.dtype = dtype
        self.config.device_id = device_id

        # Pin actor process to target CUDA GPU device
        self.device = torch.device(f"cuda:{self.device_id}")
        torch.cuda.set_device(self.device)

        # Create isolated CUDA Stream for non-blocking execution
        self.cuda_stream = torch.cuda.Stream(device=self.device)

        # Instantiate C++ Native Subsystems locally
        logger.info(
            f"[Worker {self.worker_id}] Initializing C++ Memory Subsystem on {self.device}..."
        )
        self.block_manager = _C.BlockManager(self.config)
        self.page_table = _C.PageTable(self.block_manager, self.config.block_size)
        self.async_transfer = _C.AsyncTransferManager(self.device_id)

        # Async request queues & state tracking
        self.request_pool: dict[str, SequenceRequest] = {}
        self.waiting_queue: asyncio.Queue[SequenceRequest] = asyncio.Queue()
        self.active_requests: list[SequenceRequest] = []

        # Actor control loop flag
        self._is_running = False
        self._loop_task: asyncio.Task | None = None

    async def initialize(self) -> bool:
        """Warms up the CUDA context and starts the background step execution loop."""
        with torch.cuda.stream(self.cuda_stream):
            # Pre-allocate dummy tensor to force PyTorch CUDA context instantiation
            _ = torch.empty((1,), device=self.device)

        self._is_running = True
        self._loop_task = asyncio.create_task(self._continuous_batch_loop())
        logger.info(
            f"[Worker {self.worker_id}] Continuous batch loop started successfully."
        )
        return True

    async def generate_stream(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> AsyncGenerator[int, None]:
        """
        Public Ray API: Stream token generations back to client via AsyncGenerator.
        """
        req = SequenceRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        self.request_pool[request_id] = req
        await self.waiting_queue.put(req)

        try:
            while (
                req.status != RequestStatus.FINISHED
                and req.status != RequestStatus.FAILED
            ):
                token_id = await req.output_queue.get()
                if token_id is None:  # EOS or completion signal
                    break
                yield token_id
        finally:
            # Clean up allocations on client disconnect or completion
            await self._free_request(request_id)

    async def _continuous_batch_loop(self) -> None:
        """
        Background asyncio event loop driving continuous prefill/decode iterations.
        """
        while self._is_running:
            try:
                # 1. Promote waiting requests if free KV blocks are available
                while not self.waiting_queue.empty():
                    req = self.waiting_queue.get_nowait()
                    num_prompt_blocks = (
                        len(req.prompt_token_ids) + self.config.block_size - 1
                    ) // self.config.block_size

                    if self.block_manager.get_num_free_blocks() >= num_prompt_blocks:
                        seq_id = (
                            int(req.request_id, 16)
                            if req.request_id.isdigit()
                            else hash(req.request_id) % (2**31)
                        )
                        self.page_table.register_sequence(seq_id)
                        self.page_table.append_tokens(seq_id, len(req.prompt_token_ids))

                        req.status = RequestStatus.RUNNING
                        self.active_requests.append(req)
                    else:
                        # Requeue if KV space exhausted
                        await self.waiting_queue.put(req)
                        break

                if not self.active_requests:
                    await asyncio.sleep(
                        0.001
                    )  # Yield CPU to avoid tight busy-wait loop
                    continue

                # 2. Execute forward step asynchronously on CUDA stream
                await self._execute_step()

            except Exception as e:
                logger.error(
                    f"[Worker {self.worker_id}] Error in batch loop: {e}", exc_info=True
                )
                await asyncio.sleep(0.01)

    async def _execute_step(self) -> None:
        """Executes a single forward token generation iteration for active requests."""
        with torch.cuda.stream(self.cuda_stream):
            # Extract sequence IDs for slot mapping tensor construct
            active_seq_ids = [
                (
                    int(req.request_id, 16)
                    if req.request_id.isdigit()
                    else hash(req.request_id) % (2**31)
                )
                for req in self.active_requests
            ]
            query_lens = [
                1 if req.generated_token_ids else len(req.prompt_token_ids)
                for req in self.active_requests
            ]

            # Native C++ call: Generate block table & slot mapping CUDA tensors
            self.page_table.get_slot_mapping_tensor(
                active_seq_ids, query_lens, device="cuda"
            )

            # SIMULATED FORWARD PASS & SAMPLING
            # Replace with model.forward(inputs, slot_mapping) in Phase 4
            await asyncio.sleep(0.002)  # Non-blocking async compute yield

            finished_requests: list[SequenceRequest] = []

            for req in self.active_requests:
                # Mock token generation (e.g. echo increment or sample)
                next_token = (
                    req.prompt_token_ids[0] + len(req.generated_token_ids) + 1
                ) % 32000
                req.generated_token_ids.append(next_token)

                # Append newly generated token to C++ page table
                seq_id = (
                    int(req.request_id, 16)
                    if req.request_id.isdigit()
                    else hash(req.request_id) % (2**31)
                )
                self.page_table.append_tokens(seq_id, 1)

                # Push generated token to stream queue
                await req.output_queue.put(next_token)

                # Check completion conditions
                if len(req.generated_token_ids) >= req.max_new_tokens:
                    req.status = RequestStatus.FINISHED
                    await req.output_queue.put(None)  # EOS Sentinel
                    finished_requests.append(req)

            # Reclaim finished sequence memory
            for req in finished_requests:
                self.active_requests.remove(req)

    async def _free_request(self, request_id: str) -> None:
        # Reclaims C++ block allocations and frees tracking context.
        req = self.request_pool.pop(request_id, None)
        if req:
            seq_id = (
                int(request_id, 16)
                if request_id.isdigit()
                else hash(request_id) % (2**31)
            )
            if self.page_table.has_sequence(seq_id):
                self.page_table.free_sequence(seq_id)
            logger.debug(
                f"[Worker {self.worker_id}] Reclaimed KV allocations for req {request_id}"
            )

    async def get_stats(self) -> WorkerStats:
        # Fetches telemetry snapshot of VRAM usage and memory block allocations.
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return WorkerStats(
            worker_id=self.worker_id,
            gpu_id=self.device_id,
            num_active_requests=len(self.active_requests),
            num_free_blocks=self.block_manager.get_num_free_blocks(),
            num_allocated_blocks=self.block_manager.get_num_allocated_blocks(),
            total_blocks=self.block_manager.get_total_blocks(),
            vram_used_bytes=total_bytes - free_bytes,
            vram_total_bytes=total_bytes,
        )

    async def shutdown(self) -> None:
        # terminates worker task loop.
        self._is_running = False
        if self._loop_task:
            self._loop_task.cancel()
        logger.info(f"[Worker {self.worker_id}] Shutdown complete.")
