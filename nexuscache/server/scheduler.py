"""
Iteration-level scheduler handling prefill and decode batch construction.
Manages dynamic KV-cache block allocation, sequence preemption, and priority execution.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import nexuscache._C as _C

logger = logging.getLogger("nexuscache.scheduler")


class SequenceState(Enum):
    WAITING = "WAITING"
    PREFILL = "PREFILL"
    DECODE = "DECODE"
    PREEMPTED = "PREEMPTED"
    FINISHED = "FINISHED"


@dataclass
class Sequence:
    """Represents a single sequence managed by the scheduler."""

    seq_id: int
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    block_size: int
    arrival_time: float = field(default_factory=time.perf_counter)

    # Dynamic runtime state
    generated_token_ids: list[int] = field(default_factory=list)
    state: SequenceState = SequenceState.WAITING

    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def required_num_blocks(self) -> int:
        """Calculates total physical blocks needed for current sequence length."""
        return (self.total_len + self.block_size - 1) // self.block_size

    def is_finished(self) -> bool:
        return (
            len(self.generated_token_ids) >= self.max_new_tokens
            or self.state == SequenceState.FINISHED
        )


@dataclass
class SchedulerConfig:
    """Runtime limits and capacity bounds for continuous batching."""

    max_num_batched_tokens: int = (
        4096  # Max total tokens across prefill + decode in 1 step
    )
    max_num_seqs: int = 256  # Max active sequences in flight
    max_paged_blocks: int = 1024  # Total hardware VRAM cache blocks
    block_size: int = 16


@dataclass
class BatchOutput:
    """Constructed batch payload passed to C++ execution kernels."""

    prefill_seqs: list[Sequence]
    decode_seqs: list[Sequence]
    preempted_seqs: list[Sequence]

    @property
    def is_empty(self) -> bool:
        return not self.prefill_seqs and not self.decode_seqs


class Scheduler:
    """
    Iteration-Level Continuous Batching Scheduler.

    Co-batches prefill (context processing) and decode (token generation) iterations
    while maintaining exact block usage telemetry from C++ BlockManager/PageTable.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        block_manager: _C.BlockManager,
        page_table: _C.PageTable,
    ):
        self.config = config
        self.block_manager = block_manager
        self.page_table = page_table

        # Active queue structures
        self.waiting_queue: list[Sequence] = []
        self.running_queue: list[Sequence] = []
        self.preempted_queue: list[Sequence] = []
        self.sequence_map: dict[int, Sequence] = {}

    def add_sequence(
        self, request_id: str, prompt_token_ids: list[int], max_new_tokens: int
    ) -> int:
        """Adds a new client request into the scheduler waiting queue."""
        seq_id = (
            hash(request_id) & 0x7FFFFFFF
        )  # Positive 32-bit integer for C++ compatibility
        seq = Sequence(
            seq_id=seq_id,
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            block_size=self.config.block_size,
        )
        self.waiting_queue.append(seq)
        self.sequence_map[seq_id] = seq
        logger.debug(
            f"[Scheduler] Added seq_id={seq_id} (request_id={request_id}) to WAITING queue."
        )
        return seq_id

    def schedule(self) -> BatchOutput:
        """
        Main scheduling loop called before every model execution step.
        Determines which sequences run prefill and decode, managing block budget.
        """
        prefill_batch: list[Sequence] = []
        decode_batch: list[Sequence] = []
        preempted_batch: list[Sequence] = []

        num_batched_tokens = 0
        num_free_blocks = self.block_manager.get_num_free_blocks()

        # 1. DECODE PHASE: Prioritize existing active sequences (prevent starvation)

        retained_running: list[Sequence] = []

        for seq in self.running_queue:
            if seq.is_finished():
                self._free_sequence_resources(seq)
                continue

            # Check if appending 1 token requires a new physical block
            current_len = seq.total_len
            needs_new_block = (current_len % self.config.block_size) == 0

            if needs_new_block and num_free_blocks < 1:
                # VRAM full! Preempt this sequence (FIFO/LRU tail preemption)
                logger.warning(
                    f"[Scheduler] Preempting seq_id={seq.seq_id} due to KV-cache exhaustion."
                )
                self._preempt_sequence(seq)
                preempted_batch.append(seq)
                continue

            if needs_new_block:
                num_free_blocks -= 1

            # Append token in C++ PageTable
            self.page_table.append_tokens(seq.seq_id, 1)
            seq.state = SequenceState.DECODE
            decode_batch.append(seq)
            retained_running.append(seq)
            num_batched_tokens += 1

        self.running_queue = retained_running

        # 2. PREFILL PHASE: Promote waiting & preempted sequences into batch
        # First try to restore preempted sequences
        self._schedule_preempted(prefill_batch, num_batched_tokens, num_free_blocks)

        # Next, try to schedule fresh waiting requests
        while (
            self.waiting_queue
            and len(self.running_queue) + len(prefill_batch) < self.config.max_num_seqs
        ):
            seq = self.waiting_queue[0]
            prompt_len = seq.num_prompt_tokens
            required_blocks = seq.required_num_blocks

            # Check capacity constraints
            if num_batched_tokens + prompt_len > self.config.max_num_batched_tokens:
                break
            if num_free_blocks < required_blocks:
                break

            # Allocate in C++ Engine
            self.waiting_queue.pop(0)
            self.page_table.register_sequence(seq.seq_id)
            self.page_table.append_tokens(seq.seq_id, prompt_len)

            seq.state = SequenceState.PREFILL
            prefill_batch.append(seq)
            self.running_queue.append(seq)

            num_free_blocks -= required_blocks
            num_batched_tokens += prompt_len

        return BatchOutput(
            prefill_seqs=prefill_batch,
            decode_seqs=decode_batch,
            preempted_seqs=preempted_batch,
        )

    def _schedule_preempted(
        self,
        prefill_batch: list[Sequence],
        current_batched_tokens: int,
        num_free_blocks: int,
    ) -> None:
        """Attempts to re-schedule previously preempted sequences."""
        while (
            self.preempted_queue
            and len(self.running_queue) + len(prefill_batch) < self.config.max_num_seqs
        ):
            seq = self.preempted_queue[0]
            required_blocks = seq.required_num_blocks
            seq_tokens = seq.total_len

            if current_batched_tokens + seq_tokens > self.config.max_num_batched_tokens:
                break
            if num_free_blocks < required_blocks:
                break

            # Re-register sequence in C++ PageTable
            self.preempted_queue.pop(0)
            self.page_table.register_sequence(seq.seq_id)
            self.page_table.append_tokens(seq.seq_id, seq_tokens)

            seq.state = SequenceState.PREFILL  # Full re-prefill context recovery
            prefill_batch.append(seq)
            self.running_queue.append(seq)

            num_free_blocks -= required_blocks
            current_batched_tokens += seq_tokens

    def _preempt_sequence(self, seq: Sequence) -> None:
        """Evicts a sequence's KV allocations from GPU VRAM back to PREEMPTED state."""
        if self.page_table.has_sequence(seq.seq_id):
            self.page_table.free_sequence(seq.seq_id)
        seq.state = SequenceState.PREEMPTED
        self.preempted_queue.append(seq)

    def _free_sequence_resources(self, seq: Sequence) -> None:
        """Frees C++ physical cache blocks upon sequence completion."""
        if self.page_table.has_sequence(seq.seq_id):
            self.page_table.free_sequence(seq.seq_id)
        seq.state = SequenceState.FINISHED
        self.sequence_map.pop(seq.seq_id, None)
        logger.debug(
            f"[Scheduler] Reclaimed C++ KV blocks for finished seq_id={seq.seq_id}."
        )

    def finish_sequence(self, seq_id: int) -> None:
        """External client API to force cancel/finish a running sequence."""
        seq = self.sequence_map.get(seq_id)
        if seq:
            self._free_sequence_resources(seq)
            if seq in self.running_queue:
                self.running_queue.remove(seq)
            if seq in self.waiting_queue:
                self.waiting_queue.remove(seq)

    def get_num_unfinished_sequences(self) -> int:
        return len(self.sequence_map)
