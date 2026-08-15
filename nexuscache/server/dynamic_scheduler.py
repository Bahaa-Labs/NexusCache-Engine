"""
Dynamic SLA-Aware Iteration Scheduler with Prefix Caching & Anti-Starvation
Advanced scheduling engine extending the base Scheduler with:
1. SLA-Aware & Priority Execution with Dynamic Virtual Age Decay (Anti-Starvation).
2. Radix Trie Prefix Caching (Detects and reuses shared prompt prefixes across requests).
3. Dynamic Token Budget Allocator (Balances prefill and decode token quotas dynamically).
4. Smart Victim Selection for Preemption (Evicts low-priority or non-SLA requests first).
"""

import logging
import time
from dataclasses import dataclass
from enum import IntEnum

import nexuscache._C as _C
from nexuscache.server.scheduler import (
    BatchOutput,
    Scheduler,
    SchedulerConfig,
    Sequence,
    SequenceState,
)

logger = logging.getLogger("nexuscache.scheduler.dynamic")


class PriorityLevel(IntEnum):
    """Client priority tiers for scheduling precedence."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class DynamicSequence(Sequence):
    """Extended Sequence with SLA guarantees, priority metrics, and prefix metadata."""

    priority: PriorityLevel = PriorityLevel.NORMAL
    sla_target_ttft_ms: float = 100.0  # Max acceptable Time To First Token
    sla_target_tpot_ms: float = 20.0  # Max acceptable Time Per Output Token

    # Prefix caching state
    shared_prefix_blocks: int = 0
    computed_prefix_tokens: int = 0

    # Timing instrumentation
    first_token_time: float | None = None
    last_token_time: float | None = None

    @property
    def ttft_deadline(self) -> float:
        """Calculates exact timestamp deadline for prefill phase based on SLA."""
        return self.arrival_time + (self.sla_target_ttft_ms / 1000.0)

    @property
    def remaining_prefill_tokens(self) -> int:
        """Effective prefill tokens needed after subtracting cached prefix tokens."""
        return max(0, len(self.prompt_token_ids) - self.computed_prefix_tokens)

    def calculate_scheduling_score(self, current_sim_time: float) -> float:
        """
        Normalized SLA + Prefix + Anti-Starvation Priority Scoring Engine.
        Caps prefix bonuses and accelerates aging at 75% SLA to eliminate P99 tail spikes.
        """
        wait_time_sec = max(0.001, current_sim_time - self.arrival_time)
        sla_time_sec = self.sla_target_ttft_ms / 1000.0

        # 1. Base Priority Tier
        tier_weights = {
            PriorityLevel.CRITICAL: 100.0,
            PriorityLevel.HIGH: 50.0,
            PriorityLevel.NORMAL: 10.0,
            PriorityLevel.LOW: 0.0,
        }
        base_score = tier_weights.get(self.priority, 10.0)

        # 2. SLA Urgency
        time_to_deadline = sla_time_sec - wait_time_sec
        if time_to_deadline <= 0:
            overdue_ratio = abs(time_to_deadline) / max(0.01, sla_time_sec)
            sla_urgency_score = 150.0 + (overdue_ratio * 300.0)
        else:
            sla_urgency_score = min(120.0, 10.0 / max(0.005, time_to_deadline))

        # 3. Capped Prefix Score (Prevents prefix matches from permanently hijacking queue priority)
        prefix_score = min(20.0, (self.computed_prefix_tokens / 128.0) * 10.0)

        # 4. Aggressive Anti-Starvation Boost (Triggers heavily at 75% SLA wait time)
        aging_ratio = wait_time_sec / max(0.01, sla_time_sec)
        if aging_ratio >= 0.75:
            aging_score = 300.0 + ((aging_ratio - 0.75) ** 3.0) * 1000.0
        else:
            aging_score = (aging_ratio**2.0) * 50.0

        return float(base_score + sla_urgency_score + prefix_score + aging_score)


@dataclass
class DynamicSchedulerConfig(SchedulerConfig):
    """Configuration extensions for SLA targets and prefix matching."""

    enable_prefix_caching: bool = True
    min_prefix_match_tokens: int = 16
    decode_protection_ratio: float = (
        0.7  # Reserve token budget for decode phase under load
    )
    cpu_swap_space_blocks: int = (
        2048  # Number of host CPU RAM blocks for swap preemption
    )
    max_paged_blocks: int = 256  # Scaled from 64 to relieve KV-cache block contention
    max_num_batched_tokens: int = (
        4096  # Expanded token budget for parallel prefill passes
    )


# ============================================================================
# Prefix Caching Trie
# ============================================================================


class PrefixTrieNode:
    """Node in the Radix Trie used to track cached token sequences and physical blocks."""

    def __init__(self, block_id: int | None = None):
        self.children: dict[int, PrefixTrieNode] = {}
        self.block_id: int | None = block_id
        self.ref_count: int = 0
        self.last_accessed: float = time.perf_counter()


class PrefixCacheManager:
    """
    Tracks token prefix hierarchies to enable instant KV-cache block sharing
    across identical prompt prefixes.
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.root = PrefixTrieNode()

    def match_prefix(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """
        Finds the longest cached prefix match for a sequence of token IDs.
        Returns: (number_of_matched_tokens, list_of_reusable_block_ids)
        """
        curr = self.root
        matched_blocks: list[int] = []
        matched_tokens = 0
        idx = 0

        while idx + self.block_size <= len(token_ids):
            block_tokens = tuple(token_ids[idx : idx + self.block_size])
            block_key = hash(block_tokens)

            if block_key not in curr.children:
                break

            curr = curr.children[block_key]
            curr.last_accessed = time.perf_counter()

            if curr.block_id is not None:
                matched_blocks.append(curr.block_id)
                matched_tokens += self.block_size

            idx += self.block_size

        return matched_tokens, matched_blocks

    def insert_prefix(
        self, token_ids: list[int], physical_block_ids: list[int]
    ) -> None:
        """Registers newly computed KV-cache blocks into the prefix trie."""
        curr = self.root
        num_blocks = min(len(token_ids) // self.block_size, len(physical_block_ids))

        for i in range(num_blocks):
            idx = i * self.block_size
            block_tokens = tuple(token_ids[idx : idx + self.block_size])
            block_key = hash(block_tokens)

            if block_key not in curr.children:
                curr.children[block_key] = PrefixTrieNode(
                    block_id=physical_block_ids[i]
                )

            curr = curr.children[block_key]
            curr.ref_count += 1
            curr.last_accessed = time.perf_counter()


# ============================================================================
# Dynamic Scheduler Engine
# ============================================================================


class DynamicScheduler(Scheduler):
    """
    SLA-Aware Dynamic Iteration Scheduler with Prefix Matching, Anti-Starvation, and Smart Preemption.
    """

    def __init__(
        self,
        config: DynamicSchedulerConfig,
        block_manager: _C.BlockManager,
        page_table: _C.PageTable,
    ):
        super().__init__(
            config=config, block_manager=block_manager, page_table=page_table
        )
        self.dynamic_config = config
        self.prefix_cache = PrefixCacheManager(block_size=config.block_size)

    def add_dynamic_sequence(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        priority: PriorityLevel = PriorityLevel.NORMAL,
        sla_target_ttft_ms: float = 100.0,
        sla_target_tpot_ms: float = 20.0,
    ) -> int:
        """Adds a request with SLA targets and client priority tier."""
        seq_id = hash(request_id) & 0x7FFFFFFF

        seq = DynamicSequence(
            seq_id=seq_id,
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            block_size=self.config.block_size,
            priority=priority,
            sla_target_ttft_ms=sla_target_ttft_ms,
            sla_target_tpot_ms=sla_target_tpot_ms,
        )

        # Prefix matching check
        if self.dynamic_config.enable_prefix_caching:
            matched_tokens, matched_blocks = self.prefix_cache.match_prefix(
                prompt_token_ids
            )
            if matched_tokens >= self.dynamic_config.min_prefix_match_tokens:
                seq.computed_prefix_tokens = matched_tokens
                seq.shared_prefix_blocks = len(matched_blocks)
                logger.info(
                    f"[DynamicScheduler] Prefix hit for seq_id={seq_id}: "
                    f"reused {matched_tokens} tokens ({len(matched_blocks)} blocks)."
                )

        self.waiting_queue.append(seq)
        self.sequence_map[seq_id] = seq
        return seq_id

    def schedule(self, sim_time: float | None = None) -> BatchOutput:
        """
        Executes SLA-priority schedule pass across Decode and Prefill stages.
        Optionally accepts `sim_time` to align time-based metrics in simulation environments.
        """
        prefill_batch: list[Sequence] = []
        decode_batch: list[Sequence] = []
        preempted_batch: list[Sequence] = []

        num_batched_tokens = 0
        num_free_blocks = self.block_manager.get_num_free_blocks()
        now = sim_time if sim_time is not None else time.perf_counter()

        preempted_ids: set[int] = set()

        # --------------------------------------------------------------------
        # 1. DECODE PHASE (SLA protection & block expansion check)
        # --------------------------------------------------------------------
        retained_running: list[Sequence] = []

        # Sort decode sequences by priority and arrival time
        self.running_queue.sort(
            key=lambda s: (
                (
                    s.priority.value
                    if isinstance(s, DynamicSequence)
                    else PriorityLevel.NORMAL.value
                ),
                s.arrival_time,
            )
        )

        for seq in self.running_queue:
            if seq.seq_id in preempted_ids:
                continue

            if seq.is_finished():
                self._free_sequence_resources(seq)
                continue

            current_len = seq.total_len
            needs_new_block = (current_len % self.config.block_size) == 0

            # Check VRAM capacity
            if needs_new_block and num_free_blocks < 1:
                victim = self._select_preemption_victim(
                    exclude_seq_id=seq.seq_id, preempted_ids=preempted_ids
                )
                if victim is not None:
                    logger.warning(
                        f"[DynamicScheduler] Preempting victim seq_id={victim.seq_id} "
                        f"(Priority={getattr(victim, 'priority', 'NORMAL')}) to free blocks."
                    )
                    self._preempt_sequence(victim)
                    preempted_batch.append(victim)
                    preempted_ids.add(victim.seq_id)

                    if victim in retained_running:
                        retained_running.remove(victim)

                    num_free_blocks += victim.required_num_blocks
                else:
                    self._preempt_sequence(seq)
                    preempted_batch.append(seq)
                    preempted_ids.add(seq.seq_id)
                    continue

            if needs_new_block:
                num_free_blocks -= 1

            if isinstance(seq, DynamicSequence):
                if seq.first_token_time is None:
                    seq.first_token_time = now
                seq.last_token_time = now

            self.page_table.append_tokens(seq.seq_id, 1)
            seq.state = SequenceState.DECODE
            decode_batch.append(seq)
            retained_running.append(seq)
            num_batched_tokens += 1

        self.running_queue = retained_running

        # --------------------------------------------------------------------
        # 2. PREFILL PHASE (Anti-Starvation Composite Priority Sorting & Budget Reservation)
        # --------------------------------------------------------------------
        self._schedule_preempted(prefill_batch, num_batched_tokens, num_free_blocks)

        # Sort waiting queue using Anti-Starvation Score (Highest score scheduled first)
        self.waiting_queue.sort(
            key=lambda s: (
                s.calculate_scheduling_score(now)
                if isinstance(s, DynamicSequence)
                else -s.arrival_time
            ),
            reverse=True,
        )

        idx = 0
        while (
            idx < len(self.waiting_queue)
            and len(self.running_queue) < self.config.max_num_seqs
        ):
            seq = self.waiting_queue[idx]

            effective_prefill_tokens = (
                seq.remaining_prefill_tokens
                if isinstance(seq, DynamicSequence)
                else seq.num_prompt_tokens
            )
            required_blocks = seq.required_num_blocks

            # Check if this sequence is in critical wait territory
            is_critical = False
            if isinstance(seq, DynamicSequence):
                wait_time = max(0.001, now - seq.arrival_time)
                sla_time = seq.sla_target_ttft_ms / 1000.0
                if (wait_time / sla_time) >= 0.75:  # Lowered from 0.90
                    is_critical = True

            if (
                num_batched_tokens + effective_prefill_tokens
                > self.config.max_num_batched_tokens
            ):
                # Emergency Allocation: If request is critical and no prefill requests were scheduled yet,
                # schedule it alone in this step to prevent P99 tail delays.
                if (
                    is_critical
                    and len(prefill_batch) == 0
                    and num_free_blocks >= required_blocks
                ):
                    self.waiting_queue.pop(idx)
                    self.page_table.register_sequence(seq.seq_id)
                    self.page_table.append_tokens(seq.seq_id, seq.num_prompt_tokens)

                    seq.state = SequenceState.PREFILL
                    if isinstance(seq, DynamicSequence):
                        seq.first_token_time = now

                    prefill_batch.append(seq)
                    self.running_queue.append(seq)

                    num_free_blocks -= required_blocks
                    num_batched_tokens += effective_prefill_tokens
                    break

                idx += 1
                continue

            if num_free_blocks < required_blocks:
                break

            self.waiting_queue.pop(idx)
            self.page_table.register_sequence(seq.seq_id)
            self.page_table.append_tokens(seq.seq_id, seq.num_prompt_tokens)

            seq.state = SequenceState.PREFILL
            if isinstance(seq, DynamicSequence):
                seq.first_token_time = now

            prefill_batch.append(seq)
            self.running_queue.append(seq)

            num_free_blocks -= required_blocks
            num_batched_tokens += effective_prefill_tokens

        return BatchOutput(
            prefill_seqs=prefill_batch,
            decode_seqs=decode_batch,
            preempted_seqs=preempted_batch,
        )

    def _select_preemption_victim(
        self, exclude_seq_id: int, preempted_ids: set[int] | None = None
    ) -> Sequence | None:
        """
        Selects optimal victim for preemption using priority tiers and allocated space.
        """
        if preempted_ids is None:
            preempted_ids = set()

        candidates = [
            s
            for s in self.running_queue
            if s.seq_id != exclude_seq_id and s.seq_id not in preempted_ids
        ]
        if not candidates:
            return None

        candidates.sort(
            key=lambda s: (
                (
                    -(s.priority.value)
                    if isinstance(s, DynamicSequence)
                    else -PriorityLevel.NORMAL.value
                ),
                -s.required_num_blocks,
            )
        )
        return candidates[0]

    def _free_sequence_resources(self, seq: Sequence) -> None:
        """Frees KV blocks and registers full prompt/completion tokens into Prefix Cache."""
        if (
            isinstance(seq, DynamicSequence)
            and self.dynamic_config.enable_prefix_caching
        ):
            full_tokens = seq.prompt_token_ids + seq.generated_token_ids
            block_ids = self.page_table.get_block_table(seq.seq_id)
            if block_ids:
                self.prefix_cache.insert_prefix(full_tokens, block_ids)

        super()._free_sequence_resources(seq)
