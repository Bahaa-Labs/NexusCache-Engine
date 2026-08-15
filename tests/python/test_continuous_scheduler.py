"""
Tests prefill vs. decode phase separation, KV-cache capacity constraints,
and preemption/eviction handling using PyTest fixtures.
"""

import pytest
import torch

import nexuscache._C as _C
from nexuscache.server.scheduler import Scheduler, SchedulerConfig, SequenceState


@pytest.fixture
def allocator_config() -> _C.BlockAllocatorConfig:
    """Fixture providing a small BlockAllocatorConfig to test block limits."""
    config = _C.BlockAllocatorConfig()
    config.num_blocks = 8  # Small block count to easily test preemption/exhaustion
    config.block_size = 4
    config.num_layers = 2
    config.num_heads = 4
    config.head_dim = 32
    config.dtype = torch.float16
    config.device_id = 0
    return config


@pytest.fixture
def cpp_subsystem(allocator_config: _C.BlockAllocatorConfig):
    """Fixture initializing C++ BlockManager and PageTable instances."""
    block_manager = _C.BlockManager(allocator_config)
    page_table = _C.PageTable(block_manager, allocator_config.block_size)
    return block_manager, page_table


@pytest.fixture
def scheduler(cpp_subsystem, allocator_config: _C.BlockAllocatorConfig) -> Scheduler:
    """Fixture initializing the continuous batching Scheduler."""
    block_manager, page_table = cpp_subsystem
    sched_config = SchedulerConfig(
        max_num_batched_tokens=64,
        max_num_seqs=4,
        max_paged_blocks=allocator_config.num_blocks,
        block_size=allocator_config.block_size,
    )
    return Scheduler(sched_config, block_manager, page_table)


class TestContinuousScheduler:

    def test_add_sequence_and_initial_state(self, scheduler: Scheduler):
        """Verify that adding sequences correctly populates waiting queues and sequence maps."""
        seq_id = scheduler.add_sequence("req_1", [101, 102, 103], max_new_tokens=5)

        assert seq_id in scheduler.sequence_map
        assert len(scheduler.waiting_queue) == 1
        assert scheduler.sequence_map[seq_id].state == SequenceState.WAITING
        assert scheduler.sequence_map[seq_id].required_num_blocks == 1

    def test_prefill_then_decode_transition(self, scheduler: Scheduler, cpp_subsystem):
        """Verify that a fresh sequence runs prefill in step 1 and transitions to decode in step 2."""
        _, page_table = cpp_subsystem

        # Add 2 sequences requiring 2 blocks and 1 block respectively
        s1 = scheduler.add_sequence("req_1", [1, 2, 3, 4, 5], max_new_tokens=3)
        s2 = scheduler.add_sequence("req_2", [6, 7], max_new_tokens=2)

        # Iteration 1: Prefill Phase
        batch_1 = scheduler.schedule()
        assert len(batch_1.prefill_seqs) == 2
        assert len(batch_1.decode_seqs) == 0
        assert scheduler.sequence_map[s1].state == SequenceState.PREFILL
        assert page_table.has_sequence(s1)
        assert page_table.has_sequence(s2)

        # Iteration 2: Transition into Decode Phase
        batch_2 = scheduler.schedule()
        assert len(batch_2.prefill_seqs) == 0
        assert len(batch_2.decode_seqs) == 2
        assert scheduler.sequence_map[s1].state == SequenceState.DECODE
        assert scheduler.sequence_map[s2].state == SequenceState.DECODE

    def test_sequence_completion_cleans_up_cpp_memory(
        self, scheduler: Scheduler, cpp_subsystem
    ):
        """Verify that completed sequences free their C++ block allocations."""
        block_manager, page_table = cpp_subsystem
        initial_free_blocks = block_manager.get_num_free_blocks()

        s1 = scheduler.add_sequence("req_short", [1, 2, 3], max_new_tokens=1)

        # Step 1: Prefill
        scheduler.schedule()
        assert block_manager.get_num_free_blocks() < initial_free_blocks

        # Simulate generating the single output token
        scheduler.sequence_map[s1].generated_token_ids.append(99)

        # Step 2: Decode -> Execution sees sequence finished and frees memory
        scheduler.schedule()

        assert not page_table.has_sequence(s1)
        assert block_manager.get_num_free_blocks() == initial_free_blocks
        assert scheduler.get_num_unfinished_sequences() == 0

    def test_kv_exhaustion_triggers_preemption(
        self, scheduler: Scheduler, cpp_subsystem
    ):
        """Verify that running out of C++ free blocks triggers sequence preemption."""
        block_manager, page_table = cpp_subsystem

        # Populate scheduler to fill 7 out of 8 available blocks
        # Prompt len 16 requires 4 blocks (block_size=4)
        scheduler.add_sequence("req_large_1", list(range(16)), max_new_tokens=10)
        # Prompt len 12 requires 3 blocks
        scheduler.add_sequence("req_large_2", list(range(12)), max_new_tokens=10)

        # Step 1: Prefill both sequences (Total blocks consumed: 7/8)
        scheduler.schedule()
        assert block_manager.get_num_free_blocks() == 1

        # Force decode iterations until s1 needs a new block beyond block boundary
        # s1 currently has 16 tokens (exactly 4 blocks full). Appending token 17 requires block #5.
        # s2 currently has 12 tokens (3 blocks full). Appending token 13 requires block #4.
        # Total needed: 2 new blocks, but only 1 free block exists!

        batch = scheduler.schedule()

        # One of the sequences must be preempted to accommodate the other
        assert len(batch.preempted_seqs) > 0 or len(scheduler.preempted_queue) > 0
        assert len(scheduler.running_queue) < 2
