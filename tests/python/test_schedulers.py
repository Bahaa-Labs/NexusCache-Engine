"""
Unit and Integration Test Suite for NexusCache Schedulers
==========================================================
Tests continuous batching, preemption, prefix caching, and SLA guarantees
for both `scheduler.py` and `dynamic_scheduler.py`.
"""

from typing import cast
from unittest.mock import MagicMock

import pytest

from nexuscache.server.dynamic_scheduler import (
    DynamicScheduler,
    DynamicSchedulerConfig,
    DynamicSequence,
    PriorityLevel,
)
from nexuscache.server.scheduler import (
    Scheduler,
    SchedulerConfig,
    SequenceState,
)

# ============================================================================
# Mocks & Fixtures
# ============================================================================


@pytest.fixture
def mock_cpp_engine():
    """Mocks low-level C++ BlockManager and PageTable components."""
    block_manager = MagicMock()
    page_table = MagicMock()

    # Default VRAM state: 100 free blocks available
    block_manager.get_num_free_blocks.return_value = 100
    page_table.has_sequence.return_value = True
    page_table.get_block_table.return_value = [10, 11, 12]

    return block_manager, page_table


@pytest.fixture
def base_config():
    return SchedulerConfig(
        max_num_batched_tokens=128,
        max_num_seqs=4,
        max_paged_blocks=100,
        block_size=16,
    )


@pytest.fixture
def dynamic_config():
    return DynamicSchedulerConfig(
        max_num_batched_tokens=128,
        max_num_seqs=4,
        max_paged_blocks=100,
        block_size=16,
        enable_prefix_caching=True,
        min_prefix_match_tokens=16,
    )


# ============================================================================
# 1. Tests for Base Scheduler (`scheduler.py`)
# ============================================================================


def test_add_sequence(mock_cpp_engine, base_config):
    """Test adding new requests to the waiting queue."""
    bm, pt = mock_cpp_engine
    scheduler = Scheduler(base_config, bm, pt)

    seq_id = scheduler.add_sequence(
        "req-1", prompt_token_ids=[1, 2, 3, 4], max_new_tokens=10
    )

    assert seq_id in scheduler.sequence_map
    assert len(scheduler.waiting_queue) == 1
    assert scheduler.waiting_queue[0].request_id == "req-1"


def test_prefill_and_decode_cycle(mock_cpp_engine, base_config):
    """Test standard continuous batching: Prefill -> Decode transition."""
    bm, pt = mock_cpp_engine
    scheduler = Scheduler(base_config, bm, pt)

    # Add 2 requests
    scheduler.add_sequence(
        "req-1", prompt_token_ids=[10] * 32, max_new_tokens=2
    )  # Needs 2 blocks
    scheduler.add_sequence(
        "req-2", prompt_token_ids=[20] * 16, max_new_tokens=2
    )  # Needs 1 block

    # Iteration 1: Should run PREFILL for both
    batch1 = scheduler.schedule()
    assert len(batch1.prefill_seqs) == 2
    assert len(batch1.decode_seqs) == 0
    assert batch1.prefill_seqs[0].state == SequenceState.PREFILL

    # Simulate token generation step by updating sequence objects
    for seq in scheduler.running_queue:
        seq.generated_token_ids.append(999)

    # Iteration 2: Should transition to DECODE phase
    batch2 = scheduler.schedule()
    assert len(batch2.prefill_seqs) == 0
    assert len(batch2.decode_seqs) == 2
    assert batch2.decode_seqs[0].state == SequenceState.DECODE


def test_vram_oom_preemption(mock_cpp_engine, base_config):
    """Test that sequences are preempted when VRAM runs out of blocks."""
    bm, pt = mock_cpp_engine
    scheduler = Scheduler(base_config, bm, pt)

    # Setup scenario where only 1 free block remains
    bm.get_num_free_blocks.return_value = 1

    # Add a sequence that is currently at a block boundary (needs a new block on next token)
    seq_id = scheduler.add_sequence(
        "req-oom", prompt_token_ids=[1] * 16, max_new_tokens=10
    )

    # Run prefill first
    scheduler.schedule()

    # Exhaust all free blocks to 0
    bm.get_num_free_blocks.return_value = 0

    # Next iteration decode attempt should trigger preemption
    batch = scheduler.schedule()
    assert len(batch.preempted_seqs) == 1
    assert batch.preempted_seqs[0].seq_id == seq_id
    assert batch.preempted_seqs[0].state == SequenceState.PREEMPTED


# ============================================================================
# 2. Tests for Dynamic Scheduler (`dynamic_scheduler.py`)
# ============================================================================


def test_sla_priority_sorting(mock_cpp_engine, dynamic_config):
    """Test that CRITICAL/HIGH priority requests jump ahead of LOW priority requests."""
    bm, pt = mock_cpp_engine
    scheduler = DynamicScheduler(dynamic_config, bm, pt)

    # Add low-priority request first
    scheduler.add_dynamic_sequence(
        "req-low",
        prompt_token_ids=[1] * 10,
        max_new_tokens=5,
        priority=PriorityLevel.LOW,
    )
    # Add high-priority request second
    scheduler.add_dynamic_sequence(
        "req-critical",
        prompt_token_ids=[2] * 10,
        max_new_tokens=5,
        priority=PriorityLevel.CRITICAL,
    )

    batch = scheduler.schedule()

    # The first sequence executed in prefill should be the CRITICAL priority one
    assert batch.prefill_seqs[0].request_id == "req-critical"
    assert batch.prefill_seqs[1].request_id == "req-low"


def test_prefix_caching_hit(mock_cpp_engine, dynamic_config):
    """Test that shared prompt prefixes are detected and cached."""
    bm, pt = mock_cpp_engine
    scheduler = DynamicScheduler(dynamic_config, bm, pt)

    shared_prompt = [100] * 32  # Exactly 2 blocks (block_size=16)

    # Request 1: Complete execution to populate Prefix Cache Trie
    seq1_id = scheduler.add_dynamic_sequence(
        "req-1", prompt_token_ids=shared_prompt, max_new_tokens=1
    )
    scheduler.schedule()

    # Finish seq1 to register blocks into prefix cache
    seq1 = scheduler.sequence_map[seq1_id]
    seq1.generated_token_ids = [99]
    scheduler.finish_sequence(seq1_id)

    # Request 2: Add request with identical prefix
    seq2_id = scheduler.add_dynamic_sequence(
        "req-2", prompt_token_ids=shared_prompt, max_new_tokens=5
    )
    seq2 = cast(DynamicSequence, scheduler.sequence_map[seq2_id])
    # Check prefix hit detection
    assert seq2.computed_prefix_tokens == 32
    assert seq2.shared_prefix_blocks == 2


def test_smart_victim_preemption(mock_cpp_engine, dynamic_config):
    """Test that the dynamic scheduler preempts LOW priority sequence instead of HIGH priority sequence on OOM."""
    bm, pt = mock_cpp_engine
    scheduler = DynamicScheduler(dynamic_config, bm, pt)

    # Fill running queue with LOW and HIGH priority requests
    scheduler.add_dynamic_sequence(
        "req-high",
        prompt_token_ids=[1] * 16,
        max_new_tokens=10,
        priority=PriorityLevel.HIGH,
    )
    scheduler.add_dynamic_sequence(
        "req-low",
        prompt_token_ids=[2] * 16,
        max_new_tokens=10,
        priority=PriorityLevel.LOW,
    )

    scheduler.schedule()  # Run prefill

    # Trigger OOM condition
    bm.get_num_free_blocks.return_value = 0

    batch = scheduler.schedule()

    # Verify that the LOW priority sequence was selected as the preemption victim
    assert len(batch.preempted_seqs) == 1
    assert batch.preempted_seqs[0].request_id == "req-low"
