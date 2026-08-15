"""
Validates C++ BlockManager, PageTable, raw CUDA pointer interfacing,
and memory recycling mechanisms via NexusCacheEngine wrapper.
"""

import pytest
import torch

import nexuscache._C as _C
from nexuscache.cache_engine import NexusCacheEngine

# Guard against running on non-CUDA CI environment
CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture
def cache_engine():
    """Fixture initializing NexusCacheEngine instance on GPU 0."""
    return NexusCacheEngine(
        num_blocks=128,
        block_size=16,
        num_layers=16,
        num_heads=8,
        head_dim=64,
        dtype=torch.float16,
        device_id=0,
    )


@pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA hardware is required for memory subsystem tests"
)
class TestMemorySubsystem:

    def test_raw_cuda_pointer_interfacing(self, cache_engine: NexusCacheEngine):
        """1. Verify C++ raw device pointer extraction matches PyTorch tensor addresses."""
        key_ptr = cache_engine.key_cache_ptr
        val_ptr = cache_engine.value_cache_ptr

        # Pointer validity checks
        assert key_ptr > 0, "Key cache pointer is invalid"
        assert val_ptr > 0, "Value cache pointer is invalid"
        assert (
            key_ptr != val_ptr
        ), "Key and Value caches must reside at distinct physical VRAM addresses"

        # Verify C++ pointer extraction matches PyTorch tensor data pointers
        key_cache, val_cache = cache_engine.get_physical_kv_tensors()
        assert key_cache.is_cuda and val_cache.is_cuda
        assert key_cache.dtype == torch.float16
        assert _C.get_tensor_device_ptr(key_cache) == key_ptr
        assert _C.get_tensor_device_ptr(val_cache) == val_ptr

    def test_sequence_lifecycle_and_dynamic_allocation(
        self, cache_engine: NexusCacheEngine
    ):
        """2. Verify token allocation, block mapping, and kernel metadata tensor shape generation."""
        seq_id = 99
        cache_engine.register_sequence(seq_id)

        # Allocating 33 tokens with block_size=16 requires ceil(33/16) = 3 physical blocks
        slots = cache_engine.allocate_tokens(seq_id, num_tokens=33)

        assert len(slots) == 33

        # Prepare metadata tensors for CUDA kernel execution
        block_tables, slot_mapping = cache_engine.prepare_kernel_metadata(
            sequence_ids=[seq_id], query_lens=[33]
        )

        # Batch size 1, 3 physical blocks allocated
        assert block_tables.shape[0] == 1
        assert block_tables.shape[1] == 3
        # Slot mapping tensor contains exactly 33 index mappings
        assert slot_mapping.shape[0] == 33
        assert slot_mapping.is_cuda

    def test_memory_recycling_and_block_deallocation(
        self, cache_engine: NexusCacheEngine
    ):
        """3. Verify freeing sequence returns all physical VRAM blocks back to the free pool."""
        initial_free = cache_engine.block_manager.get_num_free_blocks()
        assert initial_free == 128

        seq_id = 101
        cache_engine.register_sequence(seq_id)
        cache_engine.allocate_tokens(
            seq_id, num_tokens=64
        )  # 64 tokens = 4 blocks (64 / 16)

        assert cache_engine.block_manager.get_num_free_blocks() == 128 - 4

        # Reclaim memory
        cache_engine.free_sequence(seq_id)
        assert cache_engine.block_manager.get_num_free_blocks() == 128
