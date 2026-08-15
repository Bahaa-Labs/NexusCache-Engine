"""
End-to-End Pytest Verification for NexusCache C++/CUDA Extension
================================================================
"""

import pytest
import torch

try:
    import nexuscache._C as C_ext

    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False


@pytest.mark.skipif(
    not (HAS_EXTENSION and torch.cuda.is_available()),
    reason="Requires PyTorch GPU support and compiled nexuscache._C extension.",
)
class TestNexusCacheNativeExtension:

    def setup_method(self):
        """Build standard test configuration and initializers."""
        self.config = C_ext.BlockAllocatorConfig()
        self.config.num_blocks = 32
        self.config.block_size = 16
        self.config.num_layers = 1  # 1 layer for single-block physical stride alignment
        self.config.num_heads = 8
        self.config.head_dim = 64
        self.config.dtype = torch.float16
        self.config.device_id = 0

    def test_block_manager_and_pointer_extraction(self):
        """Test BlockManager initialization, allocation, and raw CUDA pointer extraction."""
        manager = C_ext.BlockManager(self.config)
        assert manager.get_total_blocks() == 32
        assert manager.get_num_free_blocks() == 32

        # Allocate blocks
        b_idx = manager.allocate_block()
        assert b_idx >= 0
        assert manager.get_num_allocated_blocks() == 1

        # Pointer verification
        key_ptr = manager.get_key_cache_ptr()
        val_ptr = manager.get_value_cache_ptr()
        assert key_ptr > 0
        assert val_ptr > 0

    def test_page_table_sequence_lifecycle(self):
        """Test PageTable allocation, token mapping, and CUDA tensor generation."""
        manager = C_ext.BlockManager(self.config)
        page_table = C_ext.PageTable(manager, block_size=16)

        seq_id = 101
        page_table.register_sequence(seq_id)
        assert page_table.has_sequence(seq_id)

        # Append 40 tokens (should allocate 3 blocks: ceil(40/16))
        page_table.append_tokens(seq_id, 40)
        assert page_table.get_sequence_length(seq_id) == 40

        block_table = page_table.get_block_table(seq_id)
        assert len(block_table) == 3

        # Test block table tensor conversion on CUDA
        bt_tensor = page_table.get_block_table_tensor([seq_id], "cuda")
        assert bt_tensor.is_cuda
        assert bt_tensor.shape[0] == 1

        # Free sequence and confirm blocks returned
        page_table.free_sequence(seq_id)
        assert not page_table.has_sequence(seq_id)

    def test_cuda_memset_and_copy_kernels(self):
        """Verify high-performance launch_memset_blocks and launch_copy_blocks CUDA kernels."""
        manager = C_ext.BlockManager(self.config)
        key_cache, val_cache = manager.get_physical_kv_tensors()

        key_ptr = manager.get_key_cache_ptr()
        val_ptr = manager.get_value_cache_ptr()

        # Calculate exact byte size for one block directly from allocated physical tensor
        elem_size = key_cache.element_size()
        block_bytes = key_cache[0].numel() * elem_size

        # 1. Fill Key cache with dummy values
        key_cache.fill_(1.5)
        torch.cuda.synchronize()

        # 2. Test Zeroing (Memset) Kernel on physical blocks 2 and 4
        C_ext.launch_memset_blocks(
            cache_ptr=key_ptr,
            block_indices=[2, 4],
            block_bytes=block_bytes,
            stream_ptr=torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()

        # Assert zeroed target blocks
        assert torch.all(key_cache[2] == 0.0)
        assert torch.all(key_cache[4] == 0.0)
        assert torch.all(key_cache[0] == 1.5)

        # 3. Test Copy Kernel (Gathering physical block 0 to logical index 0)
        out_key = torch.zeros_like(key_cache[0:1])
        out_val = torch.zeros_like(val_cache[0:1])

        mappings = [(0, 0)]  # (logical_0, physical_0)
        C_ext.launch_copy_blocks(
            key_cache_ptr=key_ptr,
            value_cache_ptr=val_ptr,
            out_key_ptr=out_key.data_ptr(),
            out_value_ptr=out_val.data_ptr(),
            mappings=mappings,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            block_size=self.config.block_size,
            elem_size_bytes=elem_size,
            stream_ptr=torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()

        # Confirm non-zero data was copied correctly
        assert torch.all(out_key[0] == 1.5)
