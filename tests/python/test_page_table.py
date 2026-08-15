"""
Unit Tests for NexusCache PageTable Module
=========================================
"""

import pytest
import torch

try:
    import nexuscache._C as C_ext

    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False


@pytest.mark.skipif(not HAS_EXTENSION, reason="nexuscache._C extension is required.")
class TestPageTable:

    def setup_method(self):
        """Build standard testing setup."""
        self.config = C_ext.BlockAllocatorConfig()
        self.config.num_blocks = 64
        self.config.block_size = 16
        self.config.num_layers = 1
        self.config.num_heads = 4
        self.config.head_dim = 32
        self.config.dtype = torch.float16
        self.config.device_id = 0

        self.block_manager = C_ext.BlockManager(self.config)
        self.page_table = C_ext.PageTable(self.block_manager, block_size=16)

    def test_sequence_lifecycle_and_block_growth(self):
        """Test sequence registration, token growth, and physical block allocation."""
        seq_id = 42
        self.page_table.register_sequence(seq_id)
        assert self.page_table.has_sequence(seq_id)
        assert self.page_table.get_sequence_length(seq_id) == 0

        # Append 10 tokens (requires 1 block)
        new_blocks = self.page_table.append_tokens(seq_id, 10)
        assert len(new_blocks) == 1
        assert self.page_table.get_sequence_length(seq_id) == 10
        assert len(self.page_table.get_block_table(seq_id)) == 1

        # Append 20 tokens (30 tokens total = requires 2 blocks, adds 1 new block)
        new_blocks_2 = self.page_table.append_tokens(seq_id, 20)
        assert len(new_blocks_2) == 1
        assert self.page_table.get_sequence_length(seq_id) == 30
        assert len(self.page_table.get_block_table(seq_id)) == 2

        # Free sequence and verify block cleanup
        initial_free = self.block_manager.get_num_free_blocks()
        self.page_table.free_sequence(seq_id)
        assert not self.page_table.has_sequence(seq_id)
        assert self.block_manager.get_num_free_blocks() == initial_free + 2

    def test_block_table_and_slot_mapping_tensors(self):
        """Test conversion to PyTorch CUDA block table and slot mapping tensors."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        seq1, seq2 = 1, 2
        self.page_table.register_sequence(seq1)
        self.page_table.register_sequence(seq2)

        self.page_table.append_tokens(seq1, 32)  # 2 blocks
        self.page_table.append_tokens(seq2, 16)  # 1 block

        # Verify block table tensor shape and padding
        bt_tensor = self.page_table.get_block_table_tensor([seq1, seq2], device)
        assert bt_tensor.shape == (2, 2)
        assert bt_tensor[1, 1].item() == -1  # Padding for shorter sequence

        # Verify slot mapping tensor
        slot_tensor = self.page_table.get_slot_mapping_tensor(
            [seq1, seq2], [4, 2], device
        )
        assert slot_tensor.shape == (6,)  # 4 + 2 total query tokens
