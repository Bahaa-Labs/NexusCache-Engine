import torch

try:
    import nexuscache._C as _C
except ImportError:
    # Allows Pylance/IDE to fall back cleanly if binary isn't compiled yet
    _C = None


class NexusCacheEngine:
    """High-Level Python Engine interface for NexusCache C++/CUDA Memory Subsystem."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device_id: int = 0,
    ) -> None:
        if _C is None:
            raise RuntimeError(
                "NexusCache C++ extension (_C) is not compiled. "
                "Run `pip install -e .` inside Distrobox to build native components."
            )

        self.config = _C.BlockAllocatorConfig()
        self.config.num_blocks = num_blocks
        self.config.block_size = block_size
        self.config.num_layers = num_layers
        self.config.num_heads = num_heads
        self.config.head_dim = head_dim
        self.config.dtype = dtype
        self.config.device_id = device_id

        # Instantiate C++ BlockManager & PageTable native objects
        self.block_manager = _C.BlockManager(self.config)
        self.page_table = _C.PageTable(self.block_manager, block_size)
        self.transfer_manager = _C.AsyncTransferManager(device_id)

    @property
    def key_cache_ptr(self) -> int:
        """Returns the raw CUDA memory pointer address (uintptr_t) of Key Cache Pool."""
        return self.block_manager.get_key_cache_ptr()

    @property
    def value_cache_ptr(self) -> int:
        """Returns the raw CUDA memory pointer address (uintptr_t) of Value Cache Pool."""
        return self.block_manager.get_value_cache_ptr()

    def get_physical_kv_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (Key, Value) PyTorch Tensors on GPU VRAM."""
        return self.block_manager.get_physical_kv_tensors()

    def register_sequence(self, sequence_id: int) -> None:
        self.page_table.register_sequence(sequence_id)

    def allocate_tokens(self, sequence_id: int, num_tokens: int) -> list[int]:
        return self.page_table.append_tokens(sequence_id, num_tokens)

    def free_sequence(self, sequence_id: int) -> None:
        self.page_table.free_sequence(sequence_id)

    def prepare_kernel_metadata(
        self, sequence_ids: list[int], query_lens: list[int], device="cuda"
    ):
        # Convert torch.device instance or object to string if necessary
        device_str = str(device)  # e.g., converts torch.device("cuda:0") -> "cuda:0"

        block_tables = self.page_table.get_block_table_tensor(
            sequence_ids=sequence_ids, device=device_str  # <--- Pass device as string!
        )

        slot_mapping = self.page_table.get_slot_mapping_tensor(
            sequence_ids=sequence_ids,
            query_lens=query_lens,
            device=device_str,  # <--- Pass device as string!
        )

        return block_tables, slot_mapping
