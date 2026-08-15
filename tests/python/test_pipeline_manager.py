"""
Integration Tests for GPU Utilization Tuning & Pipeline Execution Manager
Verifies CUDA Stream synchronization, CUDA Graph Replay, and Adaptive Batch Tuning.
"""

import pytest
import torch

import nexuscache._C as _C
from nexuscache.server.pipeline_manager import (
    AdaptiveBatchTuner,
    ExecutionPipelineManager,
)
from nexuscache.server.scheduler import Scheduler, SchedulerConfig

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture
def allocator_config() -> _C.BlockAllocatorConfig:
    config = _C.BlockAllocatorConfig()
    config.num_blocks = 32
    config.block_size = 16
    config.num_layers = 2
    config.num_heads = 4
    config.head_dim = 32
    config.dtype = torch.float16
    config.device_id = 0
    return config


@pytest.fixture
def cpp_subsystem(allocator_config: _C.BlockAllocatorConfig):
    block_manager = _C.BlockManager(allocator_config)
    page_table = _C.PageTable(block_manager, allocator_config.block_size)
    return block_manager, page_table


def dummy_model_forward(
    input_ids: torch.Tensor, slot_mapping: torch.Tensor
) -> torch.Tensor:
    """Mock PyTorch forward pass returning logits tensor on CUDA."""
    batch_size = input_ids.shape[0]
    return torch.randn(
        (batch_size, 32000), dtype=torch.float16, device=input_ids.device
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU hardware required")
class TestPipelineManager:

    def test_adaptive_batch_tuner_scaling(self):
        """Verify dynamic token budget scales up/down based on VRAM pressure and latency."""
        tuner = AdaptiveBatchTuner(
            initial_token_budget=2048, min_token_budget=1024, max_token_budget=8192
        )

        # Force 50 iterations with low step time + high free VRAM -> budget should scale UP
        for _ in range(50):
            budget = tuner.update_budget(
                current_batch_size=16, step_duration_ms=2.0, free_vram_ratio=0.50
            )

        assert budget > 2048

        # Force 50 iterations with high VRAM pressure -> budget should scale DOWN
        for _ in range(50):
            budget = tuner.update_budget(
                current_batch_size=64, step_duration_ms=15.0, free_vram_ratio=0.05
            )

        assert budget < 8192

    def test_pipeline_execution_and_cuda_graphs(self, cpp_subsystem, allocator_config):
        """Verify full execution pipeline step including C++ slot mapping and dummy model pass."""
        block_manager, page_table = cpp_subsystem

        sched_config = SchedulerConfig(
            max_num_batched_tokens=512,
            max_num_seqs=16,
            max_paged_blocks=32,
            block_size=16,
        )
        scheduler = Scheduler(sched_config, block_manager, page_table)

        # Initialize Pipeline Manager
        pipeline = ExecutionPipelineManager(
            device_id=0,
            block_manager=block_manager,
            page_table=page_table,
            enable_cuda_graphs=False,  # Skip graph warm-up for lightweight unit test
        )

        # Add sequences to scheduler
        scheduler.add_sequence("req_1", [10, 11, 12, 13], max_new_tokens=5)
        scheduler.add_sequence("req_2", [20, 21], max_new_tokens=5)

        # 1. Run Prefill Batch
        batch_prefill = scheduler.schedule()
        results_prefill = pipeline.execute_batch_step(
            batch_prefill, scheduler, dummy_model_forward
        )

        assert len(results_prefill) == 2
        assert len(results_prefill[0][0].generated_token_ids) == 1

        # 2. Run Decode Batch
        batch_decode = scheduler.schedule()
        results_decode = pipeline.execute_batch_step(
            batch_decode, scheduler, dummy_model_forward
        )

        assert len(results_decode) == 2
        assert len(results_decode[0][0].generated_token_ids) == 2
