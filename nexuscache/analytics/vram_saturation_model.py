"""
NexusCache Analytics: VRAM Saturation & Capacity Model
Calculates GPU memory headroom, estimates KV-cache block capacities,
evaluates dynamic VRAM saturation thresholds, and models fragmentation risk.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum

import torch

logger = logging.getLogger("nexuscache.analytics.vram_saturation_model")


class SaturationState(Enum):
    """Dynamic operational state based on KV-cache VRAM saturation."""

    NORMAL = "NORMAL"  # < 75% utilization: Unrestricted allocations
    ELEVATED = "ELEVATED"  # 75% - 85% utilization: Monitor headroom closely
    CRITICAL_THROTTLE = (
        "CRITICAL"  # 85% - 90% utilization: Throttle new prefill requests
    )
    EVICTION_REQUIRED = (
        "EVICTION"  # >= 90% utilization: Trigger cache eviction / preempt sequences
    )


@dataclass(frozen=True)
class ModelArchConfig:
    """Architectural parameters of the target LLM required for KV-cache sizing."""

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.float16
    vocab_size: int = 32000

    def __post_init__(self):
        if (
            self.num_layers <= 0
            or self.num_heads <= 0
            or self.num_kv_heads <= 0
            or self.head_dim <= 0
        ):
            raise ValueError(
                "All architectural counts (layers, heads, head_dim) must be positive integers."
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})."
            )

    @property
    def element_size_bytes(self) -> int:
        """Returns element size in bytes based on torch dtype."""
        return torch.tensor([], dtype=self.dtype).element_size()

    @property
    def bytes_per_token_all_layers(self) -> int:
        """Calculates memory bytes required to store 1 token's KV values across ALL layers.

        Formula: 2 (K and V) * num_layers * num_kv_heads * head_dim * element_size_bytes
        """
        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * self.element_size_bytes
        )


@dataclass
class VRAMPoolConfig:
    """Configures system VRAM limits and model static memory footprints."""

    total_vram_bytes: int
    model_weights_bytes: int
    cuda_workspace_bytes: int = (
        1 * 1024 * 1024 * 1024
    )  # Default 1 GB CUDA context/cuDNN workspace
    eviction_threshold: float = 0.90  # Trigger block eviction at 90% KV utilization
    throttle_threshold: float = 0.85  # Throttle prefill batches at 85% KV utilization
    elevated_threshold: float = 0.75  # Alert state at 75% KV utilization

    def __post_init__(self):
        if not (
            0.0
            < self.elevated_threshold
            < self.throttle_threshold
            < self.eviction_threshold
            <= 1.0
        ):
            raise ValueError(
                "Thresholds must strictly satisfy: 0 < elevated < throttle < eviction <= 1.0"
            )


@dataclass
class BatchAllocationEstimate:
    """Report generated when modeling an incoming request batch against available KV blocks."""

    required_blocks: int
    available_free_blocks: int
    can_accommodate: bool
    post_allocation_saturation: float
    projected_state: SaturationState
    estimated_internal_fragmentation_pct: float


@dataclass
class VRAMAnalysisReport:
    """Comprehensive snapshot of VRAM usage, KV-cache capacity, and saturation state."""

    total_vram_mb: float
    usable_kv_vram_mb: float
    block_size_tokens: int
    block_size_bytes: int
    total_physical_blocks: int
    allocated_blocks: int
    free_blocks: int
    kv_utilization_pct: float
    saturation_state: SaturationState
    internal_fragmentation_pct: float


class VRAMSaturationModel:
    """Analytical engine for predicting GPU memory headroom, KV block capacity,
    and dynamic saturation states in NexusCache.
    """

    def __init__(self, model_config: ModelArchConfig, pool_config: VRAMPoolConfig):
        self.model_config = model_config
        self.pool_config = pool_config

    def compute_block_size_bytes(self, block_size_tokens: int) -> int:
        """Computes physical memory size in bytes for a single KV-cache block."""
        if block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive.")
        return block_size_tokens * self.model_config.bytes_per_token_all_layers

    def predict_usable_kv_vram(self) -> int:
        """Predicts net available VRAM bytes dedicated specifically for the KV-Cache pool
        after subtracting model weights and CUDA workspace overheads.
        """
        reserved = (
            self.pool_config.model_weights_bytes + self.pool_config.cuda_workspace_bytes
        )
        usable = self.pool_config.total_vram_bytes - reserved
        if usable <= 0:
            raise RuntimeError(
                f"Insufficient VRAM: Total VRAM ({self.pool_config.total_vram_bytes / 1e9:.2f} GB) "
                f"is smaller than weights + workspace ({reserved / 1e9:.2f} GB)."
            )
        return usable

    def calculate_max_capacity(self, block_size_tokens: int) -> tuple[int, int]:
        """Calculates total usable KV-Cache VRAM bytes and total physical block capacity.

        Returns:
            Tuple[usable_vram_bytes, total_physical_blocks]
        """
        usable_vram = self.predict_usable_kv_vram()
        block_bytes = self.compute_block_size_bytes(block_size_tokens)
        total_blocks = usable_vram // block_bytes
        return usable_vram, total_blocks

    def evaluate_saturation(
        self, allocated_blocks: int, total_blocks: int
    ) -> tuple[float, SaturationState]:
        """Determines current KV-cache saturation percentage and operational state."""
        if total_blocks <= 0:
            return 1.0, SaturationState.EVICTION_REQUIRED

        # Guard against over-allocation bugs
        if allocated_blocks > total_blocks:
            logger.warning(
                f"Allocated blocks ({allocated_blocks}) exceeds total physical capacity ({total_blocks}). "
                "Calculations will cap utilization at max capacity state."
            )

        utilization = allocated_blocks / float(total_blocks)

        if utilization >= self.pool_config.eviction_threshold:
            state = SaturationState.EVICTION_REQUIRED
        elif utilization >= self.pool_config.throttle_threshold:
            state = SaturationState.CRITICAL_THROTTLE
        elif utilization >= self.pool_config.elevated_threshold:
            state = SaturationState.ELEVATED
        else:
            state = SaturationState.NORMAL

        return utilization, state

    def estimate_batch_requirements(
        self,
        request_token_counts: list[int],
        block_size_tokens: int,
        currently_allocated_blocks: int,
        total_physical_blocks: int,
    ) -> BatchAllocationEstimate:
        """Estimates block requirements and post-allocation saturation for an incoming request batch.

        Args:
            request_token_counts: List of token sequence lengths (prompt + max_gen) for incoming batch.
            block_size_tokens: Number of tokens stored per physical block.
            currently_allocated_blocks: Number of physical blocks currently in use.
            total_physical_blocks: Total physical block capacity in VRAM pool.

        Returns:
            BatchAllocationEstimate report detailing feasibility and fragmentation.
        """
        if block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive.")

        total_required_blocks = 0
        total_requested_tokens = 0

        for num_tokens in request_token_counts:
            if num_tokens <= 0:
                continue
            req_blocks = math.ceil(num_tokens / float(block_size_tokens))
            total_required_blocks += req_blocks
            total_requested_tokens += num_tokens

        free_blocks = max(0, total_physical_blocks - currently_allocated_blocks)
        can_accommodate = total_required_blocks <= free_blocks

        projected_allocated = currently_allocated_blocks + total_required_blocks
        projected_saturation, projected_state = self.evaluate_saturation(
            projected_allocated, total_physical_blocks
        )

        # Compute internal fragmentation: Wasted token slots in the last allocated block of each sequence
        total_allocated_token_capacity = total_required_blocks * block_size_tokens
        wasted_slots = total_allocated_token_capacity - total_requested_tokens
        fragmentation_pct = (
            (wasted_slots / float(total_allocated_token_capacity)) * 100.0
            if total_allocated_token_capacity > 0
            else 0.0
        )

        return BatchAllocationEstimate(
            required_blocks=total_required_blocks,
            available_free_blocks=free_blocks,
            can_accommodate=can_accommodate,
            post_allocation_saturation=projected_saturation,
            projected_state=projected_state,
            estimated_internal_fragmentation_pct=fragmentation_pct,
        )

    def calculate_active_fragmentation(
        self, sequence_lengths: list[int], block_size_tokens: int
    ) -> float:
        """Calculates real-time internal fragmentation percentage across active sequences."""
        if not sequence_lengths or block_size_tokens <= 0:
            return 0.0

        total_used_tokens = sum(sequence_lengths)
        total_allocated_blocks = sum(
            math.ceil(seq_len / float(block_size_tokens))
            for seq_len in sequence_lengths
        )
        total_block_capacity = total_allocated_blocks * block_size_tokens

        if total_block_capacity == 0:
            return 0.0

        wasted_tokens = total_block_capacity - total_used_tokens
        return (wasted_tokens / float(total_block_capacity)) * 100.0

    def generate_report(
        self,
        block_size_tokens: int,
        allocated_blocks: int,
        active_sequence_lengths: list[int] | None = None,
    ) -> VRAMAnalysisReport:
        """Generates a complete analytical snapshot of the engine's VRAM state."""
        usable_vram, total_blocks = self.calculate_max_capacity(block_size_tokens)

        # Clamp allocated blocks to physical bounds for safe metric reports
        safe_allocated = min(allocated_blocks, total_blocks)
        block_bytes = self.compute_block_size_bytes(block_size_tokens)
        free_blocks = max(0, total_blocks - safe_allocated)

        utilization, state = self.evaluate_saturation(allocated_blocks, total_blocks)

        fragmentation = 0.0
        if active_sequence_lengths:
            fragmentation = self.calculate_active_fragmentation(
                active_sequence_lengths, block_size_tokens
            )

        return VRAMAnalysisReport(
            total_vram_mb=self.pool_config.total_vram_bytes / (1024 * 1024),
            usable_kv_vram_mb=usable_vram / (1024 * 1024),
            block_size_tokens=block_size_tokens,
            block_size_bytes=block_bytes,
            total_physical_blocks=total_blocks,
            allocated_blocks=safe_allocated,
            free_blocks=free_blocks,
            kv_utilization_pct=utilization * 100.0,
            saturation_state=state,
            internal_fragmentation_pct=fragmentation,
        )


# Simple standalone verification when run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model_cfg = ModelArchConfig(
        num_layers=32, num_heads=32, num_kv_heads=8, head_dim=128, dtype=torch.float16
    )

    pool_cfg = VRAMPoolConfig(
        total_vram_bytes=24 * 1024 * 1024 * 1024,
        model_weights_bytes=16 * 1024 * 1024 * 1024,
    )

    sat_model = VRAMSaturationModel(model_cfg, pool_cfg)
    block_size = 16

    # Realistically allocate 2,800 blocks out of 3,584 capacity (~78.12% utilization -> ELEVATED)
    report = sat_model.generate_report(
        block_size_tokens=block_size,
        allocated_blocks=2800,
        active_sequence_lengths=[128, 256, 512, 1024, 73],
    )

    print("=== NexusCache VRAM Analytics Report ===")
    print(f"Total VRAM: {report.total_vram_mb:.1f} MB")
    print(f"Usable KV-Cache VRAM: {report.usable_kv_vram_mb:.1f} MB")
    print(
        f"Single Block Size: {report.block_size_bytes / 1024:.2f} KB ({report.block_size_tokens} tokens)"
    )
    print(f"Total Physical Capacity: {report.total_physical_blocks} blocks")
    print(f"Allocated: {report.allocated_blocks} | Free: {report.free_blocks}")
    print(
        f"Utilization: {report.kv_utilization_pct:.2f}% ({report.saturation_state.value})"
    )
    print(f"Active Internal Fragmentation: {report.internal_fragmentation_pct:.2f}%")

    # Incoming Batch Analysis
    incoming_batch = [512, 1024, 2048, 120]
    batch_est = sat_model.estimate_batch_requirements(
        request_token_counts=incoming_batch,
        block_size_tokens=block_size,
        currently_allocated_blocks=report.allocated_blocks,
        total_physical_blocks=report.total_physical_blocks,
    )

    print("\n=== Incoming Batch Analysis ===")
    print(f"Required Blocks: {batch_est.required_blocks}")
    print(f"Can Accommodate: {batch_est.can_accommodate}")
    print(
        f"Projected Saturation: {batch_est.post_allocation_saturation * 100:.2f}% ({batch_est.projected_state.value})"
    )
