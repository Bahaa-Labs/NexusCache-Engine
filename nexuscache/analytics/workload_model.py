"""
NexusCache Workload Modeling & VRAM Saturation Engine
=====================================================
Quantitative modeling tools for fitting prompt/generation sequence distributions,
computing deterministic VRAM consumption formulas, and calculating hardware saturation
capacity bounds on VRAM-constrained GPUs (e.g., RTX 3080 10GB).
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

logger = logging.getLogger("nexuscache.analytics.workload")


@dataclass
class SequenceDistributionParams:
    """Fitted parametric distribution parameters for sequence lengths."""

    dist_type: Literal["lognormal", "poisson", "gamma", "weibull"]
    params: dict[str, float]
    mean: float
    std: float
    p50: float
    p95: float
    p99: float


@dataclass
class ModelMemoryConfig:
    """Architecture memory parameters for LLM models and GPU hardware bounds."""

    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int | None = None  # Supports Grouped-Query Attention (GQA)
    head_dim: int = 128
    block_size: int = 16
    dtype_bytes: int = 2  # float16/bfloat16 = 2 bytes, fp8 = 1 byte

    # Model Weights & CUDA Overhead
    model_weights_bytes: int = 14 * (1024**3)  # e.g., 7B parameters in FP16 = 14GB
    cuda_context_overhead_bytes: int = 600 * (1024**2)  # ~600MB CUDA context overhead
    activation_workspace_bytes: int = 512 * (
        1024**2
    )  # ~512MB activation/cublas workspace

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads


@dataclass
class VRAMSaturationMetrics:
    """Quantitative snapshot of VRAM allocation breakdown and capacity limits."""

    total_gpu_vram_bytes: int
    model_weights_bytes: int
    fixed_overhead_bytes: int
    usable_kv_vram_bytes: int
    bytes_per_block: int
    total_allocatable_blocks: int
    max_active_sequences_p50: int
    max_active_sequences_p95: int
    max_active_sequences_p99: int
    kv_cache_efficiency_pct: float


class WorkloadDistributionFitter:
    """Fits empirical sequence length data to statistical parametric distributions."""

    @staticmethod
    def fit_lognormal(lengths: list[int] | np.ndarray) -> SequenceDistributionParams:
        """Fits a Lognormal distribution to empirical token sequence lengths."""
        data = np.asarray(lengths, dtype=np.float64)
        data = data[data > 0]  # Filter non-positive values

        # Explicitly unpack and convert fit parameters to standard floats
        fit_res = stats.lognorm.fit(data, floc=0)
        s = float(fit_res[0])
        loc = float(fit_res[1])
        scale = float(fit_res[2])

        # Use .item() to convert 0D NumPy arrays returned by SciPy into native Python floats
        mean_val = np.asarray(
            stats.lognorm.mean(s, loc, scale), dtype=np.float64
        ).item()
        std_val = np.asarray(stats.lognorm.std(s, loc, scale), dtype=np.float64).item()

        # Extract percentiles as scalar floats
        ppf_res = np.asarray(
            stats.lognorm.ppf([0.50, 0.95, 0.99], s, loc, scale), dtype=np.float64
        )
        p50 = float(ppf_res[0])
        p95 = float(ppf_res[1])
        p99 = float(ppf_res[2])

        return SequenceDistributionParams(
            dist_type="lognormal",
            params={"s": s, "loc": loc, "scale": scale},
            mean=mean_val,
            std=std_val,
            p50=p50,
            p95=p95,
            p99=p99,
        )

    @staticmethod
    def fit_poisson(lengths: list[int] | np.ndarray) -> SequenceDistributionParams:
        """Fits a Poisson distribution to sequence length data."""
        data = np.asarray(lengths, dtype=np.float64)
        mu = float(np.mean(data))

        mean = mu
        std = float(np.sqrt(mu))
        p50, p95, p99 = [float(x) for x in stats.poisson.ppf([0.50, 0.95, 0.99], mu)]

        return SequenceDistributionParams(
            dist_type="poisson",
            params={"mu": mu},
            mean=mean,
            std=std,
            p50=p50,
            p95=p95,
            p99=p99,
        )

    @staticmethod
    def sample_distribution(
        params: SequenceDistributionParams, num_samples: int = 1000
    ) -> np.ndarray:
        """Generates synthetic sequence length samples from fitted parameters."""
        if params.dist_type == "lognormal":
            samples = stats.lognorm.rvs(
                s=params.params["s"],
                loc=params.params["loc"],
                scale=params.params["scale"],
                size=num_samples,
            )
        elif params.dist_type == "poisson":
            samples = stats.poisson.rvs(
                mu=params.params["mu"],
                size=num_samples,
            )
        else:
            raise ValueError(f"Unsupported distribution type: {params.dist_type}")

        return np.clip(np.round(samples), 1, None).astype(int)


class VRAMSaturationModel:
    """
    Deterministic mathematical saturation model calculating total VRAM consumption,
    physical block allocations, and max batch capacities under fixed VRAM constraints.
    """

    def __init__(self, config: ModelMemoryConfig, total_gpu_vram_bytes: int):
        self.config = config
        self.total_gpu_vram_bytes = total_gpu_vram_bytes

    def calculate_bytes_per_block(self) -> int:
        """
        Calculates exact memory footprint for a single physical KV-Cache block.

        Formula:
            V_block = 2 * N_layers * N_kv_heads * D_head * S_block * sizeof(dtype)
        """
        # Narrow Optional[int] to int for Pyright static analysis
        num_kv_heads = (
            self.config.num_kv_heads
            if self.config.num_kv_heads is not None
            else self.config.num_heads
        )

        # Multiply by 2 for separate Key and Value caches
        return (
            2
            * self.config.num_layers
            * num_kv_heads
            * self.config.head_dim
            * self.config.block_size
            * self.config.dtype_bytes
        )

    def calculate_fixed_overhead_bytes(self) -> int:
        """Calculates total non-KV memory overhead (weights + CUDA context + workspaces)."""
        return (
            self.config.model_weights_bytes
            + self.config.cuda_context_overhead_bytes
            + self.config.activation_workspace_bytes
        )

    def compute_saturation_limits(
        self,
        prompt_dist: SequenceDistributionParams,
        gen_dist: SequenceDistributionParams,
    ) -> VRAMSaturationMetrics:
        """
        Computes maximum dynamic batch capacity limits given workload sequence distributions.
        """
        bytes_per_block = self.calculate_bytes_per_block()
        fixed_overhead = self.calculate_fixed_overhead_bytes()

        usable_kv_vram = self.total_gpu_vram_bytes - fixed_overhead
        if usable_kv_vram <= 0:
            logger.error("Model weights and CUDA context exceed total GPU VRAM limit!")
            return VRAMSaturationMetrics(
                total_gpu_vram_bytes=self.total_gpu_vram_bytes,
                model_weights_bytes=self.config.model_weights_bytes,
                fixed_overhead_bytes=fixed_overhead,
                usable_kv_vram_bytes=0,
                bytes_per_block=bytes_per_block,
                total_allocatable_blocks=0,
                max_active_sequences_p50=0,
                max_active_sequences_p95=0,
                max_active_sequences_p99=0,
                kv_cache_efficiency_pct=0.0,
            )

        total_blocks = usable_kv_vram // bytes_per_block

        # Helper to compute max sequences for a target sequence length quantile
        def max_seqs_for_len(total_len: float) -> int:
            req_blocks_per_seq = int(np.ceil(total_len / self.config.block_size))
            return total_blocks // max(1, req_blocks_per_seq)

        max_p50 = max_seqs_for_len(prompt_dist.p50 + gen_dist.p50)
        max_p95 = max_seqs_for_len(prompt_dist.p95 + gen_dist.p95)
        max_p99 = max_seqs_for_len(prompt_dist.p99 + gen_dist.p99)

        kv_efficiency = (
            (total_blocks * bytes_per_block) / float(self.total_gpu_vram_bytes) * 100.0
        )

        return VRAMSaturationMetrics(
            total_gpu_vram_bytes=self.total_gpu_vram_bytes,
            model_weights_bytes=self.config.model_weights_bytes,
            fixed_overhead_bytes=fixed_overhead,
            usable_kv_vram_bytes=usable_kv_vram,
            bytes_per_block=bytes_per_block,
            total_allocatable_blocks=total_blocks,
            max_active_sequences_p50=max_p50,
            max_active_sequences_p95=max_p95,
            max_active_sequences_p99=max_p99,
            kv_cache_efficiency_pct=kv_efficiency,
        )


def compute_rtx_3080_10gb_capacity(
    prompt_lengths: list[int],
    gen_lengths: list[int],
    model_weight_fp16_gb: float = 6.0,  # e.g., Quantized / Small model fitting 10GB VRAM
) -> VRAMSaturationMetrics:
    """
    Convenience helper computing batch capacity bounds specifically for NVIDIA RTX 3080 10GB.
    """
    RTX_3080_VRAM_BYTES = 10 * (1024**3)  # 10 GB

    # Fit sequence distributions
    p_dist = WorkloadDistributionFitter.fit_lognormal(prompt_lengths)
    g_dist = WorkloadDistributionFitter.fit_lognormal(gen_lengths)

    # Configure memory parameters tailored for 10GB GPU
    config = ModelMemoryConfig(
        num_layers=24,
        num_heads=16,
        num_kv_heads=16,
        head_dim=64,
        block_size=16,
        dtype_bytes=2,
        model_weights_bytes=int(model_weight_fp16_gb * (1024**3)),
        cuda_context_overhead_bytes=500 * (1024**2),
        activation_workspace_bytes=300 * (1024**2),
    )

    saturation_model = VRAMSaturationModel(
        config, total_gpu_vram_bytes=RTX_3080_VRAM_BYTES
    )
    return saturation_model.compute_saturation_limits(p_dist, g_dist)
