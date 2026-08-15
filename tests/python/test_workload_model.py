"""
Unit Tests for Workload Distribution Fitting and VRAM Saturation Modeling
"""

import numpy as np
import pytest

from nexuscache.analytics.workload_model import (
    ModelMemoryConfig,
    VRAMSaturationModel,
    WorkloadDistributionFitter,
    compute_rtx_3080_10gb_capacity,
)


class TestWorkloadModel:

    def test_fit_lognormal_distribution(self):
        """Verify Lognormal distribution fitting and percentile output."""
        np.random.seed(42)
        sample_lengths = (
            np.random.lognormal(mean=4.5, sigma=0.75, size=500).astype(int) + 1
        )

        params = WorkloadDistributionFitter.fit_lognormal(sample_lengths)

        assert params.dist_type == "lognormal"
        assert params.p50 < params.p95 < params.p99
        assert params.mean > 0

    def test_fit_poisson_distribution(self):
        """Verify Poisson distribution fitting."""
        np.random.seed(42)
        sample_lengths = np.random.poisson(lam=64, size=500) + 1

        params = WorkloadDistributionFitter.fit_poisson(sample_lengths)

        assert params.dist_type == "poisson"
        assert pytest.approx(params.mean, abs=2.0) == 64.0

    def test_calculate_bytes_per_block(self):
        """
        Verify exact bytes per KV block formula:
        2 * layers(16) * kv_heads(8) * head_dim(64) * block_size(16) * dtype_bytes(2)
        = 2 * 16 * 8 * 64 * 16 * 2 = 524,288 bytes (512 KB)
        """
        config = ModelMemoryConfig(
            num_layers=16,
            num_heads=8,
            num_kv_heads=8,
            head_dim=64,
            block_size=16,
            dtype_bytes=2,
        )
        model = VRAMSaturationModel(config, total_gpu_vram_bytes=10 * (1024**3))

        expected_bytes = 2 * 16 * 8 * 64 * 16 * 2
        assert model.calculate_bytes_per_block() == expected_bytes
        assert expected_bytes == 524288  # Exact 512 KB

    def test_rtx_3080_10gb_capacity_limits(self):
        """Verify dynamic batch capacity limits on 10GB RTX 3080 boundary."""
        prompt_lens = [128, 256, 512, 1024] * 50
        gen_lens = [32, 64, 128, 256] * 50

        # Run capacity model assuming 4.5 GB model weights
        metrics = compute_rtx_3080_10gb_capacity(
            prompt_lengths=prompt_lens,
            gen_lengths=gen_lens,
            model_weight_fp16_gb=4.5,
        )

        assert metrics.total_gpu_vram_bytes == 10 * (1024**3)
        assert metrics.usable_kv_vram_bytes > 0
        assert metrics.total_allocatable_blocks > 0
        assert metrics.max_active_sequences_p50 >= metrics.max_active_sequences_p95
        assert metrics.max_active_sequences_p95 >= metrics.max_active_sequences_p99
