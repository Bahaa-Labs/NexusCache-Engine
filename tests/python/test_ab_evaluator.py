"""
Unit Tests for A/B Testing Strategy Evaluator
==============================================
Tests timing properties, percentile aggregation, memory fragmentation delta validation,
and payload serialization.
"""

from copy import deepcopy

import pytest

from nexuscache.analytics.ab_evaluator import (
    ABEvaluatorHarness,
    RequestBenchmarkSample,
    StrategyASimulator,
    StrategyBSimulator,
    StrategyMetricsResult,
)


class TestABEvaluator:

    def test_request_benchmark_sample_metrics(self):
        """Verify TTFT, TPOT, and latency calculation accuracy under standard conditions."""
        sample = RequestBenchmarkSample(
            request_id="test_001",
            prompt_tokens=128,
            gen_tokens=4,
            arrival_time_s=10.0,
            first_token_time_s=10.05,  # 50 ms TTFT
            completion_time_s=10.25,  # 250 ms Total
            inter_token_latencies_ms=[50.0, 50.0, 50.0, 50.0],
        )

        assert pytest.approx(sample.ttft_ms, abs=1e-3) == 50.0
        assert pytest.approx(sample.tpot_ms, abs=1e-3) == 50.0
        assert pytest.approx(sample.total_latency_ms, abs=1e-3) == 250.0

    def test_request_benchmark_sample_edge_cases(self):
        """Verify graceful fallback metrics when timestamps are missing or invalid."""
        # Unfinished request sample
        incomplete_sample = RequestBenchmarkSample(
            request_id="test_incomplete",
            prompt_tokens=64,
            gen_tokens=10,
            arrival_time_s=100.0,
            first_token_time_s=None,
            completion_time_s=None,
            inter_token_latencies_ms=[],
        )

        assert incomplete_sample.ttft_ms == 0.0
        assert incomplete_sample.tpot_ms == 0.0
        assert incomplete_sample.total_latency_ms == 0.0

    def test_percentile_computation_helper(self):
        """Verify percentile calculation correctness for p50, p95, and p99."""
        # Empty array fallback
        p50, p95, p99 = ABEvaluatorHarness._compute_percentiles([])
        assert (p50, p95, p99) == (0.0, 0.0, 0.0)

        # Cast elements explicitly to float to satisfy List[float] type checking
        values = [float(x) for x in range(1, 101)]
        p50, p95, p99 = ABEvaluatorHarness._compute_percentiles(values)
        assert pytest.approx(p50, abs=1e-1) == 50.5
        assert pytest.approx(p95, abs=1e-1) == 95.05
        assert pytest.approx(p99, abs=1e-1) == 99.01

    def test_simulators_execution(self):
        """Ensure Strategy A and Strategy B simulators compute non-negative timings."""
        sample_base = RequestBenchmarkSample(
            request_id="sim_001",
            prompt_tokens=128,
            gen_tokens=16,
            arrival_time_s=0.0,
        )

        # Strategy A (Static) - pass a deep copy
        sample_a = deepcopy(sample_base)
        sim_a = StrategyASimulator(static_batch_size=2)
        samples_a, dur_a = sim_a.execute_workload([sample_a])

        assert len(samples_a) == 1
        assert dur_a >= 0.0
        assert samples_a[0].strategy_used == "Strategy_A_Static"
        assert samples_a[0].first_token_time_s is not None
        assert samples_a[0].completion_time_s is not None
        assert samples_a[0].completion_time_s > samples_a[0].first_token_time_s

        # Strategy B (Paged) - pass a fresh deep copy
        sample_b = deepcopy(sample_base)
        sim_b = StrategyBSimulator(page_size=16)
        samples_b, dur_b = sim_b.execute_workload([sample_b])

        assert len(samples_b) == 1
        assert dur_b >= 0.0
        assert samples_b[0].strategy_used == "Strategy_B_Paged"
        assert samples_b[0].first_token_time_s is not None
        assert samples_b[0].completion_time_s is not None
        assert samples_b[0].completion_time_s > samples_b[0].first_token_time_s

    def test_ab_experiment_driver_execution(self):
        """Verify driver runs both strategies and outputs percentile metrics."""
        harness = ABEvaluatorHarness()
        results = harness.run_ab_experiment(num_requests=20, seed=42)

        assert "Strategy_A" in results
        assert "Strategy_B" in results

        res_a = results["Strategy_A"]
        res_b = results["Strategy_B"]

        assert isinstance(res_a, StrategyMetricsResult)
        assert isinstance(res_b, StrategyMetricsResult)

        assert res_a.total_requests == 20
        assert res_b.total_requests == 20
        assert res_a.throughput_tokens_per_sec > 0
        assert res_b.throughput_tokens_per_sec > 0

        # Paged KV-Cache must show lower VRAM fragmentation than static contiguity
        assert res_b.avg_vram_fragmentation_pct < res_a.avg_vram_fragmentation_pct

    def test_metrics_result_serialization(self):
        """Verify StrategyMetricsResult converts seamlessly to dict representations."""
        harness = ABEvaluatorHarness()
        results = harness.run_ab_experiment(num_requests=5, seed=42)

        res_dict = results["Strategy_B"].to_dict()
        assert isinstance(res_dict, dict)
        assert res_dict["strategy_name"] == "Strategy_B_Paged"
        assert "ttft_p50_ms" in res_dict
        assert "e2e_p99_ms" in res_dict
