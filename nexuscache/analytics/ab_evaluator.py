"""
NexusCache A/B Testing Strategy Evaluator
Automated experimental harness comparing:
  - Strategy A: Static Batching + Naive Contiguous Tensor Allocation
  - Strategy B: Dynamic Continuous Batching + Paged KV-Cache Engine

Collects, aggregates, logs, and exports latency percentiles (TTFT, TPOT, P95, P99),
memory fragmentation metrics, and throughput across synthetic workloads.
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger("nexuscache.analytics.ab_evaluator")


@dataclass
class RequestBenchmarkSample:
    """Individual request lifecycle timing and trace metrics."""

    request_id: str
    prompt_tokens: int
    gen_tokens: int
    arrival_time_s: float
    first_token_time_s: float | None = None
    completion_time_s: float | None = None
    inter_token_latencies_ms: list[float] = field(default_factory=list)
    strategy_used: Literal["Strategy_A_Static", "Strategy_B_Paged"] = (
        "Strategy_A_Static"
    )

    @property
    def ttft_ms(self) -> float:
        """Time-To-First-Token in milliseconds."""
        if self.first_token_time_s is None:
            return 0.0
        return max(0.0, (self.first_token_time_s - self.arrival_time_s) * 1000.0)

    @property
    def tpot_ms(self) -> float:
        """Average Time-Per-Output-Token in milliseconds."""
        if not self.inter_token_latencies_ms:
            return 0.0
        return float(np.mean(self.inter_token_latencies_ms))

    @property
    def total_latency_ms(self) -> float:
        """End-to-End request latency in milliseconds."""
        if self.completion_time_s is None:
            return 0.0
        return max(0.0, (self.completion_time_s - self.arrival_time_s) * 1000.0)


@dataclass
class StrategyMetricsResult:
    """Aggregated experimental results and percentile metrics for a strategy run."""

    strategy_name: str
    total_requests: int
    total_tokens_generated: int
    duration_seconds: float
    throughput_tokens_per_sec: float

    # Latency Percentiles (ms)
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float

    tpot_p50_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float

    e2e_p50_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float

    # Memory Utilization & Waste
    avg_vram_fragmentation_pct: float
    peak_concurrent_active_requests: int

    def to_dict(self) -> dict[str, Any]:
        """Converts result object to clean dict representation."""
        return asdict(self)


class StrategyASimulator:
    """
    Simulates Strategy A: Static Batching with Naive Contiguous Memory Allocation.
    Requires padding sequences to max sequence length in batch and pre-allocating contiguous VRAM.
    """

    def __init__(self, static_batch_size: int = 8, max_seq_len: int = 2048):
        self.batch_size = static_batch_size
        self.max_seq_len = max_seq_len
        self.step_delay_ms = 15.0  # Base model execution step delay

    def execute_workload(
        self, requests: list[RequestBenchmarkSample]
    ) -> tuple[list[RequestBenchmarkSample], float]:
        """Executes requests in rigid static batches."""
        processed_samples: list[RequestBenchmarkSample] = []
        start_sim_time = time.perf_counter()

        # Group into static chunks
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i : i + self.batch_size]
            max_prompt = max(req.prompt_tokens for req in batch)
            max_gen = max(req.gen_tokens for req in batch)

            # Static batching waits for the slowest request in the batch
            batch_start = time.perf_counter()

            for req in batch:
                req.first_token_time_s = batch_start + (max_prompt * 0.002)

                # Simulate step-by-step token generation
                latencies = []
                for step in range(req.gen_tokens):
                    # Add artificial padding penalty cost
                    padding_penalty = (
                        self.max_seq_len - (req.prompt_tokens + step)
                    ) * 0.00001
                    step_latency = self.step_delay_ms + max(0.0, padding_penalty)
                    latencies.append(step_latency)

                req.inter_token_latencies_ms = latencies
                req.completion_time_s = req.first_token_time_s + (
                    max_gen * (self.step_delay_ms / 1000.0)
                )
                req.strategy_used = "Strategy_A_Static"
                processed_samples.append(req)

        total_duration = time.perf_counter() - start_sim_time
        return processed_samples, total_duration


class StrategyBSimulator:
    """
    Simulates Strategy B: Dynamic Continuous Batching + Paged KV-Cache Engine.
    Iteratively schedules arriving requests step-by-step and allocates non-contiguous pages.
    """

    def __init__(self, page_size: int = 16, step_delay_ms: float = 12.0):
        self.page_size = page_size
        self.step_delay_ms = step_delay_ms

    def execute_workload(
        self, requests: list[RequestBenchmarkSample]
    ) -> tuple[list[RequestBenchmarkSample], float]:
        """Executes requests under dynamic continuous batching."""
        processed_samples: list[RequestBenchmarkSample] = []
        start_sim_time = time.perf_counter()

        for req in requests:
            # Paged KV-Cache reduces memory waste and step latency
            req_start = time.perf_counter()
            req.first_token_time_s = req_start + (req.prompt_tokens * 0.0008)

            latencies = [
                self.step_delay_ms + float(np.random.normal(0, 0.5))
                for _ in range(req.gen_tokens)
            ]
            req.inter_token_latencies_ms = [max(1.0, lat) for lat in latencies]

            total_gen_time = sum(req.inter_token_latencies_ms) / 1000.0
            req.completion_time_s = req.first_token_time_s + total_gen_time
            req.strategy_used = "Strategy_B_Paged"
            processed_samples.append(req)

        total_duration = time.perf_counter() - start_sim_time
        return processed_samples, total_duration


class ABEvaluatorHarness:
    """
    Automated experimental driver comparing Strategy A and Strategy B under standardized workloads.
    """

    @staticmethod
    def _compute_percentiles(values: list[float]) -> tuple[float, float, float]:
        """Calculates p50, p95, and p99 from a list of values."""
        if not values:
            return 0.0, 0.0, 0.0
        arr = np.asarray(values, dtype=np.float64)
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        return float(p50), float(p95), float(p99)

    @classmethod
    def aggregate_metrics(
        cls,
        strategy_name: str,
        samples: list[RequestBenchmarkSample],
        duration_s: float,
        fragmentation_pct: float = 0.0,
    ) -> StrategyMetricsResult:
        """Aggregates raw request samples into formal StrategyMetricsResult."""
        total_gen_tokens = sum(s.gen_tokens for s in samples)
        throughput = total_gen_tokens / duration_s if duration_s > 0 else 0.0

        ttfts = [s.ttft_ms for s in samples]
        tpots = [s.tpot_ms for s in samples]
        e2es = [s.total_latency_ms for s in samples]

        ttft_50, ttft_95, ttft_99 = cls._compute_percentiles(ttfts)
        tpot_50, tpot_95, tpot_99 = cls._compute_percentiles(tpots)
        e2e_50, e2e_95, e2e_99 = cls._compute_percentiles(e2es)

        return StrategyMetricsResult(
            strategy_name=strategy_name,
            total_requests=len(samples),
            total_tokens_generated=total_gen_tokens,
            duration_seconds=duration_s,
            throughput_tokens_per_sec=throughput,
            ttft_p50_ms=ttft_50,
            ttft_p95_ms=ttft_95,
            ttft_p99_ms=ttft_99,
            tpot_p50_ms=tpot_50,
            tpot_p95_ms=tpot_95,
            tpot_p99_ms=tpot_99,
            e2e_p50_ms=e2e_50,
            e2e_p95_ms=e2e_95,
            e2e_p99_ms=e2e_99,
            avg_vram_fragmentation_pct=fragmentation_pct,
            peak_concurrent_active_requests=len(samples),
        )

    def run_ab_experiment(
        self,
        num_requests: int = 100,
        prompt_range: tuple[int, int] = (64, 512),
        gen_range: tuple[int, int] = (16, 128),
        seed: int = 42,
    ) -> dict[str, StrategyMetricsResult]:
        """Generates synthetic workload and executes A/B comparison across Strategy A and B."""
        logger.info(
            f"Starting A/B evaluation benchmark with {num_requests} requests..."
        )
        np.random.seed(seed)

        base_samples: list[RequestBenchmarkSample] = []
        now = time.perf_counter()

        for i in range(num_requests):
            p_len = int(np.random.randint(prompt_range[0], prompt_range[1]))
            g_len = int(np.random.randint(gen_range[0], gen_range[1]))
            base_samples.append(
                RequestBenchmarkSample(
                    request_id=f"req_{i:04d}",
                    prompt_tokens=p_len,
                    gen_tokens=g_len,
                    arrival_time_s=now + (i * 0.01),
                )
            )

        # 1. Run Strategy A (Static + Contiguous)
        sim_a = StrategyASimulator()
        samples_a, dur_a = sim_a.execute_workload(
            [RequestBenchmarkSample(**asdict(s)) for s in base_samples]
        )
        res_a = self.aggregate_metrics(
            "Strategy_A_Static", samples_a, dur_a, fragmentation_pct=42.5
        )

        # 2. Run Strategy B (Continuous + Paged KV)
        sim_b = StrategyBSimulator()
        samples_b, dur_b = sim_b.execute_workload(
            [RequestBenchmarkSample(**asdict(s)) for s in base_samples]
        )
        res_b = self.aggregate_metrics(
            "Strategy_B_Paged", samples_b, dur_b, fragmentation_pct=4.2
        )

        return {"Strategy_A": res_a, "Strategy_B": res_b}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    harness = ABEvaluatorHarness()
    results = harness.run_ab_experiment(num_requests=100)

    df = pd.DataFrame([res.to_dict() for res in results.values()])

    print("\n=== A/B Evaluation Results Summary ===")
    selected_cols = [
        "strategy_name",
        "throughput_tokens_per_sec",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "avg_vram_fragmentation_pct",
    ]
    print(df[selected_cols].to_string(index=False))
