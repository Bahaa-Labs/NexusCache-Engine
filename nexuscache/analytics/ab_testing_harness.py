"""
Benchmarking and data collection framework for running live concurrent workload
comparisons across caching paradigms (e.g., Standard Cache vs. NexusCache Paged KV-Cache).
"""

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger("nexuscache.analytics.ab_testing_harness")


class CacheStrategy(Enum):
    """Supported caching strategies for A/B benchmarking."""

    STANDARD_KV_CACHE = "STANDARD_KV_CACHE"
    NEXUS_PAGED_CACHE = "NEXUS_PAGED_CACHE"


@dataclass(frozen=True)
class RequestSpec:
    """Specification for a single inference request in the synthetic stream."""

    request_id: str
    prompt_tokens: int
    max_gen_tokens: int
    arrival_offset_sec: float
    prefix_block_ids: list[int] = field(default_factory=list)


@dataclass
class RequestMetric:
    """Detailed performance metrics for a single completed request."""

    request_id: str
    strategy: str
    prompt_tokens: int
    gen_tokens: int
    ttft_ms: float  # Time To First Token (Prefill phase latency)
    tpot_ms: float  # Time Per Output Token (Average Decode phase step latency)
    total_latency_ms: float  # End-to-end total request latency
    cache_hits: int  # Number of KV blocks/tokens matched in prefix cache
    cache_misses: int  # Number of KV blocks/tokens required to be computed
    cache_hit_ratio: float  # cache_hits / (cache_hits + cache_misses)
    success: bool
    error_message: str | None = None


@dataclass
class BenchmarkSummary:
    """Aggregated benchmarking metrics report across an entire test run."""

    strategy: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_sec: float
    system_throughput_tok_per_sec: float
    avg_ttft_ms: float
    p50_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    avg_tpot_ms: float
    p50_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float
    avg_total_latency_ms: float
    overall_cache_hit_ratio: float
    raw_metrics: list[RequestMetric] = field(default_factory=list, repr=False)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts raw request metrics into a Pandas DataFrame."""
        return pd.DataFrame([asdict(m) for m in self.raw_metrics])

    def to_dict(self) -> dict[str, Any]:
        """Returns clean structured JSON-serializable dictionary summary."""
        res = asdict(self)
        res.pop("raw_metrics", None)
        return res


class CacheEngineProtocol(Protocol):
    """Protocol interface that candidate caching backends must satisfy."""

    def process_request(self, spec: RequestSpec) -> AsyncIterator[str]:
        """Processes request and yields generated tokens asynchronously."""
        ...

    def get_cache_stats(self) -> tuple[int, int]:
        """Returns tuple of (total_hits, total_misses)."""
        ...


class SyntheticWorkloadGenerator:
    """Generates synthetic multi-tenant request streams following a Poisson arrival process."""

    @staticmethod
    def generate_poisson_stream(
        num_requests: int,
        request_rate_qps: float,
        mean_prompt_len: int = 512,
        std_prompt_len: int = 128,
        mean_gen_len: int = 128,
        std_gen_len: int = 32,
        shared_prefix_ratio: float = 0.3,
        num_shared_prefixes: int = 4,
        seed: int = 42,
    ) -> list[RequestSpec]:
        """Generates a reproducible list of request specifications with Poisson arrival delays."""
        random.seed(seed)
        np.random.seed(seed)

        requests: list[RequestSpec] = []
        current_time = 0.0

        for i in range(num_requests):
            inter_arrival = random.expovariate(request_rate_qps)
            current_time += inter_arrival

            prompt_len = int(max(16, np.random.normal(mean_prompt_len, std_prompt_len)))
            gen_len = int(max(1, np.random.normal(mean_gen_len, std_gen_len)))

            prefix_ids = []
            if random.random() < shared_prefix_ratio:
                prefix_id = random.randint(0, num_shared_prefixes - 1)
                prefix_ids = [1000 + prefix_id]

            requests.append(
                RequestSpec(
                    request_id=f"req_{i:06d}",
                    prompt_tokens=prompt_len,
                    max_gen_tokens=gen_len,
                    arrival_offset_sec=current_time,
                    prefix_block_ids=prefix_ids,
                )
            )

        return requests


class MockCacheEngine:
    """High-fidelity simulated cache engine representing Standard or NexusCache behavior."""

    def __init__(
        self, strategy: CacheStrategy, simulated_compute_ms_per_tok: float = 0.5
    ):
        self.strategy = strategy
        self.compute_speed = simulated_compute_ms_per_tok
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_stats(self) -> tuple[int, int]:
        return self.cache_hits, self.cache_misses

    async def process_request(self, spec: RequestSpec) -> AsyncIterator[str]:
        if spec.prefix_block_ids:
            if self.strategy == CacheStrategy.NEXUS_PAGED_CACHE:
                hits = min(spec.prompt_tokens, 128)
                misses = spec.prompt_tokens - hits
            else:
                hits = min(spec.prompt_tokens, 32)
                misses = spec.prompt_tokens - hits
        else:
            hits = 0
            misses = spec.prompt_tokens

        self.cache_hits += hits
        self.cache_misses += misses

        prefill_latency_sec = (misses * 0.001 * self.compute_speed) + 0.005
        await asyncio.sleep(prefill_latency_sec)
        yield "FIRST_TOKEN"

        for _ in range(spec.max_gen_tokens - 1):
            decode_latency_sec = (0.001 * self.compute_speed) + 0.002
            await asyncio.sleep(decode_latency_sec)
            yield "TOKEN"


class ABTestingHarness:
    """Main execution engine for running concurrent A/B performance benchmarks."""

    def __init__(self, concurrency_limit: int = 32):
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def _execute_single_request(
        self, engine: CacheEngineProtocol, spec: RequestSpec, strategy_label: str
    ) -> RequestMetric:
        """Executes a single request while taking precise high-resolution metrics."""
        async with self.semaphore:
            start_time = time.perf_counter()
            first_token_time: float | None = None
            generated_tokens = 0
            initial_hits, initial_misses = engine.get_cache_stats()

            try:
                async for _ in engine.process_request(spec):
                    generated_tokens += 1
                    if generated_tokens == 1:
                        first_token_time = time.perf_counter()

                end_time = time.perf_counter()

                if first_token_time is None:
                    first_token_time = end_time

                ttft_ms = (first_token_time - start_time) * 1000.0
                total_latency_ms = (end_time - start_time) * 1000.0

                decode_tokens = max(1, generated_tokens - 1)
                decode_duration_ms = max(0.0, total_latency_ms - ttft_ms)
                tpot_ms = decode_duration_ms / float(decode_tokens)

                curr_hits, curr_misses = engine.get_cache_stats()
                req_hits = curr_hits - initial_hits
                req_misses = curr_misses - initial_misses
                total_kv = req_hits + req_misses
                hit_ratio = (req_hits / float(total_kv)) if total_kv > 0 else 0.0

                return RequestMetric(
                    request_id=spec.request_id,
                    strategy=strategy_label,
                    prompt_tokens=spec.prompt_tokens,
                    gen_tokens=generated_tokens,
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    total_latency_ms=total_latency_ms,
                    cache_hits=req_hits,
                    cache_misses=req_misses,
                    cache_hit_ratio=hit_ratio,
                    success=True,
                )

            except Exception as exc:
                end_time = time.perf_counter()
                logger.error(
                    f"Request {spec.request_id} failed under {strategy_label}: {str(exc)}"
                )
                return RequestMetric(
                    request_id=spec.request_id,
                    strategy=strategy_label,
                    prompt_tokens=spec.prompt_tokens,
                    gen_tokens=generated_tokens,
                    ttft_ms=0.0,
                    tpot_ms=0.0,
                    total_latency_ms=(end_time - start_time) * 1000.0,
                    cache_hits=0,
                    cache_misses=0,
                    cache_hit_ratio=0.0,
                    success=False,
                    error_message=str(exc),
                )

    async def run_benchmark(
        self,
        engine: CacheEngineProtocol,
        strategy: CacheStrategy,
        workload: list[RequestSpec],
    ) -> BenchmarkSummary:
        """Runs concurrent workload benchmark for a specific caching engine."""
        logger.info(
            f"Starting A/B benchmark run for strategy: {strategy.value} ({len(workload)} requests)"
        )
        start_benchmark_time = time.perf_counter()

        async def scheduled_worker(spec: RequestSpec) -> RequestMetric:
            if spec.arrival_offset_sec > 0:
                await asyncio.sleep(spec.arrival_offset_sec)
            return await self._execute_single_request(engine, spec, strategy.value)

        tasks = [asyncio.create_task(scheduled_worker(req)) for req in workload]
        results: list[RequestMetric] = await asyncio.gather(*tasks)

        end_benchmark_time = time.perf_counter()
        total_duration = end_benchmark_time - start_benchmark_time

        successful = [r for r in results if r.success]
        failed_count = len(results) - len(successful)

        if not successful:
            raise RuntimeError(f"All requests failed for strategy {strategy.value}")

        total_generated_tokens = sum(r.gen_tokens for r in successful)
        throughput = (
            total_generated_tokens / total_duration if total_duration > 0 else 0.0
        )

        ttfts = [r.ttft_ms for r in successful]
        tpots = [r.tpot_ms for r in successful]
        total_latencies = [r.total_latency_ms for r in successful]

        total_hits = sum(r.cache_hits for r in successful)
        total_misses = sum(r.cache_misses for r in successful)
        denom = total_hits + total_misses
        overall_hit_ratio = (total_hits / float(denom)) if denom > 0 else 0.0

        return BenchmarkSummary(
            strategy=strategy.value,
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=failed_count,
            total_duration_sec=total_duration,
            system_throughput_tok_per_sec=throughput,
            avg_ttft_ms=float(np.mean(ttfts)),
            p50_ttft_ms=float(np.percentile(ttfts, 50)),
            p90_ttft_ms=float(np.percentile(ttfts, 90)),
            p99_ttft_ms=float(np.percentile(ttfts, 99)),
            avg_tpot_ms=float(np.mean(tpots)),
            p50_tpot_ms=float(np.percentile(tpots, 50)),
            p90_tpot_ms=float(np.percentile(tpots, 90)),
            p99_tpot_ms=float(np.percentile(tpots, 99)),
            avg_total_latency_ms=float(np.mean(total_latencies)),
            overall_cache_hit_ratio=overall_hit_ratio,
            raw_metrics=results,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def main():
        print("=== Initializing Synthetic Workload ===")
        workload = SyntheticWorkloadGenerator.generate_poisson_stream(
            num_requests=50,
            request_rate_qps=20.0,
            mean_prompt_len=256,
            mean_gen_len=64,
            shared_prefix_ratio=0.5,
        )

        harness = ABTestingHarness(concurrency_limit=16)

        std_engine = MockCacheEngine(CacheStrategy.STANDARD_KV_CACHE)
        std_summary = await harness.run_benchmark(
            std_engine, CacheStrategy.STANDARD_KV_CACHE, workload
        )

        nexus_engine = MockCacheEngine(CacheStrategy.NEXUS_PAGED_CACHE)
        nexus_summary = await harness.run_benchmark(
            nexus_engine, CacheStrategy.NEXUS_PAGED_CACHE, workload
        )

        print("\n=== A/B Test Execution Summary ===")
        comparison_df = pd.DataFrame([std_summary.to_dict(), nexus_summary.to_dict()])[
            [
                "strategy",
                "system_throughput_tok_per_sec",
                "avg_ttft_ms",
                "p90_ttft_ms",
                "avg_tpot_ms",
                "overall_cache_hit_ratio",
            ]
        ]

        print(comparison_df.to_string(index=False))

    asyncio.run(main())
