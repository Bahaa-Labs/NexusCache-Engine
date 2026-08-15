"""
Observability and Prometheus Metrics Module

Provides low-overhead, thread-safe Prometheus metrics instrumentation for monitoring:
- Request throughput and latency SLAs (TTFT, TPOT, E2E).
- VRAM and KV-Cache block allocation.
- Scheduler queue depths and backpressure rejection stats.
- Multi-GPU engine health.
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import cast

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger("nexuscache.utils.metrics")

# Default latency buckets (in seconds) tailored for LLM serving
LATENCY_BUCKETS_SECONDS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)

# TPOT latency buckets (in seconds, typically sub-100ms)
TPOT_BUCKETS_SECONDS = (
    0.001,
    0.005,
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.2,
    0.5,
)


class EngineMetrics:
    """
    Centralized Prometheus telemetry metric collection registry for the LLM engine.
    """

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry(auto_describe=True)

        # ---------------------------------------------------------------------
        # 1. Request Counters
        # ---------------------------------------------------------------------
        self.request_total = Counter(
            "llm_requests_total",
            "Total number of incoming inference requests received.",
            labelnames=["priority", "status"],
            registry=self.registry,
        )

        self.request_rejections_429 = Counter(
            "llm_request_rejections_total",
            "Total number of rejected requests due to backpressure/full queue (HTTP 429).",
            labelnames=["reason"],
            registry=self.registry,
        )

        self.token_count_total = Counter(
            "llm_processed_tokens_total",
            "Total count of processed tokens across prefill and decode stages.",
            labelnames=["type"],  # 'prompt' or 'generation'
            registry=self.registry,
        )

        # ---------------------------------------------------------------------
        # 2. Latency & Performance Histograms
        # ---------------------------------------------------------------------
        self.time_to_first_token_s = Histogram(
            "llm_time_to_first_token_seconds",
            "Latency from request arrival to generation of first token (TTFT).",
            labelnames=["priority"],
            buckets=LATENCY_BUCKETS_SECONDS,
            registry=self.registry,
        )

        self.time_per_output_token_s = Histogram(
            "llm_time_per_output_token_seconds",
            "Inter-token generation latency during decode phase (TPOT).",
            labelnames=["priority"],
            buckets=TPOT_BUCKETS_SECONDS,
            registry=self.registry,
        )

        self.request_e2e_latency_s = Histogram(
            "llm_request_e2e_latency_seconds",
            "Total end-to-end request processing time from arrival to completion.",
            labelnames=["priority"],
            buckets=LATENCY_BUCKETS_SECONDS,
            registry=self.registry,
        )

        # ---------------------------------------------------------------------
        # 3. Memory & KV Cache Gauges
        # ---------------------------------------------------------------------
        self.kv_cache_usage_percentage = Gauge(
            "llm_kv_cache_usage_fraction",
            "Current fraction of total physical GPU KV-cache blocks in use.",
            registry=self.registry,
        )

        self.gpu_vram_allocated_bytes = Gauge(
            "llm_gpu_vram_allocated_bytes",
            "Current VRAM allocation in bytes across hardware worker ranks.",
            labelnames=["device_id"],
            registry=self.registry,
        )

        self.cpu_swap_usage_percentage = Gauge(
            "llm_cpu_swap_usage_fraction",
            "Current fraction of preempted CPU swap cache memory blocks in use.",
            registry=self.registry,
        )

        # ---------------------------------------------------------------------
        # 4. Queue & Batching Gauges
        # ---------------------------------------------------------------------
        self.queue_depth = Gauge(
            "llm_queue_depth_requests",
            "Current number of pending requests waiting in scheduler queue.",
            registry=self.registry,
        )

        self.num_running_sequences = Gauge(
            "llm_running_sequences",
            "Current count of active sequences in iteration execution batch.",
            registry=self.registry,
        )

        self.batch_size = Gauge(
            "llm_current_batch_size_tokens",
            "Total prompt/decode token count in the active iteration batch pass.",
            registry=self.registry,
        )

    # -------------------------------------------------------------------------
    # Helper Context Managers & Exporters
    # -------------------------------------------------------------------------

    @contextmanager
    def measure_latency(
        self, histogram: Histogram, labels: dict[str, str] | None = None
    ) -> Generator[None, None, None]:
        """
        Context manager helper for timing code execution blocks with high precision.
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            if labels:
                histogram.labels(**labels).observe(elapsed)
            else:
                histogram.observe(elapsed)

    def export_metrics_bytes(self) -> bytes:
        """
        Renders the current Prometheus metrics snapshot into standard scrape format bytes.
        """
        return cast(bytes, generate_latest(self.registry))


# ============================================================================
# Global Singleton Accessor
# ============================================================================

_METRICS_INSTANCE: EngineMetrics | None = None


def get_metrics() -> EngineMetrics:
    """
    Retrieves or initializes the global telemetry metrics singleton instance.
    """
    global _METRICS_INSTANCE
    if _METRICS_INSTANCE is None:
        _METRICS_INSTANCE = EngineMetrics()
        logger.info("Initialized global EngineMetrics registry.")
    return _METRICS_INSTANCE
