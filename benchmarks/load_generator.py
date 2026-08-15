"""
Ultra-High-Throughput Asynchronous Load Generator for NexusCache & LLM Engines.

Optimized with:
  - aiohttp + TCPConnector (keep-alive pooling & minimal HTTP parsing overhead)
  - uvloop high-performance event loop
  - Lock-free result collection
  - Real-time TTFT & ITL tracking with zero unnecessary allocations
"""

import argparse
import asyncio
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field

import aiohttp
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Enable uvloop for C-speed async event loop execution
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

console = Console()


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class RequestResult:
    req_id: str
    prompt_len: int
    success: bool
    status_code: int
    ttft_ms: float = 0.0  # Time to First Token (ms)
    total_latency_ms: float = 0.0  # Total End-to-End Latency (ms)
    gen_tokens: int = 0
    itl_ms: list[float] = field(default_factory=list)
    error_msg: str = ""


@dataclass
class BenchmarkSummary:
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_sec: float
    actual_qps: float
    total_tokens_generated: int
    token_throughput_tps: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    e2e_p50_ms: float
    e2e_p90_ms: float
    e2e_p99_ms: float


# ============================================================================
# Core Async Load Generator Engine
# ============================================================================


class FastLoadGenerator:
    def __init__(
        self,
        target_url: str,
        request_rate: float,
        total_requests: int,
        concurrency_limit: int,
        poisson_arrival: bool = True,
        stream: bool = True,
        prompt_prefix: str = "NexusCache benchmark prompt: ",
        input_token_len: int = 128,
        max_output_tokens: int = 128,
        timeout: float = 120.0,
    ):
        self.target_url = target_url
        self.request_rate = request_rate
        self.total_requests = total_requests
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.poisson_arrival = poisson_arrival
        self.stream = stream
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

        # Pre-synthesize payload string once
        base_prompt = prompt_prefix + (
            " token" * max(0, input_token_len - len(prompt_prefix.split()))
        )
        self.prompt = base_prompt.strip()

        self.payload = {
            "model": "nexuscache-model",
            "messages": [{"role": "user", "content": self.prompt}],
            "max_tokens": self.max_output_tokens,
            "stream": self.stream,
            "temperature": 0.0,
        }

        self.results: list[RequestResult] = []

    def _get_inter_arrival_time(self) -> float:
        if self.request_rate <= 0:
            return 0.0
        if self.poisson_arrival:
            return random.expovariate(self.request_rate)
        return 1.0 / self.request_rate

    async def _send_request(
        self, session: aiohttp.ClientSession, req_id: str
    ) -> RequestResult:
        result = RequestResult(
            req_id=req_id,
            prompt_len=len(self.prompt.split()),
            success=False,
            status_code=0,
        )

        start_time = time.perf_counter()
        first_token_time: float | None = None
        last_chunk_time: float | None = None

        async with self.semaphore:
            try:
                if self.stream:
                    async with session.post(
                        self.target_url, json=self.payload
                    ) as response:
                        result.status_code = response.status
                        if response.status == 200:
                            async for line_bytes in response.content:
                                chunk_now = time.perf_counter()
                                line = line_bytes.decode("utf-8").strip()

                                if not line or not line.startswith("data:"):
                                    continue
                                if line == "data: [DONE]":
                                    break

                                # Fast TTFT & ITL calculation
                                if first_token_time is None:
                                    first_token_time = chunk_now
                                    result.ttft_ms = (
                                        first_token_time - start_time
                                    ) * 1000.0
                                elif last_chunk_time is not None:
                                    result.itl_ms.append(
                                        (chunk_now - last_chunk_time) * 1000.0
                                    )

                                last_chunk_time = chunk_now
                                result.gen_tokens += 1

                            result.success = True
                        else:
                            result.error_msg = f"HTTP {response.status}"
                else:
                    async with session.post(
                        self.target_url, json=self.payload
                    ) as response:
                        result.status_code = response.status
                        now = time.perf_counter()
                        if response.status == 200:
                            data = await response.json()
                            result.success = True
                            result.ttft_ms = (now - start_time) * 1000.0
                            result.gen_tokens = data.get("usage", {}).get(
                                "completion_tokens", self.max_output_tokens
                            )
                        else:
                            result.error_msg = f"HTTP {response.status}"

            except Exception as e:
                result.success = False
                result.error_msg = str(e)

            end_time = time.perf_counter()
            result.total_latency_ms = (end_time - start_time) * 1000.0

            if result.ttft_ms == 0.0 and result.success:
                result.ttft_ms = result.total_latency_ms

        # Lock-free append
        self.results.append(result)
        return result

    async def run(self) -> BenchmarkSummary:
        # High performance TCP connection pool
        connector = aiohttp.TCPConnector(
            limit=0,  # No limit on total pool size
            limit_per_host=0,  # No limit per host
            ttl_dns_cache=300,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout, connect=10.0)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            tasks = []
            start_benchmark_time = time.perf_counter()

            for i in range(self.total_requests):
                req_id = f"req-{i+1:05d}"
                task = asyncio.create_task(self._send_request(session, req_id))
                tasks.append(task)

                delay = self._get_inter_arrival_time()
                if delay > 0:
                    await asyncio.sleep(delay)

            await asyncio.gather(*tasks, return_exceptions=True)
            total_duration = time.perf_counter() - start_benchmark_time

        return self._compute_summary(total_duration)

    def _compute_summary(self, duration_sec: float) -> BenchmarkSummary:
        if not self.results:
            raise ValueError("No requests were completed during benchmark.")

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        def percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            sorted_v = sorted(values)
            k = (len(sorted_v) - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_v[int(k)]
            return sorted_v[int(f)] * (c - k) + sorted_v[int(c)] * (k - f)

        ttfts = [r.ttft_ms for r in successful if r.ttft_ms > 0]
        e2es = [r.total_latency_ms for r in successful]
        total_tokens = sum(r.gen_tokens for r in successful)

        return BenchmarkSummary(
            total_requests=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            duration_sec=duration_sec,
            actual_qps=len(self.results) / duration_sec if duration_sec > 0 else 0,
            total_tokens_generated=total_tokens,
            token_throughput_tps=total_tokens / duration_sec if duration_sec > 0 else 0,
            ttft_p50_ms=percentile(ttfts, 50),
            ttft_p90_ms=percentile(ttfts, 90),
            ttft_p99_ms=percentile(ttfts, 99),
            e2e_p50_ms=percentile(e2es, 50),
            e2e_p90_ms=percentile(e2es, 90),
            e2e_p99_ms=percentile(e2es, 99),
        )


# ============================================================================
# CLI & Terminal UI
# ============================================================================


def print_summary_table(summary: BenchmarkSummary) -> None:
    table = Table(
        title="NexusCache Load Generator - Benchmark Results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="dim", width=30)
    table.add_column("Value", justify="right", style="bold green")

    table.add_row("Total Duration", f"{summary.duration_sec:.2f} s")
    table.add_row(
        "Total / Successful Requests",
        f"{summary.total_requests} / {summary.successful_requests}",
    )
    table.add_row(
        "Failed Requests",
        f"{summary.failed_requests}",
        style="bold red" if summary.failed_requests > 0 else "dim",
    )
    table.add_row("Achieved Rate (QPS)", f"{summary.actual_qps:.2f} req/s")
    table.add_row("Token Throughput", f"{summary.token_throughput_tps:.2f} tokens/s")
    table.add_section()
    table.add_row("TTFT (P50)", f"{summary.ttft_p50_ms:.2f} ms")
    table.add_row("TTFT (P90)", f"{summary.ttft_p90_ms:.2f} ms")
    table.add_row("TTFT (P99)", f"{summary.ttft_p99_ms:.2f} ms")
    table.add_section()
    table.add_row("E2E Latency (P50)", f"{summary.e2e_p50_ms:.2f} ms")
    table.add_row("E2E Latency (P90)", f"{summary.e2e_p90_ms:.2f} ms")
    table.add_row("E2E Latency (P99)", f"{summary.e2e_p99_ms:.2f} ms")

    console.print(Panel(table, expand=False))


def main():
    parser = argparse.ArgumentParser(description="High-Speed NexusCache Load Generator")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1/chat/completions",
        help="Target API endpoint URL",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=0.0,
        help="Target QPS (0 = unlimited max throughput speed)",
    )
    parser.add_argument(
        "--num-requests", type=int, default=100, help="Total requests to fire"
    )
    parser.add_argument(
        "--concurrency", type=int, default=32, help="Max parallel connections"
    )
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="Use uniform pacing instead of Poisson process",
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="Disable response streaming"
    )
    parser.add_argument(
        "--prompt-tokens", type=int, default=128, help="Prompt length in tokens"
    )
    parser.add_argument(
        "--output-tokens", type=int, default=128, help="Max generation tokens"
    )
    parser.add_argument(
        "--output-json", type=str, default=None, help="Save summary to JSON file"
    )

    args = parser.parse_args()

    console.print(
        f"[bold green]Starting benchmark on[/bold green] [yellow]{args.url}[/yellow] "
        f"([cyan]QPS={args.qps}[/cyan], [cyan]Requests={args.num_requests}[/cyan], [cyan]Concurrency={args.concurrency}[/cyan])..."
    )

    generator = FastLoadGenerator(
        target_url=args.url,
        request_rate=args.qps,
        total_requests=args.num_requests,
        concurrency_limit=args.concurrency,
        poisson_arrival=not args.uniform,
        stream=not args.no_stream,
        input_token_len=args.prompt_tokens,
        max_output_tokens=args.output_tokens,
    )

    summary = asyncio.run(generator.run())

    if summary.failed_requests > 0:
        console.print("\n[bold red]Sample Failure Errors:[/bold red]")
        errors = list({r.error_msg for r in generator.results if not r.success})
        for err in errors[:5]:
            console.print(f"{err}")

    print_summary_table(summary)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(asdict(summary), f, indent=2)
        console.print(f"[dim]Results saved to [bold]{args.output_json}[/bold][/dim]")


if __name__ == "__main__":
    main()
