"""
Static vs. Dynamic Iteration Scheduler Benchmark Harness.

Compares standard continuous batching (Static Baseline) against SLA-aware
scheduling with prefix caching and smart victim preemption (Dynamic Scheduler).
"""

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

from nexuscache.server.dynamic_scheduler import (
    DynamicScheduler,
    DynamicSchedulerConfig,
    DynamicSequence,
    PriorityLevel,
)
from nexuscache.server.scheduler import (
    Scheduler,
    SchedulerConfig,
)

# Optional rich formatting for CLI results
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ============================================================================
# Memory Subsystem Fallback / C++ Interface Layer
# ============================================================================


class FallbackBlockManager:
    """Python fallback for C++ BlockManager when native binary is absent."""

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.total_blocks = num_blocks
        self.free_blocks = num_blocks
        self.block_size = block_size

    def get_num_free_blocks(self) -> int:
        return self.free_blocks

    def allocate(self, count: int = 1) -> bool:
        if self.free_blocks >= count:
            self.free_blocks -= count
            return True
        return False

    def free(self, count: int = 1) -> None:
        self.free_blocks = min(self.total_blocks, self.free_blocks + count)


class FallbackPageTable:
    """Python fallback for C++ PageTable when native binary is absent."""

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.active_sequences: dict[int, list[int]] = {}
        self._next_block_id = 1000

    def register_sequence(self, seq_id: int) -> None:
        if seq_id not in self.active_sequences:
            self.active_sequences[seq_id] = []

    def append_tokens(self, seq_id: int, num_tokens: int) -> None:
        if seq_id not in self.active_sequences:
            self.register_sequence(seq_id)
        current_blocks = len(self.active_sequences[seq_id])
        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        for _ in range(needed_blocks - current_blocks):
            self.active_sequences[seq_id].append(self._next_block_id)
            self._next_block_id += 1

    def has_sequence(self, seq_id: int) -> bool:
        return seq_id in self.active_sequences

    def free_sequence(self, seq_id: int) -> tuple[int, list[int]]:
        blocks = self.active_sequences.pop(seq_id, [])
        return len(blocks), blocks

    def get_block_table(self, seq_id: int) -> list[int]:
        return self.active_sequences.get(seq_id, [])


def create_memory_subsystem(num_blocks: int, block_size: int) -> tuple[Any, Any]:
    """Instantiates C++ BlockManager/PageTable or falls back to Python stubs."""
    try:
        import nexuscache._C as _C

        config = _C.BlockAllocatorConfig()
        config.num_blocks = num_blocks
        config.block_size = block_size

        return _C.BlockManager(config), _C.PageTable(block_size)
    except (ImportError, AttributeError, TypeError):
        return FallbackBlockManager(num_blocks, block_size), FallbackPageTable(
            block_size
        )


# ============================================================================
# Workload & Metrics Data Structures
# ============================================================================


@dataclass
class SyntheticRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    arrival_time: float
    priority: PriorityLevel = PriorityLevel.NORMAL
    sla_target_ttft_ms: float = 100.0


@dataclass
class BenchmarkMetrics:
    total_requests: int = 0
    completed_requests: int = 0
    preemption_count: int = 0
    total_execution_steps: int = 0
    total_generated_tokens: int = 0
    simulation_duration_sec: float = 0.0
    ttft_ms: list[float] = field(default_factory=list)
    tpot_ms: list[float] = field(default_factory=list)
    sla_violations: int = 0
    prefix_hit_count: int = 0
    peak_memory_blocks_used: int = 0

    def percentile(self, data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_d = sorted(data)
        k = (len(sorted_d) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_d[int(k)]
        return sorted_d[int(f)] * (c - k) + sorted_d[int(c)] * (k - f)


# ============================================================================
# Synthetic Workload Generator
# ============================================================================


def generate_workload(
    num_requests: int,
    shared_prefix_ratio: float,
    request_rate: float,
    mean_prompt_len: int = 256,
    mean_gen_len: int = 128,
    seed: int = 42,
) -> list[SyntheticRequest]:
    """Generates a realistic multi-tenant workload with shared prompt prefixes."""
    random.seed(seed)
    requests: list[SyntheticRequest] = []

    # Common shared system prompt prefix (e.g., 64 tokens)
    common_prefix = [random.randint(100, 5000) for _ in range(64)]
    current_time = 0.0

    for i in range(num_requests):
        inter_arrival = random.expovariate(request_rate) if request_rate > 0 else 0.0
        current_time += inter_arrival

        prompt_len = max(32, int(random.gauss(mean_prompt_len, 64)))
        if random.random() < shared_prefix_ratio:
            remaining_tokens = max(0, prompt_len - len(common_prefix))
            prompt_tokens = common_prefix + [
                random.randint(5001, 20000) for _ in range(remaining_tokens)
            ]
        else:
            prompt_tokens = [random.randint(5001, 20000) for _ in range(prompt_len)]

        gen_len = max(16, int(random.gauss(mean_gen_len, 32)))

        # Priority tier assignment
        rand_p = random.random()
        if rand_p < 0.05:
            priority = PriorityLevel.CRITICAL
            sla_ttft = 50.0
        elif rand_p < 0.20:
            priority = PriorityLevel.HIGH
            sla_ttft = 80.0
        elif rand_p < 0.80:
            priority = PriorityLevel.NORMAL
            sla_ttft = 150.0
        else:
            priority = PriorityLevel.LOW
            sla_ttft = 300.0

        requests.append(
            SyntheticRequest(
                request_id=f"req-{i+1:04d}",
                prompt_token_ids=prompt_tokens,
                max_new_tokens=gen_len,
                arrival_time=current_time,
                priority=priority,
                sla_target_ttft_ms=sla_ttft,
            )
        )

    return requests


# ============================================================================
# Step Execution Latency Model (Compute + Memory Bound)
# ============================================================================


def estimate_step_execution_time(
    prefill_tokens: int,
    decode_tokens: int,
    base_step_ms: float = 2.0,
    per_prefill_token_ms: float = 0.015,
    per_decode_token_ms: float = 0.08,
) -> float:
    """
    Simulates GPU step runtime accurately.
    Prefill: Compute-bound GEMM time.
    Decode: Memory bandwidth-bound lookup time.
    """
    total_ms = (
        base_step_ms
        + (prefill_tokens * per_prefill_token_ms)
        + (decode_tokens * per_decode_token_ms)
    )
    return total_ms / 1000.0  # Return in seconds


# ============================================================================
# Core Benchmark Engine
# ============================================================================


def run_simulation(
    scheduler_type: str,
    config: SchedulerConfig,
    workload: list[SyntheticRequest],
) -> BenchmarkMetrics:
    """Executes discrete event step simulation loop for static vs dynamic schedulers."""
    bm, pt = create_memory_subsystem(config.max_paged_blocks, config.block_size)

    scheduler: Scheduler
    if scheduler_type == "dynamic":
        dyn_config = DynamicSchedulerConfig(
            max_num_batched_tokens=getattr(config, "max_num_batched_tokens", 4096),
            max_num_seqs=config.max_num_seqs,
            max_paged_blocks=256,
            block_size=config.block_size,
            enable_prefix_caching=True,
        )
        scheduler = DynamicScheduler(
            config=dyn_config, block_manager=bm, page_table=pt
        )
    else:
        scheduler = Scheduler(config=config, block_manager=bm, page_table=pt)

    metrics = BenchmarkMetrics(total_requests=len(workload))
    pending_workload = list(workload)
    active_seq_meta: dict[int, dict[str, Any]] = {}

    current_sim_time = 0.0

    while pending_workload or scheduler.get_num_unfinished_sequences() > 0:
        metrics.total_execution_steps += 1

        # 1. Admit arriving requests
        arrived = [r for r in pending_workload if r.arrival_time <= current_sim_time]
        for req in arrived:
            pending_workload.remove(req)

            if isinstance(scheduler, DynamicScheduler):
                seq_id = scheduler.add_dynamic_sequence(
                    request_id=req.request_id,
                    prompt_token_ids=req.prompt_token_ids,
                    max_new_tokens=req.max_new_tokens,
                    priority=req.priority,
                    sla_target_ttft_ms=req.sla_target_ttft_ms,
                )
            else:
                seq_id = scheduler.add_sequence(
                    request_id=req.request_id,
                    prompt_token_ids=req.prompt_token_ids,
                    max_new_tokens=req.max_new_tokens,
                )

            active_seq_meta[seq_id] = {
                "arrival_time": req.arrival_time,
                "first_token_time": None,
                "last_token_time": None,
                "token_times": [],
                "sla_ttft_ms": req.sla_target_ttft_ms,
            }

        # 2. Run scheduler step pass with simulation time alignment
        if isinstance(scheduler, DynamicScheduler):
            batch = scheduler.schedule(sim_time=current_sim_time)
        else:
            batch = scheduler.schedule()

        if batch.preempted_seqs:
            metrics.preemption_count += len(batch.preempted_seqs)

        # Count active prefill/decode tokens for performance modeling
        prefill_token_count = 0
        for seq in batch.prefill_seqs:
            if isinstance(seq, DynamicSequence):
                prefill_token_count += seq.remaining_prefill_tokens
            else:
                prefill_token_count += seq.num_prompt_tokens

        decode_token_count = len(batch.decode_seqs)

        # Estimate hardware execution step duration
        step_delta_sec = estimate_step_execution_time(
            prefill_token_count, decode_token_count
        )

        # 3. Process prefill pass
        for seq in batch.prefill_seqs:
            meta = active_seq_meta[seq.seq_id]
            if meta["first_token_time"] is None:
                if isinstance(seq, DynamicSequence) and seq.computed_prefix_tokens > 0:
                    metrics.prefix_hit_count += 1

                meta["first_token_time"] = current_sim_time + step_delta_sec

                ttft = (meta["first_token_time"] - meta["arrival_time"]) * 1000.0
                metrics.ttft_ms.append(ttft)
                if ttft > meta["sla_ttft_ms"]:
                    metrics.sla_violations += 1

        # 4. Process decode pass
        for seq in batch.decode_seqs:
            meta = active_seq_meta[seq.seq_id]
            seq.generated_token_ids.append(random.randint(100, 1000))
            metrics.total_generated_tokens += 1

            if meta["last_token_time"] is not None:
                tpot = (current_sim_time - meta["last_token_time"]) * 1000.0
                metrics.tpot_ms.append(tpot)
            meta["last_token_time"] = current_sim_time

            # Handle sequence completion & clean up memory
            if seq.is_finished():
                scheduler.finish_sequence(seq.seq_id)
                if hasattr(pt, "free_sequence"):
                    freed_blocks, _ = pt.free_sequence(seq.seq_id)
                    if freed_blocks > 0 and hasattr(bm, "free"):
                        bm.free(freed_blocks)
                metrics.completed_requests += 1

        current_sim_time += step_delta_sec

    metrics.simulation_duration_sec = current_sim_time
    return metrics


# ============================================================================
# Results Display Engine
# ============================================================================


def print_results(static_m: BenchmarkMetrics, dyn_m: BenchmarkMetrics) -> None:
    """Prints comprehensive side-by-side performance comparison table."""

    def calc_delta(
        val_dyn: float, val_stat: float, lower_is_better: bool = True
    ) -> str:
        if val_stat == 0.0:
            return "N/A"
        pct = ((val_dyn - val_stat) / val_stat) * 100.0
        if (pct < 0 and lower_is_better) or (pct > 0 and not lower_is_better):
            return f"[bold green]{pct:+.1f}%[/bold green]"
        elif pct == 0:
            return "0.0%"
        else:
            return f"[bold red]{pct:+.1f}%[/bold red]"

    s_tps = (
        static_m.total_generated_tokens / static_m.simulation_duration_sec
        if static_m.simulation_duration_sec > 0
        else 0
    )
    d_tps = (
        dyn_m.total_generated_tokens / dyn_m.simulation_duration_sec
        if dyn_m.simulation_duration_sec > 0
        else 0
    )

    if HAS_RICH:
        console = Console()
        table = Table(
            title="NexusCache Engine: Static vs. Dynamic Scheduler Benchmark Results",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric Name", style="dim", width=32)
        table.add_column("Static Baseline", justify="right")
        table.add_column("Dynamic Scheduler", justify="right", style="bold green")
        table.add_column("Performance Delta", justify="right")

        table.add_row(
            "Completed Requests",
            f"{static_m.completed_requests}",
            f"{dyn_m.completed_requests}",
            "",
        )
        table.add_row(
            "Preemption Count",
            f"{static_m.preemption_count}",
            f"{dyn_m.preemption_count}",
            calc_delta(dyn_m.preemption_count, static_m.preemption_count),
        )
        table.add_row(
            "Prefix Cache Hits",
            "0",
            f"{dyn_m.prefix_hit_count}",
            "[bold green]N/A (New)[/bold green]",
        )
        table.add_row(
            "SLA TTFT Violations",
            f"{static_m.sla_violations}",
            f"{dyn_m.sla_violations}",
            calc_delta(dyn_m.sla_violations, static_m.sla_violations),
        )
        table.add_section()

        table.add_row(
            "Throughput (TPS)",
            f"{s_tps:.2f} tok/s",
            f"{d_tps:.2f} tok/s",
            calc_delta(d_tps, s_tps, lower_is_better=False),
        )
        table.add_row(
            "TTFT (P50)",
            f"{static_m.percentile(static_m.ttft_ms, 50):.2f} ms",
            f"{dyn_m.percentile(dyn_m.ttft_ms, 50):.2f} ms",
            calc_delta(
                dyn_m.percentile(dyn_m.ttft_ms, 50),
                static_m.percentile(static_m.ttft_ms, 50),
            ),
        )
        table.add_row(
            "TTFT (P90)",
            f"{static_m.percentile(static_m.ttft_ms, 90):.2f} ms",
            f"{dyn_m.percentile(dyn_m.ttft_ms, 90):.2f} ms",
            calc_delta(
                dyn_m.percentile(dyn_m.ttft_ms, 90),
                static_m.percentile(static_m.ttft_ms, 90),
            ),
        )
        table.add_row(
            "TTFT (P99)",
            f"{static_m.percentile(static_m.ttft_ms, 99):.2f} ms",
            f"{dyn_m.percentile(dyn_m.ttft_ms, 99):.2f} ms",
            calc_delta(
                dyn_m.percentile(dyn_m.ttft_ms, 99),
                static_m.percentile(static_m.ttft_ms, 99),
            ),
        )
        table.add_section()

        table.add_row(
            "TPOT (P50)",
            f"{static_m.percentile(static_m.tpot_ms, 50):.2f} ms",
            f"{dyn_m.percentile(dyn_m.tpot_ms, 50):.2f} ms",
            calc_delta(
                dyn_m.percentile(dyn_m.tpot_ms, 50),
                static_m.percentile(static_m.tpot_ms, 50),
            ),
        )
        table.add_row(
            "TPOT (P99)",
            f"{static_m.percentile(static_m.tpot_ms, 99):.2f} ms",
            f"{dyn_m.percentile(dyn_m.tpot_ms, 99):.2f} ms",
            calc_delta(
                dyn_m.percentile(dyn_m.tpot_ms, 99),
                static_m.percentile(static_m.tpot_ms, 99),
            ),
        )

        console.print(Panel(table, expand=False))
    else:
        print(
            "\n==========================================================================="
        )
        print(" NexusCache Engine: Static vs. Dynamic Scheduler Benchmark Results")
        print(
            "==========================================================================="
        )
        print(
            f"Total Completed Requests  : Static={static_m.completed_requests} | Dynamic={dyn_m.completed_requests}"
        )
        print(
            f"Preemptions               : Static={static_m.preemption_count} | Dynamic={dyn_m.preemption_count}"
        )
        print(
            f"Prefix Cache Hits         : Static=0 | Dynamic={dyn_m.prefix_hit_count}"
        )
        print(
            f"SLA Violations            : Static={static_m.sla_violations} | Dynamic={dyn_m.sla_violations}"
        )
        print(
            f"Token Throughput (TPS)    : Static={s_tps:.2f} tok/s | Dynamic={d_tps:.2f} tok/s"
        )
        print(
            f"TTFT (P50 / P99)          : Static={static_m.percentile(static_m.ttft_ms, 50):.2f} / {static_m.percentile(static_m.ttft_ms, 99):.2f} ms"
        )
        print(
            f"                            Dynamic={dyn_m.percentile(dyn_m.ttft_ms, 50):.2f} / {dyn_m.percentile(dyn_m.ttft_ms, 99):.2f} ms"
        )
        print(
            "===========================================================================\n"
        )


# ============================================================================
# CLI Entrypoint
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NexusCache Static vs. Dynamic Scheduler Benchmark"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=300,
        help="Total synthetic requests to simulate",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=60.0,
        help="Arrival QPS (Poisson distribution)",
    )
    parser.add_argument(
        "--shared-prefix-ratio",
        type=float,
        default=0.50,
        help="Ratio of requests sharing system prompt prefix",
    )
    parser.add_argument(
        "--max-vram-blocks", type=int, default=64, help="Hardware KV-cache block budget"
    )
    parser.add_argument(
        "--block-size", type=int, default=16, help="Token block allocation quantum"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Save comparison summary to JSON file",
    )

    args = parser.parse_args()

    print(
        f"Generating synthetic workload ({args.num_requests} requests, QPS={args.request_rate}, Prefix Sharing={args.shared_prefix_ratio*100:.0f}%)..."
    )
    workload = generate_workload(
        num_requests=args.num_requests,
        shared_prefix_ratio=args.shared_prefix_ratio,
        request_rate=args.request_rate,
    )

    config = SchedulerConfig(
        max_num_batched_tokens=2048,
        max_num_seqs=64,
        max_paged_blocks=args.max_vram_blocks,
        block_size=args.block_size,
    )

    print("Running Pass 1: Static Baseline Scheduler...")
    static_metrics = run_simulation("static", config, workload)

    print("Running Pass 2: Dynamic SLA & Prefix Scheduler...")
    dynamic_metrics = run_simulation("dynamic", config, workload)

    print_results(static_metrics, dynamic_metrics)

    if args.output_json:
        summary_data = {
            "static": {
                "completed": static_metrics.completed_requests,
                "preemptions": static_metrics.preemption_count,
                "ttft_p50_ms": static_metrics.percentile(static_metrics.ttft_ms, 50),
                "ttft_p99_ms": static_metrics.percentile(static_metrics.ttft_ms, 99),
                "sla_violations": static_metrics.sla_violations,
            },
            "dynamic": {
                "completed": dynamic_metrics.completed_requests,
                "preemptions": dynamic_metrics.preemption_count,
                "prefix_hits": dynamic_metrics.prefix_hit_count,
                "ttft_p50_ms": dynamic_metrics.percentile(dynamic_metrics.ttft_ms, 50),
                "ttft_p99_ms": dynamic_metrics.percentile(dynamic_metrics.ttft_ms, 99),
                "sla_violations": dynamic_metrics.sla_violations,
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Results exported to {args.output_json}")


if __name__ == "__main__":
    main()
