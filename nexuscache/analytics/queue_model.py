"""
NexusCache Queueing Theory & Drop-Rate Modeling Engine
Production-grade analytical engine modeling LLM request traffic via finite-capacity
queueing theory (M/M/1/K and M/G/1/K Pollaczek-Khinchine approximations).
Calculates queue wait times, blocking/drop probabilities, and active shedding thresholds.
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger("nexuscache.analytics.queue")


@dataclass
class QueueMetrics:
    """Quantitative performance snapshot from queueing model analysis."""

    arrival_rate_lambda: float  # Requests per second (λ)
    service_rate_mu: float  # Requests per second served (μ)
    traffic_intensity_rho: float  # Server utilization (ρ = λ / μ)
    queue_capacity_k: int  # Maximum system capacity (K = Queue + Active)
    blocking_probability_pb: float  # Request drop/rejection probability (P_b)
    effective_arrival_rate: float  # Admitted traffic rate λ_eff = λ * (1 - P_b)
    avg_queue_length_lq: float  # Expected number of requests waiting in queue (L_q)
    avg_system_length_l: float  # Expected number of requests in system (L)
    avg_wait_time_wq_ms: float  # Expected queue waiting time in milliseconds (W_q)
    avg_response_time_w_ms: float  # Expected total time in system in milliseconds (W)


@dataclass
class SheddingPolicy:
    """Configured shed thresholds and admission control rules."""

    max_queue_depth: int
    rejection_probability: float
    target_p99_latency_ms: float
    recommended_shed_action: bool


class FiniteQueueModel:
    """
    M/M/1/K and M/G/1/K Finite-Capacity Queueing Model.

    Models request arrivals as Poisson/Renewal processes with bounded queue depth
    to prevent VRAM exhaustion and calculate exact drop rates during traffic bursts.
    """

    def __init__(self, service_rate_mu: float, queue_capacity_k: int):
        if service_rate_mu <= 0:
            raise ValueError("Service rate (μ) must be positive.")
        if queue_capacity_k <= 0:
            raise ValueError("Queue capacity (K) must be positive.")

        self.mu = float(service_rate_mu)
        self.k = int(queue_capacity_k)

    def analyze_mm1k(self, arrival_rate_lambda: float) -> QueueMetrics:
        """
        Analyzes M/M/1/K queue performance for a given arrival rate (λ).
        Handles both stable (ρ < 1) and overloaded (ρ >= 1) traffic regimes.
        """
        lam = float(arrival_rate_lambda)
        if lam <= 0:
            return QueueMetrics(
                arrival_rate_lambda=0.0,
                service_rate_mu=self.mu,
                traffic_intensity_rho=0.0,
                queue_capacity_k=self.k,
                blocking_probability_pb=0.0,
                effective_arrival_rate=0.0,
                avg_queue_length_lq=0.0,
                avg_system_length_l=0.0,
                avg_wait_time_wq_ms=0.0,
                avg_response_time_w_ms=0.0,
            )

        rho = lam / self.mu

        # 1. Blocking Probability P_b (Loss Probability)
        if math.isclose(rho, 1.0):
            pb = 1.0 / (self.k + 1)
            1.0 / (self.k + 1)
            avg_l = self.k / 2.0
        else:
            pb = ((1.0 - rho) * (rho**self.k)) / (1.0 - (rho ** (self.k + 1)))
            (1.0 - rho) / (1.0 - (rho ** (self.k + 1)))

            # Expected number of items in system (L)
            num = rho * (
                1.0 - (self.k + 1) * (rho**self.k) + self.k * (rho ** (self.k + 1))
            )
            den = (1.0 - rho) * (1.0 - (rho ** (self.k + 1)))
            avg_l = num / den

        # 2. Effective arrival rate λ_eff (Traffic not dropped)
        lam_eff = lam * (1.0 - pb)

        # 3. Little's Law Metrics: W = L / λ_eff
        avg_w = avg_l / lam_eff if lam_eff > 0 else 0.0
        avg_wq = max(0.0, avg_w - (1.0 / self.mu))
        avg_lq = lam_eff * avg_wq

        return QueueMetrics(
            arrival_rate_lambda=lam,
            service_rate_mu=self.mu,
            traffic_intensity_rho=rho,
            queue_capacity_k=self.k,
            blocking_probability_pb=pb,
            effective_arrival_rate=lam_eff,
            avg_queue_length_lq=avg_lq,
            avg_system_length_l=avg_l,
            avg_wait_time_wq_ms=avg_wq * 1000.0,
            avg_response_time_w_ms=avg_w * 1000.0,
        )

    def analyze_mg1k_approximation(
        self,
        arrival_rate_lambda: float,
        service_variance: float,
    ) -> QueueMetrics:
        """
        Analyzes M/G/1/K queue using the Pollaczek-Khinchine (P-K) formula approximation
        for general service time distributions (e.g., variable prompt/decode lengths).
        """
        mm1k_base = self.analyze_mm1k(arrival_rate_lambda)
        if mm1k_base.arrival_rate_lambda == 0.0:
            return mm1k_base

        # Squared coefficient of variation C_v^2 = Var(S) / E[S]^2
        mean_service_time = 1.0 / self.mu
        cv_sq = service_variance / (mean_service_time**2)

        # Pollaczek-Khinchine adjustment factor for queue waiting time
        pk_factor = (1.0 + cv_sq) / 2.0
        adjusted_wq_ms = mm1k_base.avg_wait_time_wq_ms * pk_factor
        adjusted_w_ms = adjusted_wq_ms + (mean_service_time * 1000.0)

        adjusted_lq = (mm1k_base.effective_arrival_rate * adjusted_wq_ms) / 1000.0
        adjusted_l = adjusted_lq + (mm1k_base.effective_arrival_rate / self.mu)

        return QueueMetrics(
            arrival_rate_lambda=mm1k_base.arrival_rate_lambda,
            service_rate_mu=self.mu,
            traffic_intensity_rho=mm1k_base.traffic_intensity_rho,
            queue_capacity_k=self.k,
            blocking_probability_pb=mm1k_base.blocking_probability_pb,
            effective_arrival_rate=mm1k_base.effective_arrival_rate,
            avg_queue_length_lq=adjusted_lq,
            avg_system_length_l=adjusted_l,
            avg_wait_time_wq_ms=adjusted_wq_ms,
            avg_response_time_w_ms=adjusted_w_ms,
        )


class LoadSheddingOptimizer:
    """
    Computes optimal request rejection thresholds and admission control policies
    to uphold SLA bounds under severe traffic spikes.
    """

    def __init__(self, target_max_wait_ms: float = 100.0):
        self.target_max_wait_ms = target_max_wait_ms

    def evaluate_shedding_threshold(
        self,
        current_queue_depth: int,
        arrival_rate_lambda: float,
        service_rate_mu: float,
        max_capacity_k: int,
    ) -> SheddingPolicy:
        """
        Determines whether incoming requests should be shed (rejected) based on SLA targets.
        """
        model = FiniteQueueModel(
            service_rate_mu=service_rate_mu, queue_capacity_k=max_capacity_k
        )
        metrics = model.analyze_mm1k(arrival_rate_lambda)

        # Max acceptable queue depth before exceeding SLA target: Depth_max = W_target * μ
        max_acceptable_depth = int(
            math.floor((self.target_max_wait_ms / 1000.0) * service_rate_mu)
        )
        max_acceptable_depth = max(1, min(max_capacity_k, max_acceptable_depth))

        should_shed = False
        rejection_prob = 0.0

        if (
            current_queue_depth >= max_acceptable_depth
            or metrics.avg_wait_time_wq_ms > self.target_max_wait_ms
        ):
            should_shed = True
            # Probabilistic shedding factor based on excess arrival intensity
            overload_ratio = max(
                0.0, (arrival_rate_lambda - service_rate_mu) / arrival_rate_lambda
            )
            rejection_prob = min(1.0, max(0.1, overload_ratio))

        return SheddingPolicy(
            max_queue_depth=max_acceptable_depth,
            rejection_probability=rejection_prob,
            target_p99_latency_ms=self.target_max_wait_ms,
            recommended_shed_action=should_shed,
        )
