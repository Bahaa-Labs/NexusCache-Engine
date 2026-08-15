"""
Unit Tests for Queueing Theory & Drop-Rate Modeling Engine
"""

from nexuscache.analytics.queue_model import FiniteQueueModel, LoadSheddingOptimizer


class TestQueueModel:

    def test_mm1k_stable_queue(self):
        """Verify M/M/1/K metrics for stable traffic (λ = 80, μ = 100, K = 20)."""
        model = FiniteQueueModel(service_rate_mu=100.0, queue_capacity_k=20)
        metrics = model.analyze_mm1k(arrival_rate_lambda=80.0)

        assert metrics.traffic_intensity_rho == 0.8
        assert metrics.blocking_probability_pb < 0.05
        assert metrics.effective_arrival_rate > 75.0
        assert metrics.avg_wait_time_wq_ms > 0.0

    def test_mm1k_overloaded_queue(self):
        """Verify M/M/1/K blocking probability during heavy overload (λ = 200, μ = 100)."""
        model = FiniteQueueModel(service_rate_mu=100.0, queue_capacity_k=10)
        metrics = model.analyze_mm1k(arrival_rate_lambda=200.0)

        assert metrics.traffic_intensity_rho == 2.0
        assert metrics.blocking_probability_pb > 0.40  # Significant drop rate required
        assert metrics.effective_arrival_rate <= 100.0

    def test_mg1k_variance_impact(self):
        """Verify M/G/1/K wait times increase when service variance increases."""
        model = FiniteQueueModel(service_rate_mu=100.0, queue_capacity_k=20)

        # Low variance vs high variance service
        m_low = model.analyze_mg1k_approximation(
            arrival_rate_lambda=80.0, service_variance=0.00001
        )
        m_high = model.analyze_mg1k_approximation(
            arrival_rate_lambda=80.0, service_variance=0.001
        )

        assert m_high.avg_wait_time_wq_ms > m_low.avg_wait_time_wq_ms

    def test_load_shedding_policy_trigger(self):
        """Verify load shedding optimizer triggers shedding when queue depth violates SLA."""
        optimizer = LoadSheddingOptimizer(target_max_wait_ms=50.0)  # 50ms SLA

        # Under low load, no shedding
        policy_normal = optimizer.evaluate_shedding_threshold(
            current_queue_depth=2,
            arrival_rate_lambda=50.0,
            service_rate_mu=100.0,
            max_capacity_k=20,
        )
        assert not policy_normal.recommended_shed_action

        # Under severe spike, shedding recommended
        policy_spike = optimizer.evaluate_shedding_threshold(
            current_queue_depth=15,
            arrival_rate_lambda=250.0,
            service_rate_mu=100.0,
            max_capacity_k=20,
        )
        assert policy_spike.recommended_shed_action
        assert policy_spike.rejection_probability > 0.0
