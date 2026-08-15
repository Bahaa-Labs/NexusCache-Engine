import pytest
import ray
import torch

from nexuscache.server.worker import InferenceWorker


@pytest.mark.asyncio
async def test_inference_worker_ray_execution():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # 1. Instantiate Ray Actor with primitive arguments (avoiding serialization errors)
    worker = InferenceWorker.remote(
        worker_id=0,
        num_blocks=128,
        block_size=16,
        num_layers=4,
        num_heads=8,
        head_dim=64,
        dtype=torch.float16,
        device_id=0,
    )

    # 2. Initialize background continuous batching event loop
    init_ok = await worker.initialize.remote()
    assert init_ok is True

    # 3. Stream tokens asynchronously from Ray actor
    gen = worker.generate_stream.remote(
        request_id="req_001", prompt_token_ids=[101, 102, 103, 104], max_new_tokens=10
    )

    tokens = []
    async for token in gen:
        tokens.append(token)

    assert len(tokens) == 10

    # 4. Check Telemetry & KV Cache reclaiming
    stats = await worker.get_stats.remote()
    assert stats.num_active_requests == 0
    assert stats.num_allocated_blocks == 0

    await worker.shutdown.remote()
    ray.shutdown()
