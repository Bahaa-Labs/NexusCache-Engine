"""
Distributed Ray Actor Pool and Worker Lifecycle Manager
Wraps model engine workers as Ray actors to enable multi-GPU and multi-node scaling.

Features:
1. Actor Pool Lifecycle Management: Spawns and tracks workers across cluster nodes/GPUs.
2. Distributed State Synchronization: Propagates weights, configuration states, and engine commands.
3. Non-Blocking Concurrent Execution: Asynchronously dispatches and gathers Ray ObjectRefs.
4. Health Heartbeat & Recovery: Concurrently pings actors and automatically respawns dead actors.
"""

import asyncio
import logging
import time
from typing import Any, cast

import ray
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError, RayActorError, RayTaskError

logger = logging.getLogger("nexuscache.cluster.ray_actor")


@ray.remote
class ModelWorkerActor:
    """
    Ray Actor wrapping a localized model execution worker (e.g., KV-Cache engine, C++ bindings).
    """

    def __init__(self, worker_id: int, worker_config: dict[str, Any]):
        self.worker_id = worker_id
        self.config = worker_config
        self.is_initialized = False
        self._last_heartbeat = time.time()

        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initializes low-level execution contexts or hardware resources."""
        try:
            logger.info(
                f"[ModelWorkerActor-{self.worker_id}] Initializing execution engine..."
            )
            # Example: Initialize GPU context, C++ bindings, or model weights here
            self.is_initialized = True
            logger.info(
                f"[ModelWorkerActor-{self.worker_id}] Successfully initialized."
            )
        except Exception as e:
            logger.error(
                f"[ModelWorkerActor-{self.worker_id}] Initialization failed: {e}"
            )
            raise e

    def ping(self) -> dict[str, Any]:
        """Health check endpoint returning worker telemetry and status."""
        self._last_heartbeat = time.time()
        return {
            "worker_id": self.worker_id,
            "status": "HEALTHY" if self.is_initialized else "DEGRADED",
            "timestamp": self._last_heartbeat,
        }

    def execute_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Executes a distributed scheduling command or model forward pass."""
        if not self.is_initialized:
            raise RuntimeError(f"Worker {self.worker_id} is not initialized.")

        logger.debug(
            f"[ModelWorkerActor-{self.worker_id}] Executing command: {command}"
        )
        return {"status": "SUCCESS", "worker_id": self.worker_id, "result": None}

    def update_state(self, state_update: dict[str, Any]) -> bool:
        """Propagates distributed state sync updates to the worker."""
        logger.info(
            f"[ModelWorkerActor-{self.worker_id}] Synchronizing distributed state..."
        )
        return True


class RayWorkerPool:
    """
    Manages a cluster-wide pool of Ray worker actors, handling health checks,
    automatic failure recovery, and state propagation.
    """

    def __init__(
        self,
        num_workers: int,
        worker_config: dict[str, Any],
        heartbeat_interval_s: float = 5.0,
        actor_max_restarts: int = 3,
    ):
        self.num_workers = num_workers
        self.worker_config = worker_config
        self.heartbeat_interval_s = heartbeat_interval_s
        self.actor_max_restarts = actor_max_restarts

        self.actors: list[ActorHandle] = []
        self._restart_counts: dict[int, int] = {}
        self._monitor_task: asyncio.Task | None = None
        self._is_running = False

    async def start(self) -> None:
        """Spawns the actor pool across available Ray cluster nodes."""
        logger.info(f"[RayWorkerPool] Spawning {self.num_workers} worker actors...")

        actor_cls = cast(Any, ModelWorkerActor)
        for i in range(self.num_workers):
            actor = actor_cls.options(
                num_gpus=self.worker_config.get("num_gpus_per_worker", 1),
                max_restarts=self.actor_max_restarts,
                max_task_retries=2,
            ).remote(worker_id=i, worker_config=self.worker_config)

            self.actors.append(cast(ActorHandle, actor))
            self._restart_counts[i] = 0

        self._is_running = True
        self._monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("[RayWorkerPool] All workers spawned and monitoring loop active.")

    async def stop(self) -> None:
        """Gracefully shuts down the actor pool and monitoring loops."""
        self._is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("[RayWorkerPool] Terminating Ray worker actors...")
        for actor in self.actors:
            try:
                ray.kill(actor)
            except Exception:
                pass
        self.actors.clear()
        logger.info("[RayWorkerPool] Actor pool successfully stopped.")

    async def broadcast_command(
        self, command: str, payload: dict[str, Any]
    ) -> list[Any]:
        """Broadcasts a command across all active worker actors concurrently."""
        if not self.actors:
            return []

        object_refs = [
            actor.execute_command.remote(command, payload)  # type: ignore[attr-defined]
            for actor in self.actors
        ]
        try:
            # Wrap ray.get in a lambda to prevent asyncio.to_thread overload signature errors
            return await asyncio.to_thread(lambda: ray.get(object_refs))
        except Exception as e:
            logger.error(f"[RayWorkerPool] Error during command broadcast: {e}")
            raise e

    async def sync_state(self, state_update: dict[str, Any]) -> None:
        """Synchronizes engine states across all Ray workers."""
        if not self.actors:
            return

        logger.info("[RayWorkerPool] Propagating state synchronization to all workers.")
        object_refs = [
            actor.update_state.remote(state_update)  # type: ignore[attr-defined]
            for actor in self.actors
        ]
        await asyncio.to_thread(lambda: ray.get(object_refs))

    async def _check_single_worker_health(
        self, idx: int, actor: ActorHandle
    ) -> tuple[int, bool]:
        """Pings an individual actor with a strict timeout."""
        try:
            future = actor.ping.remote()  # type: ignore[attr-defined]
            await asyncio.to_thread(
                lambda: ray.get(future, timeout=self.heartbeat_interval_s)
            )
            return idx, True
        except (RayActorError, RayTaskError, GetTimeoutError, Exception) as e:
            logger.error(f"[RayWorkerPool] Health check failed for worker {idx}: {e}")
            return idx, False

    async def _health_monitor_loop(self) -> None:
        """Background coroutine that pings all worker actors concurrently."""
        while self._is_running:
            await asyncio.sleep(self.heartbeat_interval_s)

            if not self.actors:
                continue

            # Run all worker pings concurrently
            tasks = [
                self._check_single_worker_health(idx, actor)
                for idx, actor in enumerate(self.actors)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in results:
                if isinstance(res, tuple):
                    worker_id, is_healthy = res
                    if not is_healthy and self._is_running:
                        await self._recover_worker(worker_id)

    async def _recover_worker(self, worker_id: int) -> None:
        """Recovers or respawns a dead worker actor."""
        self._restart_counts[worker_id] += 1
        if self._restart_counts[worker_id] > self.actor_max_restarts:
            logger.critical(
                f"[RayWorkerPool] Worker {worker_id} exceeded max restart limits "
                f"({self.actor_max_restarts}). Manual intervention required."
            )
            return

        try:
            logger.warning(f"[RayWorkerPool] Respawning dead worker {worker_id}...")

            try:
                ray.kill(self.actors[worker_id])
            except Exception:
                pass

            actor_cls = cast(Any, ModelWorkerActor)
            new_actor = actor_cls.options(
                num_gpus=self.worker_config.get("num_gpus_per_worker", 1),
                max_restarts=self.actor_max_restarts,
            ).remote(worker_id=worker_id, worker_config=self.worker_config)

            self.actors[worker_id] = cast(ActorHandle, new_actor)
            logger.info(
                f"[RayWorkerPool] Successfully recovered and respawned worker {worker_id}."
            )
        except Exception as e:
            logger.error(f"[RayWorkerPool] Failed to recover worker {worker_id}: {e}")