"""
Configuration Management Module
==========================================
Provides type-safe, validated, and environment-aware configuration schemas for
distributed LLM inference, memory management, and request scheduling.

Utilizes Pydantic v2 BaseSettings for environment variable parsing, YAML ingestion,
and strict runtime validation.
"""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("nexuscache.utils.config")


# ============================================================================
# Enums for Constrained Settings
# ============================================================================


class QuantizationType(str, Enum):
    NONE = "none"
    INT8 = "int8"
    FP8 = "fp8"
    AWQ = "awq"
    GPTQ = "gptq"


class DeviceType(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"
    ROCM = "rocm"


class SchedulingPolicy(str, Enum):
    FCFS = "fcfs"
    EDF = "edf"  # Earliest Deadline First (SLA-Aware)
    PRIORITY = "priority"


# ============================================================================
# Nested Sub-Configurations
# ============================================================================


class ModelConfig(BaseSettings):
    """Configuration options governing the LLM architecture and quantization."""

    model_config = SettingsConfigDict(frozen=True)

    model_path: str = Field(
        ...,
        description="Path to HuggingFace model checkpoint directory or repository ID.",
    )
    tokenizer_path: str | None = Field(
        default=None,
        description="Optional explicit path to tokenizer (defaults to model_path).",
    )
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE,
        description="Quantization scheme applied to model weights and KV cache.",
    )
    device: DeviceType = Field(
        default=DeviceType.CUDA,
        description="Hardware acceleration backend type.",
    )
    max_model_len: int = Field(
        default=8192,
        ge=128,
        le=131072,
        description="Maximum combined prompt + completion sequence context length.",
    )
    dtype: str = Field(
        default="bfloat16",
        description="Tensor data type (e.g., float16, bfloat16, float32).",
    )

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        valid_dtypes = {"float16", "bfloat16", "float32", "fp16", "bf16"}
        if v.lower() not in valid_dtypes:
            raise ValueError(f"Invalid dtype '{v}'. Supported options: {valid_dtypes}")
        return v.lower()

    @model_validator(mode="after")
    def resolve_tokenizer(self) -> "ModelConfig":
        if self.tokenizer_path is None:
            # Set default tokenizer_path to model_path if omitted
            object.__setattr__(self, "tokenizer_path", self.model_path)
        return self


class CacheConfig(BaseSettings):
    """Configuration options for physical VRAM Page Table and KV-Cache Memory Pool."""

    model_config = SettingsConfigDict(frozen=True)

    block_size: int = Field(
        default=16,
        description="Number of tokens managed per physical KV-Cache VRAM block.",
    )
    gpu_memory_utilization: float = Field(
        default=0.90,
        gt=0.0,
        lt=1.0,
        description="Fraction of total GPU VRAM dedicated to KV cache block allocation.",
    )
    cpu_swap_space_blocks: int = Field(
        default=2048,
        ge=0,
        description="Number of CPU host RAM blocks allocated for preempted sequence swapping.",
    )
    enable_prefix_caching: bool = Field(
        default=True,
        description="Enables Radix Trie automatic prompt prefix KV-cache reuse.",
    )
    min_prefix_match_tokens: int = Field(
        default=16,
        ge=1,
        description="Minimum prefix length required to trigger KV-cache block sharing.",
    )

    @field_validator("block_size")
    @classmethod
    def validate_block_size(cls, v: int) -> int:
        if v not in (8, 16, 32, 64):
            raise ValueError(
                "block_size must be a power-of-two block unit (8, 16, 32, or 64)."
            )
        return v


class SchedulerConfig(BaseSettings):
    """Configuration parameters for the SLA-aware dynamic iteration batching engine."""

    model_config = SettingsConfigDict(frozen=True)

    max_num_batched_tokens: int = Field(
        default=8192,
        ge=128,
        description="Maximum cumulative prefill + decode tokens per single iteration batch pass.",
    )
    max_num_seqs: int = Field(
        default=256,
        ge=1,
        le=4096,
        description="Maximum concurrent active sequences supported in running decode phase.",
    )
    policy: SchedulingPolicy = Field(
        default=SchedulingPolicy.EDF,
        description="Scheduling prioritization strategy.",
    )
    max_queue_size: int = Field(
        default=1024,
        ge=1,
        description="Maximum pending requests allowed in asyncio buffer before HTTP 429 rejection.",
    )
    default_request_timeout_s: float = Field(
        default=60.0,
        gt=0.0,
        description="Time-to-live (TTL) limit for queued requests before timeout eviction.",
    )


class RayClusterConfig(BaseSettings):
    """Configuration options for multi-GPU/multi-node Ray distributed actor clusters."""

    model_config = SettingsConfigDict(frozen=True)

    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of tensor parallel GPU worker actors per pipeline stage.",
    )
    pipeline_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of pipeline parallel stages across cluster nodes.",
    )
    num_gpus_per_worker: float = Field(
        default=1.0,
        ge=0.0,
        description="GPU fraction or count required per Ray ModelWorkerActor.",
    )
    worker_heartbeat_interval_s: float = Field(
        default=5.0,
        gt=0.0,
        description="Interval for actor pool health status pings.",
    )
    actor_max_restarts: int = Field(
        default=3,
        ge=0,
        description="Maximum restart limit before marking worker unrecoverable.",
    )


# ============================================================================
# Main Engine System Configuration
# ============================================================================


class EngineConfig(BaseSettings):
    """
    Root Configuration Object consolidating all LLM server components.
    Automatically parses environment variables with prefix 'NEXUSCACHE_'.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEXUSCACHE_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    server_host: str = Field(default="0.0.0.0", description="API Server bind IP.")
    server_port: int = Field(
        default=8000, ge=1024, le=65535, description="API Server port."
    )

    # Sub-component configurations
    model: ModelConfig
    cache: CacheConfig = Field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    ray_cluster: RayClusterConfig = Field(default_factory=RayClusterConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "EngineConfig":
        """
        Loads configuration from a YAML file, allowing environment variable overrides.
        """
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration YAML file not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                raw_data = yaml.safe_load(f) or {}

            if not isinstance(raw_data, dict):
                raise ValueError(
                    f"YAML content in {path} must be a top-level dictionary/mapping."
                )

            # Ensure all keys are strings for static type checkers
            typed_data: dict[str, Any] = {str(k): v for k, v in raw_data.items()}

            logger.info(f"Loaded configuration from file: {path.resolve()}")
            return cls(**typed_data)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML configuration at {path}: {e}")
            raise ValueError(f"Malformed YAML configuration file: {e}") from e


# ============================================================================
# Global Configuration Factory Singleton Helper
# ============================================================================

_GLOBAL_CONFIG: EngineConfig | None = None


def get_config() -> EngineConfig:
    """Retrieves current global system configuration singleton."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        raise RuntimeError(
            "Global EngineConfig has not been initialized. "
            "Call 'initialize_config(yaml_path=...)' first."
        )
    return _GLOBAL_CONFIG


def initialize_config(
    yaml_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> EngineConfig:
    """
    Initializes and freezes global configuration singleton from YAML + Env Vars + Overrides.
    """
    global _GLOBAL_CONFIG

    if yaml_path:
        config = EngineConfig.from_yaml(yaml_path)
    else:
        # Fallback to pure environment variables and defaults
        config = EngineConfig(
            model=ModelConfig(
                model_path=os.getenv(
                    "NEXUSCACHE_MODEL__MODEL_PATH", "facebook/opt-125m"
                )
            )
        )

    if overrides:
        # Re-instantiate model with explicitly passed CLI/Python dictionary overrides
        current_data = config.model_dump()

        # Deep merge dictionary helper
        def _deep_update(d: dict[str, Any], u: dict[str, Any]) -> dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = _deep_update(d[k], v)
                else:
                    d[k] = v
            return d

        updated_data = _deep_update(current_data, overrides)
        config = EngineConfig(**updated_data)

    _GLOBAL_CONFIG = config
    logger.info("Global EngineConfig initialized and validated successfully.")
    return _GLOBAL_CONFIG
