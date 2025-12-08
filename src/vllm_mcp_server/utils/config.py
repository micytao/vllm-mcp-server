"""Configuration management for vLLM MCP Server."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # vLLM Server Configuration
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    model: Optional[str] = None

    # Request Defaults
    default_temperature: float = 0.7
    default_max_tokens: int = 1024
    default_timeout: float = 60.0

    # Server Control (for container mode)
    container_runtime: Optional[str] = None  # "podman", "docker", or None for auto-detect
    docker_image: str = "vllm/vllm-openai:0.11.0"
    docker_image_macos: str = "quay.io/rh_ee_micyang/vllm-service:macos"
    docker_image_cpu: str = "quay.io/rh_ee_micyang/vllm-service:cpu"
    container_name: str = "vllm-server"
    gpu_memory_utilization: float = 0.9

    @property
    def openai_base_url(self) -> str:
        """Get the OpenAI-compatible API base URL."""
        return f"{self.base_url}/v1"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

