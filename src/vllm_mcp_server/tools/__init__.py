"""MCP Tools for vLLM Server."""

from vllm_mcp_server.tools.chat import handle_chat, handle_complete
from vllm_mcp_server.tools.models import get_model_info, list_models
from vllm_mcp_server.tools.server_control import (
    get_platform_info,
    get_platform_status,
    get_vllm_logs,
    list_vllm_containers,
    restart_vllm,
    start_vllm,
    stop_vllm,
)
from vllm_mcp_server.tools.benchmark import run_benchmark

__all__ = [
    "handle_chat",
    "handle_complete",
    "list_models",
    "get_model_info",
    "start_vllm",
    "stop_vllm",
    "restart_vllm",
    "list_vllm_containers",
    "get_vllm_logs",
    "get_platform_info",
    "get_platform_status",
    "run_benchmark",
]

