"""Server status resource for vLLM MCP Server."""

import json
from typing import Any

from vllm_mcp_server.utils.vllm_client import VLLMClient, VLLMClientError


async def get_server_status() -> dict[str, Any]:
    """
    Get the current status of the vLLM server.

    Returns:
        Dictionary with server status information:
        - status: "healthy", "unhealthy", or "offline"
        - base_url: The configured vLLM base URL
        - models: List of available models (if healthy)
        - error: Error message (if any)
    """
    result: dict[str, Any] = {
        "status": "unknown",
        "base_url": "",
        "models": [],
        "error": None,
    }

    try:
        async with VLLMClient() as client:
            result["base_url"] = client.settings.base_url

            # Check health
            health = await client.health_check()
            if health.get("status") == "healthy":
                result["status"] = "healthy"

                # Get available models
                try:
                    models = await client.list_models()
                    result["models"] = [m.get("id", "unknown") for m in models]
                except VLLMClientError as e:
                    result["models_error"] = str(e)
            else:
                result["status"] = "unhealthy"
                result["error"] = f"Server returned status code: {health.get('code')}"

    except VLLMClientError as e:
        result["status"] = "offline"
        result["error"] = str(e)

    return result


async def get_server_status_text() -> str:
    """
    Get formatted server status as text.

    Returns:
        Formatted string with server status.
    """
    status = await get_server_status()

    # Status emoji
    status_emoji = {
        "healthy": "✅",
        "unhealthy": "⚠️",
        "offline": "❌",
        "unknown": "❓",
    }

    emoji = status_emoji.get(status["status"], "❓")
    
    lines = [
        f"## vLLM Server Status {emoji}",
        "",
        f"**Status:** {status['status']}",
        f"**Base URL:** {status['base_url']}",
    ]

    if status["models"]:
        lines.append(f"**Models:** {', '.join(status['models'])}")
    
    if status.get("error"):
        lines.append(f"**Error:** {status['error']}")

    if status.get("models_error"):
        lines.append(f"**Models Error:** {status['models_error']}")

    return "\n".join(lines)

