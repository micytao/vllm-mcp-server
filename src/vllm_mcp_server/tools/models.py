"""Model management tools for vLLM MCP Server."""

import json
from typing import Any, Optional

from mcp.types import TextContent

from vllm_mcp_server.utils.vllm_client import VLLMClient, VLLMClientError


async def list_models() -> list[TextContent]:
    """
    List all available models on the vLLM server.

    Returns:
        List of TextContent with model information.
    """
    try:
        async with VLLMClient() as client:
            models = await client.list_models()

            if not models:
                return [TextContent(type="text", text="No models available on the vLLM server.")]

            # Format model list
            model_list = []
            for model in models:
                model_id = model.get("id", "unknown")
                owned_by = model.get("owned_by", "unknown")
                created = model.get("created", "unknown")
                model_list.append(f"- **{model_id}** (owned by: {owned_by}, created: {created})")

            result = f"## Available Models ({len(models)} total)\n\n" + "\n".join(model_list)
            return [TextContent(type="text", text=result)]

    except VLLMClientError as e:
        return [TextContent(type="text", text=f"Error listing models: {str(e)}")]


async def get_model_info(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Get detailed information about a specific model.

    Args:
        arguments: Dictionary containing:
            - model_id: The ID of the model to get info for

    Returns:
        List of TextContent with detailed model information.
    """
    model_id = arguments.get("model_id")
    if not model_id:
        return [TextContent(type="text", text="Error: No model_id provided")]

    try:
        async with VLLMClient() as client:
            model_info = await client.get_model_info(model_id)

            if not model_info:
                return [
                    TextContent(type="text", text=f"Model '{model_id}' not found on the server.")
                ]

            # Format model info
            result = f"## Model: {model_id}\n\n"
            result += "```json\n"
            result += json.dumps(model_info, indent=2)
            result += "\n```"

            return [TextContent(type="text", text=result)]

    except VLLMClientError as e:
        return [TextContent(type="text", text=f"Error getting model info: {str(e)}")]

