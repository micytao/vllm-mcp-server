#!/usr/bin/env python3
"""vLLM MCP Server - Expose vLLM capabilities to MCP clients."""

import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    Resource,
    TextContent,
    Tool,
)

from vllm_mcp_server.prompts.system_prompts import PROMPTS, get_prompt, list_prompts
from vllm_mcp_server.resources.metrics import get_metrics_summary
from vllm_mcp_server.resources.server_status import get_server_status_text
from vllm_mcp_server.tools.benchmark import run_benchmark
from vllm_mcp_server.tools.chat import handle_chat, handle_complete
from vllm_mcp_server.tools.models import get_model_info, list_models
from vllm_mcp_server.tools.server_control import (
    get_platform_status,
    get_vllm_logs,
    list_vllm_containers,
    restart_vllm,
    start_vllm,
    stop_vllm,
)
from vllm_mcp_server.utils.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vllm-mcp-server")

# Create the MCP server
app = Server("vllm-mcp-server")


# =============================================================================
# Tool Definitions
# =============================================================================


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        # P0: Core chat/completion tools
        Tool(
            name="vllm_chat",
            description="Send a chat message to the vLLM server. Supports multi-turn conversations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "List of messages in the conversation",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["system", "user", "assistant"],
                                    "description": "The role of the message sender",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The content of the message",
                                },
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (optional, uses default if not specified)",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature (0-2)",
                        "default": 0.7,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate",
                        "default": 1024,
                    },
                },
                "required": ["messages"],
            },
        ),
        Tool(
            name="vllm_complete",
            description="Generate text completion using vLLM. Good for code completion and text generation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to complete",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (optional)",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature (0-2)",
                        "default": 0.7,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate",
                        "default": 1024,
                    },
                    "stop": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Stop sequences",
                    },
                },
                "required": ["prompt"],
            },
        ),
        # P1: Model management tools
        Tool(
            name="list_models",
            description="List all available models on the vLLM server",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_model_info",
            description="Get detailed information about a specific model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "The ID of the model to get info for",
                    },
                },
                "required": ["model_id"],
            },
        ),
        # P2: Status tool
        Tool(
            name="vllm_status",
            description="Check the health and status of the vLLM server",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        # P3: Server control tools (platform-aware)
        Tool(
            name="start_vllm",
            description="Start a vLLM server in a Docker container. Automatically detects platform (Linux/macOS/Windows) and GPU availability.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "HuggingFace model ID to serve (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')",
                    },
                    "port": {
                        "type": "integer",
                        "description": "Port to expose",
                        "default": 8000,
                    },
                    "gpu_memory_utilization": {
                        "type": "number",
                        "description": "GPU memory fraction (0-1), only used when GPU is available",
                        "default": 0.9,
                    },
                    "cpu_only": {
                        "type": "boolean",
                        "description": "Force CPU mode even if GPU is available",
                        "default": False,
                    },
                    "tensor_parallel_size": {
                        "type": "integer",
                        "description": "Number of GPUs for tensor parallelism",
                        "default": 1,
                    },
                    "max_model_len": {
                        "type": "integer",
                        "description": "Maximum model context length (optional, uses model default)",
                    },
                    "dtype": {
                        "type": "string",
                        "description": "Data type: auto, float16, bfloat16, float32",
                        "enum": ["auto", "float16", "bfloat16", "float32"],
                        "default": "auto",
                    },
                    "container_name": {
                        "type": "string",
                        "description": "Name for the Docker container",
                    },
                    "extra_args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional vLLM command-line arguments",
                    },
                },
                "required": ["model"],
            },
        ),
        Tool(
            name="stop_vllm",
            description="Stop a running vLLM Docker container",
            inputSchema={
                "type": "object",
                "properties": {
                    "container_name": {
                        "type": "string",
                        "description": "Name of the container to stop",
                    },
                    "remove": {
                        "type": "boolean",
                        "description": "Whether to remove the container after stopping",
                        "default": True,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Seconds to wait before force killing",
                        "default": 10,
                    },
                },
            },
        ),
        Tool(
            name="restart_vllm",
            description="Restart a vLLM Docker container",
            inputSchema={
                "type": "object",
                "properties": {
                    "container_name": {
                        "type": "string",
                        "description": "Name of the container to restart",
                    },
                },
            },
        ),
        Tool(
            name="list_vllm_containers",
            description="List all vLLM Docker containers",
            inputSchema={
                "type": "object",
                "properties": {
                    "all": {
                        "type": "boolean",
                        "description": "Show all containers including stopped ones",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="get_vllm_logs",
            description="Get logs from a vLLM container to check loading progress or errors",
            inputSchema={
                "type": "object",
                "properties": {
                    "container_name": {
                        "type": "string",
                        "description": "Name of the container",
                    },
                    "tail": {
                        "type": "integer",
                        "description": "Number of log lines to show",
                        "default": 50,
                    },
                },
            },
        ),
        Tool(
            name="get_platform_status",
            description="Get platform information including Docker and GPU availability",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        # P3: Benchmark tool
        Tool(
            name="run_benchmark",
            description="Run a performance benchmark against the vLLM server using GuideLLM",
            inputSchema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Target URL (default: from settings)",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to benchmark",
                    },
                    "rate": {
                        "type": "string",
                        "description": "Request rate (requests/sec) or 'sweep'",
                        "default": "sweep",
                    },
                    "max_requests": {
                        "type": "integer",
                        "description": "Maximum number of requests",
                    },
                    "max_seconds": {
                        "type": "integer",
                        "description": "Maximum duration in seconds",
                        "default": 120,
                    },
                    "data": {
                        "type": "string",
                        "description": "Dataset ('emulated' or path)",
                        "default": "emulated",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save results",
                    },
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    try:
        if name == "vllm_chat":
            return await handle_chat(arguments)
        elif name == "vllm_complete":
            return await handle_complete(arguments)
        elif name == "list_models":
            return await list_models()
        elif name == "get_model_info":
            return await get_model_info(arguments)
        elif name == "vllm_status":
            status_text = await get_server_status_text()
            return [TextContent(type="text", text=status_text)]
        elif name == "start_vllm":
            return await start_vllm(arguments)
        elif name == "stop_vllm":
            return await stop_vllm(arguments)
        elif name == "restart_vllm":
            return await restart_vllm(arguments)
        elif name == "list_vllm_containers":
            return await list_vllm_containers(arguments)
        elif name == "get_vllm_logs":
            return await get_vllm_logs(arguments)
        elif name == "get_platform_status":
            return await get_platform_status(arguments)
        elif name == "run_benchmark":
            return await run_benchmark(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# =============================================================================
# Resource Definitions
# =============================================================================


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    settings = get_settings()
    return [
        Resource(
            uri="vllm://status",
            name="vLLM Server Status",
            description="Current status and health of the vLLM server",
            mimeType="text/plain",
        ),
        Resource(
            uri="vllm://metrics",
            name="vLLM Performance Metrics",
            description="Performance metrics from the vLLM server",
            mimeType="text/plain",
        ),
        Resource(
            uri="vllm://config",
            name="vLLM MCP Configuration",
            description="Current configuration settings",
            mimeType="text/plain",
        ),
        Resource(
            uri="vllm://platform",
            name="Platform Information",
            description="Platform, Docker, and GPU status information",
            mimeType="text/plain",
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource by URI."""
    logger.info(f"Resource read: {uri}")

    if uri == "vllm://status":
        return await get_server_status_text()
    elif uri == "vllm://metrics":
        return await get_metrics_summary()
    elif uri == "vllm://config":
        settings = get_settings()
        return f"""## vLLM MCP Configuration

**Base URL:** {settings.base_url}
**Model:** {settings.model or '(auto-detect)'}
**Default Temperature:** {settings.default_temperature}
**Default Max Tokens:** {settings.default_max_tokens}
**Timeout:** {settings.default_timeout}s
**Docker Image:** {settings.docker_image}
"""
    elif uri == "vllm://platform":
        result = await get_platform_status({})
        return result[0].text
    else:
        raise ValueError(f"Unknown resource: {uri}")


# =============================================================================
# Prompt Definitions
# =============================================================================


@app.list_prompts()
async def list_prompts_handler() -> list[Prompt]:
    """List available prompts."""
    return [
        Prompt(
            name=prompt_id,
            description=prompt_data["description"],
            arguments=[
                PromptArgument(
                    name="user_message",
                    description="Your message or question",
                    required=True,
                ),
            ],
        )
        for prompt_id, prompt_data in PROMPTS.items()
    ]


@app.get_prompt()
async def get_prompt_handler(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    """Get a specific prompt."""
    logger.info(f"Prompt requested: {name}")

    prompt_data = get_prompt(name)
    if not prompt_data:
        raise ValueError(f"Unknown prompt: {name}")

    user_message = (arguments or {}).get("user_message", "")

    return GetPromptResult(
        description=prompt_data["description"],
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"{prompt_data['content']}\n\n---\n\n{user_message}",
                ),
            ),
        ],
    )


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_server():
    """Run the MCP server."""
    logger.info("Starting vLLM MCP Server...")
    settings = get_settings()
    logger.info(f"Configured vLLM base URL: {settings.base_url}")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def main():
    """Main entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()

