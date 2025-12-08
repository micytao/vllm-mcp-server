"""Benchmark tool using GuideLLM for vLLM MCP Server."""

import asyncio
import json
import shutil
from typing import Any

from mcp.types import TextContent

from vllm_mcp_server.utils.config import get_settings


async def run_benchmark(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Run a benchmark against the vLLM server using GuideLLM.

    Args:
        arguments: Dictionary containing:
            - target: Target URL (default: from settings)
            - model: Model to benchmark (optional)
            - rate: Request rate (requests/sec) or "sweep" for rate sweep
            - max_requests: Maximum number of requests
            - max_seconds: Maximum duration in seconds
            - data: Dataset to use ("emulated" or path to dataset)
            - output_path: Path to save results (optional)

    Returns:
        List of TextContent with benchmark results.
    """
    # Check if guidellm is available
    if not shutil.which("guidellm"):
        return [
            TextContent(
                type="text",
                text="Error: GuideLLM is not installed. Install it with:\n"
                     "```bash\n"
                     "pip install guidellm\n"
                     "```",
            )
        ]

    settings = get_settings()

    target = arguments.get("target", f"{settings.base_url}/v1")
    model = arguments.get("model", settings.model)
    rate = arguments.get("rate", "sweep")
    max_requests = arguments.get("max_requests")
    max_seconds = arguments.get("max_seconds", 120)
    data = arguments.get("data", "emulated")
    output_path = arguments.get("output_path")

    # Build command
    cmd = [
        "guidellm",
        "--target", target,
        "--rate", str(rate),
        "--max-seconds", str(max_seconds),
        "--data", data,
    ]

    if model:
        cmd.extend(["--model", model])

    if max_requests:
        cmd.extend(["--max-requests", str(max_requests)])

    if output_path:
        cmd.extend(["--output-path", output_path])

    # Run benchmark
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Stream output in real-time would be ideal, but for now we wait
        stdout, stderr = await process.communicate()

        stdout_str = stdout.decode("utf-8")
        stderr_str = stderr.decode("utf-8")

        if process.returncode != 0:
            return [
                TextContent(
                    type="text",
                    text=f"Benchmark failed with exit code {process.returncode}:\n\n"
                         f"**stderr:**\n```\n{stderr_str}\n```\n\n"
                         f"**stdout:**\n```\n{stdout_str}\n```",
                )
            ]

        # Format results
        result = "## Benchmark Results\n\n"
        result += f"**Command:** `{' '.join(cmd)}`\n\n"
        result += "**Output:**\n```\n"
        result += stdout_str
        result += "\n```"

        if output_path:
            result += f"\n\nResults saved to: `{output_path}`"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [
            TextContent(
                type="text",
                text=f"Error running benchmark: {str(e)}",
            )
        ]

