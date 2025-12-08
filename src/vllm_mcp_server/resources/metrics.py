"""Performance metrics resource for vLLM MCP Server."""

import re
from typing import Any

from vllm_mcp_server.utils.vllm_client import VLLMClient, VLLMClientError


def parse_metrics(metrics_text: str) -> dict[str, Any]:
    """
    Parse Prometheus metrics text into a structured dictionary.

    Args:
        metrics_text: Raw Prometheus metrics text.

    Returns:
        Dictionary with parsed metrics grouped by category.
    """
    metrics: dict[str, Any] = {
        "requests": {},
        "tokens": {},
        "latency": {},
        "cache": {},
        "gpu": {},
        "queue": {},
        "other": {},
    }

    # Patterns for metric parsing
    metric_pattern = re.compile(r'^(\w+)(?:\{([^}]*)\})?\s+(\S+)$', re.MULTILINE)

    for match in metric_pattern.finditer(metrics_text):
        name = match.group(1)
        labels = match.group(2) or ""
        value = match.group(3)

        # Try to convert value to number
        try:
            if '.' in value or 'e' in value.lower():
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass

        # Categorize metrics
        metric_entry = {"value": value}
        if labels:
            # Parse labels
            label_dict = {}
            for label in labels.split(","):
                if "=" in label:
                    k, v = label.split("=", 1)
                    label_dict[k.strip()] = v.strip().strip('"')
            metric_entry["labels"] = label_dict

        # Categorize by name prefix
        if name.startswith("vllm_request") or "request" in name.lower():
            metrics["requests"][name] = metric_entry
        elif "token" in name.lower():
            metrics["tokens"][name] = metric_entry
        elif "latency" in name.lower() or "time" in name.lower() or "duration" in name.lower():
            metrics["latency"][name] = metric_entry
        elif "cache" in name.lower() or "kv" in name.lower():
            metrics["cache"][name] = metric_entry
        elif "gpu" in name.lower() or "memory" in name.lower():
            metrics["gpu"][name] = metric_entry
        elif "queue" in name.lower() or "pending" in name.lower() or "running" in name.lower():
            metrics["queue"][name] = metric_entry
        else:
            metrics["other"][name] = metric_entry

    return metrics


async def get_metrics() -> dict[str, Any]:
    """
    Get metrics from the vLLM server.

    Returns:
        Dictionary with:
        - raw: Raw metrics text
        - parsed: Parsed and categorized metrics
        - error: Error message if any
    """
    result: dict[str, Any] = {
        "raw": None,
        "parsed": None,
        "error": None,
    }

    try:
        async with VLLMClient() as client:
            raw_metrics = await client.get_metrics()
            result["raw"] = raw_metrics
            result["parsed"] = parse_metrics(raw_metrics)

    except VLLMClientError as e:
        result["error"] = str(e)

    return result


async def get_metrics_summary() -> str:
    """
    Get a formatted summary of vLLM metrics.

    Returns:
        Formatted string with key metrics.
    """
    metrics_data = await get_metrics()

    if metrics_data["error"]:
        return f"❌ Error fetching metrics: {metrics_data['error']}"

    parsed = metrics_data.get("parsed", {})
    if not parsed:
        return "No metrics available"

    lines = ["## vLLM Performance Metrics", ""]

    # Request metrics
    requests = parsed.get("requests", {})
    if requests:
        lines.append("### Requests")
        for name, data in list(requests.items())[:5]:
            lines.append(f"- **{name}:** {data['value']}")
        lines.append("")

    # Token metrics
    tokens = parsed.get("tokens", {})
    if tokens:
        lines.append("### Tokens")
        for name, data in list(tokens.items())[:5]:
            lines.append(f"- **{name}:** {data['value']}")
        lines.append("")

    # Latency metrics
    latency = parsed.get("latency", {})
    if latency:
        lines.append("### Latency")
        for name, data in list(latency.items())[:5]:
            lines.append(f"- **{name}:** {data['value']}")
        lines.append("")

    # GPU/Memory metrics
    gpu = parsed.get("gpu", {})
    if gpu:
        lines.append("### GPU / Memory")
        for name, data in list(gpu.items())[:5]:
            lines.append(f"- **{name}:** {data['value']}")
        lines.append("")

    # Queue metrics
    queue = parsed.get("queue", {})
    if queue:
        lines.append("### Queue")
        for name, data in list(queue.items())[:5]:
            lines.append(f"- **{name}:** {data['value']}")

    return "\n".join(lines)

