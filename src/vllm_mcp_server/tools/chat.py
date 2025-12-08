"""Chat and completion tools for vLLM MCP Server."""

from typing import Any, Optional

from mcp.types import TextContent

from vllm_mcp_server.utils.vllm_client import VLLMClient, VLLMClientError


async def handle_chat(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle chat completion request.

    Args:
        arguments: Dictionary containing:
            - messages: List of message objects with 'role' and 'content'
            - model: Optional model name to use
            - temperature: Optional temperature (0-2)
            - max_tokens: Optional maximum tokens to generate
            - stream: Whether to stream the response (default: False)

    Returns:
        List of TextContent with the assistant's response.
    """
    messages = arguments.get("messages", [])
    if not messages:
        return [TextContent(type="text", text="Error: No messages provided")]

    # Validate message format
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return [
                TextContent(
                    type="text",
                    text="Error: Each message must have 'role' and 'content' fields",
                )
            ]

    model = arguments.get("model")
    temperature = arguments.get("temperature")
    max_tokens = arguments.get("max_tokens")

    try:
        async with VLLMClient() as client:
            response = await client.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            # Extract the assistant's message
            choices = response.get("choices", [])
            if not choices:
                return [TextContent(type="text", text="Error: No response from model")]

            assistant_message = choices[0].get("message", {}).get("content", "")

            # Include usage info
            usage = response.get("usage", {})
            usage_info = ""
            if usage:
                usage_info = (
                    f"\n\n---\n"
                    f"Tokens: {usage.get('prompt_tokens', 0)} prompt + "
                    f"{usage.get('completion_tokens', 0)} completion = "
                    f"{usage.get('total_tokens', 0)} total"
                )

            return [TextContent(type="text", text=assistant_message + usage_info)]

    except VLLMClientError as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_complete(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle text completion request.

    Args:
        arguments: Dictionary containing:
            - prompt: The text prompt to complete
            - model: Optional model name to use
            - temperature: Optional temperature (0-2)
            - max_tokens: Optional maximum tokens to generate
            - stop: Optional stop sequences

    Returns:
        List of TextContent with the generated completion.
    """
    prompt = arguments.get("prompt", "")
    if not prompt:
        return [TextContent(type="text", text="Error: No prompt provided")]

    model = arguments.get("model")
    temperature = arguments.get("temperature")
    max_tokens = arguments.get("max_tokens")
    stop = arguments.get("stop")

    extra_kwargs: dict[str, Any] = {}
    if stop:
        extra_kwargs["stop"] = stop

    try:
        async with VLLMClient() as client:
            response = await client.text_completion(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra_kwargs,
            )

            # Extract the completion
            choices = response.get("choices", [])
            if not choices:
                return [TextContent(type="text", text="Error: No response from model")]

            completion_text = choices[0].get("text", "")

            # Include usage info
            usage = response.get("usage", {})
            usage_info = ""
            if usage:
                usage_info = (
                    f"\n\n---\n"
                    f"Tokens: {usage.get('prompt_tokens', 0)} prompt + "
                    f"{usage.get('completion_tokens', 0)} completion = "
                    f"{usage.get('total_tokens', 0)} total"
                )

            return [TextContent(type="text", text=completion_text + usage_info)]

    except VLLMClientError as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

