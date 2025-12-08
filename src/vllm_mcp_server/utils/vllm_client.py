"""OpenAI-compatible API client for vLLM."""

import asyncio
from typing import Any, AsyncIterator, Optional

import aiohttp

from vllm_mcp_server.utils.config import Settings, get_settings


class VLLMClientError(Exception):
    """Base exception for vLLM client errors."""

    pass


class VLLMConnectionError(VLLMClientError):
    """Connection error to vLLM server."""

    pass


class VLLMAPIError(VLLMClientError):
    """API error from vLLM server."""

    def __init__(self, message: str, status_code: int, response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class VLLMClient:
    """Async client for vLLM OpenAI-compatible API."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def base_url(self) -> str:
        return self.settings.openai_base_url

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.settings.default_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "VLLMClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def health_check(self) -> dict[str, Any]:
        """Check if the vLLM server is healthy."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.settings.base_url}/health",
                headers=self.headers,
            ) as response:
                if response.status == 200:
                    return {"status": "healthy", "code": 200}
                return {"status": "unhealthy", "code": response.status}
        except aiohttp.ClientConnectorError as e:
            raise VLLMConnectionError(f"Cannot connect to vLLM server: {e}") from e
        except asyncio.TimeoutError as e:
            raise VLLMConnectionError("Connection to vLLM server timed out") from e

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.base_url}/models",
                headers=self.headers,
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    raise VLLMAPIError(
                        f"Failed to list models: {response.status}",
                        response.status,
                        body,
                    )
                data = await response.json()
                return data.get("data", [])
        except aiohttp.ClientConnectorError as e:
            raise VLLMConnectionError(f"Cannot connect to vLLM server: {e}") from e

    async def get_model_info(self, model_id: str) -> Optional[dict[str, Any]]:
        """Get information about a specific model."""
        models = await self.list_models()
        for model in models:
            if model.get("id") == model_id:
                return model
        return None

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Send a chat completion request."""
        model = model or self.settings.model
        if not model:
            # Try to get the first available model
            models = await self.list_models()
            if models:
                model = models[0].get("id")
            else:
                raise VLLMAPIError("No model specified and no models available", 400)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature or self.settings.default_temperature,
            "max_tokens": max_tokens or self.settings.default_max_tokens,
            "stream": stream,
            **kwargs,
        }

        session = await self._get_session()
        try:
            if stream:
                return self._stream_chat_completion(session, payload)
            else:
                return await self._send_chat_completion(session, payload)
        except aiohttp.ClientConnectorError as e:
            raise VLLMConnectionError(f"Cannot connect to vLLM server: {e}") from e

    async def _send_chat_completion(
        self, session: aiohttp.ClientSession, payload: dict
    ) -> dict[str, Any]:
        """Send non-streaming chat completion request."""
        async with session.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise VLLMAPIError(
                    f"Chat completion failed: {response.status}",
                    response.status,
                    body,
                )
            return await response.json()

    async def _stream_chat_completion(
        self, session: aiohttp.ClientSession, payload: dict
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chat completion response."""
        import json

        async with session.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise VLLMAPIError(
                    f"Chat completion failed: {response.status}",
                    response.status,
                    body,
                )

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    async def text_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a text completion request."""
        model = model or self.settings.model
        if not model:
            models = await self.list_models()
            if models:
                model = models[0].get("id")
            else:
                raise VLLMAPIError("No model specified and no models available", 400)

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature or self.settings.default_temperature,
            "max_tokens": max_tokens or self.settings.default_max_tokens,
            "stream": stream,
            **kwargs,
        }

        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/completions",
                headers=self.headers,
                json=payload,
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    raise VLLMAPIError(
                        f"Text completion failed: {response.status}",
                        response.status,
                        body,
                    )
                return await response.json()
        except aiohttp.ClientConnectorError as e:
            raise VLLMConnectionError(f"Cannot connect to vLLM server: {e}") from e

    async def get_metrics(self) -> str:
        """Get Prometheus metrics from vLLM server."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.settings.base_url}/metrics",
                headers=self.headers,
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    raise VLLMAPIError(
                        f"Failed to get metrics: {response.status}",
                        response.status,
                        body,
                    )
                return await response.text()
        except aiohttp.ClientConnectorError as e:
            raise VLLMConnectionError(f"Cannot connect to vLLM server: {e}") from e

