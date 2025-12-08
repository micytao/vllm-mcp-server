"""Tests for vLLM MCP Server resources."""

import pytest
from unittest.mock import AsyncMock, patch

from vllm_mcp_server.resources.server_status import get_server_status, get_server_status_text
from vllm_mcp_server.resources.metrics import parse_metrics, get_metrics, get_metrics_summary
from vllm_mcp_server.utils.vllm_client import VLLMConnectionError


class TestServerStatus:
    """Tests for server status resource."""

    @pytest.mark.asyncio
    async def test_get_server_status_healthy(self):
        """Test status when server is healthy."""
        with patch("vllm_mcp_server.resources.server_status.VLLMClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.settings.base_url = "http://localhost:8000"
            mock_client.health_check.return_value = {"status": "healthy", "code": 200}
            mock_client.list_models.return_value = [{"id": "test-model"}]
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await get_server_status()

            assert result["status"] == "healthy"
            assert result["base_url"] == "http://localhost:8000"
            assert "test-model" in result["models"]

    @pytest.mark.asyncio
    async def test_get_server_status_offline(self):
        """Test status when server is offline."""
        with patch("vllm_mcp_server.resources.server_status.VLLMClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.settings.base_url = "http://localhost:8000"
            mock_client.health_check.side_effect = VLLMConnectionError("Connection refused")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await get_server_status()

            assert result["status"] == "offline"
            assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    async def test_get_server_status_text_healthy(self):
        """Test formatted status text when healthy."""
        with patch("vllm_mcp_server.resources.server_status.VLLMClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.settings.base_url = "http://localhost:8000"
            mock_client.health_check.return_value = {"status": "healthy", "code": 200}
            mock_client.list_models.return_value = [{"id": "test-model"}]
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await get_server_status_text()

            assert "✅" in result
            assert "healthy" in result
            assert "test-model" in result


class TestMetrics:
    """Tests for metrics resource."""

    def test_parse_metrics_simple(self):
        """Test parsing simple metrics."""
        metrics_text = """
# HELP vllm_request_count Total requests
# TYPE vllm_request_count counter
vllm_request_count 100
vllm_token_count 5000
gpu_memory_used_bytes 8589934592
"""
        result = parse_metrics(metrics_text)

        assert "vllm_request_count" in result["requests"]
        assert result["requests"]["vllm_request_count"]["value"] == 100
        assert "vllm_token_count" in result["tokens"]
        assert result["tokens"]["vllm_token_count"]["value"] == 5000
        assert "gpu_memory_used_bytes" in result["gpu"]

    def test_parse_metrics_with_labels(self):
        """Test parsing metrics with labels."""
        metrics_text = """
vllm_request_count{model="gpt-3",status="success"} 100
"""
        result = parse_metrics(metrics_text)

        metric = result["requests"]["vllm_request_count"]
        assert metric["value"] == 100
        assert metric["labels"]["model"] == "gpt-3"
        assert metric["labels"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_get_metrics_success(self):
        """Test getting metrics successfully."""
        raw_metrics = "vllm_request_count 100"

        with patch("vllm_mcp_server.resources.metrics.VLLMClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get_metrics.return_value = raw_metrics
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await get_metrics()

            assert result["raw"] == raw_metrics
            assert result["parsed"] is not None
            assert result["error"] is None

    @pytest.mark.asyncio
    async def test_get_metrics_error(self):
        """Test getting metrics with error."""
        with patch("vllm_mcp_server.resources.metrics.VLLMClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get_metrics.side_effect = VLLMConnectionError("Connection refused")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await get_metrics()

            assert result["error"] is not None
            assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    async def test_get_metrics_summary_error(self):
        """Test metrics summary with error."""
        with patch("vllm_mcp_server.resources.metrics.VLLMClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get_metrics.side_effect = VLLMConnectionError("Connection refused")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await get_metrics_summary()

            assert "❌" in result
            assert "Error" in result

